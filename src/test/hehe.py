import json
import torch
from llama_index.core import Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pyvi import ViTokenizer
import numpy as np
from typing import List, Dict, Any
from llama_index.core.vector_stores import SimpleVectorStore
import weaviate
from weaviate.classes.init import Auth

# Import the Weaviate vector store
from llama_index.vector_stores.weaviate import WeaviateVectorStore

# Configuration constants
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "dangvantuan/vietnamese-document-embedding"
DATA_COLLECTION = "ND168"

# Weaviate configuration
WEAVIATE_URL = "https://eypxka08rk6gbnyt57idzq.c0.us-west3.gcp.weaviate.cloud"
WEAVIATE_API_KEY = "jWgkVYQprMDiOuvTh56iLqRK5oqITcxN27wJ"

# Improved chunking parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

def load_and_preprocess_data(file_path: str) -> List[Document]:
    """
    Load JSON data and convert to Document objects with enhanced metadata
    """
    with open(file_path) as f:
        data = json.load(f)
    
    documents = []
    
    for item in data:
        category = item.get("category", "")
        fine = item.get("mức_phạt", "")
        
        # Process each violation individually for better retrieval
        for violation in item.get("nội_dung", []):
            doc_text = f"Loại phương tiện: {category}\nMức phạt: {fine}\nNội dung vi phạm: {violation}"
            
            # Add structured metadata for better filtering
            metadata = {
                "category": category,
                "fine_amount": fine,
                "violation_type": categorize_violation(violation)
            }
            
            documents.append(Document(text=doc_text, metadata=metadata))
    
    return documents

def categorize_violation(violation_text: str) -> str:
    """
    Categorize violations based on keywords for better filtering
    
    Args:
        violation_text (str): The text of the violation in Vietnamese
        
    Returns:
        str: The category of the violation
    """
    violation_text = violation_text.lower()
    
    # Define more comprehensive violation categories with relevant keywords
    categories = {
        # Speed violations
        "tốc_độ": [
            "tốc độ", "km/h", "chạy quá", "vượt quá", "tốc độ tối thiểu", 
            "chạy dưới", "tốc độ thấp", "đuổi nhau"
        ],
        
        # Parking and stopping violations
        "đỗ_dừng_xe": [
            "dừng xe", "đỗ xe", "đậu xe", "bên trái đường", "trên cầu", 
            "vạch kẻ đường", "lề đường", "vỉa hè", "đường cao tốc", 
            "nơi giao nhau", "phạm vi", "mét", "cách lề đường", "bánh xe",
            "đỗ ngược chiều", "nơi cấm đỗ", "nơi cấm dừng"
        ],
        
        # Documentation and license violations
        "giấy_tờ": [
            "giấy phép", "chứng nhận", "đăng ký", "kiểm định", 
            "bảo hiểm", "không mang theo", "hết hạn", "quá thời hạn",
            "tẩy xóa", "không hợp lệ", "không phù hợp", "trừ hết điểm",
            "không do cơ quan có thẩm quyền cấp"
        ],
        
        # DUI/alcohol violations
        "nồng_độ_cồn": [
            "nồng độ cồn", "cồn", "trong máu", "hơi thở", "miligam", 
            "mililít", "không chấp hành yêu cầu kiểm tra"
        ],
        
        # Drug-related violations
        "ma_túy": [
            "ma túy", "chất kích thích", "pháp luật cấm sử dụng",
            "yêu cầu kiểm tra", "cơ thể có chất"
        ],
        
        # Safety equipment and precautions
        "an_toàn": [
            "dây đai", "an toàn", "khoảng cách", "thiết bị", "giám sát",
            "đèn khẩn cấp", "biển cảnh báo", "thiết bị thoát hiểm",
            "chỗ ngồi", "ghế", "khoang hành lý", "chằng buộc", "nối chắc chắn"
        ],
        
        # Light and signal violations
        "đèn_tín_hiệu": [
            "đèn", "chiếu sáng", "còi", "tín hiệu", "đèn hiệu", 
            "báo hiệu", "đèn tín hiệu", "đèn chiếu xa", "đèn sương mù",
            "đèn khẩn cấp", "hiệu lệnh"
        ],
        
        # Lane and direction violations
        "làn_đường": [
            "làn đường", "chuyển làn", "phần đường", "vạch kẻ đường",
            "làn cùng chiều", "làn ngược chiều", "ngược chiều", "đường một chiều",
            "đường cao tốc", "dải phân cách", "bên phải", "bên trái"
        ],
        
        # Child safety violations
        "trẻ_em": [
            "trẻ em", "trẻ nhỏ", "dưới 10 tuổi", "chiều cao dưới", 
            "1,35 mét", "mầm non", "học sinh", "thiết bị an toàn"
        ],
        
        # Vehicle technical violations
        "kỹ_thuật_xe": [
            "kính chắn gió", "thiết bị giám sát", "biển số", "gương chiếu hậu", 
            "bánh lốp", "thùng xe", "hệ thống hãm", "hệ thống phanh", 
            "thiết bị chữa cháy", "kích thước", "niên hạn", "cải tạo"
        ],
        
        # Environmental violations
        "môi_trường": [
            "khói", "bụi", "ô nhiễm", "rơi vãi", "phế thải", "rác", 
            "đổ trái phép", "vệ sinh", "mùi hôi", "giảm thanh"
        ],
        
        # Wrong-way driving and turning
        "lùi_quay_đầu": [
            "lùi xe", "quay đầu", "đảo chiều", "quay xe", "điểm quay", "đường cong",
            "tầm nhìn bị che khuất", "tại nơi có biển báo"
        ],
        
        # Overtaking violations
        "vượt_xe": [
            "vượt xe", "không được vượt", "vượt bên phải", "nơi cấm vượt", 
            "không có tín hiệu", "không nhường đường", "vượt không an toàn"
        ],
        
        # Traffic signals and signs
        "biển_báo": [
            "không chấp hành", "biển báo hiệu", "hiệu lệnh", "đèn tín hiệu", 
            "người điều khiển giao thông", "người kiểm soát giao thông",
            "cấm đi vào", "cấm quay đầu", "cấm rẽ", "khu vực cấm"
        ],
        
        # Passenger transport violations
        "chở_người": [
            "chở quá số người", "chở người trên mui", "chở người trên thùng xe", 
            "trong khoang hành lý", "ngoài thành xe", "buồng lái", "chở người"
        ],
        
        # Cargo transport violations
        "chở_hàng": [
            "chở hàng", "vượt trọng tải", "quá tải", "vượt kích thước",
            "làm lệch xe", "rơ moóc", "không chốt", "không đóng cố định",
            "siêu trường", "siêu trọng", "hàng nguy hiểm"
        ],
        
        # Mobile phone and distracted driving
        "điện_thoại": [
            "điện thoại", "thiết bị điện tử", "dùng tay", "khi điều khiển",
            "mất tập trung", "không quan sát"
        ],
        
        # Intersection violations
        "giao_nhau": [
            "đường giao nhau", "giao lộ", "giao cắt", "ngã ba", "ngã tư",
            "vòng xuyến", "đường ưu tiên", "đường không ưu tiên", "đường nhánh",
            "đường chính", "nhường đường"
        ],
        
        # Reckless driving
        "lái_xe_nguy_hiểm": [
            "lạng lách", "đánh võng", "dùng chân", "vô lăng", "thiếu chú ý",
            "gây tai nạn", "không giữ khoảng cách"
        ],
        
        # Emergency vehicle violations
        "xe_ưu_tiên": [
            "xe ưu tiên", "xe được quyền ưu tiên", "thiết bị phát tín hiệu", 
            "xe cứu thương", "xe cứu hỏa", "xe cứu hộ", "xe công an", "xe quân sự"
        ],
        
        # Accidents and crash handling
        "tai_nạn": [
            "tai nạn", "không dừng ngay phương tiện", "không giữ nguyên hiện trường",
            "không trợ giúp người bị nạn", "không ở lại", "không trình báo"
        ],
        
        # Highway-specific violations
        "đường_cao_tốc": [
            "đường cao tốc", "vào cao tốc", "ra cao tốc", "làn dừng xe khẩn cấp",
            "đón trả khách", "đi ngược chiều", "đỗ xe trên cao tốc"
        ]
    }
    
    # Check for violations in each category
    matched_categories = []
    for category, keywords in categories.items():
        if any(keyword in violation_text for keyword in keywords):
            matched_categories.append(category)
    
    # If multiple categories match, return the most specific one or combine them
    if len(matched_categories) > 1:
        # Prioritize certain categories
        priority_order = [
            "nồng_độ_cồn", "ma_túy", "trẻ_em", "tai_nạn", "lái_xe_nguy_hiểm",
            "tốc_độ", "đường_cao_tốc", "chở_hàng", "chở_người", "đỗ_dừng_xe"
        ]
        
        for priority in priority_order:
            if priority in matched_categories:
                return priority
        
        # If no priority match, return the first one
        return matched_categories[0]
    
    # If only one category matches, return it
    elif len(matched_categories) == 1:
        return matched_categories[0]
    
    # Default category if no match found
    return "khác"

def create_optimized_chunker() -> SentenceSplitter:
    """
    Create an optimized document chunker for Vietnamese text
    """
    return SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separator=".",  # Better for Vietnamese sentences
        include_metadata=True,
        include_prev_next_rel=True,  # Keep relationships for context
    )

def tokenize_vietnamese_text(nodes: List[Any]) -> List[Any]:
    """
    Apply Vietnamese tokenization to improve embedding quality
    """
    for node in nodes:
        # Save original text for display
        node.metadata["original_text"] = node.text
        
        # Tokenize Vietnamese text for better embedding
        tokenized_text = ViTokenizer.tokenize(node.text.lower())
        node.text = tokenized_text
        
        # Exclude original text from embedding to save tokens
        node.excluded_embed_metadata_keys.append("original_text")
        node.excluded_llm_metadata_keys.append("original_text")
    
    return nodes

def create_embedding_model() -> HuggingFaceEmbedding:
    """
    Create and configure the Vietnamese embedding model
    """
    return HuggingFaceEmbedding(
        model_name=MODEL_NAME,
        max_length=8192,
        device=DEVICE,
        trust_remote_code=True
    )

def setup_weaviate_client():
    """
    Set up and connect to Weaviate cloud
    """
    try:
        # Connect to Weaviate cloud instance
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL,
            auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
        )
        print(f"Connected to Weaviate cloud at {WEAVIATE_URL}")
        
        # Check if our schema already exists
        schema = client.schema.get()
        existing_classes = [cls["class"] for cls in schema["classes"]] if "classes" in schema and schema["classes"] else []
        
        # Create schema if it doesn't exist
        if DATA_COLLECTION not in existing_classes:
            print(f"Class {DATA_COLLECTION} not found, creating schema...")
            schema = {
                "classes": [
                    {
                        "class": DATA_COLLECTION,
                        "description": "Traffic violations for Vietnamese vehicles",
                        "properties": [
                            {
                                "name": "text",
                                "dataType": ["text"],
                                "description": "The text content of the violation"
                            },
                            {
                                "name": "original_text",
                                "dataType": ["text"],
                                "description": "Original non-tokenized text"
                            },
                            {
                                "name": "category",
                                "dataType": ["string"],
                                "description": "Vehicle category"
                            },
                            {
                                "name": "fine_amount",
                                "dataType": ["string"],
                                "description": "Amount of fine"
                            },
                            {
                                "name": "violation_type",
                                "dataType": ["string"],
                                "description": "Type of violation"
                            }
                        ]
                    }
                ]
            }
            
            client.schema.create(schema)
            print(f"Created schema for class {DATA_COLLECTION}")
        else:
            print(f"Class {DATA_COLLECTION} already exists")
        
        return client
    except Exception as e:
        print(f"Error connecting to Weaviate: {e}")
        raise

def build_optimized_index(nodes: List[Any], embed_model: HuggingFaceEmbedding) -> VectorStoreIndex:
    """
    Build and optimize the vector index using Weaviate storage
    """
    # Configure global settings
    Settings.embed_model = embed_model
    Settings.chunk_size = CHUNK_SIZE
    
    try:
        # Set up Weaviate client
        client = setup_weaviate_client()
        
        # Set up Weaviate vector store
        vector_store = WeaviateVectorStore(
            weaviate_client=client,
            index_name=DATA_COLLECTION,
            text_key="text",
        )
        
        # Create storage context with Weaviate
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create index with optimized settings
        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=True
        )
        
        print(f"Successfully built and stored index in Weaviate class: {DATA_COLLECTION}")
        return index
    
    except Exception as e:
        print(f"Error building index: {e}")
        # Fallback to simple vector store if Weaviate fails
        print("Falling back to SimpleVectorStore...")
        vector_store = SimpleVectorStore()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=True
        )
        return index

def create_optimized_retriever(index: VectorStoreIndex):
    """
    Create an optimized retriever for better search results
    """
    return index.as_retriever(
        similarity_top_k=7,  # Retrieve more candidates
    )

def perform_metadata_filtering(nodes, query):
    """
    Add comprehensive metadata-based filtering to improve relevance
    """
    query_lower = query.lower()
    
    # Detect violation types in query with expanded keywords
    violation_types = []
    
    # Child safety violations
    if any(keyword in query_lower for keyword in ["trẻ em", "trẻ nhỏ", "1,35 mét", "mầm non", "học sinh", 
                                                "dưới 10 tuổi", "chiều cao", "ghế trẻ em"]):
        violation_types.append("trẻ_em")
    
    # Speed violations
    if any(keyword in query_lower for keyword in ["tốc độ", "km/h", "chạy quá", "vượt quá tốc độ", 
                                                "tốc độ tối đa", "tốc độ tối thiểu", "chạy nhanh"]):
        violation_types.append("tốc_độ")
    
    # Alcohol violations
    if any(keyword in query_lower for keyword in ["nồng độ cồn", "cồn", "rượu", "bia", "miligam", 
                                                "nồng độ trong máu", "hơi thở", "say xỉn"]):
        violation_types.append("nồng_độ_cồn")
    
    # Drug violations
    if any(keyword in query_lower for keyword in ["ma túy", "chất kích thích", "chất gây nghiện", 
                                                "chất ma túy", "dương tính"]):
        violation_types.append("ma_túy")
    
    # Parking violations
    if any(keyword in query_lower for keyword in ["đỗ xe", "đậu xe", "dừng xe", "đỗ trái phép", 
                                                "đỗ sai quy định", "vạch kẻ đường", "lề đường", "vỉa hè"]):
        violation_types.append("đỗ_dừng_xe")
    
    # Document violations
    if any(keyword in query_lower for keyword in ["giấy phép", "chứng nhận", "đăng ký", "đăng kiểm", 
                                                "bảo hiểm", "giấy tờ", "không mang theo", "hết hạn"]):
        violation_types.append("giấy_tờ")
    
    # Highway violations
    if any(keyword in query_lower for keyword in ["đường cao tốc", "cao tốc", "làn khẩn cấp", 
                                                "vào cao tốc", "ra cao tốc"]):
        violation_types.append("đường_cao_tốc")
    
    # Traffic signals violations
    if any(keyword in query_lower for keyword in ["đèn tín hiệu", "đèn giao thông", "đèn đỏ", 
                                                "vượt đèn", "không chấp hành", "biển báo"]):
        violation_types.append("biển_báo")
    
    # Lane violations
    if any(keyword in query_lower for keyword in ["làn đường", "chuyển làn", "làn xe", "phần đường", 
                                                "vượt làn", "đi sai làn"]):
        violation_types.append("làn_đường")
    
    # Passenger violations
    if any(keyword in query_lower for keyword in ["chở người", "quá số người", "chở quá", "người ngồi", 
                                                "chở trên mui", "thùng xe"]):
        violation_types.append("chở_người")
    
    # Cargo violations
    if any(keyword in query_lower for keyword in ["chở hàng", "trọng tải", "quá tải", "hàng hóa", 
                                                "vượt kích thước", "kích thước giới hạn"]):
        violation_types.append("chở_hàng")
    
    # Phone usage violations
    if any(keyword in query_lower for keyword in ["điện thoại", "nghe điện thoại", "nhắn tin", 
                                                "sử dụng điện thoại", "thiết bị điện tử"]):
        violation_types.append("điện_thoại")
    
    # Dangerous driving
    if any(keyword in query_lower for keyword in ["lạng lách", "đánh võng", "nguy hiểm", "liều lĩnh", 
                                                "không giữ khoảng cách", "vô lăng"]):
        violation_types.append("lái_xe_nguy_hiểm")
    
    # Accident related
    if any(keyword in query_lower for keyword in ["tai nạn", "va chạm", "không dừng lại", 
                                                "bỏ chạy", "không trợ giúp", "trốn khỏi"]):
        violation_types.append("tai_nạn")
    
    # Traffic lights & signs
    if any(keyword in query_lower for keyword in ["biển báo", "biển hiệu", "không chấm hành", 
                                                "hiệu lệnh", "đèn tín hiệu"]):
        violation_types.append("biển_báo")
    
    # Overtaking violations
    if any(keyword in query_lower for keyword in ["vượt xe", "không được vượt", "vượt bên phải", 
                                                "vượt ẩu", "vượt trái phép"]):
        violation_types.append("vượt_xe")
    
    # Safety equipment violations
    if any(keyword in query_lower for keyword in ["dây an toàn", "mũ bảo hiểm", "thiết bị an toàn", 
                                                "không đội mũ", "không thắt dây"]):
        violation_types.append("an_toàn")
    
    # Technical violations
    if any(keyword in query_lower for keyword in ["kính chắn gió", "biển số", "gương chiếu hậu", 
                                                "bánh lốp", "hệ thống phanh", "thiết bị"]):
        violation_types.append("kỹ_thuật_xe")
    
    # Environmental violations
    if any(keyword in query_lower for keyword in ["khói", "bụi", "ô nhiễm", "rơi vãi", "tiếng ồn", 
                                                "xả thải", "môi trường"]):
        violation_types.append("môi_trường")
    
    # Backing up and turning around violations
    if any(keyword in query_lower for keyword in ["lùi xe", "quay đầu", "đảo chiều", "quay xe", 
                                                "điểm quay đầu", "cấm quay đầu"]):
        violation_types.append("lùi_quay_đầu")
    
    # Intersection violations
    if any(keyword in query_lower for keyword in ["giao nhau", "ngã ba", "ngã tư", "giao lộ", 
                                                "vòng xuyến", "đường ưu tiên"]):
        violation_types.append("giao_nhau")
    
    # Emergency vehicle violations
    if any(keyword in query_lower for keyword in ["xe ưu tiên", "xe cứu thương", "xe cứu hỏa", 
                                                "xe công an", "xe quân sự", "không nhường đường"]):
        violation_types.append("xe_ưu_tiên")
    
    # Remove duplicates
    violation_types = list(set(violation_types))
    
    # If violation types detected, filter results
    if violation_types:
        print(f"Detected violation types: {violation_types}")
        filtered_nodes = [
            node for node in nodes 
            if "violation_type" in node.metadata and node.metadata["violation_type"] in violation_types
        ]
        if filtered_nodes:
            print(f"Filtered results from {len(nodes)} to {len(filtered_nodes)} nodes")
            return filtered_nodes
    
    print("No specific violation type detected or filtering yielded no results. Using all results.")
    return nodes

def rank_results(nodes, query):
    """
    Rank results based on relevance to query
    """
    query_tokens = set(ViTokenizer.tokenize(query.lower()).split())
    
    scored_nodes = []
    for node in nodes:
        # Calculate term overlap
        node_tokens = set(node.text.split())
        overlap = len(query_tokens.intersection(node_tokens))
        
        # Calculate score based on vector similarity and term overlap
        combined_score = (0.7 * node.score) + (0.3 * (overlap / max(1, len(query_tokens))))
        
        scored_nodes.append((node, combined_score))
    
    # Sort by combined score
    scored_nodes.sort(key=lambda x: x[1], reverse=True)
    
    return [node for node, _ in scored_nodes]

def search_violations(retriever, query, top_k=5):
    """
    Perform optimized search with metadata filtering and ranking
    """
    # Tokenize query for better matching
    tokenized_query = ViTokenizer.tokenize(query.lower())
    
    # Retrieve initial results
    results = retriever.retrieve(tokenized_query)
    
    # Apply metadata filtering
    filtered_results = perform_metadata_filtering(results, query)
    
    # Rank results by relevance
    ranked_results = rank_results(filtered_results, query)
    
    # Return top results
    return ranked_results[:top_k]

def main():
    try:
        # Load and process data
        documents = load_and_preprocess_data("/home/ltnga/LawVN-Instructction-Gen/src/data/car.json")
        print(f"Loaded {len(documents)} documents")
        
        # Create chunker and process nodes
        chunker = create_optimized_chunker()
        nodes = chunker.get_nodes_from_documents(documents, show_progress=True)
        nodes = tokenize_vietnamese_text(nodes)
        print(f"Created {len(nodes)} nodes")
        
        # Create embedding model
        embed_model = create_embedding_model()
        
        # Build Weaviate index
        index = build_optimized_index(nodes, embed_model)
        
        # Create retriever
        retriever = create_optimized_retriever(index)
        
        # Example search
        query = "Các mức xử phạt nồng độ cồn đối với xe ô tô"
        results = search_violations(retriever, query)
        
        print(f"\nSearch Results for: {query}\n")
        for i, node in enumerate(results):
            print(f"Result {i+1}:")
            print(f"Text: {node.metadata.get('original_text', node.text)}")
            print(f"Score: {node.score}")
            print(f"Category: {node.metadata.get('category', 'N/A')}")
            print(f"Fine: {node.metadata.get('fine_amount', 'N/A')}")
            print(f"Violation Type: {node.metadata.get('violation_type', 'N/A')}")
            print("-" * 50)
        
        print("\nIndex saved successfully to Weaviate")
        print(f"Weaviate is running at {WEAVIATE_URL}")
        
    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure we close the Weaviate client connection properly
        try:
            # This is a bit tricky because we need to get the client from where it was created
            # Let's try to recreate it and close it
            client = weaviate.connect_to_weaviate_cloud(
                cluster_url=WEAVIATE_URL,
                auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
            )
            client.close()
            print("Weaviate client connection closed properly")
        except Exception as e:
            print(f"Warning: Could not close Weaviate client connection: {e}")

if __name__ == "__main__":
    main()