import json
import torch
from llama_index.core import Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pyvi import ViTokenizer
import weaviate
from weaviate.classes.init import Auth
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from typing import List, Any

# Configuration constants
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "dangvantuan/vietnamese-document-embedding"
DATA_COLLECTION = "ND168"  # Same collection as car data

# Weaviate configuration
WEAVIATE_URL = "https://eypxka08rk6gbnyt57idzq.c0.us-west3.gcp.weaviate.cloud"
WEAVIATE_API_KEY = "jWgkVYQprMDiOuvTh56iLqRK5oqITcxN27wJ"

# Chunking parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

def load_and_preprocess_moto_data(file_path: str) -> List[Document]:
    """
    Load JSON data for motorcycles and convert to Document objects with metadata
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
    """
    violation_text = violation_text.lower()
    
    # Define comprehensive violation categories with relevant keywords
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
            "mililít", "không chấp hành yêu cầu kiểm tra", "bia", "rượu"
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
            "chỗ ngồi", "ghế", "khoang hành lý", "chằng buộc", "nối chắc chắn",
            "mũ bảo hiểm", "đội mũ"
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
            "1,35 mét", "mầm non", "học sinh", "thiết bị an toàn",
            "dưới 12 tuổi"
        ],
        
        # Vehicle technical violations
        "kỹ_thuật_xe": [
            "kính chắn gió", "thiết bị giám sát", "biển số", "gương chiếu hậu", 
            "bánh lốp", "thùng xe", "hệ thống hãm", "hệ thống phanh", 
            "thiết bị chữa cháy", "kích thước", "niên hạn", "cải tạo",
            "đèn soi biển số", "đèn báo hãm", "gương chiếu hậu", "còi"
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
            "trong khoang hành lý", "ngoài thành xe", "buồng lái", "chở người",
            "chở theo", "người ngồi trên xe"
        ],
        
        # Cargo transport violations
        "chở_hàng": [
            "chở hàng", "vượt trọng tải", "quá tải", "vượt kích thước",
            "làm lệch xe", "rơ moóc", "không chốt", "không đóng cố định",
            "siêu trường", "siêu trọng", "hàng nguy hiểm", "kéo theo"
        ],
        
        # Mobile phone and distracted driving
        "điện_thoại": [
            "điện thoại", "thiết bị điện tử", "dùng tay", "khi điều khiển",
            "mất tập trung", "không quan sát", "sử dụng điện thoại"
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
            "gây tai nạn", "không giữ khoảng cách", "bám", "kéo", "đẩy xe",
            "buông cả hai tay", "dùng chân điều khiển", "nằm trên yên xe",
            "chạy bằng một bánh", "bịt mắt"
        ],
        
        # Emergency vehicle violations
        "xe_ưu_tiên": [
            "xe ưu tiên", "xe được quyền ưu tiên", "thiết bị phát tín hiệu", 
            "xe cứu thương", "xe cứu hỏa", "xe cứu hộ", "xe công an", "xe quân sự"
        ],
        
        # Accidents and crash handling
        "tai_nạn": [
            "tai nạn", "không dừng ngay phương tiện", "không giữ nguyên hiện trường",
            "không trợ giúp người bị nạn", "không ở lại", "không trình báo",
            "có liên quan trực tiếp"
        ],
        
        # Highway-specific violations
        "đường_cao_tốc": [
            "đường cao tốc", "vào cao tốc", "ra cao tốc", "làn dừng xe khẩn cấp",
            "đón trả khách", "đi ngược chiều", "đỗ xe trên cao tốc"
        ],
        
        # Motorcycle-specific violations
        "xe_máy_đặc_thù": [
            "ba bánh", "ô (dù)", "xe sản xuất", "lắp ráp trái quy định",
            "mô tô", "gắn máy", "xe hai bánh", "xe ba bánh"
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
            "tốc_độ", "đường_cao_tốc", "chở_hàng", "chở_người", "đỗ_dừng_xe",
            "xe_máy_đặc_thù"
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
    Set up and connect to Weaviate cloud using v4.x client
    """
    try:
        # Connect to Weaviate cloud instance
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL,
            auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
        )
        print(f"Connected to Weaviate cloud at {WEAVIATE_URL}")
        
        # Check if our collection already exists
        try:
            # In v4.x, we check if collection exists differently
            existing_collection = client.collections.get(DATA_COLLECTION)
            print(f"Collection {DATA_COLLECTION} already exists")
        except weaviate.exceptions.WeaviateNotFoundError:
            # Collection doesn't exist, create it
            print(f"Collection {DATA_COLLECTION} not found, creating it...")
            client.collections.create(
                name=DATA_COLLECTION,
                description="Traffic violations for Vietnamese vehicles",
                properties=[
                    {
                        "name": "text",
                        "dataType": "text",
                        "description": "The text content of the violation"
                    },
                    {
                        "name": "original_text",
                        "dataType": "text",
                        "description": "Original non-tokenized text"
                    },
                    {
                        "name": "category",
                        "dataType": "string",
                        "description": "Vehicle category"
                    },
                    {
                        "name": "fine_amount",
                        "dataType": "string",
                        "description": "Amount of fine"
                    },
                    {
                        "name": "violation_type",
                        "dataType": "string",
                        "description": "Type of violation"
                    }
                ]
            )
            print(f"Created collection {DATA_COLLECTION}")
        
        return client
    except Exception as e:
        print(f"Error connecting to Weaviate: {e}")
        raise

def add_to_index(nodes: List[Any], embed_model: HuggingFaceEmbedding):
    """
    Add motorcycle data to existing index
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
        
        # Create index with the existing vector store
        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=True
        )
        
        print(f"Successfully added motorcycle data to existing index in Weaviate class: {DATA_COLLECTION}")
        return index
    
    except Exception as e:
        print(f"Error adding to index: {e}")
        raise

def main():
    try:
        # Path to the motorcycle data file
        moto_file_path = "/home/ltnga/LawVN-Instructction-Gen/src/data/motor.json"
        
        # Load and process motorcycle data
        documents = load_and_preprocess_moto_data(moto_file_path)
        print(f"Loaded {len(documents)} motorcycle violation documents")
        
        # Create chunker and process nodes
        chunker = create_optimized_chunker()
        nodes = chunker.get_nodes_from_documents(documents, show_progress=True)
        nodes = tokenize_vietnamese_text(nodes)
        print(f"Created {len(nodes)} nodes from motorcycle data")
        
        # Create embedding model
        embed_model = create_embedding_model()
        
        # Add to existing Weaviate index
        index = add_to_index(nodes, embed_model)
        
        # Test query to verify integration
        retriever = index.as_retriever(similarity_top_k=5)
        query = "Xử phạt khi nằm trên yên xe"
        results = retriever.retrieve(ViTokenizer.tokenize(query.lower()))
        
        print(f"\nTest Search Results for: {query}\n")
        for i, node in enumerate(results):
            print(f"Result {i+1}:")
            print(f"Text: {node.metadata.get('original_text', node.text)}")
            print(f"Score: {node.score}")
            print(f"Category: {node.metadata.get('category', 'N/A')}")
            print(f"Fine: {node.metadata.get('fine_amount', 'N/A')}")
            print(f"Violation Type: {node.metadata.get('violation_type', 'N/A')}")
            print("-" * 50)
        
        print("\nMotorcycle data successfully integrated into existing index")
        
    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure we close the Weaviate client connection properly
        try:
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