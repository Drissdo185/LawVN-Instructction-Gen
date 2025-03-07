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

# Configuration constants
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "dangvantuan/vietnamese-document-embedding"

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
    """
    violation_text = violation_text.lower()
    
    categories = {
        "tốc_độ": ["tốc độ", "km/h", "chạy quá"],
        "đỗ_xe": ["dừng xe", "đỗ xe", "đậu xe"],
        "giấy_tờ": ["giấy phép", "chứng nhận", "đăng ký"],
        "cồn": ["nồng độ cồn", "cồn"],
        "an_toàn": ["dây đai", "an toàn", "khoảng cách"],
        "đèn": ["đèn", "chiếu sáng"],
        "làn_đường": ["làn đường", "chuyển làn"],
        "trẻ_em": ["trẻ em", "trẻ nhỏ"]
    }
    
    for category, keywords in categories.items():
        if any(keyword in violation_text for keyword in keywords):
            return category
    
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

def build_optimized_index(nodes: List[Any], embed_model: HuggingFaceEmbedding) -> VectorStoreIndex:
    """
    Build and optimize the vector index using in-memory storage
    """
    # Configure global settings
    Settings.embed_model = embed_model
    Settings.chunk_size = CHUNK_SIZE
    
    # Use simple in-memory vector store instead of Weaviate
    vector_store = SimpleVectorStore()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Create index with optimized settings
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
    Add metadata-based filtering to improve relevance
    """
    query_lower = query.lower()
    
    # Detect violation types in query
    violation_types = []
    if any(keyword in query_lower for keyword in ["trẻ em", "trẻ nhỏ", "1,35 mét"]):
        violation_types.append("trẻ_em")
    if any(keyword in query_lower for keyword in ["tốc độ", "km/h", "chạy quá"]):
        violation_types.append("tốc_độ")
    if any(keyword in query_lower for keyword in ["nồng độ cồn", "cồn"]):
        violation_types.append("cồn")
    
    # If violation types detected, filter results
    if violation_types:
        filtered_nodes = [
            node for node in nodes 
            if "violation_type" in node.metadata and node.metadata["violation_type"] in violation_types
        ]
        if filtered_nodes:
            return filtered_nodes
    
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
    return results[:top_k]

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
        
        # Build simple in-memory index (no Weaviate needed)
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
            
        # Save index for future use
        index.storage_context.persist("./traffic_violation_index")
        print("\nIndex saved successfully to ./traffic_violation_index")
        
    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()