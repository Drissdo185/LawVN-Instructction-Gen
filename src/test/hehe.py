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
MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"
DATA_COLLECTION = "ND168"

WEAVIATE_URL="https://mmtip1s2qwwe1q3bwih2fw.c0.us-west3.gcp.weaviate.cloud"
WEAVIATE_API_KEY="37Nj5DR1LWMDz98URsr3ib7e80pnHHAoyqn1"

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
            
            # Add simplified metadata without categorization
            metadata = {
                "category": category,
                "fine_amount": fine,
                "violation_type": "khác"  # Default type since categorize_violation is removed
            }
            
            documents.append(Document(text=doc_text, metadata=metadata))
    
    return documents

def create_optimized_chunker() -> SentenceSplitter:
    """
    Create an optimized document chunker for Vietnamese text
    """
    return SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separator=".",  
        include_metadata=True,
        include_prev_next_rel=True,
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
        client = weaviate.connect_to_local(
            host="127.0.0.1",  # Use a string to specify the host
            port=8080,
            grpc_port=50051,
        )

        print(client.is_ready())
        
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
        return index

def create_optimized_retriever(index: VectorStoreIndex):
    """
    Create an optimized retriever for better search results
    """
    return index.as_retriever(
        similarity_top_k=7,  # Retrieve more candidates
    )


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
    
    
    
    # Rank results by relevance
    ranked_results = rank_results(results, query)
    
    # Return top results
    return ranked_results[:top_k]

def main():
    try:
        # Load and process data
        documents = load_and_preprocess_data("/home/ltnga/LawVN-Instructction-Gen/src/data/raw_data_processed.json")
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
        query = "Các mức xử phạt nồng độ cồn đối với xe gắn máy"
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
            client = weaviate.connect_to_local(
            host="127.0.0.1",  # Use a string to specify the host
            port=8080,
            grpc_port=50051)
            client.close()
            print("Weaviate client connection closed properly")
        except Exception as e:
            print(f"Warning: Could not close Weaviate client connection: {e}")

if __name__ == "__main__":
    main()
