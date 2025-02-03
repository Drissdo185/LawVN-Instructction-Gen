import json 
import requests
from typing import List, Dict
import time 
import re

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import accelerate
from transformers import pipeline

def extract_articles(text: str) -> List[str]:
    """Extract individual articles from text"""
    articles = []
    parts = text.split('Điều ')
    for part in parts[1:]:
        article_text = 'Điều ' + part
        articles.append(article_text.strip())
    return articles

def chunk_article(article: str, max_tokens: int = 1500) -> List[str]:
    """Split article into smaller chunks if needed"""
    chunks = []
    paragraphs = article.split('\n')
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) > max_tokens:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = para
        else:
            current_chunk += "\n" + para if current_chunk else para
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def generate_qa(model, tokenizer, context: str) -> Dict:
    """Generate Q&A pairs using the model"""
    prompt = f"""Bạn là một chuyên gia về luật giao thông. Hãy đọc đoạn văn bản sau và tạo ra 5 cặp câu hỏi - câu trả lời về nội dung đoạn văn bản đó. Câu hỏi và câu trả lời phải chính xác, rõ ràng và có tính thực tế.

Văn bản:
{context}

Yêu cầu output format JSON:
{{
    "qas": [
        {{
            "question": "Câu hỏi...",
            "answer": "Câu trả lời...", 
            "context": "{context}",
            "metadata": {{
                "article": "Điều số...",
                "category": "Chọn 1 trong các category: Mức phạt/Hành vi vi phạm/Quy định chung",
                "tags": ["tag1", "tag2"]
            }}
        }}
    ]
}}

Chỉ trả về JSON, không thêm bất kỳ text nào khác."""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2000,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    try:
        # Extract JSON content (everything between first { and last })
        json_str = response[response.find('{'):response.rfind('}')+1]
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        print(f"Raw response: {response}")
        return None

def main():
    # Initialize model and tokenizer
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    print(f"Loading tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    print(f"Loading model from {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    
    # Load input data
    file_path = "/home/ltnga/LawVN-Instructction-Gen/src/data/data_test.json"
    print(f"Loading data from {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Initialize dataset
    dataset = {
        "version": "1.0",
        "data": []
    }
    
    # Process articles
    articles = extract_articles(data)
    print(f"\nFound {len(articles)} articles to process")
    
    for article_idx, article in enumerate(articles):
        print(f"\nProcessing article {article_idx + 1}/{len(articles)}")
        chunks = chunk_article(article)
        
        for chunk_idx, chunk in enumerate(chunks):
            print(f"Processing chunk {chunk_idx + 1}/{len(chunks)}")
            try:
                qa_data = generate_qa(model, tokenizer, chunk)
                if qa_data and 'qas' in qa_data:
                    dataset['data'].extend(qa_data['qas'])
                    print(f"Generated {len(qa_data['qas'])} Q&A pairs")
                else:
                    print("Failed to generate Q&A pairs for this chunk")
                
                # Add small delay between generations
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error processing chunk: {e}")
                continue
    
    # Save dataset
    output_path = 'traffic_law_dataset.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"\nDataset saved to {output_path}")
    print(f"Total Q&A pairs generated: {len(dataset['data'])}")

if __name__ == "__main__":
    main()