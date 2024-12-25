import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_model_and_tokenizer():
    """
    Load Qwen 2.5 14B model and tokenizer
    """
    model_name = "Qwen/Qwen1.5-14B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    return model, tokenizer

def load_json_data(file_path):
    """
    Load and parse JSON data from file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_qa(model, tokenizer, context, num_qa_pairs=3, max_length=1024):
    """
    Generate Q&A pairs for Vietnamese legal documents
    """
    prompt_template = """
    Bạn là một chuyên gia phân tích văn bản pháp luật Việt Nam. Dựa trên văn bản sau đây, hãy tạo các câu hỏi và câu trả lời chuyên sâu theo các quy tắc sau:

    Văn bản: {context}

    Quy tắc cho Câu hỏi:
    1. Tạo các loại câu hỏi đa dạng:
       - Câu hỏi về định nghĩa pháp lý và thuật ngữ
       - Câu hỏi về phạm vi áp dụng của văn bản
       - Câu hỏi về quyền và nghĩa vụ của các bên liên quan
       - Câu hỏi về thủ tục và quy trình thực hiện
       - Câu hỏi về chế tài và hậu quả pháp lý
    
    2. Yêu cầu về nội dung:
       - Tập trung vào các điểm quan trọng của văn bản
       - Đảm bảo tính chính xác về mặt pháp lý
       - Sử dụng đúng thuật ngữ pháp lý
       - Trích dẫn điều khoản cụ thể khi cần thiết
       - Phân tích mối liên hệ giữa các quy định

    Quy tắc cho Câu trả lời:
    1. Cấu trúc câu trả lời:
       - Trích dẫn chính xác điều khoản liên quan
       - Giải thích rõ ràng nội dung quy định
       - Đưa ra ví dụ minh họa khi cần thiết
       - Phân tích ý nghĩa và mục đích của quy định
       - Nêu các trường hợp áp dụng thực tế

    2. Yêu cầu về hình thức:
       - Sử dụng ngôn ngữ chính xác và chuyên nghiệp
       - Trình bày logic và có hệ thống
       - Đảm bảo tính khách quan và trung lập
       - Tham chiếu đến các văn bản liên quan (nếu có)

    Định dạng yêu cầu:
    - Bắt đầu câu hỏi bằng "Câu hỏi: "
    - Bắt đầu câu trả lời bằng "Trả lời: "
    - Phân tách mỗi cặp câu hỏi-trả lời bằng một dòng trống
    - Độ dài câu hỏi: 15-40 từ
    - Độ dài câu trả lời: 100-300 từ

    Ví dụ mẫu:
    Câu hỏi: Theo Điều X của [tên văn bản], đâu là những yêu cầu cụ thể về [chủ đề] và các trường hợp áp dụng của những yêu cầu này?
    Trả lời: Căn cứ theo Điều X của [tên văn bản], các yêu cầu về [chủ đề] bao gồm: [liệt kê các yêu cầu]. Cụ thể, [giải thích chi tiết]. Trong thực tiễn áp dụng, [ví dụ minh họa]. Điều này nhằm mục đích [phân tích ý nghĩa]. Ngoài ra, cần lưu ý mối liên hệ với [các quy định liên quan].

    Bây giờ, hãy tạo một cặp câu hỏi và trả lời theo các quy tắc trên.
    """
    
    qa_pairs = []
    
    for _ in range(num_qa_pairs):
        prompt = prompt_template.format(context=context)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_content = response.split(prompt)[-1].strip()
        qa_pairs.append(generated_content)
    
    return qa_pairs

def process_qa_output(qa_pairs):
    """
    Process and validate the generated Q&A pairs for legal content
    """
    processed_pairs = []
    for qa in qa_pairs:
        pairs = qa.split('\n\n')
        for pair in pairs:
            if 'Câu hỏi:' in pair and 'Trả lời:' in pair:
                question = pair.split('Trả lời:')[0].replace('Câu hỏi:', '').strip()
                answer = pair.split('Trả lời:')[1].strip()
                
                # Validate content length and quality
                if (15 <= len(question.split()) <= 40 and 
                    100 <= len(answer.split()) <= 300 and
                    'Điều' in answer):  # Ensure reference to specific articles
                    processed_pairs.append({
                        "question": question,
                        "answer": answer,
                        "metadata": {
                            "references": extract_legal_references(answer),
                            "key_terms": extract_key_terms(answer)
                        }
                    })
    
    return processed_pairs

def extract_legal_references(text):
    """
    Extract legal references from the text
    """
    # Simple extraction of article references
    references = []
    words = text.split()
    for i, word in enumerate(words):
        if word == "Điều" and i < len(words) - 1:
            references.append(f"Điều {words[i+1]}")
    return list(set(references))

def extract_key_terms(text):
    """
    Extract key legal terms from the text
    """
    # Add your Vietnamese legal terms dictionary/detection logic here
    common_legal_terms = [
        "nghị định", "thông tư", "quyết định", "văn bản", "pháp luật",
        "quyền", "nghĩa vụ", "trách nhiệm", "xử phạt", "thẩm quyền"
    ]
    
    found_terms = []
    text_lower = text.lower()
    for term in common_legal_terms:
        if term in text_lower:
            found_terms.append(term)
    
    return list(set(found_terms))

def main(json_file_path):
    """
    Main function to process Vietnamese legal documents and generate Q&A pairs
    """
    print("Đang tải mô hình và tokenizer...")
    model, tokenizer = load_model_and_tokenizer()
    
    print("Đang đọc dữ liệu văn bản pháp luật...")
    data = load_json_data(json_file_path)
    
    results = []
    for item in data:
        context = str(item)
        qa_pairs = generate_qa(model, tokenizer, context)
        processed_pairs = process_qa_output(qa_pairs)
        
        results.append({
            "context": context,
            "qa_pairs": processed_pairs
        })
    
    output_file = "legal_qa_generated.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Đã lưu kết quả Q&A vào file {output_file}")

if __name__ == "__main__":
    json_file_path = "your_legal_documents.json"
    main(json_file_path)