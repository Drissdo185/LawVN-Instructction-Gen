import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging
from typing import List, Dict, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QAGenerator:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True,
            padding_side='left'
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def create_prompt(self, text: str, num_questions: int = 5) -> str:
        return f"""Task: Generate {num_questions} question-answer pairs from the given text.

Rules:
1. Questions should be diverse (Who, What, Where, When, Why, How)
2. Answers must be clear and accurate
3. Output must be valid JSON
4. Each QA pair must follow this exact format:
   {{"question": "...", "answer": "..."}}

Text to process:
{text}

Generate JSON output in this format:
[
    {{"question": "First question here", "answer": "First answer here"}},
    {{"question": "Second question here", "answer": "Second answer here"}},
    // ... more QA pairs
]

JSON Output:"""

    def extract_json(self, text: str) -> Optional[str]:
        """Extract JSON string from generated text."""
        try:
            # Find the start of the JSON array
            start_idx = text.find('[')
            if start_idx == -1:
                logger.warning("No JSON array found in generated text")
                return None
                
            # Find the end of the JSON array
            end_idx = text.rfind(']') + 1
            if end_idx <= 0:
                logger.warning("No closing bracket found in generated text")
                return None
                
            return text[start_idx:end_idx]
        except Exception as e:
            logger.error(f"Error extracting JSON: {e}")
            return None

    def generate_qa(self, text: str, num_questions: int = 5) -> List[Dict[str, str]]:
        """Generate QA pairs from input text."""
        try:
            # Create prompt
            prompt = self.create_prompt(text, num_questions)
            
            # Tokenize with proper attention mask
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.model.device)
            
            # Generate response
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=4096,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            # Decode generated text
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract and parse JSON
            json_str = self.extract_json(generated_text)
            if json_str is None:
                return []
                
            qa_pairs = json.loads(json_str)
            
            # Validate format
            if not isinstance(qa_pairs, list):
                logger.error("Generated content is not a list")
                return []
                
            return qa_pairs
            
        except Exception as e:
            logger.error(f"Error in generate_qa: {e}")
            return []

def process_file(file_path: str, model_name: str) -> None:
    """Process a JSON file and generate QA pairs."""
    try:
        # Initialize generator
        qa_gen = QAGenerator(model_name)
        
        # Read input file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Process each text
        results = []
        for i, text in enumerate(data['content']):
            logger.info(f"Processing text {i+1}/{len(data['content'])}")
            qa_pairs = qa_gen.generate_qa(text)
            results.extend(qa_pairs)
            
        # Save results
        output_path = file_path.replace('.json', '_qa_pairs.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Results saved to {output_path}")
            
    except Exception as e:
        logger.error(f"Error processing file: {e}")

if __name__ == "__main__":
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    file_path = "/home/ltnga/LawVN-Instructction-Gen/src/data/data_gen.json"
    
    process_file(file_path, model_name)