import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

class RAGGenerator:
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct"):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            print("Model and Tokenizer loaded successfully.")
        except ImportError:
            print("Error: Required packages not found. Please run 'pip install transformers torch'.")
            exit()
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please make sure 'transformers' is installed.")
            exit()

    def generate_career_advice(self, query, retrieved_context):
        """
        Generates a contextualized answer based on the query and context.
        """
        
        messages = [
            {
                "role": "system",
                "content": """You are an AI-powered career guidance assistant with 15 years of experience advising fresh graduates in Singapore.
Your goal is to provide accurate and personalized guidance, grounded strictly in the retrieved context.

📌 Grounding Rules:
1. Do NOT hallucinate. Only use information in the retrieved context below.
2. Your answer must be derived only from the Retrieved Context containing:
   - Missing skills
   - Recommended courses
   - Recommended jobs"""
            },
            {
                "role": "user",
                "content": f"""### Retrieved Context
The context below contains structured outputs containing missing skills, course recommendations and job recommendations.
{retrieved_context}

### User Query
{query}

### Instructions for Your Answer
- Provide a clear, concise, and professional answer.
- Use each section of the retrieved context appropriately:
  • Use **missing skills** to explain gaps
  • Use **courses** to suggest upskilling
  • Use **jobs** to explain fit, career paths, or alternatives

- If the user asks about:
  • Skill gaps → use *missing skills*
  • Course suggestions → use *recommended courses*
  • Job opportunities → use *recommended jobs*
  • Career planning → combine all three intelligently

- If something is absent, clearly state what is missing."""
            }
        ]
        
        # Format messages using chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=8192,  # Increased to handle longer context
            truncation=True
        )
        
        # Move inputs to same device as model
        if torch.cuda.is_available():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Get input length to extract only new tokens
        input_length = inputs['input_ids'].shape[1]
        
        print(f"DEBUG Generation: Input length = {input_length} tokens")
        print(f"DEBUG Generation: Starting generation with max_new_tokens=300...")
        
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=300,  # Increased for longer responses
            do_sample=False,  # Greedy decoding
            repetition_penalty=1.3,
            pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        print(f"DEBUG Generation: Generation completed!")
        
        # Decode only the newly generated tokens (skip the input prompt)
        generated_tokens = outputs[0][input_length:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        print(f"DEBUG Generation: Raw output length = {len(generated_text)} characters")
        print(f"DEBUG Generation: Raw output preview = {generated_text[:300]}")
        
        # Simple cleanup: just limit to reasonable length
        if len(generated_text) > 2000:
            # Find last complete sentence within 2000 chars
            truncated = generated_text[:2000]
            last_period = truncated.rfind('.')
            if last_period > 0:
                generated_text = truncated[:last_period + 1]
            else:
                generated_text = truncated
        
        print(f"DEBUG Generation: Final output length = {len(generated_text)} characters")
        
        return generated_text.strip()

#test
if __name__ == "__main__":
    generator = RAGGenerator()
    
    user_query = "From the following jobs, what jobs requires Excel?"
    
    placeholder_context = """
    Job Posting 1 (Data Analyst at Lazada):
    - Responsibilities: Collect and analyze data, create dashboards.
    - Skills: SQL, Python (Pandas, NumPy), Tableau, Power BI.
    - Qualifications: Bachelor's degree in Statistics or Computer Science.
    
    Job Posting 2 (Junior Analyst at OCBC):
    - Responsibilities: Support senior analysts, clean data.
    - Skills: Strong Excel, basic SQL, good communication.
    - Qualitifactions: None.
    
    Job Posting 3 (BI Analyst at Grab):
    - Responsibilities: Develop business intelligence solutions.
    - Skills: SQL, Power BI, understanding of data warehousing.
    - Qualifications: Bachelor's degree in Business Analytics.
    """
    
    answer = generator.generate_career_advice(user_query, placeholder_context)
    print(f"Question: {user_query}\n Answer: {answer}")