import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

class RAGGenerator:
    def __init__(self, model_name="google/flan-t5-base"): #i choose this cuz its lower resource usage and faster processing but if yall want better performance then do t5-large
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
            print("Model and Tokenizer loaded successfully.")
        except ImportError: #make sure to pip install sentencepiece
            print("Error: 'sentencepiece' not found. Please run 'pip install sentencepiece'.")
            exit()
        except Exception as e: #make sure to install transformers
            print(f"Error loading model: {e}")
            print("Please make sure 'transformers' is installed.")
            exit()

    def generate_career_advice(self, query, retrieved_context):
        """
        Generates a contextualized answer based on the query and context.
        """
        
        prompt_template = f"""
        **Context:**
        You are an AI-powered career guidance system for fresh graduates in Singapore.
        Your goal is to provide personalized, actionable career advice.
        You must base your answer *only* on the relevant job information provided below.
        Do not make up information.
        
        **Retrieved Job Information:**
        {retrieved_context}
        
        **User's Question:**
        {query}
        
        **Your Answer:**
        """
        
        inputs = self.tokenizer(
            prompt_template, 
            return_tensors="pt", 
            max_length=1024, 
            truncation=True
        )
        
        outputs = self.model.generate(
            **inputs, 
            max_length=512,  
            num_beams=5,     
            early_stopping=True,
            repetition_penalty=1.2  #optional but its to prevent repetitive phrases since our output dun make sense for repeated phrases(?)
        )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

#test
if __name__ == "__main__":
    generator = RAGGenerator()
    
    user_query = "What specific skills should I learn for a Junior Analyst role in DBS?"
    
    placeholder_context = """
    Job Posting 1 (Data Analyst at Shopee):
    - Responsibilities: Collect and analyze data, create dashboards.
    - Skills: SQL, Python (Pandas, NumPy), Tableau, Power BI.
    - Qualifications: Bachelor's degree in Statistics or Computer Science.
    
    Job Posting 2 (Junior Analyst at DBS Bank):
    - Responsibilities: Support senior analysts, clean data.
    - Skills: Strong Excel, basic SQL, good communication.
    
    Job Posting 3 (BI Analyst at Grab):
    - Responsibilities: Develop business intelligence solutions.
    - Skills: SQL, Power BI, understanding of data warehousing.
    """
    
    answer = generator.generate_career_advice(user_query, placeholder_context)
    print(answer)