from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
from pathlib import Path
from datetime import datetime
import sys

# Add src directory to Python path so all modules can be imported
current_dir = Path(__file__).resolve().parent  # src/backend/
src_path = current_dir.parent  # src/
project_root = src_path.parent  # project root
rag_path = src_path / 'rag'
ner_path = src_path / 'ner'

# Add all necessary paths
sys.path.insert(0, str(rag_path))
sys.path.insert(0, str(ner_path))
sys.path.insert(0, str(src_path))

print(f"Added to sys.path:")
print(f"  - {rag_path}")
print(f"  - {ner_path}")
print(f"  - {src_path}")

# Import your existing functions
try:
    from answer_generation import answer_generation_function
    print("✓ Successfully imported answer_generation_function")
except ImportError as e:
    print(f"✗ Failed to import answer_generation_function: {e}")
    print(f"Current sys.path: {sys.path[:5]}")
    raise

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Neo4j credentials
NEO4J_URI = "neo4j+s://accc1403.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "TwKAMXkflTo1NnauW3SEjMW4iXbExFIHA6oNi4mN1h4"

# Base paths
BASE_DIR = project_root  # Use project_root instead of parent.parent
QUERY_BASE_DIR = BASE_DIR / "input_data" / "queries"
RESUME_BASE_DIR = BASE_DIR / "input_data" / "resumes"

@app.route('/api/analyze', methods=['POST'])
def analyze_career():
    """
    Accepts a resume PDF and a query text.
    Saves them to appropriate directories and processes them.
    """
    try:
        print("\n" + "="*80)
        print("STEP 1: RECEIVING DATA FROM FRONTEND")
        print("="*80)
        
        # Get query text and resume file
        query_text = request.form.get('query')
        resume_file = request.files.get('resume')
        
        print(f"✓ Query received: {query_text[:100]}..." if query_text and len(query_text) > 100 else f"✓ Query received: {query_text}")
        print(f"✓ Resume file received: {resume_file.filename if resume_file else 'None'}")
        
        if not query_text:
            print("✗ ERROR: No query text provided")
            return jsonify({'error': 'Query text is required'}), 400
        
        if not resume_file:
            print("✗ ERROR: No resume file provided")
            return jsonify({'error': 'Resume file is required'}), 400
        
        # Create timestamp for unique folder naming
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("\n" + "="*80)
        print("STEP 2: SAVING FILES")
        print("="*80)
        
        # Setup query directory and file
        query_folder_name = f"input_query_{timestamp}"
        query_dir = QUERY_BASE_DIR / query_folder_name
        query_dir.mkdir(parents=True, exist_ok=True)
        
        query_file_path = query_dir / f"{query_folder_name}.jsonl"
        with open(query_file_path, 'w', encoding='utf-8') as f:
            json.dump({"text": query_text}, f)
        
        print(f"✓ Query saved to: {query_file_path}")
        print(f"✓ Query content: {json.dumps({'text': query_text})}")
        
        # Setup resume directory and file
        resume_filename = resume_file.filename
        if not resume_filename:
            print("✗ ERROR: Resume filename is invalid")
            return jsonify({'error': 'Resume filename is invalid'}), 400
        resume_name = Path(resume_filename).stem  # Get filename without extension
        resume_folder_name = f"{resume_name}_{timestamp}"
        resume_dir = RESUME_BASE_DIR / resume_folder_name
        resume_dir.mkdir(parents=True, exist_ok=True)
        
        # Save resume with original name
        resume_file_path = resume_dir / resume_filename
        resume_file.save(str(resume_file_path))
        
        print(f"✓ Resume saved to: {resume_file_path}")
        print(f"✓ Resume file size: {resume_file_path.stat().st_size} bytes")
        
        print("\n" + "="*80)
        print("STEP 3: PROCESSING WITH ANSWER GENERATION FUNCTION")
        print("="*80)
        print(f"Calling answer_generation_function with:")
        print(f"  - query_path: {query_file_path}")
        print(f"  - resume_path: {resume_file_path}")
        print("Processing...")
        
        # Process the query and resume
        query, generated_answer, retrieved_context = answer_generation_function(
            str(query_file_path),
            str(resume_file_path),
            NEO4J_URI,
            NEO4J_USERNAME,
            NEO4J_PASSWORD
        )
        
        # Log what we're sending back
        print("\n" + "="*80)
        print("STEP 4: SENDING RESPONSE TO FRONTEND")
        print("="*80)
        print(f"✓ Query: {query[:100]}..." if len(query) > 100 else f"✓ Query: {query}")
        print(f"✓ Answer length: {len(generated_answer)} characters")
        print(f"✓ Answer preview: {generated_answer[:200]}..." if len(generated_answer) > 200 else f"✓ Answer: {generated_answer}")
        print(f"✓ Context length: {len(retrieved_context)} characters")
        print("="*80 + "\n")
        
        response_data = {
            'success': True,
            'query': query,
            'answer': generated_answer,
            'context': retrieved_context,
            'query_path': str(query_file_path),
            'resume_path': str(resume_file_path)
        }
        
        return jsonify(response_data)
    
    except Exception as e:
        print("\n" + "="*80)
        print("ERROR OCCURRED:")
        print("="*80)
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        print("="*80 + "\n")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'Career Engine API is running'})

if __name__ == '__main__':
    # Ensure directories exist
    QUERY_BASE_DIR.mkdir(parents=True, exist_ok=True)
    RESUME_BASE_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Starting Career Engine API server...")
    print(f"Query directory: {QUERY_BASE_DIR}")
    print(f"Resume directory: {RESUME_BASE_DIR}")
    app.run(debug=True, host='0.0.0.0', port=5000)
