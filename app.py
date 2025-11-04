#this is useless since we hardcoding LOL 

from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import os 

app = Flask(__name__)
CORS(app) 

extensions = {'.pdf', '.doc', '.docx', '.txt'}

def allowed_file(filename):
    return '.' in filename and \
           os.path.splitext(filename)[1].lower() in extensions

@app.route('/analyze', methods=['POST'])
def analyze_resume():
    #get data
    if 'resume' not in request.files:
        return jsonify({"error": "No resume file found"}), 400
    
    query = request.form.get('query')
    if not query:
        return jsonify({"error": "No query provided"}), 400

    resume_file = request.files['resume']

    #check correct type
    if resume_file.filename == '':
        return jsonify({"error": "No file selected"}), 400
        
    if not allowed_file(resume_file.filename):
        return jsonify({"error": "Invalid file type. Only PDF, DOCX, or TXT are allowed."}), 400

    print("...Simulating AI analysis...")
    time.sleep(2) 
    print("...Analysis complete.")

    #hardcode 
    fake_answer_generate = (
        "Based on your resume, you are a strong candidate for a Marketing Analyst role.\n\n"
        "Strengths:\n"
        "Strong quantitative background.\n"
        "Experience with marketing campaigns (from your 'Project Sales' entry).\n\n"
        "Suggested Courses to Improve:\n"
        "1.  Advanced Excel for Marketing: To boost your modeling speed.\n"
        "2.  Python for Marketing Analysis: To automate tasks and align with modern roles.\n\n"
    )

    return jsonify({
        "answer": fake_answer_generate
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)