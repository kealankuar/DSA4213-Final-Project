# DSA4213-Final-Project : Job-Seek – Resume Intelligence & Career Insights


## 📌 Overview
This repository contains the full pipeline for the **Job-Seek System**, an end-to-end platform designed to help users understand their resumes, identify missing skills, and query relevant job insights.
The system consists of three core modules:

- Named Entity Recognition (NER) → Extracts skills, qualifications, job titles, and tools
- Retrieval-Augmented Generation (RAG) → Answers user queries about job fit
- Frontend UI → User-facing interface for uploading resumes and interacting with the system
---
## Installation
Install the required dependencies for the project:
```bash
pip install -r requirements.txt
```
---
##  Named Entity Recognition (NER)

The NER module is responsible for extracting structured information such as skills, job titles, qualifications, organisations, and tools from resumes and job descriptions. The training script loads a token-label CSV file, prepares it into HuggingFace Dataset format, aligns labels to subword tokens, and runs an ablation study across multiple transformer models and learning rates. Each model is evaluated using SeqEval, and both the trained model weights and tokenizer are saved for downstream inference. A summary CSV is also generated for quick comparison.

## Running NER Training
To train all NER models:
```bash
cd src/ner
python train_ner.py
```
Running this will:
- preprocess the dataset
- train all model × learning rate combinations
- evaluate using precision, recall, F1, and accuracy
- save each trained model under ```models/ner_ablation_results/<model>_<lr>/``` 
- generate ablation_summary.csv with all results
---
## Retrieval-Augmented Generation (RAG)

The RAG module takes the entities and text extracted by the NER component and uses them to answer user questions about resumes and job descriptions. It is designed to let users ask things like “What skills am I missing for this role?” or “Does my experience match this job?” by retrieving relevant chunks from indexed job postings or knowledge sources and passing them to a language model for response generation.

## Run RAG 
```bash
cd src/rag
python FILL THIS UP
```
---

## Frontend Web Application

The frontend provides the user interface for the Job-Seek system. It allows users to upload their resumes, view extracted entities from the NER module, and interact with the RAG backend by asking questions about job fit, missing skills, and role requirements. The frontend communicates with the backend APIs exposed by the NER and RAG modules.

## Install Frontend Dependencies

```bash
cd frontend
npm install
# or
yarn install
```
## Run Frontend 
```bash
cd frontend
npm run dev
# or (pls fill this up)
yarn dev
```

### Example Use case
1. Upload your resume.  
2. Enter a query:  "What roles in finance fit my background?"
3. The app will:
  - Match your skills with relevant job postings  
  - Highlight missing skills or gaps  
  - Suggest relevant upskilling courses  

---
## 📂 Repository Structure
```bash
DSA4213-Final-Project/
│
├── data/                # Synthetic and Kaggle job postings & resumes
│
├── src/                 # Core source code
│   ├── ner/             # Entity extraction modules
│   ├── retrieval/       # Hybrid retrieval modules
│   ├── rag/             # RAG integration with Flan-T5
│   └── utils/           # Helper functions
│
├── results/             # Evaluation report
│
├── frontend/            # Web application 
│   ├── index.html
│   ├── script.js        
│   └── style.css         
│
├── app.py               #  
│
├── requirements.txt     # Dependencies
├── README.md            # Project overview
└── .gitignore           # Ignore cache/large files
```

## 👥 Contributors

- Tan Hwee Li Rachel
- Caleb Tan Yong Yuan
- Teo Jing Kiat
- Kealan Kuar Wei Hao
