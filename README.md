# DSA4213-Final-Project : An NLP-Powered Job Matching System


## 📌 Overview
This project develops an **AI-powered career guidance system** to support fresh graduates in navigating Singapore’s competitive job market.  
The system leverages **Natural Language Processing (NLP)** techniques such as **Named Entity Recognition (NER)**, **semantic search**, and **Retrieval-Augmented Generation (RAG)** to:

- Match graduates’ skills with job postings.  
- Identify skill gaps and recommend relevant upskilling opportunities.  
- Provide **personalized, actionable career advice**.

---
## ✨ Features
- **NER Model (Fine-Tuned BERT)**  
  Extracts structured entities such as skills, qualifications, and tools from resumes and job postings.  

- **Hybrid Retrieval System**  
  Combines semantic search (dense embeddings via sentence-transformers) with keyword-based sparse retrieval for robust performance.  

- **Retrieval-Augmented Generation (RAG)**  
  Uses an instruction-tuned LLM (Flan-T5) to generate contextualized career advice in natural language.  

- **Evaluation Pipeline**  
  - NER: Precision, Recall, F1  
  - Retrieval: Precision@K, Mean Reciprocal Rank (MRR)  
  - System: Human evaluation on faithfulness, relevance, and helpfulness  

---
## 📂 Repository Structure
```bash
DSA4213-Final-Project/
│
├── data/                # Sample/anonymized job postings & resumes
├── notebooks/           # Jupyter notebooks for experiments
│   ├── 01-data-collection.ipynb
│   ├── 02-ner-training.ipynb
│   └── 03-rag-pipeline.ipynb
│
├── src/                 # Core source code
│   ├── ner/             # Entity extraction modules
│   ├── retrieval/       # Hybrid retrieval modules
│   ├── rag/             # RAG integration with Flan-T5
│   └── utils/           # Helper functions
│
├── tests/               # Unit tests
├── results/             # Evaluation reports, ablation studies
|
├── app/                   # Web application (Gradio / Streamlit)
│   ├── app.py
│   ├── components/        # UI components
│   └── static/            # Images, logos, styles
│
├── requirements.txt     # Dependencies
├── environment.yml      # (Optional) Conda environment file
├── README.md            # Project overview
├── LICENSE              # License (MIT)
└── .gitignore           # Ignore cache/large files
```

## 🌐 Demo

The system is deployed as an interactive web application.  
You can try it via:

- **Gradio Interface** (quick test queries, shareable link)  
- **Streamlit Dashboard** (full evaluation + visualizations)  

### Example
1. Upload your resume or enter your skills.  
2. Enter a query:  "What roles in finance fit my background?"
3. The app will:
  - Match your skills with relevant job postings  
  - Highlight missing skills or gaps  
  - Suggest relevant upskilling courses  

  👉 [Live Demo Link](https://your-demo-url-here) (to be added once deployed)

## 👥 Contributors

- Tan Hwee Li Rachel
- Caleb Tan Yong Yuan
- Teo Jing Kiat
- Kealan Kuar Wei Hao
