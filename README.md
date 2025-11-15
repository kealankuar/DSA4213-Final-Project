# DSA4213-Final-Project : Job-Seek – Resume Intelligence & Career Insights


## 📌 Overview
This repository contains the full pipeline for the **Job-Seek System**, an end-to-end platform designed to help users understand their resumes, identify missing skills, and query relevant job insights.
The system consists of three core modules:

- Named Entity Recognition (NER) → Extracts skills, qualifications, job titles, and tools
- Retrieval-Augmented Generation (RAG) → Answers user queries about job fit
- Frontend UI → User-facing interface for uploading resumes and interacting with the system

---
## 🧠 Named Entity Recognition (NER)

The NER module is responsible for extracting structured information such as skills, job titles, qualifications, organisations, and tools from resumes and job descriptions. The training script loads a token-label CSV file, prepares it into HuggingFace Dataset format, aligns labels to subword tokens, and runs an ablation study across multiple transformer models and learning rates. Each model is evaluated using SeqEval, and both the trained model weights and tokenizer are saved for downstream inference. A summary CSV is also generated for quick comparison.

---
## 📦 Installation
Install the required dependencies for the project:
```bash
pip install -r requirements.txt
```
---
## 🚀 Running NER Training
To train all NER models:
```bash
python train_ner.py
```
Running this will:
- preprocess the dataset
- train all model × learning rate combinations
- evaluate using precision, recall, F1, and accuracy
- save each trained model under ```models/ner_ablation_results/<model>_<lr>/``` 
- generate ablation_summary.csv with all results
---

---
## 📂 Repository Structure
```bash
DSA4213-Final-Project/
│
├── data/                # Sample/anonymized job postings & resumes
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
