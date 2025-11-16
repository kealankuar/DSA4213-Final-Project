# DSA4213-Final-Project : Job-Seek – Resume Intelligence & Career Insights


## 📌 Overview
This repository contains the full pipeline for the **Job-Seek System**, an end-to-end platform designed to help users understand their resumes, identify missing skills, and query relevant job insights.
The system consists of three core modules:

- Named Entity Recognition (NER) → Extracts skills, qualifications, job titles, and tools
- Retrieval-Augmented Generation (RAG) → Answers user queries about job fit
- Frontend UI → User-facing interface for uploading resumes and interacting with the system
---
## Getting Started

### Clone the Repository
```bash
git clone https://github.com/kealankuar/DSA4213-Final-Project.git
cd DSA4213-Final-Project
```

---
## Installation
Install the required dependencies for the project:
```bash
python -m pip install -r requirements.txt
```

### Configure Neo4j Database
Update your Neo4j credentials in the relevant scripts or create a configuration file with (currently configured to use our Neo4j database):
- URI: Your Neo4j connection string
- Username: Your database username
- Password: Your database password

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

## Generating Graph Nodes

Before running the RAG system, you need to populate the Neo4j graph database with resumes, job postings and course data. This creates the knowledge graph used for recommendations.

### Prerequisites
- Neo4j database running and accessible
- Course data in `data/courses/course_data.jsonl`
- Resume NER + embedding data in `data/embeddings/resume_embeddings.jsonl`
- Job NER + embedding data in `data/embeddings/job_embeddings.jsonl`

### Run Graph Generation
```bash
cd src/rag
python generate_graph_nodes.py
```

This will:
- Connect to your Neo4j database
- Create Resume, Job and Course nodes with their properties
- Generate embeddings for each node
- Create skill, tool, org and domain relationships
- Build the graph structure for retrieval

---

## 📊 Evaluation

The evaluation module assesses the quality of the RAG system's generated responses. It processes multiple test queries with their corresponding resumes and generates answers, which can then be analyzed for relevance, accuracy, and helpfulness.

### Running Evaluation
```bash
cd src/rag
python evaluation.py
```

This will:
- Load test queries from `input_data/evaluation/collated_evaluation_path.jsonl`
- Process each query-resume pair through the full RAG pipeline
- Generate answers using the LLM
- Save results to `results/evaluation_results_<timestamp>.csv`
- Include query, generated answer, context used

### Evaluation Metrics
The evaluation results can be analyzed for:
- Response relevance to the user query
- Quality of job and course recommendations
- Response coherence and helpfulness

---

## 🚀 Running the Application

### 1. Start the Backend Server
```bash
cd src/backend
python app.py
```
The Flask backend will start on `http://localhost:5000`

### 2. Start the Frontend Server
```bash
cd src/frontend
python -m http.server 8080
```
Open your browser and navigate to `http://localhost:8080/index.html`

### 3. Using the Application
1. Upload your resume (PDF format)
2. Enter a query (e.g., "What skills am I missing for finance roles?")
3. Click "Analyze Resume"
4. Wait for the AI to generate personalized career advice (~60-120 seconds)
5. View your results with job matches, missing skills, and course recommendations

---

### Example Use case
1. Upload your resume (PDF format)
2. Enter a query:  "What roles in finance fit my background?"
3. The app will:
  - Extract skills, tools, and qualifications from your resume using NER
  - Match your profile with relevant job postings from the Neo4j knowledge graph
  - Highlight missing skills or gaps for your target roles
  - Suggest relevant upskilling courses to bridge the gaps
  - Generate personalized career advice using AI

---
## 📂 Repository Structure
```bash
DSA4213-Final-Project/
│
├── data/                          # Training and source data
│   ├── courses/                   # Course data for recommendations
│   ├── embeddings/                # Pre-computed embeddings
│   ├── ner/                       # NER training datasets
│   └── resumes/                   # Sample resume PDFs
│
├── src/                           # Core source code
│   ├── backend/                   # Flask API server
│   │   └── app.py                 # Main backend application
│   ├── frontend/                  # Web UI
│   │   ├── index.html
│   │   ├── script.js        
│   │   └── style.css         
│   ├── ner/                       # Named Entity Recognition
│   │   ├── train_ner.py           # NER model training
│   │   ├── run_inference_fx.py    # NER inference functions
│   │   └── resume_pdf_parsing.py  # PDF text extraction
│   └── rag/                       # Retrieval-Augmented Generation
│       ├── answer_generation.py   # Main RAG pipeline
│       ├── context_generation.py  # Context retrieval
│       ├── evaluation.py          # RAG system evaluation
│       ├── generate_embeddings.py # Generate embeddings for entities
│       ├── generate_graph_nodes.py # Populate Neo4j database
│       ├── generation.py          # LLM text generation
│       ├── graph_retrieval.py     # Neo4j graph queries
│       ├── input_processing.py    # Process user inputs
│       └── retrieval_functions.py # Retrieval utility functions
│
├── input_data/                    # User-uploaded data
│   ├── evaluation/                # Evaluation test cases
│   │   └── collated_evaluation_path.jsonl  # Test query-resume pairs
│   ├── queries/                   # User query history
│   └── resumes/                   # Uploaded resume PDFs
|    
│
├── results/                       # NER and evaluation results
│   ├── evaluation_results_*.csv   # RAG evaluation outputs
│   └── *.csv                      # NER inference results
│
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
└── .gitignore                     # Git ignore rules
```

## 👥 Contributors

- Tan Hwee Li Rachel
- Caleb Tan Yong Yuan
- Teo Jing Kiat
- Kealan Kuar Wei Hao
