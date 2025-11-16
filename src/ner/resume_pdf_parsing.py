from pdfminer.high_level import extract_text
import re
from pathlib import Path
import pandas as pd
import sys
import json
from typing import Dict, List, Any, Optional

# --- CONFIGURATION ---
# Define the directory path where all your resume PDFs are located
RESUMES_DIR = Path("../../data/resumes/banking").resolve()

# Output file names
PARSED_CSV_FILE = "../../data/ner/parsed_resumes_dataframe.csv"
DOCANNO_JSONL_FILE = "../../data/ner/resume_test_data.jsonl"
# ---------------------

# --------------------------------------------------------
# CORE EXTRACTION & PARSING FUNCTIONS
# --------------------------------------------------------

def extract_text_from_pdf(pdf_path: Path) -> Optional[str]:
    """Extracts raw text from a PDF file and cleans it."""
    try:
        text = extract_text(pdf_path)
        # Clean up excessive whitespace/newlines
        return re.sub(r'\s+', ' ', text).strip()
    except Exception:
        return None

def extract_minimal_data(resume_text: str) -> Dict[str, Any]:
    """
    Extracts only the Full_Text. All other fields are discarded.
    """
    data: Dict[str, Any] = {}
    data['Full_Text'] = resume_text
    
    # We explicitly ignore Email, Skills, Name, and Organizations here.
    
    return data

# --------------------------------------------------------
# ANNOTATION PREPARATION FUNCTION (Tool-Agnostic)
# --------------------------------------------------------

def prepare_data_for_annotation(data: List[Dict[str, Any]], output_jsonl_path: Path):
    """
    Converts the raw text into the universal JSON Lines (JSONL) 
    format expected by annotation tools.
    """
    print(f"\n--- 3. Preparing {len(data)} documents for Annotation JSONL ---")
    
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for doc_data in data:
            # Only include documents with successfully extracted text
            if doc_data.get('Full_Text') and doc_data['Full_Text'] != 'N/A':
                document = {
                    "text": str(doc_data['Full_Text']),
                    "meta": {"file_name": doc_data.get('File_Name', 'N/A')}
                }
                f.write(json.dumps(document, ensure_ascii=False) + '\n')

    print(f"✅ Universal JSONL data saved to: {output_jsonl_path.name}")

# Processes PDF resumes to structured CSV + JSONL for annotation
def pdf_processing_pipeline(RESUMES_DIR, output_csv_path, output_jsonl_path):
    if not RESUMES_DIR.is_dir():
        print(f"❌ Error: Directory not found at {RESUMES_DIR}")
        sys.exit(1)
            
    all_resumes_data: List[Dict[str, Any]] = []
    pdf_files = list(RESUMES_DIR.glob("*.pdf"))
    print(f"Searching for PDF files in: {RESUMES_DIR}")

    print("\n" + "="*70)
    print(f"STARTING BATCH CONVERSION: {RESUMES_DIR.name} (Found {len(pdf_files)} PDFs)")
    print("="*70)
    
    if not pdf_files:
        print("No PDF files found. Process stopped.")
        sys.exit(0)

    # Extract Text and Parse Minimal Data ---
    for i, pdf_path in enumerate(pdf_files, 1):
        file_name = pdf_path.name
        print(f"\n[{i}/{len(pdf_files)}] Processing: {file_name}")
        
        resume_text = extract_text_from_pdf(pdf_path)
        
        if resume_text:
            extracted_info = extract_minimal_data(resume_text)
            extracted_info['File_Name'] = file_name
            all_resumes_data.append(extracted_info)
            
            print("    -> Extracted raw text only.")
        else:
            print("    -> FAILED to extract text.")
            all_resumes_data.append({'File_Name': file_name, 'Full_Text': 'N/A'})

    # 3. Final Output: Convert to Pandas DataFrame and save (Structured Data)
    if all_resumes_data:
        df = pd.DataFrame(all_resumes_data)
        
        # Define columns for the CSV output (Only File_Name and Full_Text)
        cols = ['File_Name', 'Full_Text']

        df = df.reindex(columns=cols) # Use .reindex to ensure only the desired columns exist
        df.to_csv(output_csv_path, index=False)
        
        print("\n" + "="*70)
        print(f"✅ CONVERSION COMPLETE: Minimal CSV saved to: {PARSED_CSV_FILE}")
        
        # --- Task 4: Prepare Data for Annotation: Outputs JSONL ---
        prepare_data_for_annotation(all_resumes_data, output_jsonl_path)
        print("="*70)

# --------------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------------


if __name__ == "__main__":
    pdf_processing_pipeline(RESUMES_DIR, Path(PARSED_CSV_FILE), Path(DOCANNO_JSONL_FILE))
