import json
from pathlib import Path
from datetime import datetime
from answer_generation import answer_generation_function

NEO4J_URI="neo4j+s://accc1403.databases.neo4j.io"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="TwKAMXkflTo1NnauW3SEjMW4iXbExFIHA6oNi4mN1h4"
EVALUATION_PATH="../../input_data/evaluation/collated_evaluation_path.jsonl"

def evaluation_function(collated_evaluation_path, uri, user, password):
    """
    Evaluates the answer generation function on a set of queries and resumes.
    """
    # Read JSONL file (multiple JSON objects, one per line)
    evaluation_data = []
    with open(collated_evaluation_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                evaluation_data.append(json.loads(line.strip()))
    
    results = []
    for entry in evaluation_data:
        query_path = Path(entry['query_path']).resolve()
        resume_path = Path(entry['resume_path']).resolve()
        query, generated_answer, retrieved_context = answer_generation_function(
            query_path,
            resume_path,
            uri,
            user,
            password
        )
        
        results.append({
            'query': query,
            'generated_answer': generated_answer,
            'retrieved_context': retrieved_context
        })

    # Always create timestamped file to avoid overwriting previous results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    evaluation_results_path_csv = f"../../input_data/evaluation/evaluation_results_{timestamp}.csv"
    
    with open(evaluation_results_path_csv, 'w', encoding='utf-8') as f:
        f.write("Query,Generated Answer,Retrieved Context\n")
        for result in results:
            query = result['query'].replace('"', '""')
            answer = result['generated_answer'].replace('"', '""')
            context = result['retrieved_context'].replace('"', '""')
            f.write(f'"{query}","{answer}","{context}"\n')
    
    print(f"Evaluation results saved to: {evaluation_results_path_csv}")



if __name__ == "__main__":
    evaluation_function(EVALUATION_PATH, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
