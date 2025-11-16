from generation import RAGGenerator
from context_generation import context_generation_function
import json
from pathlib import Path

NEO4J_URI="neo4j+s://accc1403.databases.neo4j.io"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="TwKAMXkflTo1NnauW3SEjMW4iXbExFIHA6oNi4mN1h4"


def answer_generation_function(query_path, resume_path, uri, user, password):
    """
    Generates an answer based on the query using RAGGenerator.
    """
    generator = RAGGenerator()
    with open(query_path, 'r', encoding='utf-8') as f:
        query_data = json.load(f)
        query = query_data['text']
    # Generate context using the context generation function
    retrieved_context = context_generation_function(query_path, resume_path, uri, user, password)
    # Generate answer using the retrieved context
    answer = generator.generate_career_advice(query, retrieved_context)
    return query, answer, retrieved_context

if __name__ == "__main__":
    query_path = Path("../../input_data/queries/test_query/test_query.json").resolve()
    resume_path = Path("../../input_data/resumes/test_resume/test_resume.pdf").resolve()
    uri = NEO4J_URI
    user = NEO4J_USERNAME
    password = NEO4J_PASSWORD
    query, answer, retrieved_context= answer_generation_function(query_path, resume_path, uri, user, password)
    print("Generated Answer:")
    print(f"Query: \n{query}")
    print(f"Flan-T5 Model Generated Answer:\n{answer}")
    print(f"Retrieved Context:\n{retrieved_context}")