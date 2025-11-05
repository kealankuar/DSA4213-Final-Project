from neo4j import GraphDatabase
import json
import os
import ast
from tqdm import tqdm
from pathlib import Path

# -----------------------
# CONFIGURATION
# -----------------------

# Path to your embedding JSONL files
os.chdir(os.path.dirname(os.path.abspath(__file__)))
RESUME_FILE = Path("../../data/embeddings/resume_embeddings.jsonl").resolve()
JOB_FILE = Path("../../data/embeddings/job_embeddings.jsonl").resolve()
COURSE_FILE = Path("../../data/embeddings/course_embeddings.jsonl").resolve()

# Neo4j connection details 
NEO4J_URI="neo4j+s://accc1403.databases.neo4j.io"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="TwKAMXkflTo1NnauW3SEjMW4iXbExFIHA6oNi4mN1h4"


# ====== CONNECT TO DATABASE ======
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))


# ====== HELPER FUNCTIONS ======
def read_jsonl(path):
    """Read JSONL file and return list of dicts."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# --- Delete everything before reload (for testing) ---
def clear_graph(tx):
    tx.run("MATCH (n) DETACH DELETE n")
    print("Cleared all existing nodes and relationships.")

# Create contraints to ensure uniqueness
def create_constraints(tx):
    """Ensure uniqueness for essential entities only."""
    # Drop any existing constraints and indexes
    try:
        tx.run("DROP CONSTRAINT resume_text_unique IF EXISTS")
        tx.run("DROP CONSTRAINT job_text_unique IF EXISTS")
        tx.run("DROP CONSTRAINT course_title_unique IF EXISTS")
        tx.run("DROP INDEX resume_text IF EXISTS")
        tx.run("DROP INDEX job_text IF EXISTS")
    except Exception as e:
        print(f"Dropping constraints/indexes failed: {e}")

    # Create constraints only for small, frequently referenced entities
    constraints = [
            "CREATE CONSTRAINT skill_name_unique IF NOT EXISTS FOR (s:SKILL) REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT tool_name_unique IF NOT EXISTS FOR (t:TOOL) REQUIRE t.name IS UNIQUE",
            "CREATE CONSTRAINT org_name_unique IF NOT EXISTS FOR (o:ORG) REQUIRE o.name IS UNIQUE",
            "CREATE CONSTRAINT domain_name_unique IF NOT EXISTS FOR (d:DOMAIN) REQUIRE d.name IS UNIQUE"
    ]
    
    for c in constraints:
        try:
            tx.run(c)
        except Exception as e:
            print(f"Constraint creation failed: {e}")
            
    print("Constraints created or verified.")

def create_resume(tx, resume):
    """Create RESUME node and relationships with proper type handling."""
    try:
        # Create the resume node first - using CREATE since we don't require uniqueness
        result = tx.run(
            """
            CREATE (r:RESUME {text: $text, embedding: $embedding})
            RETURN elementId(r) as node_id
            """,
            text=resume["text"],
            embedding=resume.get("embedding", []),
        )
        resume_node_id = result.single()["node_id"]

        # Extract and parse the entities
        entities_str = resume.get('extracted_entities', '{}')
        if isinstance(entities_str, str):
            try:
                # Try ast.literal_eval first
                entities = ast.literal_eval(entities_str)
            except (ValueError, SyntaxError):
                # If that fails, try json.loads with quote replacement
                json_str = entities_str.replace("'", '"')
                entities = json.loads(json_str)
        else:
            entities = entities_str  # In case it's already a dictionary
            
        if not isinstance(entities, dict):
            raise ValueError(f"Entities must be a dictionary, got {type(entities)}")

        # Process each entity type
        for entity_type, ents in entities.items():
            for ent in ents:
                word = ent["word"].strip()
                if not word:
                    continue
                    
                if entity_type.upper() == "SKILL":
                    tx.run(
                        """
                        MERGE (s:SKILL {name: $word})
                        WITH s
                        MATCH (r:RESUME) WHERE elementId(r) = $resume_id
                        MERGE (r)-[:HAS_SKILL]->(s)
                        """,
                        word=word, resume_id=resume_node_id
                    )
                elif entity_type.upper() == "TOOLS":
                    tx.run(
                        """
                        MERGE (t:TOOL {name: $word})
                        WITH t
                        MATCH (r:RESUME) WHERE elementId(r) = $resume_id
                        MERGE (r)-[:HAS_TOOL]->(t)
                        """,
                        word=word, resume_id=resume_node_id
                    )
                elif entity_type.upper() == "ORG":
                    tx.run(
                        """
                        MERGE (o:ORG {name: $word})
                        WITH o
                        MATCH (r:RESUME) WHERE elementId(r) = $resume_id
                        MERGE (r)-[:WORKED_IN]->(o)
                        """,
                        word=word, resume_id=resume_node_id
                    )
    except Exception as e:
        print(f"Error processing resume: {str(e)}")
        print(f"Resume text: {resume.get('text', '')}")
        print(f"Entities: {resume.get('extracted_entities', '')}")
        raise


def create_job(tx, job):
    """Create JOB node and relationships with proper type handling."""
    try:
        # Create the job node first - using CREATE since we don't require uniqueness
        result = tx.run(
            """
            CREATE (j:JOB {text: $text, embedding: $embedding, document_type: $doc_type})
            RETURN elementId(j) as node_id
            """,
            text=job["text"],
            embedding=job.get("embedding", []),
            doc_type="job_description"
        )
        job_node_id = result.single()["node_id"]

        # Extract and parse the entities
        entities_str = job.get('extracted_entities', '{}')
        if isinstance(entities_str, str):
            try:
                # Try ast.literal_eval first
                entities = ast.literal_eval(entities_str)
            except (ValueError, SyntaxError):
                # If that fails, try json.loads with quote replacement
                json_str = entities_str.replace("'", '"')
                entities = json.loads(json_str)
        else:
            entities = entities_str  # In case it's already a dictionary
            
        if not isinstance(entities, dict):
            raise ValueError(f"Entities must be a dictionary, got {type(entities)}")

        # Process each entity type
        for entity_type, ents in entities.items():
            for ent in ents:
                word = ent["word"].strip()
                if not word:
                    continue
                    
                if entity_type.upper() == "SKILL":
                    tx.run(
                        """
                        MERGE (s:SKILL {name: $word})
                        WITH s
                        MATCH (j:JOB) WHERE elementId(j) = $job_id
                        MERGE (j)-[:REQUIRES_SKILL]->(s)
                        """,
                        word=word, job_id=job_node_id
                    )
                elif entity_type.upper() == "TOOLS":
                    tx.run(
                        """
                        MERGE (t:TOOL {name: $word})
                        WITH t
                        MATCH (j:JOB) WHERE elementId(j) = $job_id
                        MERGE (j)-[:REQUIRES_TOOL]->(t)
                        """,
                        word=word, job_id=job_node_id
                    )
                elif entity_type.upper() == "ORG":
                    tx.run(
                        """
                        MERGE (o:ORG {name: $word})
                        WITH o
                        MATCH (j:JOB) WHERE elementId(j) = $job_id
                        MERGE (j)-[:WORKING_IN]->(o)
                        """,
                        word=word, job_id=job_node_id
                    )
    except Exception as e:
        print(f"Error processing job: {str(e)}")
        print(f"Job text: {job.get('text', '')}")
        print(f"Entities: {job.get('extracted_entities', '')}")
        raise


def create_course(tx, course):
    """Create COURSE node and relationships with proper type handling."""
    try:
        # Create the course node first
        tx.run(
            """
            MERGE (c:COURSE {title: $title})
            SET c.url = $url,
                c.description = $description,
                c.embedding = $embedding
            """,
            title=course["title"],
            url=course.get("url"),
            description=course.get("description"),
            embedding=course.get("embedding", []),
            document_type="course"
        )

        # Create domain nodes
        domain = course.get("category")
        subdomain = course.get("sub_category")

        if domain:
            tx.run(
                """
                MERGE (d:DOMAIN {name: $domain})
                MERGE (c:COURSE {title: $title})
                MERGE (c)-[:COVERS_DOMAIN]->(d)
                """,
                domain=domain, title=course["title"]
            )

        if subdomain:
            tx.run(
                """
                MERGE (sd:DOMAIN {name: $subdomain})
                MERGE (c:COURSE {title: $title})
                MERGE (c)-[:COVERS_DOMAIN]->(sd)
                """,
                subdomain=subdomain, title=course["title"]
            )

        # Extract and parse the entities
        entities_str = course.get('extracted_entities', '{}')
        if isinstance(entities_str, str):
            try:
                # Try ast.literal_eval first
                entities = ast.literal_eval(entities_str)
            except (ValueError, SyntaxError):
                # If that fails, try json.loads with quote replacement
                json_str = entities_str.replace("'", '"')
                entities = json.loads(json_str)
        else:
            entities = entities_str  # In case it's already a dictionary
            
        if not isinstance(entities, dict):
            raise ValueError(f"Entities must be a dictionary, got {type(entities)}")

        # Process each entity type
        for entity_type, ents in entities.items():
            for ent in ents:
                word = ent["word"].strip()
                if not word:
                    continue
                    
                if entity_type.upper() == "SKILL":
                    tx.run(
                        """
                        MERGE (s:SKILL {name: $word})
                        MERGE (c:COURSE {title: $title})
                        MERGE (c)-[:TEACHES_SKILL]->(s)
                        """,
                        word=word, title=course["title"]
                    )
                elif entity_type.upper() == "TOOLS":
                    tx.run(
                        """
                        MERGE (t:TOOL {name: $word})
                        MERGE (c:COURSE {title: $title})
                        MERGE (c)-[:TEACHES_TOOL]->(t)
                        """,
                        word=word, title=course["title"]
                    )
    except Exception as e:
        print(f"Error processing course: {str(e)}")
        print(f"Course title: {course.get('title', '')}")
        print(f"Entities: {course.get('extracted_entities', '')}")
        raise


# ====== MAIN INGESTION LOGIC ======
def ingest_data():
    with driver.session() as session:
        print("Resetting graph database...")
        session.execute_write(clear_graph)

        print("Creating constraints...")
        session.execute_write(create_constraints)

        print("Inserting resumes...")
        for resume in tqdm(read_jsonl(RESUME_FILE)):
            session.execute_write(create_resume, resume)

        print("Inserting jobs...")
        for job in tqdm(read_jsonl(JOB_FILE)):
            session.execute_write(create_job, job)

        print("Inserting courses...")
        for course in tqdm(read_jsonl(COURSE_FILE)):
            session.execute_write(create_course, course)

    print("All data successfully inserted into Neo4j!")


if __name__ == "__main__":
    ingest_data()
    driver.close()
