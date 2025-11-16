from neo4j import GraphDatabase
import numpy as np
import json
import ast

class CareerEngine:
    def flatten_entities(self, extracted_entities, min_score=0.6):
        """
        Flattens all entity types (e.g. SKILL, TOOL, ORG, etc.)
        into a single list of unique keywords above min_score.
        """
        keywords = []
        for ent_type, items in extracted_entities.items():
            for item in items:
                if item["score"] >= min_score:
                    keywords.append(item["word"])
        return list(set(keywords))  # remove duplicates

    def __init__(self, uri, user, password, query_embedding_path, resume_embedding_path):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.query_embedding_path = query_embedding_path
        self.resume_embedding_path = resume_embedding_path
        
        # Read query data (assumes single-line JSONL with all fields)
        with open(self.query_embedding_path, 'r', encoding='utf-8') as f:
            query_data = json.loads(f.readline().strip())
            self.query_text = query_data['text']
            self.query_embedding = query_data['embedding']
            # Parse extracted_entities (handles both JSON and Python dict strings)
            entities_raw = query_data['extracted_entities']
            if isinstance(entities_raw, str):
                try:
                    self.query_keywords_raw = json.loads(entities_raw)
                except json.JSONDecodeError:
                    # Fall back to ast.literal_eval for Python dict strings
                    self.query_keywords_raw = ast.literal_eval(entities_raw)
            else:
                self.query_keywords_raw = entities_raw
            self.query_keywords = self.flatten_entities(self.query_keywords_raw)
        
        # Read resume data (assumes single-line JSONL with all fields)
        with open(self.resume_embedding_path, 'r', encoding='utf-8') as f:
            resume_data = json.loads(f.readline().strip())
            self.resume_text = resume_data['text']
            self.resume_embedding = resume_data['embedding']
            # Parse extracted_entities (handles both JSON and Python dict strings)
            entities_raw = resume_data['extracted_entities']
            if isinstance(entities_raw, str):
                try:
                    self.resume_keywords_raw = json.loads(entities_raw)
                except json.JSONDecodeError:
                    # Fall back to ast.literal_eval for Python dict strings
                    self.resume_keywords_raw = ast.literal_eval(entities_raw)
            else:
                self.resume_keywords_raw = entities_raw
            self.resume_keywords = self.flatten_entities(self.resume_keywords_raw)
        
        # Ensure required indexes exist
        self._create_indexes_if_not_exist()
    
    def _create_indexes_if_not_exist(self):
        """Create required indexes if they don't exist."""
        with self.driver.session() as session:
            try:
                # 1. Fulltext index for job keyword search
                check_fulltext = "SHOW INDEXES YIELD name WHERE name = 'jobTextIndex'"
                if not list(session.run(check_fulltext)):
                    session.run("""
                        CREATE FULLTEXT INDEX jobTextIndex IF NOT EXISTS
                        FOR (j:JOB) ON EACH [j.text]
                    """)
                    print("Created fulltext index 'jobTextIndex'")
                
                # 2. Vector index for job semantic search (384 dimensions for all-MiniLM-L6-v2)
                check_job_vector = "SHOW INDEXES YIELD name WHERE name = 'jobEmbeddingIndex'"
                if not list(session.run(check_job_vector)):
                    session.run("""
                        CREATE VECTOR INDEX jobEmbeddingIndex IF NOT EXISTS
                        FOR (j:JOB) ON (j.embedding)
                        OPTIONS {indexConfig: {
                            `vector.dimensions`: 384,
                            `vector.similarity_function`: 'cosine'
                        }}
                    """)
                    print("Created vector index 'jobEmbeddingIndex'")
                    
            except Exception as e:
                print(f"Index creation warning: {e}")

    # ----------------------------------------------------------------------
    # 1. KEYWORD SEARCH (Neo4j full-text index)
    # ----------------------------------------------------------------------
    # Returns dict of {Neo4j node: keyword_score}
    def keyword_search_jobs(self, keywords, top_k=6):
        if not keywords:
            return {}

        # Escape Lucene special characters to prevent query parser errors
        def escape_lucene(text):
            # Lucene special chars: + - && || ! ( ) { } [ ] ^ " ~ * ? : \ /
            special_chars = r'+-&|!(){}[]^"~*?:\/'
            for char in special_chars:
                text = text.replace(char, f'\\{char}')
            return text
        
        # Escape each keyword and join with OR
        escaped_keywords = [escape_lucene(k) for k in keywords]
        query_string = " OR ".join(escaped_keywords)

        query = """
        CALL db.index.fulltext.queryNodes('jobTextIndex', $q)
        YIELD node, score
        RETURN node, score
        ORDER BY score DESC LIMIT $k
        """

        with self.driver.session() as session:
            res = session.run(query, q=query_string, k=top_k)
            return {r["node"]: r["score"] for r in res}

    # ----------------------------------------------------------------------
    # 2. SEMANTIC SEARCH (cosine similarity)
    # ----------------------------------------------------------------------
    # Returns dict of {Neo4j node: semantic_score}
    def semantic_search_jobs(self, embedding, top_k=6):
        # Use vector index for faster search
        query = """
        CALL db.index.vector.queryNodes('jobEmbeddingIndex', $k, $vec)
        YIELD node, score
        RETURN node, score
        """

        with self.driver.session() as session:
            try:
                res = session.run(query, vec=embedding, k=top_k)
                return {r["node"]: r["score"] for r in res}
            except Exception as e:
                # Fallback to non-indexed cosine similarity if index doesn't exist
                print(f"Vector index not available, using fallback: {e}")
                fallback_query = """
                MATCH (j:JOB)
                RETURN j AS node, vector.similarity.cosine(j.embedding, $vec) AS score
                ORDER BY score DESC LIMIT $k
                """
                res = session.run(fallback_query, vec=embedding, k=top_k)
                return {r["node"]: r["score"] for r in res}

    # ----------------------------------------------------------------------
    # 3. HYBRID SEARCH (semantic + keyword on combined keywords)
    # ----------------------------------------------------------------------
    # Returns list of (Neo4j node, combined_score) tuples, sorted by score descending
    def hybrid_search_jobs(self, embedding, keywords, resume_text=None, resume_entities=None, alpha=0.6, top_k=6):
        """
        Combines hybrid search (semantic + keyword) with graph traversal.
        
        Args:
            embedding: Resume embedding vector
            keywords: Combined keywords from resume and query
            resume_text: Resume text for creating temporary resume node (optional)
            resume_entities: Dict of resume entities {'skills': set(), 'tools': set(), ...} (optional)
            alpha: Weight for semantic search (1-alpha for keyword)
            top_k: Total number of jobs to return
            
        Returns: 
            list of (Neo4j node, combined_score) tuples, sorted by score descending
            - Half from hybrid search (semantic + keyword)
            - Half from graph traversal (entity matching)
        """
        
        half_k = top_k // 2
        remainder = top_k % 2
        
        # Part 1: Traditional hybrid search (semantic + keyword)
        semantic_scores = self.semantic_search_jobs(embedding, top_k=top_k)
        keyword_scores = self.keyword_search_jobs(keywords, top_k=top_k)

        # Normalize both sets using softmax
        semantic_norm = self._softmax_normalize(semantic_scores)
        keyword_norm = self._softmax_normalize(keyword_scores)

        # Combine
        hybrid_scores = {}
        for job in set(list(semantic_norm.keys()) + list(keyword_norm.keys())):
            s = semantic_norm.get(job, 0)
            k = keyword_norm.get(job, 0)
            hybrid_scores[job] = alpha * s + (1 - alpha) * k

        # Get top half from hybrid search
        hybrid_ranked = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        top_hybrid = hybrid_ranked[:half_k + remainder]
        
        # Part 2: Graph traversal based on entity matching
        graph_jobs = []
        resume_text = self.resume_text
        resume_entities = self.resume_keywords_raw
        if resume_text and resume_entities:
            print(f"DEBUG: Calling graph traversal with resume_entities type: {type(resume_entities)}")
            print(f"DEBUG: resume_entities keys: {resume_entities.keys() if isinstance(resume_entities, dict) else 'NOT A DICT'}")
            if isinstance(resume_entities, dict):
                print(f"DEBUG: Skills: {resume_entities.get('skills', 'NO SKILLS KEY')}")
                print(f"DEBUG: Tools: {resume_entities.get('tools', 'NO TOOLS KEY')}")
            graph_jobs = self._graph_traversal_search(resume_text, resume_entities, top_k=half_k, 
                                                     exclude_jobs=set(job for job, _ in top_hybrid))
        else:
            print(f"DEBUG: Graph traversal skipped - resume_text={bool(resume_text)}, resume_entities={bool(resume_entities)}")
        
        # Combine both results
        combined_results = list(top_hybrid) + graph_jobs
        
        return combined_results

    # ----------------------------------------------------------------------
    # 3b. GRAPH TRAVERSAL SEARCH
    # ----------------------------------------------------------------------
    def _graph_traversal_search(self, resume_text, resume_entities, top_k=6, exclude_jobs=None):
        """
        Create temporary resume node and find jobs through graph traversal.
        
        Args:
            resume_text: Text of the resume
            resume_entities: Dict with 'skills', 'tools', etc.
            top_k: Number of jobs to return
            exclude_jobs: Set of job nodes to exclude
            
        Returns:
            list of (job_node, match_score) tuples
        """
        if exclude_jobs is None:
            exclude_jobs = set()
        
        # Convert raw entities to standardized format if needed
        if isinstance(resume_entities, dict):
            # Check if it's raw format (uppercase keys like 'SKILL', 'TOOL')
            if 'SKILL' in resume_entities or 'TOOL' in resume_entities:
                print(f"DEBUG: Converting raw entities to standardized format")
                resume_entities = self.extract_skills_from_entities(resume_entities)
        
        # Extract entity lists
        skills = list(resume_entities.get('skills', set()))
        tools = list(resume_entities.get('tools', set()))
        orgs = list(resume_entities.get('orgs', set())) if 'orgs' in resume_entities else []
        
        print(f"DEBUG: Graph traversal extracting - skills count: {len(skills)}, tools count: {len(tools)}, orgs count: {len(orgs)}")
        if skills:
            print(f"DEBUG: First 5 skills: {skills[:5]}")
        if tools:
            print(f"DEBUG: First 5 tools: {tools[:5]}")
        
        # Create temporary resume node and find matching jobs via graph traversal
        query = """
        // Create temporary resume node with unique timestamp
        CREATE (temp_resume:TEMP_RESUME {text: $resume_text, timestamp: $timestamp})
        
        // Connect to SKILL nodes
        WITH temp_resume
        FOREACH (skill_name IN $skills |
            MERGE (s:SKILL {name: skill_name})
            MERGE (temp_resume)-[:HAS_SKILL]->(s)
        )
        
        // Connect to TOOL nodes
        WITH temp_resume
        FOREACH (tool_name IN $tools |
            MERGE (t:TOOL {name: tool_name})
            MERGE (temp_resume)-[:HAS_TOOL]->(t)
        )
        
        // Connect to ORG nodes
        WITH temp_resume
        FOREACH (org_name IN $orgs |
            MERGE (o:ORG {name: org_name})
            MERGE (temp_resume)-[:WORKED_IN]->(o)
        )
        
        // Find jobs that require matching skills
        WITH temp_resume
        OPTIONAL MATCH (temp_resume)-[:HAS_SKILL]->(s:SKILL)<-[:REQUIRES_SKILL]-(j:JOB)
        WITH temp_resume, j, COUNT(DISTINCT s) AS skill_matches
        
        // Find jobs that require matching tools
        OPTIONAL MATCH (temp_resume)-[:HAS_TOOL]->(t:TOOL)<-[:REQUIRES_TOOL]-(j)
        WITH temp_resume, j, skill_matches, COUNT(DISTINCT t) AS tool_matches
        
        // Find jobs from matching orgs
        OPTIONAL MATCH (temp_resume)-[:WORKED_IN]->(o:ORG)<-[:WORKING_IN]-(j)
        WITH temp_resume, j, skill_matches, tool_matches, COUNT(DISTINCT o) AS org_matches
        
        // Calculate total matches and filter
        WITH temp_resume, j, (skill_matches + tool_matches + org_matches) AS total_matches
        WHERE j IS NOT NULL AND total_matches > 0 AND NOT elementId(j) IN $exclude_jobs
        
        // Clean up temp node
        WITH j, total_matches, temp_resume
        DETACH DELETE temp_resume
        
        RETURN j AS job, total_matches AS score
        ORDER BY score DESC
        LIMIT $k
        """
        
        with self.driver.session() as session:
            # Generate unique timestamp
            import time
            current_timestamp = int(time.time() * 1000)
            
            try:
                result = session.run(
                    query,
                    resume_text=resume_text,
                    skills=skills if skills else [],
                    tools=tools if tools else [],
                    orgs=orgs if orgs else [],
                    exclude_jobs=[job.element_id if hasattr(job, 'element_id') else str(job) for job in exclude_jobs],
                    timestamp=current_timestamp,
                    k=top_k
                )
                
                jobs = [(r["job"], float(r["score"])) for r in result]
                print(f"Graph traversal search found {len(jobs)} jobs with skills={len(skills)}, tools={len(tools)}")
                return jobs
            except Exception as e:
                print(f"Graph traversal search error: {e}")
                # Fallback: try cleanup
                try:
                    session.run("MATCH (temp:TEMP_RESUME) DETACH DELETE temp")
                except:
                    pass
                return []

    # ----------------------------------------------------------------------
    # NORMALIZATION UTIL
    # ----------------------------------------------------------------------
    # Returns dict of {key: softmax_normalized_value}
    def _softmax_normalize(self, scores_dict):
        """Apply softmax normalization to scores."""
        if not scores_dict:
            return {}

        values = np.array(list(scores_dict.values()))
        
        # Apply softmax: exp(x) / sum(exp(x))
        exp_values = np.exp(values - np.max(values))  # Subtract max for numerical stability
        softmax_values = exp_values / np.sum(exp_values)
        
        return {k: softmax_values[i] for i, k in enumerate(scores_dict.keys())}

    # ----------------------------------------------------------------------
    # ENTITY CONVERSION HELPER
    # ----------------------------------------------------------------------
    def extract_skills_from_entities(self, resume_entities, min_score=0.6):
        """
        Convert resume_entities dict to resume_skills format.
        
        Args:
            resume_entities: Dict with entity types as keys (e.g., 'SKILL', 'TOOL', 'ORG')
                            Each value is a list of dicts with 'word' and 'score' keys
                            Example: {'SKILL': [{'word': 'Python', 'score': 0.95}, ...],
                                     'TOOL': [{'word': 'Docker', 'score': 0.88}, ...]}
            min_score: Minimum confidence score to include an entity (default: 0.6)
        
        Returns:
            Dict with 'skills' and 'tools' keys, each containing a set of entity names
            Example: {'skills': {'Python', 'Java'}, 'tools': {'Docker', 'Git'}}
        """
        skills = set()
        tools = set()
        
        # Extract SKILL entities
        if 'SKILL' in resume_entities:
            for item in resume_entities['SKILL']:
                if item.get('score', 0) >= min_score:
                    skills.add(item['word'])
        
        # Extract TOOL entities
        if 'TOOL' in resume_entities:
            for item in resume_entities['TOOL']:
                if item.get('score', 0) >= min_score:
                    tools.add(item['word'])
        
        return {
            'skills': skills,
            'tools': tools
        }

    # ----------------------------------------------------------------------
    # SKILL EXTRACTION FROM GRAPH
    # ----------------------------------------------------------------------
    # Returns dict with 'skills' and 'tools' sets from resume node
    def get_resume_skills(self, resume_node):
        skills_query = """
        MATCH (r:RESUME)-[:HAS_SKILL]->(s:SKILL)
        WHERE elementId(r) = $id
        RETURN s.name AS name
        """
        tools_query = """
        MATCH (r:RESUME)-[:HAS_TOOL]->(t:TOOL)
        WHERE elementId(r) = $id
        RETURN t.name AS name
        """
        with self.driver.session() as session:
            skills_res = session.run(skills_query, id=resume_node.element_id)
            tools_res = session.run(tools_query, id=resume_node.element_id)
            return {
                'skills': {r["name"] for r in skills_res},
                'tools': {r["name"] for r in tools_res}
            }
        
    # Returns dict with 'skills' and 'tools' sets from job node
    def get_job_required_skills(self, job_node):
        skills_query = """
        MATCH (j:JOB)-[:REQUIRES_SKILL]->(s:SKILL)
        WHERE elementId(j) = $id
        RETURN s.name AS name
        """
        tools_query = """
        MATCH (j:JOB)-[:REQUIRES_TOOL]->(t:TOOL)
        WHERE elementId(j) = $id
        RETURN t.name AS name
        """
        with self.driver.session() as session:
            skills_res = session.run(skills_query, id=job_node.element_id)
            tools_res = session.run(tools_query, id=job_node.element_id)
            return {
                'skills': {r["name"] for r in skills_res},
                'tools': {r["name"] for r in tools_res}
            }

    # ----------------------------------------------------------------------
    # COMPUTE MISSING SKILLS
    # ----------------------------------------------------------------------
    # Returns dict with 'skills' and 'tools' sets of missing items
    def compute_missing_skills(self, resume_skills, similar_jobs):
        required_skills = set()
        required_tools = set()

        for job, _score in similar_jobs:
            job_req = self.get_job_required_skills(job)
            required_skills |= job_req['skills']
            required_tools |= job_req['tools']

        missing_skills = required_skills - resume_skills['skills']
        missing_tools = required_tools - resume_skills['tools']
        
        return {
            'skills': missing_skills,
            'tools': missing_tools
        }

    # ----------------------------------------------------------------------
    # RECOMMEND COURSES FROM MISSING SKILLS
    # ----------------------------------------------------------------------
    # Returns list of (course_node, match_count) tuples, sorted by match count descending
    def recommend_courses(self, missing_skills, top_k=6):
        # Separate skills and tools
        missing_skill_list = list(missing_skills['skills'])
        missing_tool_list = list(missing_skills['tools'])
        
        if not missing_skill_list and not missing_tool_list:
            return []

        # Query for courses that teach either missing skills or tools
        query = """
        OPTIONAL MATCH (c:COURSE)-[:TEACHES_SKILL]->(s:SKILL)
        WHERE s.name IN $skills
        WITH c, COUNT(DISTINCT s) AS skill_matches
        OPTIONAL MATCH (c)-[:TEACHES_TOOL]->(t:TOOL)
        WHERE t.name IN $tools
        WITH c, skill_matches, COUNT(DISTINCT t) AS tool_matches
        WHERE skill_matches > 0 OR tool_matches > 0
        RETURN c AS course, (skill_matches + tool_matches) AS matches
        ORDER BY matches DESC
        LIMIT $k
        """

        with self.driver.session() as session:
            res = session.run(query, skills=missing_skill_list, tools=missing_tool_list, k=top_k)
            return [(r["course"], r["matches"]) for r in res]

    # ----------------------------------------------------------------------
    # HIGH-LEVEL PIPELINES
    # ----------------------------------------------------------------------

    # 1. Courses to Recommend
    # Returns list of (course_node, match_count) tuples, sorted by match count descending
    def courses_to_recommend(self, resume_embedding, resume_keywords, query_keywords, resume_skills, resume_text=None, resume_entities=None):
        combined_keywords = list(set(resume_keywords + query_keywords))

        # Retrieve similar jobs using hybrid search + graph traversal
        similar_jobs = self.hybrid_search_jobs(
            embedding=resume_embedding,
            keywords=combined_keywords,
            resume_text=resume_text,
            resume_entities=resume_entities,
            alpha=0.6,
            top_k=30
        )

        missing = self.compute_missing_skills(resume_skills, similar_jobs)
        return self.recommend_courses(missing)

    # 2. Missing Skills
    # Returns dict with 'skills' and 'tools' sets of missing items
    def get_missing_skills_from_resume(self, resume_embedding, resume_keywords, query_keywords, resume_skills, resume_text=None, resume_entities=None):
        combined_keywords = list(set(resume_keywords + query_keywords))

        similar_jobs = self.hybrid_search_jobs(
            embedding=resume_embedding,
            keywords=combined_keywords,
            resume_text=resume_text,
            resume_entities=resume_entities,
            alpha=0.6,
            top_k=6
        )

        return self.compute_missing_skills(resume_skills, similar_jobs)

    # 3. Job Recommendation
    # Returns list of (job_node, combined_score) tuples, sorted by score descending
    def recommend_jobs(self, resume_embedding, resume_keywords, query_keywords, resume_text=None, resume_entities=None):
        combined_keywords = list(set(resume_keywords + query_keywords))

        resume_text = self.resume_text if resume_text is None else resume_text
        resume_entities = self.resume_keywords_raw if resume_entities is None else resume_entities
        ranked_jobs = self.hybrid_search_jobs(
            embedding=resume_embedding,
            keywords=combined_keywords,
            resume_text=resume_text,
            resume_entities=resume_entities,
            alpha=0.6,
            top_k=6
        )

        return ranked_jobs

    # ----------------------------------------------------------------------
    # 4. GET FULL CONTEXT (Combined Function)
    # ----------------------------------------------------------------------
    def get_full_context(self, resume_embedding=None, resume_keywords=None, query_keywords=None, 
                        resume_skills=None, resume_text=None, resume_entities=None, 
                        top_courses=6, top_jobs=6):
        """
        Get comprehensive career recommendations including missing skills, courses, and jobs.
        All parameters default to CareerEngine attributes if not provided.
        
        Args:
            resume_embedding: Embedding vector from resume (default: self.resume_embedding)
            resume_keywords: List of keywords from resume (default: self.resume_keywords)
            query_keywords: List of keywords from query (default: self.query_keywords)
            resume_skills: Dict with 'skills' and 'tools' sets from resume (optional, must be provided)
            resume_text: Resume text for graph traversal (optional)
            resume_entities: Dict with 'skills', 'tools', 'orgs' sets (optional)
            top_courses: Number of courses to recommend (default 10)
            top_jobs: Number of jobs to recommend (default 10)
        
        Returns:
            str: Formatted string with all recommendations
        """
        
        # Use defaults from CareerEngine attributes if not provided
        if resume_embedding is None:
            resume_embedding = self.resume_embedding
        if resume_keywords is None:
            resume_keywords = self.resume_keywords
        if query_keywords is None:
            query_keywords = self.query_keywords
        if resume_text is None:
            resume_text = self.resume_text
        if resume_entities is None:
            resume_entities = self.resume_keywords_raw  # Use raw extracted entities
        
        # Extract resume_skills from resume_entities if not provided
        if resume_skills is None:
            resume_skills = self.extract_skills_from_entities(resume_entities)
        
        # 1. Get missing skills
        missing_skills_dict = self.get_missing_skills_from_resume(
            resume_embedding=resume_embedding,
            resume_keywords=resume_keywords,
            query_keywords=query_keywords,
            resume_skills=resume_skills,
            resume_text=resume_text,
            resume_entities=resume_entities
        )
        
        # 2. Get recommended courses
        recommended_courses = self.courses_to_recommend(
            resume_embedding=resume_embedding,
            resume_keywords=resume_keywords,
            query_keywords=query_keywords,
            resume_skills=resume_skills,
            resume_text=resume_text,
            resume_entities=resume_entities
        )[:top_courses]
        
        # 3. Get recommended jobs
        recommended_jobs = self.recommend_jobs(
            resume_embedding=resume_embedding,
            resume_keywords=resume_keywords,
            query_keywords=query_keywords,
            resume_text=resume_text,
            resume_entities=resume_entities
        )[:top_jobs]
        
        # Format the output
        output = []
        output.append("=" * 80)
        output.append("CAREER RECOMMENDATIONS")
        output.append("=" * 80)
        output.append("")
        
        # Missing skills from resume
        output.append("Missing skills from resume:")
        if missing_skills_dict['skills'] or missing_skills_dict['tools']:
            if missing_skills_dict['skills']:
                output.append("  Skills:")
                for skill in sorted(missing_skills_dict['skills']):
                    output.append(f"    - {skill}")
            if missing_skills_dict['tools']:
                output.append("  Tools:")
                for tool in sorted(missing_skills_dict['tools']):
                    output.append(f"    - {tool}")
        else:
            output.append("  No missing skills or tools identified")
        output.append("")
        
        # Courses to recommend
        output.append("Courses to recommend:")
        if recommended_courses:
            for idx, (course, match_count) in enumerate(recommended_courses, 1):
                course_title = course.get('title', 'N/A')
                course_url = course.get('url', 'N/A')
                course_description = course.get('description', 'N/A')
                output.append(f"  {idx}. {course_title}")
                output.append(f"     Skills/Tools Matched: {match_count}")
                output.append(f"     URL: {course_url}")
                output.append(f"     Description: {course_description}")
                output.append("")
        else:
            output.append("  No courses to recommend")
            output.append("")
    
        # Jobs to recommend
        output.append("Jobs to recommend:")
        if recommended_jobs:
            for idx, (job, score) in enumerate(recommended_jobs, 1):
                job_description = job.get('text', 'N/A')
                output.append(f"  {idx}. Job Description: {job_description}...")
                if idx < len(recommended_jobs)/2 + 1:
                    output.append(f"     Match Score: {score:.4f}")
                else:
                    output.append(f"     Number of SKILLS/TOOLS Matched: {score:.4f}")
                output.append("")
        else:
            output.append("  No jobs to recommend")
            output.append("")
        
        output.append("=" * 80)
        
        return "\n".join(output)
