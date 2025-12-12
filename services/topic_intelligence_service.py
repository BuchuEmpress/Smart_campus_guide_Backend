"""
Topic Intelligence Service - AI-Powered Features

This module contains the TopicIntelligenceService class, which provides
AI-driven features for topic management, including suggestion generation (via Gemini)
and similarity checking (via Sentence Transformers embeddings).

The embedding model is loaded lazily to prevent memory exhaustion during application startup or testing.
"""

import logging
import json
from typing import List, Dict, Optional, Any, Union
import numpy as np

# Keep the import, but move the initialization out of __init__
from sentence_transformers import SentenceTransformer

# Assuming these are defined in the project structure
from services.gemini_service import GeminiService
from services.topic_service import TopicService 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TopicIntelligenceService:
    """
    AI-powered topic intelligence service.
    
    Provides smart features for topic management using Gemini AI.
    The SentenceTransformer model is loaded lazily to conserve memory during startup/testing.
    """
    
    def __init__(
        self,
        topic_service: Optional[TopicService] = None,
        gemini_service: Optional[GeminiService] = None
    ):
        """Initialize intelligence service."""
        self.topic_service = topic_service or TopicService()
        self.gemini = gemini_service or GeminiService()
        
        # Set model to None for lazy loading
        self.model: Optional[SentenceTransformer] = None
        self.embedding_size = 384  # all-MiniLM-L6-v2 size
        
        logger.info("Topic intelligence service initialized. Embedding model will be loaded lazily.")
        
    
    # ====================================================================
    # LAZY LOADING METHOD
    # ====================================================================
    def _ensure_model_loaded(self):
        """
        Loads the SentenceTransformer model (all-MiniLM-L6-v2) if it hasn't been loaded yet.
        
        This prevents the memory-intensive operation from running during application import,
        fixing the memory/paging file OSErrors during test collection.
        
        Raises:
            Exception: If the model fails to load.
        """
        if self.model is None:
            logger.info("Loading embedding model for similarity checking (Lazy Load)...")
            try:
                # The heavy operation, now deferred
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Embedding model loaded successfully ✅")
            except Exception as e:
                logger.error(f"Failed to load SentenceTransformer model: {e}")
                raise
    
    # ====================================================================
    # CENTRALIZED EMBEDDING METHOD (for reuse by check_similarity)
    # ====================================================================
    def get_embedding(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generates a vector embedding (or batch embeddings) for the given text(s).

        The method ensures the underlying SentenceTransformer model is loaded first.
        
        Args:
            text: A single string or a list of strings to be embedded.

        Returns:
            A NumPy array representing the embedding vector(s).
        """
        self._ensure_model_loaded() # Load model if necessary
        # The SentenceTransformer encode method is synchronous and returns a numpy array
        return self.model.encode(text)

    # ====================================================================
    # GEMINI AI METHODS
    # ====================================================================
    
    def suggest_topics(
        self,
        category: str,
        department: str,
        count: int = 5,
        keywords: Optional[List[str]] = None,
        user_request: Optional[str] = None
    ) -> List[Dict]:
        """
        Generate unique topic suggestions using Gemini AI.
        """
        try:
            logger.info(f"Generating {count} topic suggestions for {category}")
            
            # Smart handling: If keywords contain a long sentence, treat it as user_request
            # This handles cases where the frontend might send the query as a keyword
            if keywords and not user_request:
                # Check if any keyword is long and has spaces (likely a sentence)
                sentence_keywords = [k for k in keywords if len(str(k)) > 20 and ' ' in str(k)]
                if sentence_keywords:
                    user_request = " ".join(sentence_keywords)
                    # Remove used keywords from the list to avoid duplication
                    keywords = [k for k in keywords if k not in sentence_keywords]
                    logger.info(f"Extracted user request from keywords: {user_request}")
            
            # Get existing topics to avoid duplicates
            existing_topics = self.topic_service.list_topics(
                filter_query={'category': category},
                limit=100
            )
            
            existing_titles = [t['title'] for t in existing_topics]
            
            # Build prompt for Gemini
            keywords_text = ""
            if keywords:
                # Clean keywords
                clean_keywords = [str(k).strip() for k in keywords if str(k).strip()]
                if clean_keywords:
                    keywords_text = f"\nFocus areas/Keywords: {', '.join(clean_keywords)}"
            
            # Handle user request if provided
            user_request_section = ""
            if user_request:
                user_request_section = f"""
The user wants project topics for this specific request:
"{user_request}"

Generate {count} detailed project topics aligned with what the user is asking.
Do not repeat input literally. Create real academic topics.
"""
            else:
                user_request_section = f"Generate {count} unique, innovative final year project topics."
            
            prompt = f"""
You are an expert academic advisor helping students choose final year project topics.

Context:
- Department: {department}
- Specialization Code: {category} (If this is an abbreviation like SEN, SWE, AI, etc., interpret it as the full field name, e.g., Software Engineering, Artificial Intelligence).
{keywords_text}

{user_request_section}

Requirements:
1. **Intelligent Interpretation**: Do not just keyword-match. Understand the *intent* of the request.
2. **Academic Quality**: Topics must be suitable for a final year defense.
3. **Originality**: NOT similar to these existing topics:
{chr(10).join(f"  - {title}" for title in existing_titles[:20])}

4. **Structure**: Each topic must have a professional title, a detailed technical description, and metadata.
5. **Hybrid Topics**: Where feasible, combine the specialization ({category}) with modern fields like AI, Blockchain, IoT, Cloud, etc.

Return ONLY a JSON array with this exact structure:
[
  {{
    "title": "Professional Topic Title",
    "description": "Detailed technical description (50-100 words) explaining the problem, solution, and technology.",
    "difficulty": "beginner|intermediate|advanced",
    "tags": ["tag1", "tag2", "tag3"],
    "prerequisites": ["Required Skill 1", "Required Skill 2"],
    "estimated_duration": "4-6 months"
  }}
]

NO markdown, NO explanations, ONLY the JSON array.
"""
            
            # Call Gemini and handle response cleanup
            response = self.gemini.model.generate_content(prompt)
            response_text = response.text.strip()
            
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            response_text = response_text.strip()
            
            # Parse JSON
            suggestions = json.loads(response_text)
            
            logger.info(f"Generated {len(suggestions)} topic suggestions")
            return suggestions
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini response: {str(e)}")
            return self._fallback_suggestions(category, count)
        except Exception as e:
            logger.error(f"Error generating suggestions: {str(e)}")
            return self._fallback_suggestions(category, count)
    
    def improve_topic(
        self, 
        title: str, 
        description: str, 
        category: str,
        user_instruction: Optional[str] = None
    ) -> Dict:
        """
        Use Gemini AI to improve a topic's title and description.
        
        Args:
            title: Current topic title
            description: Current topic description
            category: Topic category
            user_instruction: Optional instruction from the user
        
        Returns:
            Dictionary with improved title, description, keywords, and difficulty
        """
        try:
            logger.info(f"Improving topic: {title}")
            
            user_instruction_text = ""
            if user_instruction:
                user_instruction_text = f"""
User Instruction for Improvement:
"{user_instruction}"
(Use this instruction to guide the improvement, but DO NOT include this text in the output description.)
"""
            
            prompt = f"""
You are an expert academic editor improving a final year project topic.

Original Topic:
- Title: "{title}"
- Description: "{description}"
- Specialization: {category}

{user_instruction_text}

Task:
1. **Refine Title**: Make it professional, academic, and specific. Avoid generic terms.
2. **Expand Description**: Write a comprehensive 100-150 word abstract. It should cover the problem statement, proposed solution, methodology, and expected impact.
3. **Keywords**: Suggest 5-7 relevant technical tags.
4. **Difficulty**: Assess the technical complexity.

Requirements:
- The output must be polished and ready for a project defense proposal.
- Do NOT repeat the user instruction in the description.
- Interpret abbreviations in the specialization (e.g., SEN -> Software Engineering).

Example of Desired Transformation:
Input Title: "Smart Dress"
Input Description: "app that helps people dress well"
Output Title: "AI-Powered Smart Fashion Recommendation and Outfit Optimization System"
Output Description: "This project proposes an intelligent fashion recommendation application that leverages artificial intelligence to analyze user preferences, body profile, clothing inventory, and current fashion trends. The system will generate personalized outfit suggestions, color-matching guidance, and occasion-based styling recommendations. It also incorporates machine-learning techniques to continuously refine suggestions based on user feedback, seasonal trends, and real-time environmental factors."

Return ONLY a JSON object with this exact structure:
{{
  "improved_title": "Enhanced Professional Title",
  "improved_description": "Comprehensive academic description...",
  "suggested_tags": ["tag1", "tag2", "tag3", "tag4", "tag5"],
  "suggested_difficulty": "intermediate"
}}

NO markdown, NO explanations, ONLY the JSON object.
"""
            
            # Call Gemini
            response = self.gemini.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Clean response
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            response_text = response_text.strip()
            
            # Parse JSON
            result = json.loads(response_text)
            
            logger.info(f"Successfully improved topic: {title}")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini response: {str(e)}")
            # Return original with minimal changes
            return {
                'improved_title': title,
                'improved_description': description,
                'suggested_tags': [],
                'suggested_difficulty': 'intermediate'
            }
        except Exception as e:
            logger.error(f"Error improving topic: {str(e)}")
            return {
                'improved_title': title,
                'improved_description': description,
                'suggested_tags': [],
                'suggested_difficulty': 'intermediate'
            }
    
    # ====================================================================
    # EMBEDDING & SIMILARITY METHODS
    # ====================================================================
    
    def check_similarity(self, title: str, category: str, threshold: float = 0.8) -> List[Dict]:
        """
        Check if topic is similar to existing topics using embeddings.

        Args:
            title: Topic title to check.
            category: Category to search in.
            threshold: Similarity threshold (0-1).
        
        Returns:
            List of similar topics with similarity scores.
        """
        try:
            logger.info(f"Checking similarity for: {title}")
            
            # 1. Get existing topics
            existing = self.topic_service.list_topics(
                filter_query={'category': category},
                limit=100
            )
            
            if not existing:
                return []
            
            existing_titles = [topic['title'] for topic in existing]
            
            # 2. Generate embeddings in efficient batch calls (Now uses the centralized method)
            logger.info(f"Generating batch embeddings for {len(existing_titles)} existing topics (Lazy Load Triggered)...")
            
            # New topic embedding (uses the new centralized method)
            new_embedding = self.get_embedding(title)
            
            # Batch encode existing topics
            existing_embeddings = self.get_embedding(existing_titles)
            
            # 3. Calculate similarities (Cosine similarity: dot product of L2-normalized vectors)
            # Normalize vectors for correct dot product similarity calculation
            new_embedding_norm = new_embedding / np.linalg.norm(new_embedding)
            # Normalize existing embeddings matrix (axis=1 normalizes rows)
            existing_embeddings_norm = existing_embeddings / np.linalg.norm(existing_embeddings, axis=1, keepdims=True)
            
            # Calculate cosine similarity (dot product of normalized vectors)
            similarities = np.dot(existing_embeddings_norm, new_embedding_norm)

            # 4. Filter results
            similar_topics = []
            for i, similarity in enumerate(similarities):
                if similarity >= threshold:
                    topic = existing[i]
                    similar_topics.append({
                        'topic_id': str(topic['_id']),
                        'title': topic['title'],
                        'similarity_score': float(similarity),
                        'status': topic.get('status', 'unknown')
                    })
            
            # 5. Sort and return
            similar_topics.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            logger.info(f"Found {len(similar_topics)} similar topics (threshold: {threshold})")
            return similar_topics
            
        except Exception as e:
            # We catch the potential failure from _ensure_model_loaded here too
            logger.error(f"Error checking similarity: {str(e)}")
            return []
    
    
    def _fallback_suggestions(self, category: str, count: int) -> List[Dict]:
        """
        Provide basic fallback suggestions when Gemini fails.
        
        Args:
            category: Topic category
            count: Number of suggestions to generate
        
        Returns:
            List of basic topic suggestions
        """
        logger.warning(f"Using fallback suggestions for {category}")
        
        # Basic template-based suggestions
        templates = [
            {
                "title": f"{category} Application Development",
                "description": f"Develop a comprehensive {category} application that addresses a real-world problem. The project should demonstrate proficiency in modern development practices and technologies.",
                "difficulty": "intermediate",
                "tags": [category.lower(), "application", "development"],
                "prerequisites": ["Programming fundamentals", "Software engineering"],
                "estimated_duration": "4-5 months"
            },
            {
                "title": f"Data Analysis System for {category}",
                "description": f"Build a data analysis system tailored for {category} applications. Focus on data collection, processing, visualization, and insights generation.",
                "difficulty": "intermediate",
                "tags": [category.lower(), "data-analysis", "visualization"],
                "prerequisites": ["Data structures", "Statistics basics"],
                "estimated_duration": "4 months"
            },
            {
                "title": f"Machine Learning Integration in {category}",
                "description": f"Integrate machine learning capabilities into a {category} system. Implement predictive models and intelligent features to enhance functionality.",
                "difficulty": "advanced",
                "tags": [category.lower(), "machine-learning", "ai"],
                "prerequisites": ["Machine learning basics", "Python programming"],
                "estimated_duration": "5-6 months"
            },
            {
                "title": f"Mobile Platform for {category}",
                "description": f"Create a mobile application focused on {category}. Implement cross-platform compatibility and modern UI/UX principles.",
                "difficulty": "intermediate",
                "tags": [category.lower(), "mobile", "cross-platform"],
                "prerequisites": ["Mobile development", "UI/UX design"],
                "estimated_duration": "4 months"
            },
            {
                "title": f"IoT Solution for {category}",
                "description": f"Develop an Internet of Things solution for {category} applications. Include sensor integration, data collection, and real-time monitoring.",
                "difficulty": "advanced",
                "tags": [category.lower(), "iot", "sensors", "real-time"],
                "prerequisites": ["Embedded systems", "Networking"],
                "estimated_duration": "5 months"
            }
        ]
        
        # Return requested number of suggestions
        return templates[:min(count, len(templates))]


if __name__ == "__main__":
    print("Topic Intelligence Service - AI Features ✅")