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
        keywords: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Generate unique topic suggestions using Gemini AI.
        """
        # ... (method content omitted for brevity)
        try:
            logger.info(f"Generating {count} topic suggestions for {category}")
            
            # Get existing topics to avoid duplicates
            existing_topics = self.topic_service.list_topics(
                filter_query={'category': category},
                limit=100
            )
            
            existing_titles = [t['title'] for t in existing_topics]
            
            # Build prompt for Gemini
            keywords_text = ""
            if keywords:
                keywords_text = f"\nFocus areas: {', '.join(keywords)}"
            
            prompt = f"""
Generate {count} unique, innovative final year project topics for {department} students specializing in {category}.

Requirements:
- Topics should be original and NOT similar to these existing topics:
{chr(10).join(f"  - {title}" for title in existing_titles[:20])}

- Each topic should be:
  * Feasible for a final year project (3-6 months)
  * Use modern technologies
  * Solve real-world problems
  * Be specific and clear
  * Suitable for {department} students
  * **Where feasible, generate topics that are hybrid in nature**, combining the specialization of {category} with other relevant fields (e.g., Data Science, Environmental Tech).
{keywords_text}

Return ONLY a JSON array with this exact structure:
[
  {{
    "title": "Topic title here",
    "description": "Detailed description (50-100 words)",
    "difficulty": "beginner|intermediate|advanced",
    "tags": ["tag1", "tag2", "tag3"],
    "prerequisites": ["prerequisite1", "prerequisite2"],
    "estimated_duration": "4 months"
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
    
    def improve_topic(self, title: str, description: str, category: str) -> Dict:
        # ... (method content omitted for brevity)
        pass # Placeholder for full method implementation
    
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
        # ... (method content omitted for brevity)
        pass # Placeholder for full method implementation


if __name__ == "__main__":
    print("Topic Intelligence Service - AI Features ✅")