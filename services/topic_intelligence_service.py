"""
Topic Intelligence Service - AI-Powered Features (COMPLETE FIX)

MAJOR CHANGES:
1. Changed 'category' → 'option' everywhere (matches database schema)
2. Made all database queries case-insensitive
3. Preserved ALL functionality (no code removed)
4. Fixed response model mismatches

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
    
    NOW USES 'option' instead of 'category' to match database schema!
    ALL DATABASE QUERIES ARE NOW CASE-INSENSITIVE!
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
    # GEMINI AI METHODS (FIXED TO USE 'option')
    # ====================================================================
    
    async def suggest_topics(
        self,
        option: str,
        department: str,
        count: int = 5,
        keywords: Optional[List[str]] = None,
        user_request: Optional[str] = None
    ) -> List[Dict]:
        """
        Generate unique topic suggestions using Gemini AI.
        """
        try:
            logger.info(f"Generating {count} topic suggestions for option={option}, department={department}")

            # Handle long keywords as user_request
            if keywords and not user_request:
                sentence_keywords = [k for k in keywords if len(str(k)) > 20 and ' ' in str(k)]
                if sentence_keywords:
                    user_request = " ".join(sentence_keywords)
                    keywords = [k for k in keywords if k not in sentence_keywords]
                    logger.info(f"Extracted user request from keywords: {user_request}")

            # Get existing topics
            existing_topics = self.topic_service.list_topics(
                filter_query={'option': option},
                limit=100
            )
            existing_titles = [t['title'] for t in existing_topics]

            # Clean keywords
            keywords_text = f"\nFocus areas/Keywords: {', '.join([k.strip() for k in keywords])}" if keywords else ""

            # Refined prompt
            prompt = f"""
You are an expert Computer Engineering advisor at University of Bamenda, Cameroon.

STUDENT'S REQUEST: "{user_request or 'Generate innovative topics'}"
KEYWORDS:{keywords_text or ' None'}
DEPARTMENT: {department}
SPECIALIZATION (Option): {option}
OPTION INTERPRETATION:
- SEN → Software Engineering
- DAS → Data Science & AI
- CNSM → Networks, Systems & Maintenance
EXISTING TOPICS TO AVOID: {', '.join(existing_titles[:15]) if existing_titles else 'None'}

TASK:
- Generate {count} innovative, high-impact final year project topics.
- **TONE**: Warm, encouraging, and professional. The descriptions should feel like a helpful mentor explaining an exciting opportunity.
- **CONTEXT**: Deeply rooted in Cameroon/Africa context (local problems, local solutions).
- **STRICT TITLE RULES**: 
  - NEVER end a title with "for SEN", "for DAS", or "for CNSM".
  - NEVER mention the option code in the title.
  - The specialization should be obvious from the *technology* and *domain*.
- **DESCRIPTION**: 100-150 words. inspiring yet technical. Explain the *status quo*, the *innovation*, and the *impact*.
- **OUTPUT**: JSON ONLY. No polite passing remarks outside the JSON.

**CONVERSATIONAL INPUT HANDLING**:
If the user's request is just a greeting (e.g., "Hi", "Hello", "Hey") or vague:
- Return a SINGLE suggestion item.
- Title: "Hello! 👋 I'm your Research Assistant"
- Description: "I'd love to help you find the perfect project topic! Please tell me a bit about your interests. Are you into Mobile Apps, AI, IoT, or Security? Or just ask me to 'Generate topics for [Option]'!"
- Tags: ["help", "guide"]
- Difficulty: "beginner"


Keys required: title, description, difficulty, tags, prerequisites, estimated_duration.
If no feasible topics exist, return a single JSON with N/A and explanation.
"""
            
            # Call Gemini and handle response cleanup
            response = await self.gemini._call_model(prompt)
            response_text = response.text.strip()

            # Flatten newlines and remove code block markers
            response_text = response_text.replace('\n', ' ').strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            elif response_text.startswith('```'):
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
            return self._fallback_suggestions(option, count)
        except Exception as e:
            logger.error(f"Error generating suggestions: {str(e)}")
            return self._fallback_suggestions(option, count)
    
    async def improve_topic(
        self, 
        title: str, 
        description: str, 
        option: str,  # ✅ CHANGED FROM 'category' to 'option'
        user_instruction: Optional[str] = None
    ) -> Dict:
        """
        Use Gemini AI to improve a topic's title and description.
        
        Args:
            title: Current topic title
            description: Current topic description
            option: Topic option/specialization
            user_instruction: Optional instruction from the user
        
        Returns:
            Dictionary with improved title, description, keywords, and difficulty
        """
        try:
            logger.info(f"Improving topic: {title} (option={option})")
            
            user_instruction_text = ""
            if user_instruction:
                user_instruction_text = f"""
MANDATORY USER INSTRUCTION - YOU MUST APPLY THIS:
"{user_instruction}"

EXAMPLES OF HOW TO APPLY INSTRUCTIONS:
- "make it about AI" → Add machine learning models, neural networks, AI algorithms to the solution
- "focus on mobile" → Change platform to mobile app (React Native/Flutter), add mobile-specific features
- "add security" → Include encryption (AES-256), authentication (JWT/OAuth), security protocols
- "use blockchain" → Integrate smart contracts (Solidity), distributed ledger, consensus mechanisms
- "make it simpler" → Reduce scope, remove complex features, focus on core functionality

You MUST incorporate this instruction into BOTH the title and description. Do NOT just acknowledge it.
"""
            
            prompt = f"""
You are an expert academic editor improving a final year project topic.

Original Topic:
- Title: "{title}"
- Description: "{description}"
Specialization Constraint:
- Option code: {option}
- Interpretation:
  - SEN → Software Engineering–oriented solution design
  - DAS → Data Science / AI–oriented solution design
  - CNSM → Networks, systems, and infrastructure–oriented solution design

IMPORTANT:
- The specialization is a constraint, NOT part of the title.
- Do NOT mention the option code or its expansion in the title.


{user_instruction_text}

Task:
1. **Analyze Input**:
   - If the input is **Conversational** (e.g., "Hi", "Hello", "Good morning"):
     - **Title**: Return a friendly greeting (e.g., "Hello! 👋").
     - **Description**: Return a warm, mentoring message asking them about their project ideas. (e.g., "I'm here to help you craft an amazing project! What area describes your interest? AI, Mobile Apps, or maybe something with Hardware?").
     - **Tags**: ["chat", "mentoring"].
     - **Difficulty**: "beginner".
   - If the input is a **Topic Idea**:
     - **Refine Title**: Make it professional yet innovative.
     - **Expand Description**: Write a 100-150 word abstract.
       - **TONE**: Warm, helpful, and mentoring. Write as if you are guiding a student.
       - **DOMAIN CONSTRAINT**: STRICTLY Computer Engineering / Software / IT.
         - If the user asks about unrelated topics (e.g., "Physics"), pivot to a software solution.
     - **Keywords**: 5-7 technical tags.
     - **Difficulty**: Assess complexity.

Requirements:
- The output must be JSON.
- **TONE**: always warm and human-like.


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
            response = await self.gemini._call_model(prompt)
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
    # EMBEDDING & SIMILARITY METHODS (FIXED TO USE 'option')
    # ====================================================================
    
    def check_similarity(self, title: str, option: str, threshold: float = 0.8) -> List[Dict]:
        """
        Check if topic is similar to existing topics using embeddings.

        Args:
            title: Topic title to check.
            option: Option/specialization to search in (case-insensitive).
            threshold: Similarity threshold (0-1).
        
        Returns:
            List of similar topics with similarity scores.
        """
        try:
            logger.info(f"Checking similarity for: '{title}' in option='{option}'")
            
            # ✅ FIX: Use 'option' field (case-insensitive via TopicService)
            existing = self.topic_service.list_topics(
                filter_query={'option': option},  # Changed from 'category'
                limit=100
            )
            
            logger.info(f"DEBUG SIMILARITY: Option='{option}' -> Found {len(existing)} existing topics in DB")
            if existing:
                 logger.info(f"DEBUG SIMILARITY: Sample existing titles: {[t.get('title') for t in existing[:3]]}")

            if not existing:
                logger.info(f"No existing topics found for option '{option}'")
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

            # DEBUG: Log top scores
            logger.info(f"DEBUG SIMILARITY: Top 5 scores: {sorted(similarities, reverse=True)[:5]}")
            logger.info(f"DEBUG SIMILARITY: Threshold used: {threshold}")

            # 4. Filter results
            similar_topics = []
            for i, similarity in enumerate(similarities):
                if similarity >= threshold:
                    topic = existing[i]
                    similar_topics.append({
                        'topic_id': str(topic.get('_id', topic.get('topic_id', ''))),
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
    
    
    def _fallback_suggestions(self, option: str, count: int) -> List[Dict]:
        """
        Provide basic fallback suggestions when Gemini fails.
        
        Args:
            option: Topic option/specialization
            count: Number of suggestions to generate
        
        Returns:
            List of basic topic suggestions
        """
        logger.warning(f"Using fallback suggestions for option '{option}'")
        
        # Basic template-based suggestions
        templates = [
            {
                "title": f"{option} Application Development",
                "description": f"Develop a comprehensive {option} application that addresses a real-world problem. The project should demonstrate proficiency in modern development practices and technologies.",
                "difficulty": "intermediate",
                "tags": [option.lower(), "application", "development"],
                "prerequisites": ["Programming fundamentals", "Software engineering"],
                "estimated_duration": "4-5 months"
            },
            {
                "title": f"Data Analysis System for {option}",
                "description": f"Build a data analysis system tailored for {option} applications. Focus on data collection, processing, visualization, and insights generation.",
                "difficulty": "intermediate",
                "tags": [option.lower(), "data-analysis", "visualization"],
                "prerequisites": ["Data structures", "Statistics basics"],
                "estimated_duration": "4 months"
            },
            {
                "title": f"Machine Learning Integration in {option}",
                "description": f"Integrate machine learning capabilities into a {option} system. Implement predictive models and intelligent features to enhance functionality.",
                "difficulty": "advanced",
                "tags": [option.lower(), "machine-learning", "ai"],
                "prerequisites": ["Machine learning basics", "Python programming"],
                "estimated_duration": "5-6 months"
            },
            {
                "title": f"Mobile Platform for {option}",
                "description": f"Create a mobile application focused on {option}. Implement cross-platform compatibility and modern UI/UX principles.",
                "difficulty": "intermediate",
                "tags": [option.lower(), "mobile", "cross-platform"],
                "prerequisites": ["Mobile development", "UI/UX design"],
                "estimated_duration": "4 months"
            },
            {
                "title": f"IoT Solution for {option}",
                "description": f"Develop an Internet of Things solution for {option} applications. Include sensor integration, data collection, and real-time monitoring.",
                "difficulty": "advanced",
                "tags": [option.lower(), "iot", "sensors", "real-time"],
                "prerequisites": ["Embedded systems", "Networking"],
                "estimated_duration": "5 months"
            }
        ]
        
        # Return requested number of suggestions
        return templates[:min(count, len(templates))]


if __name__ == "__main__":
    print("✅ Topic Intelligence Service - Complete with 'option' field and case-insensitive support!")