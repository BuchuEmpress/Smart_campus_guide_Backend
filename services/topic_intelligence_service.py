"""
Topic Intelligence Service - AI-Powered Features

Uses Gemini AI and embeddings for intelligent topic management:
- Generate unique topic suggestions
- Improve existing topics
- Check similarity (avoid duplicates)
- Analyze trends

This is the "smart" layer on top of TopicService.
"""

import logging
import json
from typing import List, Dict, Optional
import numpy as np

from services.gemini_service import GeminiService
from services.topic_service import TopicService
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TopicIntelligenceService:
    """
    AI-powered topic intelligence service.
    
    Provides smart features for topic management using Gemini AI.
    """
    
    def __init__(
        self,
        topic_service: Optional[TopicService] = None,
        gemini_service: Optional[GeminiService] = None
    ):
        """Initialize intelligence service."""
        self.topic_service = topic_service or TopicService()
        self.gemini = gemini_service or GeminiService()
        
        # Load embedding model for similarity checking
        logger.info("Loading embedding model for similarity checking...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Embedding model loaded ✅")
    
    def suggest_topics(
        self,
        category: str,
        department: str,
        count: int = 5,
        keywords: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Generate unique topic suggestions using Gemini AI.
        
        Args:
            category: Topic category (e.g., "Software Engineering")
            department: Department name
            count: Number of suggestions to generate
            keywords: Optional keywords to guide suggestions
        
        Returns:
            List of suggested topics with descriptions
        """
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
{chr(10).join(f"  - {title}" for title in existing_titles[:20])}

- Each topic should be:
  * Feasible for a final year project (3-6 months)
  * Use modern technologies
  * Solve real-world problems
  * Be specific and clear
  * Suitable for {department} students
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
        """
        Improve a topic idea using Gemini AI.
        
        Args:
            title: Original topic title
            description: Original description
            category: Topic category
        
        Returns:
            Dictionary with improved title and description
        """
        try:
            logger.info(f"Improving topic: {title}")
            
            prompt = f"""
Improve this final year project topic to make it more specific, innovative, and impactful:

Original Title: {title}
Original Description: {description}
Category: {category}

Provide:
1. An improved, more specific title
2. An enhanced description (100-150 words) that:
   - Makes the project scope clearer
   - Highlights innovative aspects
   - Mentions specific technologies/methods
   - Explains real-world impact

Return ONLY a JSON object:
{{
  "improved_title": "improved title here",
  "improved_description": "enhanced description here",
  "suggested_tags": ["tag1", "tag2", "tag3"],
  "suggested_difficulty": "beginner|intermediate|advanced"
}}

NO markdown, NO explanations, ONLY JSON.
"""
            
            response = self.gemini.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Clean response
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            result = json.loads(response_text.strip())
            
            logger.info("Topic improved successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error improving topic: {str(e)}")
            return {
                'improved_title': title,
                'improved_description': description,
                'suggested_tags': [],
                'suggested_difficulty': 'intermediate'
            }
    
    def check_similarity(self, title: str, category: str, threshold: float = 0.8) -> List[Dict]:
        """
        Check if topic is similar to existing topics using embeddings.
        
        Args:
            title: Topic title to check
            category: Category to search in
            threshold: Similarity threshold (0-1)
        
        Returns:
            List of similar topics with similarity scores
        """
        try:
            logger.info(f"Checking similarity for: {title}")
            
            # Get existing topics
            existing = self.topic_service.list_topics(
                filter_query={'category': category},
                limit=100
            )
            
            if not existing:
                return []
            
            # Generate embedding for new title
            new_embedding = self.model.encode(title)
            
            # Calculate similarities
            similar_topics = []
            
            for topic in existing:
                existing_title = topic['title']
                existing_embedding = self.model.encode(existing_title)
                
                # Cosine similarity
                similarity = np.dot(new_embedding, existing_embedding) / (
                    np.linalg.norm(new_embedding) * np.linalg.norm(existing_embedding)
                )
                
                if similarity >= threshold:
                    similar_topics.append({
                        'topic_id': str(topic['_id']),
                        'title': existing_title,
                        'similarity_score': float(similarity),
                        'status': topic.get('status', 'unknown')
                    })
            
            # Sort by similarity
            similar_topics.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            logger.info(f"Found {len(similar_topics)} similar topics (threshold: {threshold})")
            return similar_topics
            
        except Exception as e:
            logger.error(f"Error checking similarity: {str(e)}")
            return []
    
    def _fallback_suggestions(self, category: str, count: int) -> List[Dict]:
        """Fallback suggestions if AI fails."""
        return [{
            'title': f'Suggested {category} Project {i+1}',
            'description': f'A {category} project focusing on modern technologies and real-world applications.',
            'difficulty': 'intermediate',
            'tags': [category.lower()],
            'prerequisites': [],
            'estimated_duration': '4 months'
        } for i in range(count)]


if __name__ == "__main__":
    print("Topic Intelligence Service - AI Features ✅")