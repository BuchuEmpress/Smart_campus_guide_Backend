"""
Gemini AI Service - FIXED VERSION

Key fixes:
1. Correct API call syntax for google-genai SDK
2. Proper async/await patterns
3. Better error handling with detailed logging
4. Fallback responses that work
"""

import os

import json
import logging
from typing import Optional, Dict, List, Any
from functools import lru_cache
from dotenv import load_dotenv

try:
    from google import genai
except ImportError:
    raise RuntimeError("google-genai package not installed. Run: pip install google-genai")

load_dotenv(override=True)

logger = logging.getLogger("services.gemini_service")
logging.basicConfig(level=logging.INFO)


class GeminiService:
    """Async Gemini AI service with proper error handling."""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        """Initialize GeminiService."""
        # print("GEMINI_API_KEY:", os.getenv("GEMINI_API_KEY"))  # Security: Don't log API keys
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY must be set in environment or passed to GeminiService.")

        self.model_name = model_name or os.getenv("GEMINI_MODEL_NAME", "gemini-flash-latest")
        
        # Initialize client
        try:
            self.client = genai.Client(api_key=self.api_key)
            logger.info(f"✅ Gemini Client initialized with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise

    async def _call_model(self, prompt: str, **kwargs) -> str:
        """
        Call Gemini model with proper error handling.
        
        Args:
            prompt: The prompt to send
            **kwargs: Additional generation config
            
            Returns:
            Generated text response
        """
        try:
            logger.debug(f"Calling Gemini with prompt length: {len(prompt)}")
            
            # CORRECT API SYNTAX for google-genai
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=kwargs.get('config', {})
            )
            
            # Extract text from response
            if hasattr(response, 'text'):
                text = response.text
            elif hasattr(response, 'candidates') and response.candidates:
                text = response.candidates[0].content.parts[0].text
            else:
                text = str(response)
            
            logger.debug(f"Gemini response length: {len(text)}")
            return text.strip()
            
        except Exception as e:
            logger.error(f"Gemini API call failed: {type(e).__name__}: {str(e)}")
            raise

    async def embed_text(self, text: str, task_type: str = "retrieval_query") -> List[float]:
        """
        Generate embedding for a single string using Gemini.
        Uses caching to speed up repeated queries.
        Default model: text-embedding-004 (768 dimensions)
        """
        # Use a simple in-memory cache
        cache_key = f"{text}:{task_type}"
        if hasattr(self, '_embed_cache') and cache_key in self._embed_cache:
            return self._embed_cache[cache_key]
        
        try:
            model = "text-embedding-004"
            response = await self.client.aio.models.embed_content(
                model=model,
                contents=text,
                config={
                    "task_type": task_type,
                    "output_dimensionality": 768
                }
            )
            embedding = response.embeddings[0].values
            
            # Cache the result
            if not hasattr(self, '_embed_cache'):
                self._embed_cache = {}
            self._embed_cache[cache_key] = embedding
            
            return embedding
        except Exception as e:
            logger.error(f"Gemini embedding failed: {e}")
            raise

    async def embed_batch(self, texts: List[str], task_type: str = "retrieval_document") -> List[List[float]]:
        """
        Generate embeddings for a list of strings.
        """
        try:
            model = "text-embedding-004"
            response = await self.client.aio.models.embed_content(
                model=model,
                contents=texts,
                config={
                    "task_type": task_type,
                    "output_dimensionality": 768
                }
            )
            return [e.values for e in response.embeddings]
        except Exception as e:
            logger.error(f"Gemini batch embedding failed: {e}")
            raise

    async def extract_intent(self, user_query: str) -> Dict[str, Any]:
        """
        Extract intent from user query.
        
        Returns structured intent with action, location_query, etc.
        """
        try:
            if not user_query or not user_query.strip():
                logger.warning("Empty query received")
                return {
                    "action": "chat",
                    "location_query": "",
                    "location_type": "other",
                    "preferences": {},
                    "on_campus": None,
                }

            prompt = f"""Analyze this user query and return ONLY a JSON object (no markdown, no explanation).

User query: "{user_query}"

Return JSON with these exact keys:
- "action": either "navigate", "search", or "chat"
- "location_query": the location name if mentioned (empty string if none)
- "location_type": "building", "class", "landmark", "office", "food", or "other"
- "preferences": empty object {{}}
- "on_campus": true, false, or null

Examples:
Query: "Where is the library?" 
Response: {{"action": "search", "location_query": "library", "location_type": "building", "preferences": {{}}, "on_campus": true}}

Query: "How do I get to the cafeteria?"
Response: {{"action": "navigate", "location_query": "cafeteria", "location_type": "food", "preferences": {{}}, "on_campus": true}}

Query: "Hello, how are you?"
Response: {{"action": "chat", "location_query": "", "location_type": "other", "preferences": {{}}, "on_campus": null}}

Now analyze: "{user_query}"
Return ONLY the JSON object:"""

            text = await self._call_model(prompt)
            
            # Clean markdown formatting
            text = text.strip()
            if text.startswith('```json'):
                text = text[7:]
            if text.startswith('```'):
                text = text[3:]
            if text.endswith('```'):
                text = text[:-3]
            text = text.strip()
            
            logger.info(f"Raw Gemini response: {text[:200]}")
            
            # Parse JSON
            intent = json.loads(text)
            logger.info(f"✅ Intent extracted: action={intent.get('action')}, query={intent.get('location_query')}")
            
            return intent

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini JSON response: {e}")
            logger.error(f"Raw response was: {text[:500]}")
            # Fallback: try to detect intent from keywords
            return self._fallback_intent(user_query)
            
        except Exception as e:
            logger.error(f"Intent extraction error: {type(e).__name__}: {str(e)}")
            return self._fallback_intent(user_query)

    async def extract_topic_intent(self, user_query: str) -> Dict[str, Any]:
        """Extract intent from user query for final year project topics."""
        try:
            if not user_query or not user_query.strip():
                return {"action": "chat", "query": ""}

            prompt = f"""Analyze this request from a student regarding final year project topics.
            Return ONLY a JSON object.

            User query: "{user_query}"

            Return JSON with these keys:
            - "action": one of ["search", "suggest", "improve", "chat"]
            - "query": the main topic or keyword mentioned
            - "department": department name if mentioned
            - "option": option name if mentioned
            - "title": if they provided a title to improve

            Examples:
            "Show me some software engineering topics" -> {{"action": "search", "query": "software engineering", "department": "Computer Engineering", "option": "Software Engineering"}}
            "Can you suggest 5 new AI topics?" -> {{"action": "suggest", "query": "AI", "count": 5}}
            "Make this title better: Smart Campus Guide" -> {{"action": "improve", "title": "Smart Campus Guide"}}

            Return ONLY the JSON object:"""

            text = await self._call_model(prompt)
            # Clean JSON
            text = text.strip('` \n').replace('json\n', '', 1)
            return json.loads(text)
            
        except Exception as e:
            logger.error(f"Topic intent extraction error: {e}")
            return {"action": "chat", "query": user_query}

    async def extract_master_intent(self, user_query: str) -> Dict[str, str]:
        """
        Classify the user's query into broad categories: 'topic_query', 'navigation_query', or 'general_chat'.
        This acts as a high-level router for incoming user messages.
        """
        try:
            if not user_query or not user_query.strip():
                return {"intent_type": "general_chat"}

            prompt = f"""Analyze the following user query and determine its primary intent.
            Return ONLY a JSON object with a single key: "intent_type".

            Possible intent_types:
            - "topic_query": The user is asking about final year projects, academic topics, research, project ideas, courses, departments, or anything related to academic guidance.
            - "navigation_query": The user is asking for directions, location of a building, office, classroom, or any campus facility, or general campus information.
            - "general_chat": The user is greeting, making a general statement, or the query does not fit into the above categories.

            User query: "{user_query}"

            Return ONLY the JSON object, for example: {{"intent_type": "topic_query"}}"""

            text = await self._call_model(prompt)
            # Clean markdown formatting and parse JSON
            text = text.strip()
            if text.startswith('```json'):
                text = text[7:]
            if text.startswith('```'):
                text = text[3:]
            if text.endswith('```'):
                text = text[:-3]
            text = text.strip()
            
            intent = json.loads(text)
            intent_type = intent.get("intent_type", "general_chat")
            logger.info(f"✅ Master intent extracted: {intent_type}")
            return {"intent_type": intent_type}

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini JSON response for master intent: {e}. Raw response: {text[:500]}")
            return {"intent_type": "general_chat"}
        except Exception as e:
            logger.error(f"Master intent extraction error: {e}")
            return {"intent_type": "general_chat"}

    def _fallback_intent(self, query: str) -> Dict[str, Any]:
        """Fallback intent detection using simple keyword matching."""
        query_lower = query.lower()
        
        # Navigation keywords
        if any(word in query_lower for word in ['how', 'get to', 'directions', 'navigate', 'way to', 'route']):
            # Extract potential location
            location_query = query_lower
            for word in ['how do i get to', 'how to get to', 'directions to', 'navigate to', 'way to']:
                if word in query_lower:
                    location_query = query_lower.split(word)[-1].strip('? .')
                    break
            
            return {
                "action": "navigate",
                "location_query": location_query,
                "location_type": "other",
                "preferences": {},
                "on_campus": True
            }
        
        # Search keywords
        if any(word in query_lower for word in ['where', 'find', 'locate', 'location of', 'show me']):
            # Extract location
            location_query = query_lower
            # Sort by length descending to match longest phrases first
            separators = ['where is the', 'where is', 'where\'s the', 'where\'s', 'find the', 'find', 'locate the', 'locate', 'location of', 'show me the', 'show me']
            for word in separators:
                if word in query_lower:
                    try:
                        location_query = query_lower.split(word, 1)[-1].strip('? .')
                    except IndexError:
                        pass
                    break
            
            return {
                "action": "search",
                "location_query": location_query,
                "location_type": "other",
                "preferences": {},
                "on_campus": True
            }
        
        # Default to chat
        return {
            "action": "chat",
            "location_query": "",
            "location_type": "other",
            "preferences": {},
            "on_campus": None
        }

    async def humanize_directions(
        self,
        route_data: Dict,
        campus_context: Optional[Dict] = None,
        nearby_landmarks: Optional[List[str]] = None
    ) -> str:
        """Convert route to human-friendly directions."""
        try:
            steps = route_data.get("steps", [])
            total_duration = route_data.get("duration_text", "a few minutes")

            if not steps:
                return f"The walk takes about {total_duration}. Follow the main path to your destination."

            # Build simple prompt
            steps_text = "\n".join([
                f"{i+1}. {step.get('instruction', 'Continue')} ({step.get('distance_text', 'short distance')})"
                for i, step in enumerate(steps[:5])  # Limit to 5 steps
            ])

            prompt = f"""You are a helpful student guide at the University of Bamenda. Give natural, human-like walking directions based on this route.
            
Route steps:
{steps_text}

Total time: {total_duration}

Instructions:
- Don't say "Head north" or "turn east". Use "Turn left", "Turn right", "Go straight".
- Mention landmarks if possible.
- Keep it encouraging and simple, like you're talking to a friend.
- Format: "Okay, to get there: First..." or similar natural phrasing.
- Keep it under 3-4 sentences.

Your Directions:"""

            text = await self._call_model(prompt)
            return text if text else f"Walk for about {total_duration} following the main path."

        except Exception as e:
            logger.error(f"Humanize directions error: {e}")
            return f"The walk takes about {route_data.get('duration_text', 'a few minutes')}. Follow the main path."

    async def generate_response(self, prompt: str, context: Optional[List[Dict]] = None) -> str:
        """Generate conversational response."""
        try:
            if not prompt or not prompt.strip():
                return "Could you please provide more details about what you're looking for?"

            full_prompt = prompt
            if context:
                history = "\n".join([
                    f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                    for msg in context[-3:]
                ])
                full_prompt = f"Conversation:\n{history}\n\nCurrent: {prompt}\n\nResponse:"

            text = await self._call_model(full_prompt)
            return text if text else "I'm here to help with campus navigation. What location are you looking for?"

        except Exception as e:
            logger.error(f"Generate response error: {e}")
            return "I'm having trouble right now. Please try asking about a specific campus location."

    async def enhance_description(self, location: Optional[Dict], context: Optional[Dict] = None) -> str:
        """Enhance location description."""
        try:
            if not location or not location.get("name"):
                return "No description available."

            name = location.get("name")
            desc = location.get("description", "")
            
            prompt = f"""Make this location description more helpful for students (1-2 sentences):

Location: {name}
Current: {desc}

Better description:"""

            text = await self._call_model(prompt)
            return text if text else desc

        except Exception as e:
            logger.error(f"Enhance description error: {e}")
            return location.get("description", "No description available.")


# Test
if __name__ == "__main__":
    import asyncio
    
    async def test():
        try:
            service = GeminiService()
            print("✅ Service initialized\n")
            
            # Test intent extraction
            result = await service.extract_intent("Where is the library?")
            print(f"Intent: {result}\n")
            
            # Test response
            response = await service.generate_response("Hello!")
            print(f"Response: {response[:100]}\n")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    asyncio.run(test())