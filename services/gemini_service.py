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
        if any(word in query_lower for word in ['where', 'find', 'locate', 'location of']):
            # Extract location
            location_query = query_lower
            for word in ['where is', 'where is the', 'find', 'locate']:
                if word in query_lower:
                    location_query = query_lower.split(word)[-1].strip('? .')
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

            prompt = f"""Convert these directions into friendly, natural language. No compass directions (north/south/east/west). Use simple language.

Route steps:
{steps_text}

Total time: {total_duration}

Write 2-3 friendly sentences describing this walk:"""

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