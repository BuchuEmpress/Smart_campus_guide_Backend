"""
Gemini AI Service Module

This module provides integration with Google's Gemini AI for:
- Converting robotic GPS directions into natural, conversational language
- Extracting user intent from natural language queries
- Enhancing location descriptions
- General conversational AI responses

The key feature is humanizing directions to sound like a friendly student
helping another student, NOT like a GPS device.

"""

import os
import logging
import json
from typing import Optional, Dict, List

# Import environment variable support
from dotenv import load_dotenv

# Import Google Generative AI library
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Setup logging for this module
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GeminiService:
    """
    Service for interacting with Google's Gemini AI.
    
    This service specializes in:
    - Humanizing robotic GPS directions into natural language
    - Understanding user intent from queries
    - Enhancing descriptions with context
    - Conversational interactions
    
    Attributes:
        api_key: Gemini API key
        model: Gemini generative model instance
    
    Example:
        >>> gemini = GeminiService()
        >>> natural_directions = gemini.humanize_directions(route_data, campus_context)
        >>> print(natural_directions)
        "Walk straight ahead from the main gate. You'll pass the cafeteria..."
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Gemini Service.
        
        Args:
            api_key: Gemini API key. If None, reads from environment variable.
        
        Raises:
            ValueError: If API key is not provided or found in environment.
        """
        # Get API key from parameter or environment variable
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        
        # Validate that we have an API key
        if not self.api_key:
            logger.error("Gemini API key not found")
            raise ValueError(
                "Gemini API key is required. "
                "Set GEMINI_API_KEY environment variable or pass api_key parameter."
            )
        
        try:
            # Configure the Gemini API with our key
            genai.configure(api_key=self.api_key)
            
            # Initialize the Gemini model (using the latest stable version)
            # gemini-pro is good for text generation
            self.model = genai.GenerativeModel('gemini-pro')
            
            logger.info("Gemini AI service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini AI: {str(e)}")
            raise
    
    def humanize_directions(
        self,
        route_data: Dict,
        campus_context: Optional[Dict] = None,
        nearby_landmarks: Optional[List[str]] = None
    ) -> str:
        """
        Convert robotic GPS directions into natural, conversational language.
        
        This is the MOST IMPORTANT function in this service. It transforms
        technical route data from Google Maps into friendly, campus-specific
        directions that sound like a student helping another student.
        
        Args:
            route_data: Dictionary containing route information from Google Maps
                       Must include 'steps', 'total_distance', 'total_duration'
            campus_context: Optional context about the campus destination
                          e.g., {'name': 'Library', 'description': '...'}
            nearby_landmarks: Optional list of landmark names near the route
                            e.g., ['Cafeteria', 'Fountain', 'Admin Block']
        
        Returns:
            Natural, conversational directions as a string
        
        Example:
            >>> route = maps.get_directions(origin, destination)
            >>> landmarks = ['Cafeteria', 'Fountain', 'Main Admin Building']
            >>> directions = gemini.humanize_directions(route, landmarks=landmarks)
            >>> print(directions)
            "Okay, so from where you are, just walk straight ahead for about 5 minutes.
            You'll pass the cafeteria on your right - you can't miss the smell!
            Keep going until you see the big fountain in the courtyard..."
        """
        try:
            logger.info("Generating humanized directions")
            
            # Extract key information from route data
            steps = route_data.get('steps', [])
            total_distance = route_data.get('total_distance', {})
            total_duration = route_data.get('total_duration', {})
            
            # Build the context for Gemini
            # This is CRITICAL - the prompt engineering determines output quality
            
            # Start with system-level instructions
            system_prompt = """
You are a friendly university student giving directions to a fellow student on campus.

CRITICAL RULES - NEVER BREAK THESE:
1. NEVER use compass directions (north, south, east, west, northeast, etc.)
2. NEVER use exact measurements (200 meters, 0.5 kilometers, etc.)
3. ALWAYS use visible landmarks and buildings students can actually see
4. ALWAYS use natural time estimates ("about 5 minutes walk", "a short stroll")
5. Write in friendly, conversational English like you're talking to a friend
6. Make directions easy to follow using things people can actually see

GOOD EXAMPLES:
"Walk straight ahead from where you are. You'll pass the cafeteria on your right - 
you'll probably smell the food! Keep going until you reach the big fountain in the 
courtyard. The library is right there, the tall building with lots of glass windows."

"From here, just head towards that tall tree you can see ahead. When you get there, 
take the path on your left. You'll walk past the sports field, and the Computer 
Science building is the white one at the end."

BAD EXAMPLES (NEVER DO THIS):
"Head northeast for 200 meters. Turn left and proceed north for 150 meters."
"Walk 0.3 kilometers in a northwestern direction."

Your tone should be:
- Friendly and helpful (like talking to a friend)
- Confident but not bossy
- Reassuring ("you can't miss it", "it's easy to find")
- Use conversational phrases ("okay so", "from here", "you'll see")
"""
            
            # Build the specific request with route data
            landmarks_text = ""
            if nearby_landmarks and len(nearby_landmarks) > 0:
                landmarks_text = f"\n\nLandmarks nearby that you can reference:\n"
                for landmark in nearby_landmarks:
                    landmarks_text += f"- {landmark}\n"
            
            # Extract simplified step information
            simplified_steps = []
            for i, step in enumerate(steps, 1):
                # Get distance and duration in readable format
                distance_text = step.get('distance', {}).get('text', 'a short distance')
                duration_text = step.get('duration', {}).get('text', 'a few minutes')
                
                # Get HTML instruction (Google provides this)
                instruction = step.get('instruction', 'Continue')
                
                # Build simplified step description
                step_text = f"Step {i}: {instruction} (takes about {duration_text})"
                simplified_steps.append(step_text)
            
            # Add destination context if available
            destination_info = ""
            if campus_context:
                dest_name = campus_context.get('name', 'your destination')
                dest_desc = campus_context.get('description', '')
                destination_info = f"\n\nDestination: {dest_name}"
                if dest_desc:
                    destination_info += f"\nDescription: {dest_desc}"
            
            # Combine everything into the full prompt
            user_prompt = f"""
Please convert these GPS directions into natural, friendly directions:

Total journey: About {total_duration.get('text', '5 minutes')} walking

Route steps:
{chr(10).join(simplified_steps)}
{landmarks_text}
{destination_info}

Remember:
- NO compass directions (north/south/east/west)
- NO exact measurements
- USE visible landmarks
- USE natural time ("about 5 minutes", "a short walk")
- Sound like a helpful student, not a GPS device

Generate conversational, easy-to-follow directions:
"""
            
            # Generate the humanized directions using Gemini
            logger.debug(f"Sending prompt to Gemini (length: {len(user_prompt)} chars)")
            
            # Create the full prompt with system instructions
            full_prompt = system_prompt + "\n\n" + user_prompt
            
            # Call Gemini API
            response = self.model.generate_content(full_prompt)
            
            # Extract the generated text
            humanized_text = response.text.strip()
            
            logger.info(f"Successfully generated humanized directions ({len(humanized_text)} chars)")
            
            return humanized_text
            
        except Exception as e:
            logger.error(f"Error humanizing directions: {str(e)}")
            # Return a fallback message if AI fails
            return (
                "I can help you get there! The route is about "
                f"{route_data.get('total_duration', {}).get('text', '5 minutes')} walk. "
                "Follow the path ahead and look for landmarks along the way."
            )
    
    def extract_intent(self, user_query: str) -> Dict:
        """
        Extract the user's intent from their natural language query.
        
        This helps us understand what the user wants:
        - Are they looking for a location?
        - What type of location? (building, landmark, facility)
        - Is it urgent?
        - Any preferences? (wheelchair accessible, nearest, etc.)
        
        Args:
            user_query: The user's natural language query
                       e.g., "Where is the library?"
                       e.g., "I need to find a restaurant near campus"
        
        Returns:
            Dictionary with extracted intent:
            {
                'action': 'navigate' | 'search' | 'info' | 'chat',
                'location_query': str,
                'location_type': 'building' | 'landmark' | 'facility' | 'food' | etc.,
                'preferences': {
                    'wheelchair_accessible': bool,
                    'nearest': bool,
                    'urgency': 'high' | 'medium' | 'low'
                },
                'on_campus': bool (True if explicitly on-campus, False if off-campus)
            }
        
        Example:
            >>> gemini = GeminiService()
            >>> intent = gemini.extract_intent("Where's the nearest wheelchair accessible restroom?")
            >>> print(intent)
            {
                'action': 'navigate',
                'location_query': 'restroom',
                'location_type': 'facility',
                'preferences': {'wheelchair_accessible': True, 'nearest': True},
                'on_campus': True
            }
        """
        try:
            logger.info(f"Extracting intent from query: {user_query}")
            
            # Build prompt for intent extraction
            prompt = f"""
Analyze this user query and extract their intent in JSON format:

User query: "{user_query}"

Return a JSON object with these fields:
- action: One of ["navigate", "search", "info", "chat"]
  * navigate: User wants directions to a specific place
  * search: User wants to find places matching criteria
  * info: User wants information about a place
  * chat: General conversation/question
  
- location_query: The location name or description being searched for
- location_type: One of ["building", "landmark", "facility", "food", "office", "classroom", "other"]
- preferences: Object with:
  * wheelchair_accessible: true/false (if mentioned)
  * nearest: true/false (if user wants nearest option)
  * urgency: "high"/"medium"/"low" (based on language like "urgent", "quickly", "whenever")
- on_campus: true if clearly on-campus, false if explicitly off-campus, null if unclear

Example:
Query: "Where's the nearest wheelchair accessible restroom?"
Response: {{
  "action": "navigate",
  "location_query": "restroom",
  "location_type": "facility",
  "preferences": {{"wheelchair_accessible": true, "nearest": true, "urgency": "medium"}},
  "on_campus": null
}}

Now analyze: "{user_query}"

Return ONLY the JSON object, no other text.
"""
            
            # Call Gemini API
            response = self.model.generate_content(prompt)
            
            # Parse the JSON response
            # Gemini might wrap it in markdown code blocks, so clean it
            response_text = response.text.strip()
            
            # Remove markdown code block markers if present
            if response_text.startswith('```json'):
                response_text = response_text[7:]  # Remove ```json
            if response_text.startswith('```'):
                response_text = response_text[3:]  # Remove ```
            if response_text.endswith('```'):
                response_text = response_text[:-3]  # Remove closing ```
            
            response_text = response_text.strip()
            
            # Parse JSON
            intent = json.loads(response_text)
            
            logger.info(f"Extracted intent: action={intent.get('action')}, query={intent.get('location_query')}")
            
            return intent
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse intent JSON: {str(e)}")
            # Return default intent on parse error
            return {
                'action': 'search',
                'location_query': user_query,
                'location_type': 'other',
                'preferences': {'wheelchair_accessible': False, 'nearest': False, 'urgency': 'medium'},
                'on_campus': None
            }
        except Exception as e:
            logger.error(f"Error extracting intent: {str(e)}")
            # Return default intent on any error
            return {
                'action': 'search',
                'location_query': user_query,
                'location_type': 'other',
                'preferences': {'wheelchair_accessible': False, 'nearest': False, 'urgency': 'medium'},
                'on_campus': None
            }
    
    def enhance_description(
        self,
        location: Dict,
        context: Optional[Dict] = None
    ) -> str:
        """
        Enhance a location description to be more natural and contextual.
        
        Takes a basic location description and makes it more friendly,
        detailed, and useful for students.
        
        Args:
            location: Dictionary with location info (name, type, description, etc.)
            context: Optional additional context (time of day, weather, etc.)
        
        Returns:
            Enhanced, natural language description
        
        Example:
            >>> location = {
            ...     'name': 'University Library',
            ...     'type': 'building',
            ...     'description': 'Main library building'
            ... }
            >>> enhanced = gemini.enhance_description(location)
            >>> print(enhanced)
            "The University Library is the main library building on campus.
            It's a great spot for studying, with quiet reading areas and
            computer labs. You'll recognize it by its large glass windows..."
        """
        try:
            logger.info(f"Enhancing description for: {location.get('name', 'unknown')}")
            
            # Extract location details
            name = location.get('name', 'This location')
            loc_type = location.get('type', 'place')
            basic_desc = location.get('description', '')
            
            # Build context information
            context_text = ""
            if context:
                time_of_day = context.get('time_of_day', '')
                if time_of_day:
                    context_text += f"Current time: {time_of_day}\n"
            
            # Build prompt
            prompt = f"""
Make this location description more natural, friendly, and helpful for students:

Location name: {name}
Type: {loc_type}
Current description: {basic_desc}
{context_text}

Requirements:
- Write in a friendly, conversational tone
- Add useful details students would want to know
- Keep it concise (2-3 sentences)
- Make it sound natural, not robotic
- Include any visual identifiers if relevant

Generate an enhanced description:
"""
            
            # Call Gemini API
            response = self.model.generate_content(prompt)
            
            # Extract the enhanced description
            enhanced = response.text.strip()
            
            logger.info(f"Successfully enhanced description ({len(enhanced)} chars)")
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Error enhancing description: {str(e)}")
            # Return original description on error
            return location.get('description', f"{location.get('name', 'This location')} on campus.")
    
    def generate_response(
        self,
        prompt: str,
        context: Optional[List[Dict]] = None
    ) -> str:
        """
        Generate a general conversational response.
        
        This is for general chatbot functionality - answering questions,
        providing information, having natural conversations.
        
        Args:
            prompt: The user's message or question
            context: Optional conversation history for context
                    List of dicts with 'role' and 'content' keys
        
        Returns:
            AI-generated response as string
        
        Example:
            >>> gemini = GeminiService()
            >>> response = gemini.generate_response("What are the library hours?")
            >>> print(response)
        """
        try:
            logger.info("Generating conversational response")
            
            # Build full prompt with context if provided
            full_prompt = prompt
            
            if context and len(context) > 0:
                # Add conversation history
                history_text = "Previous conversation:\n"
                for msg in context[-5:]:  # Last 5 messages for context
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')
                    history_text += f"{role}: {content}\n"
                
                full_prompt = history_text + f"\nCurrent message: {prompt}\n\nYour response:"
            
            # Call Gemini API
            response = self.model.generate_content(full_prompt)
            
            # Extract response text
            response_text = response.text.strip()
            
            logger.info(f"Generated response ({len(response_text)} chars)")
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I'm having trouble processing that right now. Could you try rephrasing your question?"


# Testing and usage examples
if __name__ == "__main__":
    print("=" * 70)
    print("GEMINI AI SERVICE TEST")
    print("=" * 70)
    
    try:
        # Initialize service
        gemini = GeminiService()
        print("✅ Service initialized successfully\n")
        
        # Test 1: Humanize directions
        print("Test 1: Humanize Directions")
        print("-" * 70)
        
        # Sample route data (like what Google Maps returns)
        route_data = {
            'steps': [
                {
                    'instruction': 'Head south on Main Road',
                    'distance': {'text': '150 m', 'value': 150},
                    'duration': {'text': '2 mins', 'value': 120}
                },
                {
                    'instruction': 'Turn right onto Campus Drive',
                    'distance': {'text': '300 m', 'value': 300},
                    'duration': {'text': '4 mins', 'value': 240}
                }
            ],
            'total_distance': {'text': '450 m', 'value': 450},
            'total_duration': {'text': '6 mins', 'value': 360}
        }
        
        landmarks = ['Cafeteria', 'Main Fountain', 'Admin Building']
        
        directions = gemini.humanize_directions(
            route_data,
            nearby_landmarks=landmarks
        )
        
        print("Generated directions:")
        print(directions)
        
        # Test 2: Extract intent
        print("\n\nTest 2: Extract Intent")
        print("-" * 70)
        
        queries = [
            "Where is the library?",
            "I need to find the nearest wheelchair accessible restroom",
            "Show me restaurants near campus"
        ]
        
        for query in queries:
            intent = gemini.extract_intent(query)
            print(f"\nQuery: {query}")
            print(f"Action: {intent.get('action')}")
            print(f"Location: {intent.get('location_query')}")
            print(f"Type: {intent.get('location_type')}")
        
        # Test 3: Enhance description
        print("\n\nTest 3: Enhance Description")
        print("-" * 70)
        
        location = {
            'name': 'University Library',
            'type': 'building',
            'description': 'Main library building on campus'
        }
        
        enhanced = gemini.enhance_description(location)
        print(f"Original: {location['description']}")
        print(f"Enhanced: {enhanced}")
        
        print("\n" + "=" * 70)
        print("✅ All tests completed!")
        print("=" * 70)
        
    except ValueError as e:
        print(f"❌ Configuration error: {str(e)}")
        print("Make sure GEMINI_API_KEY is set in your .env file")
    except Exception as e:
        print(f"❌ Error: {str(e)}")