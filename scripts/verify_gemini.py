
import asyncio
import os
import logging
from services.gemini_service import GeminiService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_gemini")

async def test_gemini():
    candidates = [
        "gemini-1.5-flash-8b",
        "gemini-pro"
    ]
    
    # Use default client (v1beta usually)
    service = GeminiService(model_name="default")
    
    for model in candidates:
        try:
            print(f"\nTesting: {model}")
            service.model_name = model
            response = await service.generate_response("Hello")
            print(f"✅ SUCCESS with {model}: {response[:50]}")
            break
        except Exception as e:
            print(f"❌ FAILED {model}: {e}")

if __name__ == "__main__":
    asyncio.run(test_gemini())
