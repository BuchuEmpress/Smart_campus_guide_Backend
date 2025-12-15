
import asyncio
import os
import sys
import logging

# Add current directory to path
sys.path.append(os.getcwd())

from services.gemini_service import GeminiService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_gemini():
    print("Testing Gemini Service...")
    
    # Check API Key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY is NOT set in environment variables.")
        return
    else:
        print(f"✅ GEMINI_API_KEY found (length: {len(api_key)})")

    try:
        service = GeminiService()
        print("✅ GeminiService initialized.")
        
        prompt = "Hello, are you working?"
        print(f"Sending prompt: {prompt}")
        
        response = await service._call_model(prompt)
        print(f"✅ Response received: {response}")
        
    except Exception as e:
        print(f"❌ Error during Gemini call: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    asyncio.run(test_gemini())
