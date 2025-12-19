
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

async def test():
    print("Testing google-genai import...")
    try:
        from google import genai
        print("✅ Import successful: from google import genai")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return

    print("Testing Client initialization...")
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("❌ GEMINI_API_KEY not found in env")
            return
        
        client = genai.Client(api_key=api_key)
        print("✅ Client initialized")
        
        print("Testing basic generation...")
        response = await client.aio.models.generate_content(
            model="gemini-1.5-flash",
            contents="Say hello"
        )
        print(f"✅ Generation success: {response.text}")
        
    except Exception as e:
        print(f"❌ Runtime error: {type(e).__name__}: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test())
