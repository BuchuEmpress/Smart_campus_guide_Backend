
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

async def list_models():
    print("Listing available Gemini models...")
    try:
        from google import genai
        # Initialize client
        api_key = os.getenv("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key)
        
        # List models (using v1beta which seems to be default/required)
        # We need to find what `v1beta` actually supports.
        # But wait, google-genai SDK abstracts this? 
        # Let's try to just hit the list_models endpoint if it exists in this SDK.
        
        # The SDK structure is tricky without docs, but let's try the standard pattern
        # or just fallback to listing known names.
        
        print("Attempting to generate content with 'gemini-1.5-flash-latest'...")
        try:
             response = await client.aio.models.generate_content(
                model="gemini-1.5-flash-latest",
                contents="Hello"
            )
             print("✅ gemini-1.5-flash-latest WORKS")
        except Exception as e:
            print(f"❌ gemini-1.5-flash-latest failed: {e}")

        print("Attempting to generate content with 'gemini-1.5-flash-001'...")
        try:
             response = await client.aio.models.generate_content(
                model="gemini-1.5-flash-001",
                contents="Hello"
            )
             print("✅ gemini-1.5-flash-001 WORKS")
        except Exception as e:
            print(f"❌ gemini-1.5-flash-001 failed: {e}")

    except ImportError:
        print("google-genai not installed")
    except Exception as e:
        print(f"Global error: {e}")

if __name__ == "__main__":
    asyncio.run(list_models())
