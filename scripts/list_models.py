
import asyncio
import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

async def list_models():
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    
    print("Listing models...")
    try:
        # Await the list call first
        pager = await client.aio.models.list(config={"page_size": 100})
        
        # Iterate over the pager (it might be async iterable or just iterable)
        async for model in pager:
             print(f"Model: {model.name}")
             print(f"  DisplayName: {model.display_name}")
             print(f"  Supported: {model.supported_generation_methods}")
             print("-" * 20)
    except TypeError:
        # Fallback if not async iterable
        for model in pager:
             print(f"Model: {model.name}")
             print("-" * 20)
    except Exception as e:
        print(f"Error listing models: {e}")

if __name__ == "__main__":
    asyncio.run(list_models())
