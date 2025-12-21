import os
import asyncio
from dotenv import load_dotenv

try:
    from google import genai
except ImportError:
    print("google-genai package not installed.")
    exit(1)

load_dotenv(override=True)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("GEMINI_API_KEY not found in environment.")
    exit(1)

print(f"Using API Key: {api_key[:5]}...{api_key[-5:]}")

async def list_models():
    try:
        client = genai.Client(api_key=api_key)
        print("Client initialized. Fetching models...")
        
        # Accessing models differently depending on the SDK version structure
        # Attempting the structure seen in the user's service first
        if hasattr(client, 'aio'):
            try:
                # Based on user code structure
                # The user code calls: client.aio.models.generate_content
                # We want to list models.
                # Usually: client.models.list_models() or similar.
                
                # Let's try synchronous client first for simplicity if available
                # or just use the aio client
                
                # In google-genai v0.3+:
                # client.models.list()
                
                # But let's verify what `client` is.
                print(f"Client type: {type(client)}")
                
                # Try to list models
                async for model in client.aio.models.list():
                    print(f"Model: {model.name}")
                    print(f"  DisplayName: {model.display_name}")
                    print(f"  Supported Actions: {model.supported_generation_methods}")
                    print("-" * 20)
                    
            except Exception as e:
                print(f"Error listing (async): {e}")

                # Fallback to sync if async fails weirdly
                try:
                   for model in client.models.list():
                        print(f"Model (sync): {model.name}")
                except Exception as e2:
                     print(f"Error listing (sync): {e2}")

        else:
             print("Client has no 'aio' attribute.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(list_models())
