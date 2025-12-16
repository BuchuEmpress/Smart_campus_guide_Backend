
import os
import google.generativeai as genai
from dotenv import load_dotenv

def list_models():
    load_dotenv(override=True)
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("❌ No API Key found in .env")
        return

    print(f"Checking models for API Key: {api_key[:5]}...{api_key[-5:]}")
    
    try:
        genai.configure(api_key=api_key)
        print("\nAvailable Models:")
        found_any = False
        for m in genai.list_models():
            found_any = True
            print(f"- {m.name}")
            print(f"  Description: {m.description}")
            print(f"  Input Token Limit: {m.input_token_limit}")
            print(f"  Output Token Limit: {m.output_token_limit}")
            print("-" * 30)
            
        if not found_any:
            print("❌ No models found. Your API key might be invalid or has no access.")
            
    except Exception as e:
        print(f"❌ Error listing models: {str(e)}")

if __name__ == "__main__":
    list_models()
