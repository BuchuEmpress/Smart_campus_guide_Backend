from google.genai import Client
import os

# Read your API key from the correct environment variable
API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    print("❌ ERROR: GEMINI_API_KEY is not set.")
    print("Set it using:  set GEMINI_API_KEY=YOUR_KEY")
    exit()

def list_models():
    client = Client(api_key=API_KEY)

    print("\n=====================================")
    print("       AVAILABLE GEMINI MODELS")
    print("=====================================\n")

    models = client.models.list()

    for model in models:
        print(f"📌 Model ID: {model.name}")
        print(f"   Display Name: {model.display_name}")
        print(f"   Supported Methods:")

        if hasattr(model, 'supported_generation_methods'):
            for m in model.supported_generation_methods:
                print(f"     - {m}")

        print("-------------------------------------")

if __name__ == "__main__":
    list_models()
