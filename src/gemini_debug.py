import os
import inspect
from google import genai
from google.genai import types

def list_models(client):
    print("\n--- Available Models ---")
    for model in client.models.list():
        print(f" - {model.name}")

def inspect_config_types():
    print("\n--- ThinkingConfig Fields ---")
    try:
        print(f" Fields: {list(types.ThinkingConfig.__annotations__.keys())}")
    except:
        print(" Could not inspect ThinkingConfig")

    print("\n--- ThinkingLevel Enum Members ---")
    try:
        for level in types.ThinkingLevel:
            print(f" - {level}")
    except:
        print(" Could not inspect ThinkingLevel")

def inspect_method_signatures(client):
    print("\n--- Method Signatures ---")
    print(f" Files.upload: {inspect.signature(client.files.upload)}")
    print(f" Models.generate_content: {inspect.signature(client.models.generate_content)}")

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found.")
        return

    client = genai.Client(api_key=api_key)
    
    print("Gemini SDK Debug Tool")
    print("=====================")
    
    list_models(client)
    inspect_config_types()
    inspect_method_signatures(client)

if __name__ == "__main__":
    main()
