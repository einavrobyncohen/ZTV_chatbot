import openai
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("AZURE_API_KEY")
ENDPOINT = os.getenv("AZURE_API_ENDPOINT")
DEPLOYMENT_ID = os.getenv("AZURE_API_MODEL")  # This is the deployment ID, not the model name

if not API_KEY or not ENDPOINT or not DEPLOYMENT_ID:
    raise ValueError("Missing Azure API credentials. Check your .env file.")

openai.api_type = "azure"
openai.api_base = ENDPOINT
openai.api_version = "2023-06-01-preview"
openai.api_key = API_KEY

def test_azure_gpt():
    try:
        response = openai.ChatCompletion.create(
            engine=DEPLOYMENT_ID, 
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello! Can you confirm you're working?"},
            ],
            max_tokens=50,
            temperature=0.5
        )
        print("Response from Azure GPT-4o:")
        print(response['choices'][0]['message']['content'])
    except Exception as e:
        print(f"Error communicating with Azure GPT-4o: {e}")

if __name__ == "__main__":
    test_azure_gpt()
