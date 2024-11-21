import openai
import os
from dotenv import load_dotenv

##### THIS IS A TEST FILE I USED TO CHECK IF THE API IS WORKING #### 

load_dotenv()

endpoint = os.getenv("AZURE_OPENAI_API_BASE")
key = os.getenv("AZURE_OPENAI_API_KEY")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
openai.api_type = "azure"
openai.api_base = endpoint
openai.api_version = api_version
openai.api_key = key

query = "What are the key principles of the MAMA Framework for autonomy management?"

try:
    response = openai.ChatCompletion.create(
        engine=deployment_name,  # Use `engine` for Azure deployments
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query}
        ]
    )
    print("Response:", response["choices"][0]["message"]["content"])

except Exception as e:
    print("Error communicating with Azure GPT-4o:", e)
