import os
import requests
import json

API_KEY = os.getenv("OPENAI_API_KEY")  # Make sure this is set

response = requests.post(
    url="https://api.openai.com/v1/chat/completions",
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    },
    
    data=json.dumps({
        "model": "openai/gpt-oss-120b:free",
        "messages": [
            {"role": "user", "content": "Hello, how are you?"}
        ],
    })
)


print(response.json())