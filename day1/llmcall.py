import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

my_api_key = os.getenv("groq_api_key")

if not my_api_key:
    raise ValueError("Api error")

client = Groq(api_key=my_api_key)

model = "openai/gpt-oss-120b"
role = "user"
prompt = "Do you know CR7 ?"
message = {
    "role" : role,
    "content" : prompt
}

messages = [message]
response = client.chat.completions.create(
    model = model, messages = messages
)

# print(response)

print("----------------------")
answer=response.choices[0].message.content
print(answer)