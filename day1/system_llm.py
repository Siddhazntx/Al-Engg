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
# prompt = "can you code zigzag traversal ?"
prompt = "Create a plan from Delhi to Chandratal for 4 days and give a short response"

message_system = {
    "role":"system",
    # "content":"you just know theory but you try to code in c++ DSA"
    "content":"You are a trip planner"
}
message = {
    "role" : role,
    "content" : prompt
}

messages = [message_system,message]
response = client.chat.completions.create(
    model = model, messages = messages, temperature=2
)

# print(response)

print("----------------------")
answer=response.choices[0].message.content
print(answer)