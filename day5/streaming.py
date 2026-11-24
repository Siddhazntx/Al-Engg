import os
import json
import re
import time
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, Field 
from time import sleep
# from pypdf import PdfReader
# from docx import Document

load_dotenv()
my_api_key = os.getenv("groq_api_key")

if not my_api_key:
    raise ValueError("Api key not found")

client = Groq(api_key=my_api_key)
model="openai/gpt-oss-120b"

prompt = "Explain how you work"
message = {
    "role":"system",
    "content":prompt
}     

messages = [message]
response = client.chat.completions.create(
    model = model, messages = messages
)
answer = response.choices[0].message.content
print(answer)

# streaming = client.chat.completions.create(
#     model = model, messages=messages, stream = True
# )

# for chunk in streaming:
#     content = chunk.choices[0].delta.content
#     if content:
#         print(content, end="",flush=True)