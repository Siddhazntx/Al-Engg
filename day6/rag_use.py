import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
my_api_key = os.getenv("groq_api_key")

if not my_api_key:
    raise ValueError("Api key not found")

client = Groq(api_key=my_api_key)
model="openai/gpt-oss-120b"

knowledge_base = {
    "age":"Siddhant's age is 21",
    "status":"He is a student at BVCOE in IT department"
}

def retrie(question):
    question = question.lower()
    if "age" in question:
        return knowledge_base["age"]
    elif "status" in question:
        return knowledge_base["status"]
    else:
        return None
    
def ask_llm(question):
    context = retrie(question)
    system_prompt = f"""answer in one line only. Answer only based on this context, do not hallucinate. Context: {context}"""
    system_message = {
        "role":"system",
        "content":system_prompt
    }
    message={
        "role":"user",
        "content":question
    }
    messages = [system_message,message]
    response=client.chat.completions.create(model = model, messages=messages)
    answer = response.choices[0].message.content
    return answer

question = "Do you know what's the status of Siddhant Mahapatra from BVCOE"
print(ask_llm(question))