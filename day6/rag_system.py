import os
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

load_dotenv()

my_api_key = os.getenv("groq_api_key")
if not my_api_key:
    raise ValueError("Api key not found")
client = Groq(api_key=my_api_key)
groqmodel="openai/gpt-oss-120b"

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Employees receive 24 days of paid leave per year.",
   
    "Employees work from the office on Tuesday, Wednesday and Thursday. "
    "Monday and Friday are optional work-from-home days.",
   
    "Employees receive Rs 3000 per month for gym reimbursement.",
   
    "Employees can claim Rs 2000 per month for home internet.",
   
    "Employees have a 90 day notice period."
]

document_embedding = model.encode(documents)

def cosine_similarity(a,b):
    return np.dot(a,b)/(np.linalg.norm(a) * np.linalg.norm(b))



def retrieve(q_emb):
    scores = []
    for i, doc_emb in enumerate(document_embedding):
        score = cosine_similarity(q_emb, doc_emb)
        
        scores.append((score, documents[i]))
    scores.sort(reverse=True)
    return scores[0]

query = "How much holidays do it get ?"
embedding = model.encode(query)

def ask_llm(question,context):

    sys_prompt=f"""answer in one line only. Answer only based on this context. do not hallucinate. Context: {context}"""
    system_message={
        "role": "system",
        "content": sys_prompt

    }
    message={
        "role": "user",
        "content": question
    }
    messages=[system_message, message]
    response=client.chat.completions.create(model=groqmodel, messages=messages)
    answer=response.choices[0].message.content
    return answer

score,context = retrieve(embedding)
print(score)
# print(context) 

answer = ask_llm(query,context)
print(answer)