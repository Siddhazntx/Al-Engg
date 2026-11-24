import os
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
text = "Machine Learing is tough"

def cosine_similarity(a,b):
    return np.dot(a,b)/(np.linalg.norm(a) * np.linalg.norm(b))
# embedding = model.encode(text)
# print(embedding.shape)
# print(embedding[:10])

t1 = "ther are 24 paid leaves"
t2 = "there are 24 vacation"

e1 = model.encode(t1)
e2 = model.encode(t2)
print(cosine_similarity(e1,e2))