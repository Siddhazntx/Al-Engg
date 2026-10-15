import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, Field 
# from pypdf import PdfReader
# from docx import Document

load_dotenv()
my_api_key = os.getenv("groq_api_key")

if not my_api_key:
    raise ValueError("Api key not found")

client = Groq(api_key=my_api_key)
model="openai/gpt-oss-120b"

def llm_ans(prompt):
    message={
        "role":"user",
        "content":prompt
    }
    messages=[message]
    response = client.chat.completions.create(
        model=model, messages=messages
    )
    answers=response.choices[0].message.content
    return answers

# bad_prompt="""
# This is a user complaint:
# My laptop is not working and my girlfriend is sad
# Handle this or classify this
# """
role_prompt="""
This is a user complaint:
My laptop is not working and my girlfriend is sad
#Role:
You are a customer support technical staff for laptop/mobiles
#Task:
You have to classify the issue in a category
#Constraint:
You have to classify the issue in one of the given categories->Billing,Technical,Network error,Return
#Output format:
You should answer in one word only . The one word should be  one of the categories given in constraint. Then as a follow up question -> Would you like to get a support call or text, then ask contact details to fill
#Example:
For instance, if an user wants a refund, then category is return. If the laptop's OS is not functioning then the category is technical. If user paid the bill and facing newtork issue then category is newtwork issue
#Fallback:
If the issue is not related to any of the categories given in constraint then return The issue is not relavent to the functionality of our application and break the prompt,
and Ask Please share any related issue, Don't spam!


This is a user complaint:
my screen got a green line
"""

print(llm_ans(role_prompt))
