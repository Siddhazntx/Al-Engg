import os
import json
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel

load_dotenv()

my_api_key = os.getenv("groq_api_key")

if not my_api_key:
    raise ValueError("Api error")

client = Groq(api_key=my_api_key)

class Ticket(BaseModel):
    name : str
    email : str
    phone : int
    address : str

schema = Ticket.model_json_schema()
response_format={
    "type":"json_object"
}

system_prompt=f"""Extract the information ticket bases on the schema. {schema}"""

message_system={
    "role":"system",
    "content":system_prompt
}

model = "openai/gpt-oss-120b"
role = "user"
text="Hello ! My name is Sidd, i have bought an laptop of HP and it is not working properly. My address is Delhi 110063, number is 11342342, mail is asdwd@gmail.com"
prompt = f""" 
This is a customer issue ticket. Please extract personal information from this and give me a json output
{text}
"""
message = {
    "role" : role,
    "content" : prompt
}

messages = [message, message_system]
response = client.chat.completions.create(
    model = model, messages = messages, response_format=response_format
)

# print(response)

print("----------------------")
answer=response.choices[0].message.content

formatted_json=json.loads(answer)
print(json.dumps(formatted_json,indent=4))
# print(answer)
print("----------------------")