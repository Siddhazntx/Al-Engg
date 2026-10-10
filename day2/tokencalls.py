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
prompt1 = "Hello"
prompt2 = "Explain time dilation with an example"
prompt3 = "Write a 100 words essay on time travel and the eassy shouldnt be acvailable in internet and it should contain the best research words by scientist till now"
# prompt = "Do you know CR7 ?"
prompts = [prompt1,prompt2,prompt3]
for prompt in prompts:
    message = {
    "role" : role,
    "content" : prompt
    }
    messages = [message]
    response = client.chat.completions.create(
        model = model, messages = messages, max_tokens=50
    )
    usage = response.usage
    print(f"Prompt:{prompt} --> your tokens : {usage.prompt_tokens} completions tokens : {usage.completion_tokens}  total tokens : {usage.total_tokens} Response: {response.choices[0].finish_reason}")
    print("END")

    # print("----------------------")
    # answer=response.choices[0].message.content
    # print(answer)
    # print("\n")



# print(response)