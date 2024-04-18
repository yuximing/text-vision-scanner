from openai import OpenAI
from dotenv import load_dotenv
import os
# from openAIConnect import OpenAI
load_dotenv()
client = OpenAI(
    api_key = os.environ.get("OPENAI_API_KEY")
)

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "you are a nerdy assistant, answer with big nerdy words"},
    {"role": "user", "content": "explain what is apple as food"}
  ]
)

print(completion.choices[0].message.content)

