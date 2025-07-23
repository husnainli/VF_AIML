from openai import OpenAI
from dotenv import load_dotenv
import os
from utils.prompts import system_prompt_template

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_code(user_input, df_columns, chat_history):
    column_string = str(df_columns.tolist())
    system_prompt = system_prompt_template.format(columns=column_string)

    messages = [{"role": "system", "content": system_prompt}] + chat_history
    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0
    )

    assistant_message = response.choices[0].message.content
    return assistant_message.strip("```python").strip("```").strip(), {"role": "assistant", "content": assistant_message}

