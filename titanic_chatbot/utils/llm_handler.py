from openai import OpenAI
from dotenv import load_dotenv
import os
import time
from utils.prompts import system_prompt_template, result_explanation_prompt_template

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_code(user_input, chat_history, system_prompt):
    start_time = time.time()

    messages = [{"role": "system", "content": system_prompt}] + chat_history
    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0,
    )

    elapsed = time.time() - start_time
    usage = response.usage
    print(f"[generate_code] ⏱️ Time: {elapsed:.2f}s | 🦰 Tokens used: {usage.total_tokens} (Prompt: {usage.prompt_tokens}, Completion: {usage.completion_tokens})")

    assistant_message = response.choices[0].message.content
    code_str = assistant_message.strip("```python").strip("```").strip()

    return code_str, {"role": "assistant", "content": assistant_message}, usage, elapsed

def generate_code_with_cache(user_input, chat_history, system_prompt, cache_dict):
    if user_input in cache_dict:
        print("[Code Cache Hit]")
        return (*cache_dict[user_input], True)  # Return with cache hit flag

    code, assistant_msg, usage, duration = generate_code(user_input, chat_history, system_prompt)
    cache_dict[user_input] = (code, assistant_msg, usage, duration)
    return code, assistant_msg, usage, duration, False  # Cache miss

def generate_explanation(question, result_summary, plot_metadata="", code=""):
    start_time = time.time()
    clean_summary = str(result_summary).strip()
    if len(clean_summary) > 500:
        clean_summary = clean_summary[:500] + "..."

    if len(plot_metadata) > 500:
        plot_metadata = str(plot_metadata)[:500] + "..."

    prompt = result_explanation_prompt_template.format(
        question=question,
        result_summary=clean_summary,
        plot_metadata=plot_metadata,
        code=code
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=100
    )

    elapsed = time.time() - start_time
    usage = response.usage
    print(f"[generate_explanation] ⏱️ Time: {elapsed:.2f}s | 🦰 Tokens used: {usage.total_tokens} (Prompt: {usage.prompt_tokens}, Completion: {usage.completion_tokens})")

    explanation = response.choices[0].message.content.strip()
    return explanation, {"role": "assistant", "content": explanation}, usage, elapsed

def generate_explanation_with_cache(question, code, output, plot_metadata, cache_dict):
    cache_key = (question, code, str(output), str(plot_metadata))

    if cache_key in cache_dict:
        print("[Explanation Cache Hit]")
        return (*cache_dict[cache_key], True)

    explanation, _, usage, duration = generate_explanation(question, output, plot_metadata, code)
    cache_dict[cache_key] = (explanation, usage, duration)
    return explanation, usage, duration, False

