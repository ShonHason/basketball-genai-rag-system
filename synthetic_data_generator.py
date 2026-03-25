import os
import json
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

openai_api_key = os.getenv("OPENAI_API_KEY")
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

print("Initializing LLM clients")

openrouter = OpenAI(
    api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1"
)

openai_client = OpenAI(
    api_key=openai_api_key
)

# Model priority tiers
MODEL_TIERS = {
    "cheap": [
        "openai/gpt-4o-mini"
    ],
    "free": [
        "deepseek/deepseek-chat"
    ],
    "premium": [
        "openai/gpt-4o"
    ]
}


def try_model(client, model, system_prompt, user_prompt):

    response = client.chat.completions.create(
        model=model,
        temperature=0.3,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return safe_json_load(response.choices[0].message.content)

def safe_json_load(text):

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1

        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end])
            except:
                pass

    raise ValueError("Invalid JSON")

def generate_data(msg: str, retries=4):

    system_prompt = """
You are an advanced synthetic data generator.

Your task is to generate completely fictional, internally consistent data
that strictly follows a provided schema.

Rules:
- All generated data must be fictional
- Do not reference real players, teams, or leagues
- The response MUST be a single valid JSON object
- Do not include explanations
- Do not include markdown
- Do not include text before or after the JSON

Before returning the final answer:

1. Ensure the JSON is valid.
2. Ensure all required fields from the schema exist.
3. Ensure the structure matches the schema exactly.
4. If anything is wrong, fix the JSON before returning it.

Return ONLY the corrected JSON object.
"""

    for tier, models in MODEL_TIERS.items():

        print(f"\nTrying {tier} models")

        for model in models:

            for attempt in range(retries):

                try:

                    print(f"Model: {model} | attempt {attempt+1}")

                    client = openrouter

                    result = try_model(
                        client,
                        model,
                        system_prompt,
                        msg
                    )

                    print(f"Success with {model}")

                    return result

                except Exception as e:

                    print(f"{model} failed:", str(e))

                    time.sleep(2 * (attempt + 1))

    raise Exception("All models failed")