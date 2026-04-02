import requests

def generate_answer(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "gemma:2b",
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": 180,
                "temperature": 0.2,
                "top_p": 0.8
            }
        },
        timeout=120
    )

    return response.json()["response"]