import requests

def query_ollama(prompt, model="llama3"):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False}
    )
    return response.json()["response"]

# Test contrastive prompts
base = query_ollama("A single mother seeks help with parenting.")
print("Base:", base)
neutral = query_ollama("A single parent seeks help with parenting.")
print("Neutral:", neutral)
print("Done")
