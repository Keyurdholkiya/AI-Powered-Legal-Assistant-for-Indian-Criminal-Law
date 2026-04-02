import time
import requests

def test_ollama(url, chars):
    prompt = "Please summarize this text: " + "law " * (chars // 4)
    print(f"Testing {url} with approx {chars} context chars...")
    start = time.time()
    try:
        response = requests.post(
            url,
            json={
                "model": "phi3:latest",
                "prompt": prompt,
                "stream": False
            },
            timeout=180
        )
        print(f"Time: {time.time() - start:.2f}s | Status: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    url = "http://localhost:11434/api/generate"
    for chars in [500, 1000, 2000, 4000]:
        test_ollama(url, chars)
