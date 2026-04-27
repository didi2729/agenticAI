import requests
import sys
sys.path.insert(0, "/Users/didi/AgenticAI")
import agenticAI.api_error as api_error

BASE_URL = "http://127.0.0.1:8000"

samples = [
    {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},  # setosa
    {"sepal_length": 6.7, "sepal_width": 3.1, "petal_length": 4.7, "petal_width": 1.5},  # versicolor
    {"sepal_length": 6.3, "sepal_width": 3.3, "petal_length": 6.0, "petal_width": 2.5},  # virginica
]

species = {0: "setosa", 1: "versicolor", 2: "virginica"}

def treat_http_error(response: requests.Response) -> None:
    """Converteste erorile in exceptii"""
    if response.ok:
        return

    cod = response.status_code
    try:
        details = response.json().get("error",{}).get("message", response.text)
    except Exception:
        details = response.text
    if cod == 401:
        raise api_error.AuthenticationError(f"Authentification failuire: {details}", cod)
    if cod == 429:
        raise api_error.RateLimitError(f"Rate limit reached: {details}", cod)
    if cod in (400, 422):
        raise api_error.ValidationError(f"Invalid request: {details}", cod)
    if cod in (500, 502, 503):
        raise api_error.APIError(f"Server error: {details}", cod)
    else:
        raise api_error.APIError(f"API error({cod}):{details}", cod)
    

print("Checking API status...")
try:
    resp = requests.get(f"{BASE_URL}/")
    treat_http_error(response=resp)
    print("Sending prediction requests:")
    print("-" * 50)
    for i, sample in enumerate(samples):
        response = requests.post(f"{BASE_URL}/predict", json=sample)
        result = response.json()
        predicted_class = result["prediction"]
        print(f"Sample {i+1}: {sample}")
        print(f"  -> Prediction: {predicted_class} ({species[predicted_class]}), model version: {result['model_version']}\n")

except api_error.RateLimitError as e:
    print(f"Rate limit atins : {e}\n")
except api_error.APIError as e:
    print(f"Eroare API: {e}\n")
except requests.exceptions.ConnectionError:
    print("Server error. Restart uvicorn")
   



