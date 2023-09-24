import requests

url = 'http://127.0.0.1:5001/predict'
data = {
    'Stats': [7.4, 2.6, 7.6, 34.7, 0.5, 2.1, 25.0, 1.6, 2.3, 69.9, 0.7, 3.4, 4.1, 1.9, 0.4, 0.4, 1.3, 0.0]
}
response = requests.post(url, json=data)
result = response.json()

print(f"Prediction: {'More than 5 years' if result['prediction'] == 1 else 'Less than 5 years'}")
