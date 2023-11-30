import requests
import json
# Replace with the URL of your API
url = "http://localhost:5000/query"

# Replace the payload with the data you want to send
payload = {"query": "buy soap"}

# Make the POST request
response = requests.post(url, json=payload)
# Check if the request was successful (status code 200)
if response.status_code == 200:
    print("Request was successful!")
    print(json.dumps(response.json(), indent=4))
else:
    print("Error:", response.status_code, response.text)