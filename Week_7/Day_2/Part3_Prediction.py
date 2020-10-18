import requests
import json

url = 'http://localhost:5000/api/'

data = [[140.34, 1.68, 2.7, 0, 98.0, 2.8, 1.31, 5.53, 2.7, 130.0, 4.57, 1.96, 60.0]]
j_data = json.dumps(data)
headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
r = requests.post(url, data=j_data, headers=headers)
print(r)
print("Your wine belongs to class: " + r.text)