import time

import requests

def test_bert():
    url = "http://localhost:8000/most_similar"
    payload = {
        "text": "Want to go to the vesting bar?",
        "categories": ["activity", "eating"]
    }
    response = requests.post(url, json=payload)
    print(response.json())

if __name__ == "__main__":
    # Check timing of api
    timeBefore = time.time()
    test_bert()
    timeAfter = time.time()
    print(f"API call took {timeAfter - timeBefore:.2f} seconds")