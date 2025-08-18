# agent_a.py
import requests
import pprint

AGENT_B_URL = "http://127.0.0.1:9001/a2a/message"

payload = {
    "task": {
        "action": "convert_currency",
        "amount": 123.45,
        "from": "USD",
        "to": "EUR"
    }
}

def start_agent_a():
    """
    Start Agent A and call Agent B to perform a task.
    """
    print("Starting Agent A...")
    
    # Call Agent B with the payload
    print("Calling Agent B with payload:")
    pprint.pprint(payload)
    
    # Send the request to Agent B
    resp = requests.post(AGENT_B_URL, json=payload, timeout=10)
    resp.raise_for_status()
    pprint.pprint(resp.json())

