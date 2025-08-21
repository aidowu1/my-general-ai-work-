# agent_a.py
import requests
import pprint

import configs

class AgentA:
    """
    Agent A is a simple client that interacts with Agent B to convert currency and do some simple datetime computations.
    It sends a request to Agent B and prints the response.
    """

    @staticmethod
    def create_payload(
        amount: float, 
        from_currency: str, 
        to_currency: str,
        timezone: str
    ):
        """
        Prepare the payload to send to Agent B.
        This is a simple example that converts 123.45 USD to EUR.
        :param amount: Amount to convert
        :param from_currency: Currency to convert from (default is USD)
        :param to_currency: Currency to convert to (default is EUR)
        :param timezone: Timezone for datetime calculation (default is UTC)
        :return: Dictionary with the task to perform
        """
        return {
            configs.CURRENCY_CONVERTER_TASK_NAME: {                
                "amount": amount,
                "from": from_currency,                
                "to": to_currency,
            },
            configs.DATETIME_CALCULATOR_TASK_NAME: {
                "timezone": timezone
            }
        }
    
    @staticmethod
    def start_agent_a(
        amount: float = 123.45, 
        from_currency: str = "USD", 
        to_currency: str = "EUR",
        timezone: str = "UTC"
    ):
        """
        Start Agent A and call Agent B to perform a task.
        :param amount: Amount to convert
        :param from_currency: Currency to convert from (default is USD)
        :param to_currency: Currency to convert to (default is EUR)
        :param timezone: Timezone for datetime calculation (default is UTC)
        :return: None
        """
        print("Starting Agent A...")
        payload = AgentA.create_payload(
            amount=amount, 
            from_currency=from_currency, 
            to_currency=to_currency,
            timezone=timezone
        )
        
        # Call Agent B with the payload
        print("Calling Agent B with payload:")
        pprint.pprint(payload)
        
        # Send the request to Agent B
        response = requests.post(configs.AGENT_B_URL, json=payload, timeout=10)
        response.raise_for_status()
        response_object = response.json()
        print(f"{configs.NEW_LINE}{configs.NEW_LINE}")
        
        currency_task_response = response_object.get("result", {}).get("currency", {})
        datetime_task_response = response_object.get("result", {}).get("datetime", {})
        
        print(f"Currency Conversion Response for {amount} from {from_currency} to {to_currency}:")
        pprint.pprint(currency_task_response)
        print(f"{configs.LINE_DIVIDER}{configs.NEW_LINE}{configs.NEW_LINE}")
        print(f"Datetime Timezone Calculation for {timezone} is:")
        pprint.pprint(datetime_task_response)
        print(f"{configs.LINE_DIVIDER}{configs.NEW_LINE}{configs.NEW_LINE}")

