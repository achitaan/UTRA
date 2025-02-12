import json
import requests

def send_json_to_backend(json_data, url):
    """
    Sends json_data to the specified backend URL via an HTTP POST request.
    
    Args:
        json_data (dict): A Python dictionary that will be converted to JSON.
        url (str): The endpoint to which the JSON should be sent.
        
    Returns:
        requests.Response: The response object from the POST request.
    """
    headers = {
        'Content-Type': 'application/json'  # Tells the server we are sending JSON
    }

    # Convert Python dict to JSON string and send in the request body
    response = requests.post(url, headers=headers, json=json_data)

    return response

if __name__ == "__main__":
    # Example JSON data (Python dictionary)
    data_to_send = {
        "session_id": 1234,
        "movement_score": 85,
        "coverage_score": 92,
        "overall_score": 88
    }

    # Your backend endpoint
    backend_url = "http://your-backend-server.com/api/handwash-scores"

    # Send the JSON
    try:
        response = send_json_to_backend(data_to_send, backend_url)
        
        # Check status code or parse response
        if response.status_code == 200:
            print("Data posted successfully!", response.json())
        else:
            print("Failed to post data. Status code:", response.status_code)
            print("Response text:", response.text)
    except requests.exceptions.RequestException as e:
        print("Error sending data:", e)
