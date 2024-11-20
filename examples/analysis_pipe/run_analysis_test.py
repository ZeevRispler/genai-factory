import time
import requests
import json


def realtime_text_processing(file_path: str):
    """
    Process a text file line by line with simulated real-time behavior.
    Each line will be sent to the API endpoint with a small delay between lines.

    :param file_path: Path to the text file to process
    """
    API_ENDPOINT = "http://localhost:8001/api/projects/default/workflows/default/infer"

    print("[System] Processing text file...")

    with open(file_path, 'r') as file:
        for line in file:
            # Remove trailing whitespace and newlines
            line = line.strip()

            # Create the request data
            data = {
                "question": line or "empty line",
            }

            # Send the request
            requests.post(url=API_ENDPOINT, data=json.dumps(data))

            # Simulate real-time delay (adjust the sleep time as needed)
            time.sleep(2)  # 2 second delay between lines

    # Send final flush request
    request = {
        "question": "final message"
    }
    print("ping")
    requests.post(url=API_ENDPOINT, data=json.dumps(request))

def main():
    # Run the realtime transcription
    realtime_text_processing(file_path="./test.txt")

if __name__ == "__main__":
    main()