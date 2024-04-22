import json
import os
import logging
import requests

def Consent(logPath):
    """
    Reads the log.txt file and sends the data to a honeypot API
    """

    if not os.path.exists(logPath):
        logging.info("No log.txt file was found, exiting")
        return False

    data = {}
    section = None

    jsonPath = logPath.replace("log.txt", "consent.json")
    apiUrl = "https://116.203.28.198:8080/api" # Have fun abusing this, it's a honeypot
    with open(logPath, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith("============== Arguments ==============") or line.startswith("============== System Checker ==============") or line.startswith("============== Processing Outputs =============="):
                section = line.strip("=").strip()
                data[section] = {}
                continue

            if section:
                if line == "" or line.startswith("INPUT: ") or line.startswith("OUTPUT: "): # Skip empty lines and input/output lines for user privacy reasons
                    continue

                key, value = line.split(": ", 1)
                data[section][key] = value

    with open(jsonPath, "w") as f:
        json.dump(data, f)

    with open(jsonPath, "r") as f:
        data = json.load(f)

    try:
        response = requests.post(apiUrl, json=data)
        return response.status_code == 200
    except Exception as e:
        logging.error(f"Failed to send data to API: {e}")