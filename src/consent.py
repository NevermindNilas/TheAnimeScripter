import json
import os
import logging
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def cleanLine(line):
    return line.strip()


def isSectionHeader(line):
    return line.startswith("==============")


def isKeyValuePair(line):
    return ": " in line


def extractKeyValue(line):
    key, value = line.split(": ", 1)
    return key, value


def Consent(logPath):
    if not os.path.exists(logPath):
        logging.info("No log.txt file was found, exiting")
        return

    with open(logPath, "r") as f:
        lines = f.readlines()

    cleaned_lines = [cleanLine(line) for line in lines if line.strip() != ""]
    data = {}
    section = None
    ignore_section = False

    for line in cleaned_lines:
        if isSectionHeader(line):
            section_name = line.strip("=").strip()
            if section_name in [
                "Arguments Checker",
                "Command Line Arguments",
                "Video Metadata",
            ]:
                ignore_section = True
            else:
                ignore_section = False
                section = section_name
                data[section] = {}
        elif not ignore_section:
            if section and isKeyValuePair(line):
                key, value = extractKeyValue(line)
                if section == "Arguments" and key in ["INPUT", "OUTPUT"]:
                    continue
                elif value.strip() == "":
                    continue

                # Ignore "Decoding options", "Encoding options", "Mering audio with" in "Processing Outputs"
                if section == "Processing Outputs" and key in [
                    "Decoding options",
                    "Encoding options",
                    "Merging audio with",
                ]:
                    continue

                data[section][key] = value
            elif section == "Processing Outputs":
                # If the line is not a key-value pair and the current section is "Processing Outputs",
                # add the line to a "details" list in the "Processing Outputs" section.
                if "details" not in data[section]:
                    data[section]["details"] = []
                data[section]["details"].append(line)

    jsonPath = logPath.replace("log.txt", "consent.json")
    with open(jsonPath, "w") as f:
        json.dump(data, f, indent=4)

    apiUrl = "https://116.203.28.198:8080/api"  # Have fun abusing this
    with open(jsonPath, "r") as f:
        data = json.load(f)

    session = Session()
    retries = Retry(total=1, backoff_factor=0, status_forcelist=[ 500, 502, 503, 504 ])
    session.mount('https://', HTTPAdapter(max_retries=retries))

    try:
        response = session.post(apiUrl, json=data, verify=False)
        return response.status_code == 200
    except Exception as e:
        error = str(e).replace(apiUrl, "API_URL")
        logging.error(f"Failed to send data to API: {error}")
