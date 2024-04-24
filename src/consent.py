import json
import os
import logging
import requests

def clean_line(line):
    return line.strip()

def is_section_header(line):
    return line.startswith("==============")

def is_key_value_pair(line):
    return ": " in line

def extract_key_value(line):
    key, value = line.split(": ", 1)
    return key, value

def Consent(logPath):
    if not os.path.exists(logPath):
        logging.info("No log.txt file was found, exiting")
        return

    with open(logPath, "r") as f:
        lines = f.readlines()

    cleaned_lines = [clean_line(line) for line in lines if line.strip() != ""]
    data = {}
    section = None
    ignore_section = False

    for line in cleaned_lines:
        if is_section_header(line):
            section_name = line.strip("=").strip()
            if section_name in ["Arguments Checker", "Command Line Arguments", "Video Metadata"]:
                ignore_section = True
            else:
                ignore_section = False
                section = section_name
                data[section] = {}
        elif not ignore_section:
            if section and is_key_value_pair(line):
                key, value = extract_key_value(line)
                if section == "Arguments" and key in ["INPUT", "OUTPUT"]:
                    continue
                elif value.strip() == "":
                    continue
                
                # Ignore "Decoding options" and "Encoding options" in "Processing Outputs"
                if section == "Processing Outputs" and key in ["Decoding options", "Encoding options", "Merging audio with"]:
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

    apiUrl = "https://116.203.28.198:8080/api"  # Have fun abusing this, it's a honeypot
    with open(jsonPath, "r") as f:
        data = json.load(f)

    try:
        response = requests.post(apiUrl, json=data, verify=False)
        return response.status_code == 200
    except Exception as e:
        logging.error(f"Failed to send data to API: {e}")