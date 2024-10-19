import json
import random
import string

jsonPath = r"G:\TheAnimeScripter\benchmarkResults.json"


def jsonToMarkdownTable(jsonData):
    markdown = "# Benchmark Results\n\n"

    markdown += "## System Information\n\n"
    for key, value in jsonData["System Info"].items():
        markdown += f"- **{key}:** {value}\n"
    markdown += "\n"

    markdown += f"**Version:** {jsonData['Version']}  \n"
    markdown += f"**Testing Methodology:** {jsonData['Testing Methodology']}\n\n"

    for category, models in jsonData["Results"].items():
        markdown += f"## {category} Results\n\n"
        markdown += "| Model | Time (s) | FPS |\n"
        markdown += "|-------|-----------|-----|\n"

        for model, results in models.items():
            timeMs, fps = results
            markdown += f"| {model} | {timeMs:.2f} | {fps:.2f} |\n"

        markdown += "\n"

    return markdown


def generateRandomFilename(extension="md"):
    randomString = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
    return f"{randomString}.{extension}"


with open(jsonPath, "r") as file:
    jsonData = json.load(file)

markdownOutput = jsonToMarkdownTable(jsonData)

randomFilename = generateRandomFilename()
with open(randomFilename, "w") as file:
    file.write(markdownOutput)

print(f"{randomFilename} has been generated successfully.")
