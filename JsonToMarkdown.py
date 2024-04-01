import json
import pandas as pd

try:
    data = json.loads(open('benchmarkResults.json').read())
except FileNotFoundError:
    print("benchmarkResults.json not found")
    exit()

htmlTables = '<div style="display: flex; justify-content: space-between;">\n'
for category, methods in data.items():
    df = pd.DataFrame(list(methods.items()), columns=['Method', 'FPS'])
    htmlTables += f'<div style="flex: 1;">\n\n## {category}\n\n{df.to_html(index=False)}\n\n</div>\n'
htmlTables += '</div>\n'

with open('result.md', 'w') as f:
    f.write(htmlTables)