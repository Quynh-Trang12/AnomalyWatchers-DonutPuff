
import json
nb_path = r'c:\Users\quynh\Desktop\COS30049\code\fraud-simulator-demo\notebooks\01_primary_analysis.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        if 'from sklearn.pipeline import Pipeline' in "".join(cell['source']):
            print("VERIFIED: Pipeline import exists.")
            exit(0)
print("FAILED: Pipeline import NOT found.")
exit(1)
