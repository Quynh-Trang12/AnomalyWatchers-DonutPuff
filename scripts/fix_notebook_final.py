
import json
import os

nb_path = r'c:\Users\quynh\Desktop\COS30049\code\fraud-simulator-demo\notebooks\01_primary_analysis.ipynb'

print(f"Reading {nb_path}...")
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

found_import = False
changes_made = False

# We will simply look for the cell containing 'from sklearn.compose import ColumnTransformer'
# and verify if 'from sklearn.pipeline import Pipeline' is in that same cell's source.

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        # Join source to check easily, but we need to modify the list 'source'
        source_text = "".join(source)
        
        if 'from sklearn.compose import ColumnTransformer' in source_text:
            if 'from sklearn.pipeline import Pipeline' in source_text:
                print("Pipeline import ALREADY EXISTS in this cell.")
                found_import = True
            else:
                print("Pipeline import MISSING in this cell. Adding it.")
                # Find the line index to insert after
                insert_idx = -1
                for i, line in enumerate(source):
                    if 'from sklearn.compose import ColumnTransformer' in line:
                        insert_idx = i
                        break
                
                if insert_idx != -1:
                    source.insert(insert_idx + 1, "from sklearn.pipeline import Pipeline\n")
                    changes_made = True
                    found_import = True
                    print(f"Inserted Pipeline import at line {insert_idx + 1}")
            break

if not found_import:
    print("Could not find the target cell to insert imports!")

if changes_made:
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
    print("Notebook saved successfully.")
else:
    print("No changes required (or failed to apply changes).")
