import pandas as pd
import json

def load_json_dataset(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)