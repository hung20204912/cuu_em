import torch
import numpy as np
import platform

from sklearn.model_selection import train_test_split

import sys 
sys.path.insert(0, r'/content/drive/MyDrive/LabRISE/CodeSmell/DeepSmells/program/dl_models')
import inputs
import input_data

import pickle
from sklearn.model_selection import StratifiedKFold

def write_file(file, str):
    file = open(file, mode="a+")
    file.write(str)
    file.close()

def device():
    if torch.cuda.is_available():
        print(f'[INFO] Using GPU: {torch.cuda.get_device_name()}\n')
        device = torch.device('cuda')
    else:
        print(f'\n[INFO] GPU not found. Using CPU: {platform.processor()}\n')
        device = torch.device('cpu')
    return device

# Function K fold cross validation
def get_data_pickle(data_path):
    '''
        structure of pickle file: 3 attributes
        - embedding
        - name: the identifier of the file (ex: 123456.java --> name = 123456)
        - label
    '''
    pklFile = open(data_path, 'rb')
    df = pickle.load(pklFile)
    X = np.array([row for row in df['embedding']])
    y = df['label'].values

    kFold = StratifiedKFold(n_splits= 5, shuffle=True, random_state=42)

    return (
        input_data.Input_data(X[train], y[train], X[test], y[test], None) for train, test in kFold.split(X, y)
    )

def get_smell_and_model(path):
    smell = None
    model = None

    if "longmethod" in path.lower():
        smell = "LongMethod"
    elif "featureenvy" in path.lower():
        smell = "FeatureEnvy"
    elif "godclass" in path.lower():
        smell = "GodClass"
    elif "dataclass" in path.lower():
        smell = "DataClass"
    else:
        raise ValueError("Smell not found - CHECK FILENAME OF DATASET")
    
    if "code2seq" in path.lower():
        model = "Code2Seq"
    elif "code2vec" in path.lower():
        model = "Code2Vec"
    elif "codebert" in path.lower():
        model = "CodeBERT"
    elif "cubert" in path.lower():
        model = "CuBERT"
    elif "codet5" in path.lower():
        model = "CodeT5"
    elif "codegen" in path.lower():
        model = "CodeGen"
    elif "starcoder" in path.lower():
        model = "StarCoder"
    elif "incoder" in path.lower():
        model = "InCoder" 
    else:
        raise ValueError("Model not found - CHECK FILENAME OF DATASET")
    return smell, model