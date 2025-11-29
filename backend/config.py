import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')

BERT_PATH = os.path.join(MODELS_DIR, 'bert_model')
CLASSIC_PATH = os.path.join(MODELS_DIR, 'classic_model')

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

DEVICE = get_device()