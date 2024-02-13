from transformers import AutoModelForTokenClassification, AutoConfig, AutoTokenizer, DataCollatorForTokenClassification
import torch
import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score
from src.data.data_loader import *
from torch.nn.functional import cross_entropy
from typing import List
import logging
logging.disable(logging.INFO) # disable INFO and DEBUG logging everywhere
    
class ModelLoader():
    def __init__(self, model_name: str):
        self.model_name = model_name
        
    def load_model(self, num_labels: int):
        # Choose the gpu to load model on (if available)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('The device to run the model:', device)
        
        # Load the model
        print('Load the pretrained model ...')
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        config = AutoConfig.from_pretrained(self.model_name)
        config.num_labels = num_labels
        QA_model = AutoModelForTokenClassification.from_pretrained(self.model_name, num_labels=num_labels)
        print('The model has ' + str(QA_model.num_parameters()/1e6) + 'millions parameters.')
        
        return QA_model, tokenizer, config
