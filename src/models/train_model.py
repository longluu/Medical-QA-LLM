from src.data.data_loader import *
from src.models.model import *
from transformers import DataCollatorForTokenClassification, TrainingArguments, Trainer
from seqeval.metrics import f1_score
import argparse
import numpy as np

class ModelTrainer():
    def __init__(self, dataset_name: str, model_name: str, outdir: str, max_length: int, doc_stride: int,\
                learning_rate: float, epoch: int, batch_size: int, weight_decay: float):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.outdir = outdir
        self.max_length = max_length
        self.doc_stride = doc_stride
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        
    def train_model(self):
        # Load and preprocess the data
        dataset_loader = DatasetLoader(dataset_name=self.dataset_name, model_name=self.model_name, \
                                       max_length=self.max_length, doc_stride=self.doc_stride)
        train_dataset, validation_dataset, validation_dataset_raw = data_loader.load_dataset()


def main():
    # Parse the arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True,
                        help="names of pretrained model, e.g. 'UFNLP/gatortron-base'")
    # resume training on a NER model if set it will overwrite pretrained_model
    parser.add_argument("--resume_from_model", type=str, default=None,
                        help="The NER model file or directory for continuous fine tuning.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="The input data directory.")
    parser.add_argument("--outdir", type=str, required=True,
                        help="directory for saving new model checkpoints")
    parser.add_argument("--max_length", default=512, type=int,
                        help="max length to split the context")
    parser.add_argument("--doc_stride", default=250, type=int,
                        help="the stride of windowing to split the context")
    parser.add_argument("--batch_size", default=32, type=int,
                        help="The batch size for training and evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for optimizer.")
    parser.add_argument("--num_train_epochs", default=3, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    
    global_args = parser.parse_args()
    
    # Initiate a trainer instance
    model_trainer = ModelTrainer(dataset_name=global_args.data_dir,
                                 path_umls_semtype=global_args.path_umls_semtype,
                                 model_name=global_args.model_name, 
                                 outdir=global_args.new_model_dir, 
                                 epoch=global_args.num_train_epochs, 
                                 batch_size=global_args.batch_size, 
                                 weight_decay=global_args.weight_decay,
                                 learning_rate=global_args.learning_rate)
    
    # Start training
    model_trainer.train_model()
    
if __name__ == '__main__':
    main()