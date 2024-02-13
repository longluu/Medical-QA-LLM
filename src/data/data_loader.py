from itertools import islice
from datasets import load_dataset, Dataset, DatasetDict, ClassLabel
from transformers import AutoTokenizer
from typing import List
import os
import pandas as pd
import csv


class DatasetLoader():
    def __init__(self, dataset_name: str, model_name: str, max_length: int, doc_stride: int):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.max_length = max_length
        self.doc_stride = doc_stride
        
    def load_dataset(self):
        # Load the model tokenizer
        global tokenizer, max_length, doc_stride
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        max_length = self.max_length
        doc_stride = self.doc_stride
        
        # Load the dataset
        print('Loading and preprocessing the dataset ...')
        print(self.dataset_name)
        if 'covid_qa_deepset' in self.dataset_name:
            # Load and split
            dataset = load_dataset(self.dataset_name)
            dataset = dataset['train'].train_test_split(test_size=0.1)
            train_dataset = dataset['train']
            validation_dataset_raw = dataset['test']
            
            # Preprocess for model training
            train_dataset = train_dataset.map(
                self.preprocess_training_examples,
                batched=True,
                remove_columns=train_dataset.column_names,
            )
            validation_dataset = validation_dataset_raw.map(
                self.preprocess_validation_examples,
                batched=True,
                remove_columns=validation_dataset_raw.column_names,
            )
            
        return train_dataset, validation_dataset, validation_dataset_raw

    def preprocess_training_examples(self, examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=max_length,
            truncation="only_second",
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label is (0, 0)
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs
    
    def preprocess_validation_examples(self, examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=max_length,
            truncation="only_second",
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_map = inputs.pop("overflow_to_sample_mapping")
        example_ids = []

        for i in range(len(inputs["input_ids"])):
            sample_idx = sample_map[i]
            example_ids.append(examples["id"][sample_idx])

            sequence_ids = inputs.sequence_ids(i)
            offset = inputs["offset_mapping"][i]
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
            ]

        inputs["example_id"] = example_ids
        return inputs