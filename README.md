Medical-QA-extractive
==============================

The purpose of this repositories is to train LLMs on extractive Question-Answering in biomedical domain.

# 1. Model

# 2. Dataset
## 2.1. COVID-QA
This dataset contains 2,019 question/answer pairs annotated by volunteer biomedical experts on scientific articles regarding COVID-19 and other medical issues. The dataset can be found here: https://github.com/deepset-ai/COVID-QA. The preprocessed data can be found here https://huggingface.co/datasets/covid_qa_deepset.

## 2.2 BioASQ
This dataset consists of many biomedical tasks. For QA, they have a dataset with question-answer pairs on PubMed articles.
>Task 9B will use benchmark datasets containing development and test questions, in English, along with gold standard (reference) answers. The benchmark datasets are being constructed by a team of biomedical experts from around Europe.
The benchmark datasets contain four types of questions:
Yes/no questions: These are questions that, strictly speaking, require "yes" or "no" answers, though of course in practice longer answers will often be desirable. For example, "Do CpG islands colocalise with transcription start sites?" is a yes/no question.
Factoid questions: These are questions that, strictly speaking, require a particular entity name (e.g., of a disease, drug, or gene), a number, or a similar short expression as an answer, though again a longer answer may be desirable in practice. For example, "Which virus is best known as the cause of infectious mononucleosis?" is a factoid question.
List questions: These are questions that, strictly speaking, require a list of entity names (e.g., a list of gene names), numbers, or similar short expressions as an answer; again, in practice additional information may be desirable. For example, "Which are the Raf kinase inhibitors?" is a list question.
Summary questions: These are questions that do not belong in any of the previous categories and can only be answered by producing a short text summarizing the most prominent relevant information. For example, "What is the treatment of infectious mononucleosis?" is a summary question.

More details can be found here: http://participants-area.bioasq.org/general_information/Task9b/

# 3. Training setup
## 3.1 Environment
Here I use Python version 3.9.2. All the dependencies are listed in requirements.txt.
You also need to install the repo as a package `pip install -e .`.

## 3.2 Run the code
An example to run the training code is
```
python src/models/run_qa.py \
    --model_name_or_path 'UFNLP/gatortrons' \
    --dataset_name 'longluu/covid-qa-split' \
    --do_train \
    --do_eval\
    --per_device_train_batch_size 4 \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 512 \
    --doc_stride 250 \
    --max_answer_length 200 \
    --output_dir "/home/ec2-user/SageMaker/Medical-QA-extractive/models/COVID-QA/gatortrons/" \
    --overwrite_output_dir
```

# 4. Results
The fine-tuned models and brief results can be found at my huggingface page https://huggingface.co/longluu.
You can also look at the notebooks folder for training and test results.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py

