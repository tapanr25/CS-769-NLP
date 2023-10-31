"""Retrain ESM-2 on Olga's data.
"""
import sys

from datasets import Dataset, DatasetDict
import evaluate
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, EsmForSequenceClassification, logging, \
    Trainer, TrainingArguments
import wandb
from sklearn.model_selection import train_test_split
def configure_trainer(tokenizer, tokenized_dataset, model_name, sep):
    """
    Configure the Trainer.

    parameters
    ----------
    tokenizer: torch.Tokenizer
        Tokenizer.
    tokenized_dataset : torch.Dataset
        Dataset containing the tokenized training and test datasets.
    model_name : str
        Name of the ESM-2 model.
    seq: str
        Separator sequence.

    Returns
    -------
    trainer : torch.Trainer
        Trainer used to retrain ESM-2.

    """
    # Configure model
    model = EsmForSequenceClassification.from_pretrained(f'facebook/{model_name}', num_labels=2)
    model.resize_token_embeddings(len(tokenizer))

    # Initialize wandb
    wandb.init(project='stapler_esm', name=f'{model_name}_{sep}_ep_pc_v2_aabb')

    # Configure training arguments
    training_args = TrainingArguments(output_dir=f'tmp/stapler_{model_name}_{sep}_epv2_aabb',
                                      evaluation_strategy='epoch',
                                      per_device_train_batch_size=64,
                                      per_device_eval_batch_size=64,
                                      num_train_epochs=100,
                                      logging_strategy='epoch',
                                      learning_rate=0.000001,
                                      save_total_limit=1,
                                      
                                      report_to='wandb',
                                      load_best_model_at_end=True,
                                      metric_for_best_model="accuracy",
                                      save_strategy='epoch',)

    # Configure metrics
    metric = evaluate.load('accuracy')
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Instantiate Trainer
    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=tokenized_dataset['train'],
                      eval_dataset=tokenized_dataset['test'],
                      compute_metrics=compute_metrics)
    return trainer


def main(model_name, sep):
    """
    Entry point of the program.

    """

    # Load data
    print('No. of cuda devices:', torch.cuda.device_count())
    df = pd.read_csv('train-set_full-seq.csv')
    # test_df = pd.read_csv('train-set_full-seq.csv')[0.7*23544:]
    df['label_true_pair']=df['label_true_pair'].astype('int')
    
    def insert_1_after_characters(s):
        return '1'.join(s) + '1'
    # train_df['seq_2'] = train_df['seq_2'].apply(insert_1_after_characters)
    # test_df['seq_2'] = test_df['seq_2'].apply(insert_1_after_characters)

    ### comment for 1 vocab
    df['epitope_aa'] = df['epitope_aa'].apply(insert_1_after_characters)


    train_df, test_df = train_test_split(df, test_size=0.3)
    # print((train_df).head())
    # Format data
    
    train_df = pd.DataFrame({'seq': train_df['cdr3_alpha_aa'] + sep + train_df['epitope_aa']+ sep +train_df['cdr3_beta_aa'],
                             'label': train_df['label_true_pair']})
    test_df = pd.DataFrame({'seq': test_df['cdr3_alpha_aa'] + sep + test_df['epitope_aa']+ sep +test_df['cdr3_beta_aa'],
                            'label': test_df['label_true_pair']})
    
    dataset = DatasetDict({
        'train': Dataset.from_pandas(train_df),
        'test': Dataset.from_pandas(test_df)
    })
    print('loading model')
    # Load tokenizer and add custom tokens
    tokenizer = AutoTokenizer.from_pretrained(f'facebook/{model_name}')
    tokenizer.add_tokens([sep])
    epitope_vocab = ["A1", "C1", "D1", "E1", "F1", "G1", "H1", "I1", "K1", "L1", "M1", "N1", "P1", "Q1", "R1", "S1", "T1", "V1", "W1", "Y1"]

    ###########  comment for 1 vocab
    tokenizer.add_tokens(epitope_vocab)
    
    # Tokenize sequences
    def tokenize_function(dataset):
        return tokenizer(dataset['seq'], return_tensors='pt', max_length=len(tokenizer), padding='max_length', truncation=True)
    tokenized_dataset = dataset.map(tokenize_function, batched=True, batch_size=16)


    print('loading trainer')
    # Configure Trainer
    trainer = configure_trainer(tokenizer, tokenized_dataset, model_name, sep)
    print('training')
    # Run model
    trainer.train()

if __name__ == '__main__':
    main('esm2_t6_8M_UR50D', '0')
