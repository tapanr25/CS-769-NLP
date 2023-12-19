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
from LF_esm import EsmMM
import wandb
import random
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
    model = EsmMM.from_pretrained(f'facebook/{model_name}', num_labels=2)
    # model.resize_token_embeddings(len(tokenizer))

    # Initialize wandb
    wandb.init(project='NetTcr_esm', name=f'{model_name}_{sep}__mm_freez')

    # Configure training arguments
    training_args = TrainingArguments(output_dir=f'1012/NetTcr_{model_name}_{sep}_mm_freez_lrsmal',
                                      evaluation_strategy='steps',
                                      per_device_train_batch_size=64,
                                      per_device_eval_batch_size=64,
                                      num_train_epochs=100,
                                      learning_rate=0.000000001,
                                      logging_strategy='steps',
                                      save_steps=1000,
                                      save_total_limit=1,
                                      eval_steps=1000,
                                      report_to='wandb',
                                      load_best_model_at_end=True,
                                      metric_for_best_model="accuracy")

    # Configure metrics
    metric = evaluate.load('accuracy', experiment_id=random.randint(1,100))
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
    train_df = pd.read_csv('tcrB_dataset/train_beta_90.csv')
    test_df = pd.read_csv('tcrB_dataset/mira_eval_threshold90.csv')
    train_df['binder']=train_df['binder'].astype('int')
    test_df['binder']=test_df['binder'].astype('int')
    
    def insert_1_after_characters(s):
        return '1'.join(s) + '1'
    # train_df['seq_2'] = train_df['seq_2'].apply(insert_1_after_characters)
    # test_df['seq_2'] = test_df['seq_2'].apply(insert_1_after_characters)

    ### comment for 1 vocab
    # df['epitope_aa'] = df['epitope_aa'].apply(insert_1_after_characters)


    # train_df, test_df = train_test_split(df, test_size=0.3, random_state=101)
    # print((train_df).head())
    # Format data
    
    # train_df = pd.DataFrame({'seq1': train_df['cdr3_alpha_aa'] + sep + train_df['cdr3_beta_aa']+sep, 'seq2': train_df['epitope_aa'],
    #                          'label': train_df['label_true_pair']})
    # test_df = pd.DataFrame({'seq1': test_df['cdr3_alpha_aa'] + sep +test_df['cdr3_beta_aa']+sep,'seq2': test_df['epitope_aa'],
    #                         'label': test_df['label_true_pair']})
    train_df = pd.DataFrame({'seq1': train_df['CDR3b']+sep, 'seq2': train_df['peptide'],
                             'label': train_df['binder']})
    test_df = pd.DataFrame({'seq1': test_df['CDR3b']+sep,'seq2': test_df['peptide'],
                            'label': test_df['binder']})
    
    # train_df = pd.DataFrame({'seq1': train_df['seq_1'] + sep ,'seq2': train_df['seq_2'],
    #                          'label': train_df['label']})
    # test_df = pd.DataFrame({'seq1': test_df['seq_1'] + sep ,'seq2': test_df['seq_2'],
    #                         'label': test_df['label']})
    
    dataset = DatasetDict({
        'train': Dataset.from_pandas(train_df),
        'test': Dataset.from_pandas(test_df)
    })
    print('loading model')
    # Load tokenizer and add custom tokens
    tokenizer = AutoTokenizer.from_pretrained(f'facebook/{model_name}')
    tokenizer.add_tokens(['0'])
    
    # Tokenize sequences
    def tokenize_function(dataset):
        seq1 = tokenizer(dataset['seq1'], return_tensors='pt', max_length=len(tokenizer), padding='max_length', truncation=True)
        seq2 = tokenizer(dataset['seq2'], return_tensors='pt', max_length=len(tokenizer), padding='max_length', truncation=True)
        return {'input_ids': seq1['input_ids'],
                'input_ids2':seq2['input_ids'],
                'attention_mask': seq1['attention_mask'],
                'attention_mask2': seq2['attention_mask']
                }    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, batch_size=16)


    print('loading trainer')
    # Configure Trainer
    trainer = configure_trainer(tokenizer, tokenized_dataset, model_name, sep)
    print('training')
    # Run model
    trainer.train()


if __name__ == '__main__':
    main('esm2_t6_8M_UR50D', 'AAAAA')