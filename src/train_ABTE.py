import os, sys
sys.path.insert(1, '../dataset')
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import torch
import ast
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int, default=8, help='batch size')
parser.add_argument('--epochs', type=int, default=5, help='number of epochs')
parser.add_argument('--lr', type=float, default=3*1e-5, help='learning rate')
parser.add_argument('--lr_schedule', type=bool, default=True, help='learning rate scheduler')
parser.add_argument('--adapter', type=bool, default=False, help='adapter')


def main (batch, epochs, lr, lr_schedule, adapter):

    #load
    data = pd.read_csv('../dataset/df_label_539980_train_20240705.csv')

    data['bio_tags'] = data['bio_tags'].apply(utils.convert_to_array)
    data['sentiment_tags'] = data['sentiment_tags'].apply(utils.convert_to_array)
    data['tokens'] = data['tokens'].apply(utils.convert_to_array)

    data['bio_tags'] = data['bio_tags'].apply(utils.replace_SEP).apply(utils.replace_tags)
    data['sentiment_tags'] = data['sentiment_tags'].apply(utils.replace_SEP).apply(utils.replace_sentiment_tags)

    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")

    from abte import ABTEModel
    modelABTE = ABTEModel(tokenizer, adapter)
    modelABTE.train(data, batch_size=batch, lr=lr, epochs=epochs, device=DEVICE, lr_schedule=lr_schedule)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args.batch, args.epochs, args.lr, args.lr_schedule, args.adapter)
    print('Done')
