import argparse

import torch
from transformers import BertConfig

from model import SentimentClassifier
from torch.utils.data import DataLoader, Dataset, TensorDataset

def train(model: SentimentClassifier):
    pass


def evaluate(model: SentimentClassifier):
    pass


def prepare_dataset():
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--num_labels', type=int, default=2)
    parser.add_argument('--do_lower_case', action='store_true', default=False)
    parser.add_argument('--do_train', action='store_true', default=False)
    parser.add_argument('--do_eval', action='store_true', default=False)
    parser.add_argument('--dataset', choices=['imdb'], required=True)
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()

    device = torch.device('cuda') if args.cuda else torch.device('cpu')
    config = BertConfig.from_pretrained(args.pretrained_model_name, **vars(args))
    model = SentimentClassifier(config)
    model = model.to(device)

    if args.do_train:
        train(model)

    if args.do_eval:
        evaluate(model)
