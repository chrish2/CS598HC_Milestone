import random
from torch.utils.data import Dataset
from operator import itemgetter
import torch
import string
from nltk import word_tokenize

class Language(object):
    def __init__(self, data, vocab_limit):
        self.data = data
        self.tokenized_vocab = []
        self.vocab = self.create_vocab()

        self.truncated_vocab = sorted(self.vocab.items(), key=itemgetter(1), reverse=True)[:vocab_limit]
        self.tok_to_idx = dict()
        self.tok_to_idx['<UNK>'] = 0
        for idx, tok in enumerate(self.truncated_vocab):
            self.tok_to_idx[tok[0]] = idx + 1
        self.idx_to_tok = {idx: tok[0] for tok, idx in self.tok_to_idx.items()}

    def create_vocab(self):
        vocab = dict()
        for line in self.data:
            line = line.split()
            text = line[0].lower()

            text = text.translate(str.maketrans('', '', string.punctuation))
            note_tokenized = word_tokenize(text)

            for token in note_tokenized:
                vocab[token] = vocab.get(token, 0) + 1

        return vocab


class Dataset(Dataset):
    def __init__(self,
                 data_path,
                 vocab_size=100,
                 lang=None,
                 val_size=0.5,
                 seed=42,
                 is_val=False,
                 shuffle=True):

        self.data_path = data_path
        self.vocab_size = vocab_size
        self.parser = None
        self.val_size = val_size
        self.seed = seed
        self.is_val = is_val
        self.shuffle = shuffle
        self.labels = []


        self.data = []
        with open(self.data_path, 'r') as fr:
            for line in fr:

                [text, label] = line.split('>>><<<')
                text = text.translate(str.maketrans('', '', string.punctuation))
                text = text.lower()
                self.data.append(text.strip())
                self.labels.append(int(label))

        idxs = list(range(len(self.data)))
        num_val = int(len(idxs) * self.val_size)

        if self.is_val:
            idxs = idxs[:num_val]
        else:
            idxs = idxs[num_val:]


        random.seed(self.seed)
        if self.shuffle:
            random.shuffle(idxs)

        num_val = int(len(idxs) * self.val_size)
        if self.is_val:
            idxs = idxs[:num_val]
        else:
            idxs = idxs[num_val:]

        self.data = [self.data[idx] for idx in idxs]

        if lang is None:
            lang = Language(data=self.data, vocab_limit=self.vocab_size)

        self.lang = lang

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_token_list = self.data[idx].split()
        input_seq = tokens_to_seq(input_token_list, self.lang.tok_to_idx, self.vocab_size)
        label = self.labels[idx]
        return input_seq, label






def tokens_to_seq(tokens, tok_to_idx, max_length):
    seq = torch.zeros(max_length).long()
    for pos, token in enumerate(tokens):
        if token in tok_to_idx:
            idx = tok_to_idx[token]
        else:
            idx = tok_to_idx['<UNK>']


        try:
            seq[pos] = idx
        except:
            a=1

    return seq

