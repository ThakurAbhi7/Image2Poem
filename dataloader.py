import os, re, json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
import random

class POSDataset(Dataset):

    def __init__(self, split, data, tokenizer, SPECIAL_TOKENS, MAXLEN, randomize=True, MAXKEY=4):

        self.split = split
        self.tokenizer = tokenizer 
        self.data = data
        self.SPECIAL_TOKENS = SPECIAL_TOKENS
        self.MAXLEN = MAXLEN
        self.randomize = randomize
        self.MAXKEY = MAXKEY

    def __len__(self):
        return len(self.split)

    
    def __getitem__(self, i):
        img = self.split[i].replace(".npy", '')
        poem = self.data[img]['poem']
        keyword = self.data[img]['noun'] + self.data[img]['verb']
        if self.randomize:
            random.shuffle(keyword)

        input = self.SPECIAL_TOKENS['bos_token'] + " ".join(keyword[:self.MAXKEY]) + self.SPECIAL_TOKENS['sep_token'] + poem + self.SPECIAL_TOKENS['eos_token']
        encodings_dict = self.tokenizer(input,                                   
                                   truncation=True, 
                                   max_length=self.MAXLEN, 
                                   padding="max_length")   
        
        input_ids = encodings_dict['input_ids']
        attention_mask = encodings_dict['attention_mask']
        
        return {'label': torch.tensor(input_ids),
                'input_ids': torch.tensor(input_ids), 
                'attention_mask': torch.tensor(attention_mask)}



class TitleDataset(Dataset):

    def __init__(self, split, data, tokenizer, SPECIAL_TOKENS, MAXLEN, randomize=True):

        self.split = split
        self.tokenizer = tokenizer 
        self.data = data
        self.SPECIAL_TOKENS = SPECIAL_TOKENS
        self.MAXLEN = MAXLEN
        self.randomize = randomize

    def __len__(self):
        return len(self.split)

    
    def __getitem__(self, i):
        title = self.split[i]
        poem = self.data[title]

        input = self.SPECIAL_TOKENS['bos_token'] + poem + self.SPECIAL_TOKENS['sep_token'] + title + self.SPECIAL_TOKENS['eos_token']
        encodings_dict = self.tokenizer(input,                                   
                                   truncation=True, 
                                   max_length=self.MAXLEN, 
                                   padding="max_length")   
        
        input_ids = encodings_dict['input_ids']
        attention_mask = encodings_dict['attention_mask']
        
        return {'label': torch.tensor(input_ids),
                'input_ids': torch.tensor(input_ids), 
                'attention_mask': torch.tensor(attention_mask)}



