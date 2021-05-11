import os, json, argparse
import numpy as np
import re
import random
import time
import csv
import datetime
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dataloader import *

from transformers import AutoTokenizer, AutoConfig, AutoModelForPreTraining, \
                         AdamW, get_linear_schedule_with_warmup, \
                         TrainingArguments, BeamScorer, Trainer
                         
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, random_split, DataLoader, \
                             RandomSampler, SequentialSampler

INPUT_DIR       = 'data/feat/multi_m/'
MODEL           = 'gpt2'
UNFREEZE_LAST_N = 6 

MAXLEN          = 400

TRAIN_BATCHSIZE = 8
BATCH_UPDATE    = 32
EPOCHS          = 8
LR              = 1e-4
EPS             = 1e-8
WARMUP_STEPS    = 1e2
SEED            = 42

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_tokenier(special_tokens=None):
    tokenizer = AutoTokenizer.from_pretrained(MODEL) #GPT2Tokenizer

    if special_tokens:
        tokenizer.add_special_tokens(special_tokens)
        print("Special tokens added")
    return tokenizer

def get_model(tokenizer, special_tokens=None, load_model_path=None, MODEL='gpt2'):

    if special_tokens:
        config = AutoConfig.from_pretrained(MODEL, 
                                            bos_token_id=tokenizer.bos_token_id,
                                            eos_token_id=tokenizer.eos_token_id,
                                            sep_token_id=tokenizer.sep_token_id,
                                            pad_token_id=tokenizer.pad_token_id,
                                            output_hidden_states=False)
    else: 
        config = AutoConfig.from_pretrained(MODEL,                                     
                                            pad_token_id=tokenizer.eos_token_id,
                                            output_hidden_states=False)    

    model = AutoModelForPreTraining.from_pretrained(MODEL, config=config)

    if special_tokens:
        model.resize_token_embeddings(len(tokenizer))

    if load_model_path:
        model.load_state_dict(torch.load(load_model_path))

    model.cuda()
    return model




with open('data/feat/multi_pos.json', 'rb') as myfile:
    data = json.load(myfile)


seed_everything(42)
tokenizer = get_tokenier(special_tokens=SPECIAL_TOKENS)
model = get_model(tokenizer, 
                  special_tokens=SPECIAL_TOKENS,
                  load_model_path='checkpoints/combined2/pytorch_model.bin')


predictions = {}
for img in tqdm(list(data.keys())):
    seed_everything(42)
    title = data[img.replace(".npy", '')]["noun"] + data[img.replace(".npy", '')]["verb"]

    prompt = SPECIAL_TOKENS['bos_token'] + " ".join(title) + SPECIAL_TOKENS['sep_token']
            
    generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    device = torch.device("cuda")
    generated = generated.to(device)

    model.eval()
    sample_outputs = model.generate(generated, 
                                    do_sample=True,   
                                    min_length=100, 
                                    max_length=MAXLEN,
                                    top_k=30,                                 
                                    top_p=0.7,        
                                    temperature=0.9,
                                    repetition_penalty=2.0,
                                    num_return_sequences=1
                                    )

    text = tokenizer.decode(sample_outputs[0], skip_special_tokens=True).replace(" ".join(title), '')
    predictions[img.replace(".npy",  '')] = text

json.dump(predictions, open("full_multi_combined2.json","w"))

def train(args):
    seed_everything(args.seed)
    SPECIAL_TOKENS  = { "bos_token": "<|BOS|>",
                    "eos_token": "<|EOS|>",
                    "unk_token": "<|UNK|>",                    
                    "pad_token": "<|PAD|>",
                    "sep_token": "<|SEP|>"}
    tokenizer = get_tokenier(special_tokens=SPECIAL_TOKENS)
    model = get_model(tokenizer, special_tokens=SPECIAL_TOKENS, MODEL=args.model)
    for parameter in model.parameters():
        parameter.requires_grad = False

    for i, m in enumerate(model.transformer.h):        
        #Only un-freeze the last n transformer blocks
        if i+1 > 12 - args.unfreeze_layer:
            for parameter in m.parameters():
                parameter.requires_grad = True 

    for parameter in model.transformer.ln_f.parameters():        
        parameter.requires_grad = True

    for parameter in model.lm_head.parameters():        
        parameter.requires_grad = True

    with open(args.input_dir+'titleuni.json', 'rb') as myfile:
        uni = json.load(myfile)

    full_split = list(uni.keys())
    random.shuffle(full_split)
    train_size = int(len(full_split)*0.8)
    train_split = full_split[:train_size]
    val_split = full_split[train_size:]

    train_dataset = POSDataset(train_split, uni, tokenizer, SPECIAL_TOKENS, args.max_token, True, 4)
    val_dataset = POSDataset(val_split, uni, tokenizer, SPECIAL_TOKENS, args.max_token, False, 4)

    
    training_args = TrainingArguments(
        output_dir=args.checkpoint_poem,
        num_train_epochs=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.batch_update,
        evaluation_strategy="epoch",
        fp16=False,
        warmup_steps=args.wu,    
        learning_rate=args.lr,
        adam_epsilon=args.eps,
        weight_decay=0.01,        
        save_total_limit=1,
        load_best_model_at_end=True,     
    )

    trainer = Trainer(
        model=model,
        args=training_args,    
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer
    )


    trainer.train()
    trainer.save_model()

    seed_everything(args.seed)
    model = get_model(tokenizer, special_tokens=SPECIAL_TOKENS, MODEL=args.model)
    for parameter in model.parameters():
        parameter.requires_grad = False

    for i, m in enumerate(model.transformer.h):        
        #Only un-freeze the last n transformer blocks
        if i+1 > 12 - args.unfreeze_layer:
            for parameter in m.parameters():
                parameter.requires_grad = True 

    for parameter in model.transformer.ln_f.parameters():        
        parameter.requires_grad = True

    for parameter in model.lm_head.parameters():        
        parameter.requires_grad = True

    with open(args.input_dir+'title.json', 'rb') as myfile:
        title = json.load(myfile)

    full_split = list(title.keys())
    random.shuffle(full_split)
    train_size = int(len(full_split)*0.8)
    train_split = full_split[:train_size]
    val_split = full_split[train_size:]

    train_dataset = TitleDataset(train_split, title, tokenizer, SPECIAL_TOKENS, MAXLEN, True)
    val_dataset = TitleDataset(val_split, title, tokenizer, SPECIAL_TOKENS, MAXLEN, False)

    training_args = TrainingArguments(
        output_dir=args.checkpoint_title,
        num_train_epochs=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.batch_update,
        evaluation_strategy="epoch",
        fp16=False,
        warmup_steps=args.wu,    
        learning_rate=args.lr,
        adam_epsilon=args.eps,
        weight_decay=0.01,        
        save_total_limit=1,
        load_best_model_at_end=True,     
    )

    trainer = Trainer(
        model=model,
        args=training_args,    
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer
    )

    trainer.train()
    trainer.save_model()



def main():
    parser = argparse.ArgumentParser(description='Poem Generator')
    parser.add_argument('--checkpoint_poem', type=str, help='path to checkpoint for peom', required=True)
    parser.add_argument('--checkpoint_title', type=str, help='path to checkpoint for title', required=True)
    parser.add_argument('--seed', type=int, default=42, help='path to image')
    parser.add_argument('--input_dir', type=str, default="data/", help='path to poem data')
    parser.add_argument('--model', type=str, default="gpt2", help='model to be used')
    parser.add_argument('--unfreeze_layer', type=int, default=6, help='layer to be finetuned')
    parser.add_argument('--batch', type=int, default=8, help='batch size')
    parser.add_argument('--batch_update', type=int, default=32, help='batch update size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--eps', type=float, default=1e-8, help='Epsilon')
    parser.add_argument('--wu', type=float, default=1e2, help='WarmUp steps')
    parser.add_argument('--max_token', type=int, default=400, help='Max token in poem')
    args = parser.parse_args()

    train(args)

if __name__ == '__main__':
    main()