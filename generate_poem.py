import torch
import os
import argparse
import nltk, inflect, spacy
from transformers import AutoTokenizer, AutoConfig, AutoModelForPreTraining
import numpy as np
import random


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def caption_fn(path):
    cmd = "python3 ../predict.py --path " + path + " > caption.txt"
    os.system(cmd)
    f = open("caption.txt", "r")
    caption = f.read()
    cmd = "rm caption.txt"
    os.system(cmd)

    nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner'])
    p = inflect.engine()
    noun = []
    verb = []
    doc = nltk.word_tokenize(caption)
    for token in nltk.pos_tag(doc):
        if len(token[0]) <2:
            continue
        if token[1] == "NN":
            noun.append(token[0])
        elif token[1] == "NNS":
            if p.singular_noun(token[0]) == False:
                noun.append(token[0])
            else:
                noun.append(p.singular_noun(token[0]))
        elif "VB" in token[1]:
            sentence = token[0]
            doc = nlp(sentence)
            verb.append("".join([token.lemma_ for token in doc]))
    return noun, verb, caption


def get_tokenier(special_tokens=None, MODEL ='gpt2'):
    tokenizer = AutoTokenizer.from_pretrained(MODEL) #GPT2Tokenizer

    if special_tokens:
        tokenizer.add_special_tokens(special_tokens)
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

def poem_fn(noun, verb, path):
    SPECIAL_TOKENS  = { "bos_token": "<|BOS|>",
                    "eos_token": "<|EOS|>",
                    "unk_token": "<|UNK|>",                    
                    "pad_token": "<|PAD|>",
                    "sep_token": "<|SEP|>"}
    tokenizer = get_tokenier(special_tokens=SPECIAL_TOKENS)
    model = get_model(tokenizer, 
                    special_tokens=SPECIAL_TOKENS,
                    load_model_path=path)
    keywords = noun + verb

    prompt = SPECIAL_TOKENS['bos_token'] + " ".join(keywords) + SPECIAL_TOKENS['sep_token']
    generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    device = torch.device("cuda")
    generated = generated.to(device)

    model.eval()
    sample_outputs = model.generate(generated, 
                                    do_sample=True,   
                                    min_length=40, 
                                    max_length=200,
                                    top_k=30,                                 
                                    top_p=0.7,        
                                    temperature=0.9,
                                    repetition_penalty=2.0,
                                    num_return_sequences=1
                                    )

    poem = tokenizer.decode(sample_outputs[0], skip_special_tokens=True).replace(" ".join(keywords), '')
    return poem

def poem_title_fn(poem, path_title):
    SPECIAL_TOKENS  = { "bos_token": "<|BOS|>",
                    "eos_token": "<|EOS|>",
                    "unk_token": "<|UNK|>",                    
                    "pad_token": "<|PAD|>",
                    "sep_token": "<|SEP|>"}
    tokenizer = get_tokenier(special_tokens=SPECIAL_TOKENS)
    model = get_model(tokenizer, 
                    special_tokens=SPECIAL_TOKENS,
                    load_model_path=path_title)

    prompt =  SPECIAL_TOKENS['bos_token'] + poem + SPECIAL_TOKENS['sep_token']
    generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    device = torch.device("cuda")
    generated = generated.to(device)

    model.eval()
    sample_outputs = model.generate(generated, 
                                    do_sample=True,   
                                    min_length=len(poem)+1, 
                                    max_length=400,
                                    top_k=30,                                 
                                    top_p=0.7,        
                                    temperature=0.9,
                                    repetition_penalty=2.0,
                                    num_return_sequences=1
                                    )

    title = tokenizer.decode(sample_outputs[0], skip_special_tokens=True)[len(poem):]
    return title.split('\n')[0]


def main():
    parser = argparse.ArgumentParser(description='Poem Generator')
    parser.add_argument('--path', type=str, help='path to image', required=True)
    parser.add_argument('--checkpoint_poem', type=str, help='path to gpt2 checkpoint for peom', required=True)
    parser.add_argument('--checkpoint_title', type=str, help='path to gpt2 checkpoint for title', required=True)
    parser.add_argument('--seed', type=int, default=42, help='path to image')
    args = parser.parse_args()
    path = args.path
    path_poem = args.checkpoint_poem
    path_title = args.checkpoint_title
    seed_everything(args.seed)
    noun, verb, caption = caption_fn(path)
    poem = poem_fn(noun, verb, path_poem)
    title = poem_title_fn(poem, path_title)
    print(poem)
    file = open("poem.txt", "w")
    content = [title+"\n", poem.strip()]
    file.writelines(content)
    file.close()
    # print("Poem Title:\t", title, "\nPoem:\n", poem.strip())

if __name__ == '__main__':
    main()