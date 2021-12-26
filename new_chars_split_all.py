import os
import hanzidentifier
from pathlib import Path
from tqdm import tqdm

dataset_txt = [d for d in open("new_chars/dataset.txt","r").read().split('\n') if d!='']
dataset_txt_train = open("new_chars/dataset_train_all.txt","w+")

char_len = {}
max_char_len = {}
max_len = 250
for d in tqdm(dataset_txt):
    path,text = d.split(',')
    vs = max_char_len.get(text)
    if vs is None:
        max_char_len[text] = 1
    else:
        max_char_len[text] += 1
for d in tqdm(dataset_txt):
    path,text = d.split(',')
    vs = char_len.get(text)
    if vs is None:
        char_len[text] = 1
    else:
        char_len[text] += 1
    if char_len[text]%max(max_char_len[text]//max_len,1)==0:
        print(f"{path},{text}",file=dataset_txt_train)
dataset_txt_train.close()
