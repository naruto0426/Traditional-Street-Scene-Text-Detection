import os
import hanzidentifier
from pathlib import Path
from tqdm import tqdm

dataset_txt = [d for d in open("new_chars/dataset.txt","r").read().split('\n') if d!='']
dataset_txt_train = open("new_chars/dataset_train.txt","w+")
chars = [char for char in open("chars_list_with_ctw.txt",'r').read().split('\n') if char!='']

char_len = {}
for d in tqdm(dataset_txt):
    path,text = d.split(',')
    if text in chars:
        vs = char_len.get(text)
        if vs is None:
            char_len[text] = 1
        else:
            char_len[text] += 1
        #if char_len[text]<=500:
        print(f"{path},{text}",file=dataset_txt_train)
dataset_txt_train.close()
