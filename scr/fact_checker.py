import numpy as np
import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import sys
import re
import warnings
print("******************************\nSTART\n******************************")
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

from transformers import EarlyStoppingCallback

from transformers import AutoTokenizer, AutoConfig, AutoModelForPreTraining, AdamW, get_linear_schedule_with_warmup, TrainingArguments, BeamScorer, Trainer
import torch
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoConfig, AutoModelForPreTraining, AdamW, get_linear_schedule_with_warmup, \
    TrainingArguments, BeamScorer, Trainer
from torch.utils.data import Dataset, DataLoader


from bs4 import BeautifulSoup
import unicodedata


path = "./climate-fever.csv"
#path = r"C:\Saeid\Prj100\SA_33_txt_analytics\for_Markus_\fact"
df = pd.read_csv(path)
print(df.head())

print(df['evidences/0/evidence_id'].nunique())
SPECIAL_TOKENS  = { "bos_token": "<|BOS|>",
                    "eos_token": "<|EOS|>",
                    "unk_token": "<|UNK|>",
                    "pad_token": "<|PAD|>",
                    "sep_token": "<|SEP|>"}
claims = []
evidences = []
x = []
label = []
dic = {"SUPPORTS": 0, "REFUTES": 1, "NOT_ENOUGH_INFO": 2}
for i in range(df.shape[0]):
    claim = df.iloc[i]['claim']

    ce1 = df.iloc[i]['evidences/0/evidence']
    cel1 = df.iloc[i]['evidences/0/evidence_label']

    claims.append(claim)
    evidences.append(ce1)
    x.append(SPECIAL_TOKENS['bos_token'] + claim + SPECIAL_TOKENS['sep_token'] + ce1 + SPECIAL_TOKENS['eos_token'])
    label.append(dic[cel1])

    ce2 = df.iloc[i]['evidences/1/evidence']
    cel2 = df.iloc[i]['evidences/1/evidence_label']

    claims.append(claim)
    evidences.append(ce2)
    x.append(SPECIAL_TOKENS['bos_token'] + claim + SPECIAL_TOKENS['sep_token'] + ce2 + SPECIAL_TOKENS['eos_token'])
    label.append(dic[cel2])

    ce3 = df.iloc[i]['evidences/2/evidence']
    cel3 = df.iloc[i]['evidences/2/evidence_label']

    claims.append(claim)
    evidences.append(ce3)
    x.append(SPECIAL_TOKENS['bos_token'] + claim + SPECIAL_TOKENS['sep_token'] + ce3 + SPECIAL_TOKENS['eos_token'])
    label.append(dic[cel3])

    ce4 = df.iloc[i]['evidences/3/evidence']
    cel4 = df.iloc[i]['evidences/3/evidence_label']

    claims.append(claim)
    evidences.append(ce4)
    x.append(SPECIAL_TOKENS['bos_token'] + claim + SPECIAL_TOKENS['sep_token'] + ce4 + SPECIAL_TOKENS['eos_token'])
    label.append(dic[cel4])

    ce5 = df.iloc[i]['evidences/4/evidence']
    cel5 = df.iloc[i]['evidences/4/evidence_label']

    claims.append(claim)
    evidences.append(ce5)
    x.append(SPECIAL_TOKENS['bos_token'] + claim + SPECIAL_TOKENS['sep_token'] + ce5 + SPECIAL_TOKENS['eos_token'])
    label.append(dic[cel5])

xd = pd.DataFrame({'claim': claims, 'evidence': evidences, 'label': label})
cd = pd.DataFrame({'x': x, 'label': label})
print(xd.head())
print(cd.head())


base_model = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.add_special_tokens(SPECIAL_TOKENS)
model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=3)
model.resize_token_embeddings(len(tokenizer))

model.config.pad_token_id = model.config.eos_token_id

#path = "./results/gpt2climate/hyper/notok/pytorch_model.bin"
path = r"C:\Saeid\Prj100\SA_33_txt_analytics\for_Markus_\fact\pytorch_model.bin"

model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
print(cd.sample(1)['x'].values[0])
test = cd.sample(1)
# generated = torch.tensor(tokenizer.encode(test['x'].values[0]).unsqueeze(0))
# device = torch.device("cpu")
# generated = generated.to(device)

logits = model(torch.tensor(tokenizer.encode(test['x'].values[0])).unsqueeze(0)).logits
print("guess:", int(np.argmax(logits.detach().numpy(), axis=-1)[0]))
print("real:", test['label'].values[0])
print(dic)