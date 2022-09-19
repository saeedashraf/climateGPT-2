import numpy as np
import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import sys
import re
import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
from transformers import GPT2Tokenizer, GPT2Model,AutoTokenizer

from transformers import AutoTokenizer, AutoConfig, AutoModelForPreTraining, AdamW, get_linear_schedule_with_warmup, TrainingArguments, BeamScorer, Trainer
import torch
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

SPECIAL_TOKENS  = { "bos_token": "<|BOS|>",
                    "eos_token": "<|EOS|>",
                    "unk_token": "<|UNK|>",
                    "pad_token": "<|PAD|>",
                    "sep_token": "<|SEP|>"}

from torch.utils.data import Dataset, DataLoader
class myDataset(Dataset):

    def __init__(self, data, tokenizer, randomize=True):
        title = data["Title"].tolist()
        text = data["Text"].tolist()
        keywords = data["key"].tolist()

        self.randomize = randomize
        self.tokenizer = tokenizer
        self.title = title
        self.text = text
        self.keywords = keywords

        # ---------------------------------------------#

    @staticmethod
    def join_keywords(keywords, randomize=True):
        N = len(keywords)

        # random sampling and shuffle
        if randomize:
            M = random.choice(range(N + 1))
            keywords = keywords[:M]
            random.shuffle(keywords)

        return ','.join(keywords)

    # ---------------------------------------------#

    def __len__(self):
        return len(self.text)

    # ---------------------------------------------#

    def __getitem__(self, i):
        keywords = self.keywords[i].copy()
        kw = self.join_keywords(keywords, self.randomize)

        input = SPECIAL_TOKENS['bos_token'] + self.title[i] + \
                SPECIAL_TOKENS['sep_token'] + kw + SPECIAL_TOKENS['sep_token'] + \
                self.text[i] + SPECIAL_TOKENS['eos_token']

        encodings_dict = tokenizer(input,
                                   truncation=True,
                                   max_length=750,
                                   padding="max_length")

        input_ids = encodings_dict['input_ids']
        attention_mask = encodings_dict['attention_mask']

        return {'label': torch.tensor(input_ids),
                'input_ids': torch.tensor(input_ids),
                'attention_mask': torch.tensor(attention_mask)}
base_model = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(base_model) #GPT2Tokenizer

tokenizer.add_special_tokens(SPECIAL_TOKENS)
print("Special tokens added")

config = AutoConfig.from_pretrained(base_model,
                                            bos_token_id=tokenizer.bos_token_id,
                                            eos_token_id=tokenizer.eos_token_id,
                                            sep_token_id=tokenizer.sep_token_id,
                                            pad_token_id=tokenizer.pad_token_id,
                                            output_hidden_states=False)

model = AutoModelForPreTraining.from_pretrained(base_model, config=config)


model.resize_token_embeddings(len(tokenizer))

#model.load_state_dict(torch.load("./tekrar/content/pytorch_model.bin"))
#model.load_state_dict(torch.load(r"C:\Saeid\Prj100\SA_33_txt_analytics\for_Markus_\claim\ckeygpt\pytorch_model.bin"))
model.load_state_dict(torch.load(r"C:\Saeid\Prj100\SA_33_txt_analytics\for_Markus_\claim\ckeygpt\pytorch_model.bin", map_location=torch.device('cpu')))




title = "Climate change can affect hydropower operations through changes in the timing and magnitude of precipitation patterns"
keywording = ['climate change', 'hydropower', 'energy', 'mitigate']
kw = myDataset.join_keywords(keywording, randomize=False)

prompt = SPECIAL_TOKENS['bos_token'] + title + \
         SPECIAL_TOKENS['sep_token'] + kw + SPECIAL_TOKENS['sep_token']

generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
device = torch.device("cpu")
generated = generated.to(device)

model.eval();

sample_outputs = model.generate(generated,
                                do_sample=True,
                                min_length=50,
                                max_length=750,
                                top_k=30,
                                top_p=0.7,
                                temperature=0.9,
                                repetition_penalty=2.0,
                                num_return_sequences=10
                                )

for i, sample_output in enumerate(sample_outputs):
    text = tokenizer.decode(sample_output, skip_special_tokens=True)
    a = len(title) + len(','.join(keywording))
    print("{}: {}\n\n".format(i+1,  text[a:]))