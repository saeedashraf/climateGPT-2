import numpy as np
import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import sys
import re
import warnings
print("*****START*****")
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


def remove_text_inside_brackets(text, brackets="()[]"):
    count = [0] * (len(brackets) // 2) # count open/close brackets
    saved_chars = []
    for character in text:
        for i, b in enumerate(brackets):
            if character == b: # found bracket
                kind, is_close = divmod(i, 2)
                count[kind] += (-1)**is_close # `+1`: open, `-1`: close
                if count[kind] < 0: # unbalanced bracket
                    count[kind] = 0  # keep it
                else:  # found bracket to remove
                    break
        else: # character is not a [balanced] bracket
            if not any(count): # outside brackets
                saved_chars.append(character)
    return ''.join(saved_chars)

def steaming(text):
    text_string = BeautifulSoup(text, "lxml").text
    LOL = unicodedata.normalize("NFKD", text_string)
    LOL = re.sub(r"\\n|\n", r" ", LOL)
    LOL = re.sub(
        r"A Correction to this paper has been published:(.*)|\[This corrects the article DOI:(.*)|\[Figure: see text\].",
        r"", LOL)
    LOL = re.sub(r"Abstract. |Abstract |Introduction: ", r"", LOL)
    LOL = remove_text_inside_brackets(LOL, brackets="()[]")
    LOL = re.sub(
        r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''',
        "", LOL)

    LOL = re.sub(r"\d+", "", LOL)
    return LOL

#path = r"/home/saeidashraf/Saeidproj/nlp_uzh/authors"
#print(path)
#authors = [f for f in listdir(path) if isfile(join(path, f))]
#print(authors[0:10])
set_epoch = 100
year = 2000
# year=2014
# year = 2020
# abst = []
# titl = []
# for i, auth in enumerate(authors[:]):
#     temp = pd.read_csv(path+"/"+auth)
#     if temp.shape[0]>0:
#         year_cut = temp[temp["Publication Year"]>=year]
#         if year_cut.shape[0]>0:
#             year_cut.dropna(subset = ["Abstract"], inplace=True)
#             for j in range(year_cut.shape[0]):
#                 abstr = steaming(year_cut.iloc[j]['Abstract'])
#                 ti = steaming(year_cut.iloc[j]['Title'])
#                 if(len(abstr)>400 and len(ti)>5):
#                   abst.append(abstr)
#                   titl.append(ti)
#             #for i in range(year_cut.shape[0]):
#            #     abst.append(steaming(year_cut.iloc[i]['Abstract']))
#             #    titl.append(steaming(year_cut.iloc[i]['Title']))
#
# data = pd.DataFrame({"Title":titl, "Text":abst})
# print(data.head())


base_model = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(base_model)
print(torch.cuda.is_available())
SPECIAL_TOKENS  = { "bos_token": "<|BOS|>",
                    "eos_token": "<|EOS|>",
                    "unk_token": "<|UNK|>",
                    "pad_token": "<|PAD|>",
                    "sep_token": "<|SEP|>"}

tokenizer.add_special_tokens(SPECIAL_TOKENS)
config = AutoConfig.from_pretrained(base_model,
                                        bos_token_id=tokenizer.bos_token_id,
                                        eos_token_id=tokenizer.eos_token_id,
                                        sep_token_id=tokenizer.sep_token_id,
                                        pad_token_id=tokenizer.pad_token_id,
                                        output_hidden_states=False)
gpt_base = AutoModelForPreTraining.from_pretrained(base_model, config=config)
gpt_base.resize_token_embeddings(len(tokenizer))
#gpt_base.load_state_dict(torch.load(r"/home/saeidashraf/Saeidproj/nlp_uzh/testing/wa/ga/mama/pytorch_model.bin"))
#gpt_base.load_state_dict(torch.load(r"C:\Saeid\Prj100\SA_33_txt_analytics\for_Markus_\claim\cgpt\pytorch_model.bin"))
gpt_base.load_state_dict(torch.load(r"C:\Saeid\Prj100\SA_33_txt_analytics\for_Markus_\claim\cgpt\pytorch_model.bin", map_location=torch.device('cpu')))

# print(gpt_base)
# print(torch.load(r"/home/saeidashraf/Saeidproj/nlp_uzh/testing/wa/pytorch_model.bin"))
# print(gpt_base.parameters)
print(len(gpt_base.state_dict()))
print(type(gpt_base.state_dict()))

title = "Climate change can affect hydropower operations through changes in the timing and magnitude of precipitation patterns"

prompt = SPECIAL_TOKENS['bos_token'] + title + SPECIAL_TOKENS['sep_token']+SPECIAL_TOKENS['sep_token']

generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
device = torch.device("cpu")
generated = generated.to(device)

gpt_base.eval();

sample_outputs = gpt_base.generate(generated,
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
    a = len(title)
    print("{}: {}\n\n".format(i+1,  text[a:]))
