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

base_model = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(base_model)

config = AutoConfig.from_pretrained(base_model, pad_token_id=tokenizer.eos_token_id, output_hidden_states=False)
model = TFGPT2LMHeadModel.from_pretrained(base_model, pad_token_id=tokenizer.eos_token_id)
title = "Climate change can affect hydropower operations through changes in the timing and magnitude of precipitation patterns"

input_ids = tokenizer.encode(title, return_tensors='tf')
greedy_output = model.generate(input_ids, max_length=50)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))



sample_outputs = model.generate(input_ids,
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
    print("{}: {}\n\n".format(i+1,  text))