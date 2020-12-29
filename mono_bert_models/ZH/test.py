import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import torch.nn.functional as F
from torch.nn.utils import rnn
from transformers import BertTokenizer, BertForMaskedLM, AdamW, BertConfig, get_linear_schedule_with_warmup, pipeline, DataCollatorForWholeWordMask, DataCollatorForLanguageModeling
import json
from tqdm import tqdm
import numpy as np
import pickle
import time
import datetime
import random
import argparse

# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
if torch.cuda.is_available():    
    torch.cuda.manual_seed_all(seed_val)


print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('hfl/chinese-bert-wwm', do_lower_case=True)
print('Loaded BERT tokenizer')
