import logging
import torch
from torch.nn.utils import rnn
from transformers import BertTokenizer, DataCollatorForWholeWordMask
import json
from tqdm import tqdm
import numpy as np
import pickle
import time
import datetime
import random
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--pretrained_name", help="the pretrained model that is used")
args = parser.parse_args()

os.makedirs(os.path.dirname(args.pretrained_name.replace('/', '-') + "_data/"), exist_ok=True)

seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
if torch.cuda.is_available():    
    torch.cuda.manual_seed_all(seed_val)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("format_data.log", mode='w'),
        logging.StreamHandler()
    ]
)

logging.info('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained(args.pretrained_name, do_lower_case=True)


def wer(reference, hypothesis):
    r = reference.split()
    h = hypothesis.split()
    d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint8).reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        d[i][0] = i
    for j in range(len(h)+1):
        d[0][j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitute = d[i-1][j-1] + 1
                insert = d[i][j-1] + 1
                delete = d[i-1][j] + 1
                d[i][j] = min(substitute, insert, delete)
    return d[len(r)][len(h)]/float(len(r))


def format_sent(sent):
    tokens = []
    for x in sent.split():
        split = x.split("__")
        if len(split) < 2:
            tokens += split
        elif split[1] == "zh":
            tokens += list(split[0])
        elif split[1] == "en":
            tokens.append(split[0])
    return " ".join(tokens)


def length(sent):
    words = []
    for w in sent.split():
        if w[-4:] == "__zh":
            words += [c for c in w[:-4]]
        elif w[-4:] == "__en":
            words.append(w[:-4])
    return len(words)


def load_data(sents_f, evaluate):
  datasets = []
  if evaluate:
      # val or test set
      for test_set in tqdm(sents_f, desc="Loading eval dataset"):
        sents = []
        sent_inputs = []
        
        formatted_orig = format_sent(test_set["orig"])
        for sent in [test_set["orig"]] + test_set["en_alternatives"]:
            formatted_sent = format_sent(sent)
            if len(sent_inputs) > 0 and formatted_sent == formatted_orig:
                continue
            sents.append(sent)
            encoded_sent = tokenizer(
                formatted_sent,
                truncation = True,
                add_special_tokens = True,
                max_length=64,
                return_tensors = 'pt',
                return_special_tokens_mask = True,
            )
            sent_inputs.append(encoded_sent)
            sent_inputs[-1]["attention_mask"] = encoded_sent["attention_mask"]
            
        padded_input_ids = rnn.pad_sequence([encoded_sent["input_ids"].squeeze() for encoded_sent in sent_inputs], batch_first=True, padding_value=0)
        padded_attention_masks = rnn.pad_sequence([encoded_sent["attention_mask"].squeeze() for encoded_sent in sent_inputs], batch_first=True, padding_value=0)
        datasets.append({"input_ids": padded_input_ids, "attention": padded_attention_masks, "raw": sents})

      return datasets
  else:
      # train set
      data_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
      for test_set in tqdm(sents_f, desc="Loading train dataset"):
          sents = []
          sent_inputs = []
          formatted_orig = format_sent(test_set["orig"])

          sents_wer = [0] + [wer(test_set["orig"], s) for s in test_set["en_alternatives"]]

          max_len_sent = min(max([length(sent) for sent in [test_set["orig"]] + test_set["en_alternatives"]]), 64)
          for sent in [test_set["orig"]] + test_set["en_alternatives"]:
              formatted_sent = format_sent(sent)
              if len(sent_inputs) > 0 and formatted_sent == formatted_orig:
                  continue
              sents.append(sent)
              encoded_sent = tokenizer(
                  formatted_sent,
                  truncation=True,
                  add_special_tokens=True,
                  max_length=64,
                  return_special_tokens_mask = True,
              )
              sent_inputs.append(data_collator([encoded_sent]))
              sent_inputs[-1]["attention_mask"] = torch.Tensor(encoded_sent["attention_mask"])
          padded_input_ids = rnn.pad_sequence([encoded_sent["input_ids"].squeeze() for encoded_sent in sent_inputs], batch_first=True, padding_value=0)
          padded_labels = rnn.pad_sequence([encoded_sent["labels"].squeeze() for encoded_sent in sent_inputs], batch_first=True, padding_value=-100)
          padded_attention_masks = rnn.pad_sequence([encoded_sent["attention_mask"].squeeze() for encoded_sent in sent_inputs], batch_first=True, padding_value=0)
          datasets.append({"input_ids": padded_input_ids, "labels": padded_labels, "attention": padded_attention_masks, "wer": sents_wer, "raw": sents})

      return datasets


def lang(s):
    en = "__en" in s
    zh = "__zh" in s
    if en and zh:
        return "cs"
    if en:
        return "en"
    if zh:
        return "zh"
    return "error!"


sents_f = json.load(open("../data/alternate_sents_seame_filtered_en_only.json", 'r'))

en = []
for x in sents_f:
    lang_type = lang(x["orig"])
    if lang_type == "en":
        en.append(x)
    else:
        print("broken sentence", x["orig"])
        continue

print(len(en), "en alt sets")

random.shuffle(en)

train_data = load_data(en[:-2000], evaluate=False)
val_data = load_data(en[-2000:-1000], evaluate=True)
test_data = load_data(en[-1000:], evaluate=True)

logging.info('\n{:>5,} train samples'.format(len(train_data)))
logging.info('\n{:>5,} val samples'.format(len(val_data)))
logging.info('\n{:>5,} test samples'.format(len(test_data)))

torch.save(train_data, args.pretrained_name.replace('/', '-') + "_data/train_data.pt")
torch.save(val_data, args.pretrained_name.replace('/', '-') + "_data/val_data.pt")
torch.save(test_data, args.pretrained_name.replace('/', '-') + "_data/test_data.pt")
