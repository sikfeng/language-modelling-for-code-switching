import logging
import json
import pickle
import time
import datetime
import random

seed_val = 42

random.seed(seed_val)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("split_data.log", mode='w'),
        logging.StreamHandler()
    ]
)

def format_sent(sent):
    tokens = []
    for x in sent.split():
        split = x.split("__")
        if split[1] == "zh":
            tokens += list(split[0])
        elif split[1] == "en":
            tokens.append(split[0])
    return " ".join(tokens)

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

sents_f = json.load(open("data/alternate_sents_seame_filtered.json", 'r'))

cs = []
en = []
zh = []
for x in sents_f:
    lang_type = lang(x["orig"])
    if lang_type == "cs":
        cs.append(x)
    elif lang_type == "en":
        en.append(x)
    elif lang_type == "zh":
        zh.append(x)
    else:
        logging.error("broken sentence", x["orig"])
        continue

logging.info(f"{len(cs)} cs alt sets")
logging.info(f"{len(en)} en alt sets")
logging.info(f"{len(zh)} zh alt sets")

random.shuffle(cs)
random.shuffle(en)
random.shuffle(zh)

train_data = en[:-2000] + zh[:-2000] + cs[:-2000]
val_data = en[-2000: -1000] + zh[-2000: -1000] + cs[-2000:-1000]
test_data = en[-1000:] + zh[-1000:] + cs[-1000:]
json.dump(train_data, open("data/alternate_sents_seame_train_filtered.json", 'w'), indent=2, ensure_ascii=False)
json.dump(val_data, open("data/alternate_sents_seame_val_filtered.json", 'w'), indent=2, ensure_ascii=False)
json.dump(test_data, open("data/alternate_sents_seame_test_filtered.json", 'w'), indent=2, ensure_ascii=False)
logging.info('\n{:>5,} train samples'.format(len(train_data)))
logging.info('\n{:>5,} val samples'.format(len(val_data)))
logging.info('\n{:>5,} test samples'.format(len(test_data)))
