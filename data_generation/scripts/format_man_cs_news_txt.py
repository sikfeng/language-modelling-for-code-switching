import jieba
from collections import defaultdict
import glob
import json
import re

jieba.enable_paddle()

word_count = defaultdict(int)

zh_regex = re.compile(r'([\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+)')
en_regex = re.compile(r'([a-zA-Z])+')


def tokenize_sent(sent):
    # to seperate digits and english words from chinese
    tokens = sent.split()
    seperated_tokens = []
    for token in tokens:
        seperated_tokens += re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+|[0-9]+|[^0-9\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\s]+", token, re.UNICODE)
    return seperated_tokens


with open("../extracted/raw_ch_radio_sents.txt", "r") as ch_radio_sents_f, open("../extracted/ch_radio_sents.txt", "w") as ch_radio_sents_labeled_f:
    for sent in ch_radio_sents_f:
        tokens = tokenize_sent(sent)
        valid = True
        for token in tokens:
            word_count[token] += 1
        for idx, token in enumerate(tokens):
            if zh_regex.match(token):
                tokens[idx] = token + "__zh"
            elif en_regex.match(token):
                valid = False
                break
            else:
                print(f"broken token {token}")
                valid = False
                break
                # continue
        if valid:
            ch_radio_sents_labeled_f.write(" ".join(tokens) + '\n')
            
with open("../extracted/raw_p_daily_sents.txt", "r") as p_daily_sents_f, open("../extracted/p_daily_sents.txt", "w") as p_daily_sents_labeled_f:
    for sent in p_daily_sents_f:
        tokens = tokenize_sent(sent)
        valid = True
        for token in tokens:
            word_count[token] += 1
        for idx, token in enumerate(tokens):
            if zh_regex.match(token):
                tokens[idx] = token + "__zh"
            elif en_regex.match(token):
                valid = False
                break
            else:
                print(f"broken token {token}")
                valid = False
                break
                # continue
        if valid:
            p_daily_sents_labeled_f.write(" ".join(tokens) + '\n')

with open("../extracted/raw_xinhua_sents.txt", "r") as xinhua_sents_f, open("../extracted/xinhua_sents.txt", "w") as xinhua_sents_labeled_f:
    for sent in xinhua_sents_f:
        tokens = tokenize_sent(sent)
        valid = True
        for token in tokens:
            word_count[token] += 1
        for idx, token in enumerate(tokens):
            if zh_regex.match(token):
                tokens[idx] = token + "__zh"
            elif en_regex.match(token):
                valid = False
                break
            else:
                print(f"broken token {token}")
                valid = False
                break
                # continue
        if valid:
            xinhua_sents_labeled_f.write(" ".join(tokens) + '\n')

with open("../extracted/zh_probs.txt", "w") as probs_f:
    for key, value in word_count.items():
        if value < 40:
            value = 0
        if not zh_regex.match(key):
            continue
        if en_regex.match(key):
            continue
        probs_f.write(f'{key}\t{value}\n')
