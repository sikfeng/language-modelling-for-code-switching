import dynet as dy
import argparse
import time
import random
import numpy as np
from collections import defaultdict
import math
import pickle
import json
from tqdm import tqdm
from model import LM

parser = argparse.ArgumentParser()
parser.add_argument("--dynet-mem", help="allocate memory for dynet")
parser.add_argument("--dynet-gpu", help="use GPU")
parser.add_argument("--dynet-gpus", help="use GPU")
parser.add_argument("--dynet-seed", help="set random seed for dynet")
parser.add_argument("--dynet-gpu-ids", default=3, help="choose which GPU to use")
parser.add_argument("--dynet-autobatch", default=0, help="choose which GPU to use")
parser.add_argument("--dynet-devices", help="set random seed for dynet")
parser.add_argument("--dynet-weight-decay", help="choose weight decay")

parser.add_argument("--train", default="alternate_sents_seame_train_filtered.json", help="location of training file")
parser.add_argument("--dev", default="alternate_sents_seame_val_filtered.json", help="new dev file")
parser.add_argument("--test", default="alternate_sents_seame_test_filtered.json", help="new test file")
parser.add_argument("--train_finetune", default="alternate_sents_mono_all_filtered.json", help="location of training file for finetune")
parser.add_argument("--epochs", default=25, type=int, help="number of epochs")
parser.add_argument("--batch_size", default=20, type=int, help="size of batches")
parser.add_argument("--trainer", default="sgd", help="choose trainer for the optimization")
parser.add_argument("--num_layers", default=2, type=int, help="number of layers in RNN")
parser.add_argument("--input_dim", default=300, type=int, help="dimension of the input to the RNN")
parser.add_argument("--hidden_dim", default=650, type=int, help="dimension of the hidden layer of the RNN")
parser.add_argument("--x_dropout", default=0.35, type=float, help="value for rnn dropout")
parser.add_argument("--h_dropout", default=0.35, type=float, help="value for rnn dropout")
parser.add_argument("--w_dropout_rate", default=0.2, type=float, help="value for rnn dropout")
parser.add_argument("--logfile", default="log.txt", help="location of log file for debugging")
parser.add_argument("--learning_rate", default=1, type = float, help="set initial learning rate")
parser.add_argument("--lr_decay_factor", default=2.5, type = float, help="set clipping threshold")
parser.add_argument("--clip_thr", default=1, type=float, help="set clipping threshold")
parser.add_argument("--init_scale_rnn", default=0.05, type=float, help="scale to init rnn")
parser.add_argument("--init_scale_params", default=None, type=float, help="scale to init params")
parser.add_argument("--check_freq", default=None, help="frequency of checking perp on dev and updating lr")
parser.add_argument("--finetune_p1", action='store_true',help="train a model from monolingual data")
parser.add_argument("--finetune_p2", help="name of model learned from momnolingual data")
parser.add_argument("--resume_train", default=False, action="store_true", help="resume training from model")
parser.add_argument("--lamb", default=1, type=float, help="value of lambda")
parser.add_argument("--margin", default="wer", help="value of margin for loss")
parser.add_argument("--evaluate_model", help="model to load and evaluate, no training")


class Set:
    orig = None
    orig_type = None
    cs = None
    en = None
    zh = None
    distances = None
    raw = None


def remove_tag(sentence):
    no_tag = ""
    for token in sentence.split():
        no_tag += token.split("__")[0] + " "

    return no_tag


def detect_lang(sentence):
    tokens = sentence.split()
    en = False
    zh = False
    for token in tokens:
        if token.endswith("__en"):
            en = True
        elif token.endswith("__zh"):
            zh = True

    if en and zh:
        # code switched
        return "cs"
    if en:
        # english
        return "en"
    if zh:
        # chinese
        return "zh"

    # something wrong
    return None


def read_corpus(data_file, train = False):
    all_sets = []
    with open(data_file) as f:
        data = json.load(f)
    for s in tqdm(data):
        new_s = Set()
        new_s.cs = []
        new_s.en = []
        new_s.zh = []
        new_s.distances = []
        new_s.raw = []
        new_s.orig_type = detect_lang(s["orig"])

        new_s.orig = [w2i[w.split("__")[0].lower().lower()] for w in [START_TOKEN] + s["orig"].split() + [END_TOKEN]]
        new_s.raw.append(s["orig"])
        for idx, lang_type in enumerate(["cs", "en", "zh"]):
            for v in s[lang_type + "_alternatives"]:
                #assert(detect_lang(v) == idx)
                sent = []
                for w in [START_TOKEN] + v.split() + [END_TOKEN]:
                    sent.append(w2i[w.split("__")[0].lower()])
                getattr(new_s, lang_type).append(sent)
                new_s.raw.append(v)
        all_sets.append(new_s)
    assert(len(all_sets) == len(data))
    if train:
        print("length of train:", len(all_sets))
    return all_sets, len(all_sets)


def evaluate(test):
    all_scores = []
    all_norms = []
    ranks = []
    for item in tqdm(test):
        batch = [item.orig] + item.cs + item.en + item.zh
        output = [s.npvalue().flatten().tolist() for s in get_batch_scores(batch, True)]
        indices = list(range(len(scores)))
        indices.sort(key=lambda x: scores[x], reverse=True)
        ranks.append(indices.index(0)+1)
        all_norms.append(norms)
        all_scores.append(scores)

    cs = []
    en = []
    zh = []
    for rank, item in zip(ranks, test):
        if item.orig_type == "cs":
            cs.append(rank)
        elif item.orig_type == "en":
            en.append(rank)
        elif item.orig_type == "zh":
            zh.append(rank)

    tot_acc = sum([1 for r in ranks if r == 1])*100/float(len(ranks))
    cs_acc = sum([1 for r in cs if r == 1])*100/float(len(cs))
    en_acc = sum([1 for r in en if r == 1])*100/float(len(en))
    zh_acc = sum([1 for r in zh if r == 1])*100/float(len(zh))

    return tot_acc, cs_acc, en_acc, zh_acc, all_norms, all_scores


def logstr(f, s):
    print(f'[{str(time.time())}]  {s}')
    f.write(f'[{str(time.time())}]  {s}')


if __name__ == '__main__':

    args = parser.parse_args()
    random.seed(10)
    np.random.seed(10)

    print(str(args) + "\n\n")

    START_TOKEN = "<s>"
    END_TOKEN = "</s>"
    w2i = defaultdict(lambda: len(w2i))
    UNK = w2i["<unk>"]

    if args.finetune_p1 or args.finetune_p2:
        # need vocabulary in w2i for both p1 and p2
        train_set_finetune, train_nbatches_finetune = read_corpus(args.train_finetune, True)
        print("Loaded monolingual train set")

    train_set, train_nbatches = read_corpus(args.train, True)
    print("Loaded code switched train set")
    dev_set, dev_nbatches = read_corpus("data/" + args.dev)
    print("Loaded validation set")
    test_set, test_nbatches = read_corpus("data/" + args.test)
    print("Loaded test set")
    
    # create model
    # NOTE: need to read corpus before creating model since read_corpus() will fill up w2i
    word_num = len(w2i)
    model = LM(args.num_layers, args.input_dim, args.hidden_dim, word_num, args.init_scale_rnn, args.init_scale_params,
               args.x_dropout, args.h_dropout, args.w_dropout_rate, args.learning_rate, args.clip_thr)

    model.load("models/" + args.evaluate_model + "_model")
    print("loaded model", args.evaluate_model)
    dev_acc, dev_cs_acc, dev_en_acc, dev_zh_acc, dev_scores = evaluate(dev_set)
    test_acc, test_cs_acc, test_en_acc, test_zh_acc, test_scores = evaluate(test_set)
    print("dev accs", dev_acc, dev_cs_acc, dev_en_acc, dev_zh_acc)
    print("test accs", test_acc, test_cs_acc, test_en_acc, test_zh_acc)

    json.dump(dev_scores, open("dev_scores.json", 'w'), indent=2, ensure_ascii=False)
    logging.info("Saved dev scores to dev_scores.txt")
    json.dump(test_scores, open("test_scores.json", 'w'), indent=2, ensure_ascii=False)
    logging.info("Saved test scores to test_scores.txt")
