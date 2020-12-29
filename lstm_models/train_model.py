import dynet as dy
import argparse
import time
import datetime
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

parser.add_argument("--train", default="data/alternate_sents_seame_train_filtered.json", help="location of training file")
parser.add_argument("--dev", default="data/alternate_sents_seame_val_filtered.json", help="new dev file")
parser.add_argument("--test", default="data/alternate_sents_seame_test_filtered.json", help="new test file")
parser.add_argument("--train_finetune", default="data/alternate_sents_mono_all_filtered.json", help="location of training file for finetune")
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
parser.add_argument("--lamb", default=1, type=float, help="value of lambda")
parser.add_argument("--margin", default="wer", help="value of margin for loss")
parser.add_argument("--evaluate_model", help="model to load and evaluate, no training")


t0 = time.time()

def wer(reference, hypothesis):
    # initialisation
    reference = reference.split()
    hypothesis = hypothesis.split()
    d = np.zeros((len(reference) + 1)*(len(hypothesis) + 1), dtype=np.uint16)
    d = d.reshape((len(reference) + 1, len(hypothesis) + 1))

    # computation
    for i in range(len(reference) + 1):
        for j in range(len(hypothesis) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i
            elif reference[i - 1] == hypothesis[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion    = d[i][j-1] + 1
                deletion     = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    res = d[len(reference)][len(hypothesis)]/float(len(reference))
    assert(res > 0)
    return res


class Set:
    orig = None
    orig_type = None
    cs = None
    en = None
    zh = None
    distances = None


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
        return 0
    if en:
        # english
        return 1
    if zh:
        # chinese
        return 2

    # something wrong
    return None


def read_corpus(data_file, train=False):
    all_sets = []
    with open(data_file) as f:
        data = json.load(f)
    for s in tqdm(data):
        new_s = Set()
        new_s.cs = []
        new_s.en = []
        new_s.zh = []
        new_s.distances = []
        new_s.orig_type = detect_lang(s["orig"])

        new_s.orig = [w2i[w.split("__")[0].lower().lower()] for w in [START_TOKEN] + s["orig"].split() + [END_TOKEN]]
        for idx, lang_type in enumerate(["cs", "en", "zh"]):
            for v in s[lang_type + "_alternatives"]:
                assert(detect_lang(v) == idx)
                sent = []
                for w in [START_TOKEN] + v.split() + [END_TOKEN]:
                    sent.append(w2i[w.split("__")[0].lower()])
                getattr(new_s, lang_type).append(sent)
                if train:
                    if args.margin == "wer":
                        new_s.distances.append(wer(s["orig"], v))
                    else:
                        new_s.distances.append(float(args.margin))
        all_sets.append(new_s)
    assert(len(all_sets) == len(data))
    if train:
        print("length of train:", len(all_sets))
    return all_sets, len(all_sets)


def evaluate(test):
    ranks = []
    for item in test:
        batch = [item.orig] + item.cs + item.en + item.zh
        scores = [s.npvalue() for s in get_batch_scores(batch, True)]
        indices = list(range(len(scores)))
        indices.sort(key=lambda x: scores[x], reverse=True)
        ranks.append(indices.index(0)+1)

    cs = []
    mono = []
    for rank, item in zip(ranks, test):
        if item.orig_type:
            mono.append(rank)
        else:
            cs.append(rank)

    tot_acc = sum([1 for r in ranks if r == 1])*100/float(len(ranks))
    cs_acc = sum([1 for r in cs if r == 1])*100/float(len(cs))
    mono_acc = sum([1 for r in mono if r == 1])*100/float(len(mono))

    return tot_acc, cs_acc, mono_acc


def get_batch_scores(batch, evaluate = False):
    return model.get_batch_scores(batch, evaluate)


def calc_loss(batch_distances, batch_scores, lamb):
    batch_losses = [lamb * dy.scalarInput(d) - (batch_scores[0] - s) for s, d in zip(batch_scores[1:], batch_distances[1:])]
    losses_pos = [l if l.npvalue() >= 0 else dy.scalarInput(0) for l in batch_losses]
    
    if len(losses_pos) == 0:
        return 0

    return dy.esum(losses_pos)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def logstr(f, s):
    print(f'[{format_time(time.time() - t0)}]  {s}')
    f.write(f'[{format_time(time.time() - t0)}]  {s}')


def check_performance(epoch, train_acc, best_acc, best_epoch, batch_n = None):

    if args.check_freq:
        logstr(f_log, "batch number {}. note: train acc is only on last {} batches\n\n".format(str(batch_n), args.check_freq))

    dev_acc, dev_cs_acc, dev_mono_acc = evaluate(dev_set)
    logstr(f_log, "dev_cs_acc {}\n".format(dev_cs_acc))
    logstr(f_log, "dev_mono_acc {}\n".format(dev_mono_acc))

    test_acc, test_cs_acc, test_mono_acc = evaluate(test_set)
    logstr(f_log, "test_cs_acc {}\n".format(test_cs_acc))
    logstr(f_log, "test_mono_acc {}\n".format(test_mono_acc))

    if dev_acc > best_acc:
        best_acc = dev_acc
        best_epoch = epoch
        if args.finetune_p1:
            model.save(args.logfile, vocab=w2i)
        else:
            model.save(args.logfile)

        with open("models/" + args.logfile + "_model_info", 'w') as f:
            f.write("{}\n{}\n{}\n".format(model.get_learning_rate(), epoch, best_acc))

    else:
        model.update_lr(args.lr_decay_factor)

    logstr(f_log, "train_acc " + str(train_acc)+"\n")
    logstr(f_log, "dev_acc " + str(dev_acc)+"\n")
    logstr(f_log, "test_acc " + str(test_acc)+"\n")
    logstr(f_log, "best_so_far " + str(best_acc) + "\n")
    logstr(f_log, "learning_rate " + str(model.get_learning_rate()) + "\n\n")

    return dev_acc, best_acc, best_epoch


def train(train_set, dev_set, test_set, best_acc=0, best_epoch=-1):
    train_start_time = time.time()

    try:
        for epoch in range(best_epoch + 1, args.epochs):
            epoch_start_time = time.time()
            random.shuffle(train_set)
            train_acc = 0
            train_loss = 0
            for idx, t_set in enumerate(train_set):
                batch = [t_set.orig] + t_set.cs + t_set.en + t_set.zh

                if args.check_freq and (idx > 0) and (idx % int(args.check_freq) == 0):
                    train_acc = (train_acc/float(args.check_freq))*100
                    dev_acc, best_acc, best_epoch = check_performance(epoch, train_acc, best_acc, best_epoch, idx)
                    train_acc = 0

                batch_scores = get_batch_scores(batch)

                if np.argmax([s.npvalue() for s in batch_scores]) == 0:
                    train_acc += 1

                batch_loss = calc_loss(t_set.distances, batch_scores, args.lamb)
                ll = batch_loss.npvalue()
                train_loss += ll
                if (idx > 0) and (idx % 500 == 0):
                    logstr(f_log, f"avg loss is {train_loss/float(500)}\n")
                    train_loss = 0
                batch_loss.backward()
                model.trainer_update()

            if not args.check_freq:
                train_acc = (train_acc/float(train_nbatches))*100
                dev_acc, best_acc, best_epoch = check_performance(epoch, train_acc, best_acc, best_epoch)

            logstr(f_log, "time for epoch number " + str(epoch) + " is: " + str(time.time() - epoch_start_time) + "\n\n")


    except KeyboardInterrupt:
        logstr(f_log, "Exiting from training early\n\n")

    logstr(f_log, "time for training is: " + str(time.time() - train_start_time) + "\n\n")

    # results
    logstr(f_log, "best acc is: " + str(best_acc) + "\n\n")
    logstr(f_log, "best epoch: " + str(best_epoch) + "\n\n")


if __name__ == '__main__':

    args = parser.parse_args()
    random.seed(10)
    np.random.seed(10)

    f_log = open("models/" + args.logfile + ".txt", "w")

    logstr(f_log, str(args) + "\n\n")

    START_TOKEN = "<s>"
    END_TOKEN = "</s>"
    w2i = defaultdict(lambda: len(w2i))
    UNK = w2i["<unk>"]

    if args.finetune_p1 or args.finetune_p2:
        # need vocabulary in w2i for both p1 and p2
        train_set_finetune, train_nbatches_finetune = read_corpus(args.train_finetune, True)
        logstr(f_log, "Loaded monolingual train set")

    train_set, train_nbatches = read_corpus(args.train, True)
    logstr(f_log, "Loaded code switched train set")
    dev_set, dev_nbatches = read_corpus(args.dev)
    logstr(f_log, "Loaded validation set")
    test_set, test_nbatches = read_corpus(args.test)
    logstr(f_log, "Loaded test set")
    
    # create model
    word_num = len(w2i)
    model = LM(args.num_layers, args.input_dim, args.hidden_dim, word_num, args.init_scale_rnn, args.init_scale_params,
               args.x_dropout, args.h_dropout, args.w_dropout_rate, args.learning_rate, args.clip_thr)

    if args.evaluate_model:
        # load pre-trained model
        model.load("models/" + args.evaluate_model + "_model")
        print("loaded model", args.evaluate_model)
        dev_acc, dev_cs_acc, dev_mono_acc = evaluate(dev_set)
        test_acc, test_cs_acc, test_mono_acc = evaluate(test_set)
        print("dev accs", dev_acc, dev_cs_acc, dev_mono_acc)
        print("test accs", test_acc, test_cs_acc, test_mono_acc)
        exit()

    if args.finetune_p1:
        # use the monolingual data instead of the cs data
        train_set = train_set_finetune
    
    if args.finetune_p2:
        # load pre-trained model
        model.load("models/" + args.finetune_p2 + "_model")
        print("loaded model", args.finetune_p2)

    best_epoch = -1
    best_acc = 0

    train(train_set, dev_set, test_set, best_acc, best_epoch)
