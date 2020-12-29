import logging
import torch
import json
from tqdm import tqdm
import numpy as np
import pickle
import time
import datetime
import random
import argparse
import math

from model import BertModelForCSModelling

parser = argparse.ArgumentParser()
parser.add_argument("--pretrained_name", help="the pretrained model that is used")
parser.add_argument("--model_name", help="name of model to evaluate")
args = parser.parse_args()

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
        logging.FileHandler("evaluate_model.log", mode='w'),
        logging.StreamHandler()
    ]
)

if torch.cuda.is_available():    
    device = torch.device("cuda")
    logging.info(f'There are {torch.cuda.device_count()} GPU(s) available.')
    logging.info('We will use the GPU: ' + str(torch.cuda.get_device_name(0)))
else:
    logging.info('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

val_data = torch.load(args.pretrained_name.replace('/', '-') + "_data/val_data.pt")
test_data = torch.load(args.pretrained_name.replace('/', '-') + "_data/test_data.pt")


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def lang(sent):
    zh = "__zh" in sent
    en = "__en" in sent
    if zh and en:
        return "cs"
    elif zh:
        return "zh"
    elif en:
        return "en"
    else:
        logging.error("Invalid language!")
        return "ERROR"


def evaluate_model(model, eval_data):
    logging.info("")
    t0 = time.time()
    model.eval()

    stats = {"cs_total": 0, "cs_correct": 0, "zh_total": 0, "zh_correct": 0, "en_total": 0, "en_correct": 0}
    total_acc = 0
    pred_score_eval = []
    for step, test_set in enumerate(eval_data):
      scores = []
      b_input_ids = test_set["input_ids"].to(device)
      b_attention_masks = test_set["attention"].to(device)

      with torch.no_grad():
        scores = model(b_input_ids, labels=b_input_ids, attention_mask=b_attention_masks).to('cpu').numpy()

      pred_score_eval.append(scores)

      test_set_lang = lang(test_set["raw"][0])

      if scores[np.argmax(np.array(scores[1:])) + 1] < scores[0]:
        stats[f"{test_set_lang}_correct"] += 1
      stats[f"{test_set_lang}_total"] += 1

    avg_eval_accuracy = 100 * (stats["cs_correct"] + stats["zh_correct"] + stats["en_correct"]) / (stats["cs_total"] + stats["zh_total"] + stats["en_total"])

    evaluation_time = format_time(time.time() - t0)
    return avg_eval_accuracy, evaluation_time, stats, pred_score_eval

total_t0 = time.time()

if args.model_name:
    model = BertModelForCSModelling.from_pretrained("models/" + args.model_name)
    if torch.cuda.is_available():    
        model.cuda()
    model.eval()
else:
    logging.error("model name not provided in arguments! use --model_name <name of model>")

cs_val = 0
zh_val = 0
en_val = 0
for test_set in val_data:
    set_lang = lang(test_set["raw"][0])
    if set_lang == "cs":
        cs_val += 1
    elif set_lang == "zh":
        zh_val += 1
    elif set_lang == "en":
        en_val += 1

logging.info(f"{cs_val} CS sets, {zh_val} ZH sets, {en_val} EN sets")

cs_test = 0
zh_test = 0
en_test = 0
for test_set in test_data:
    set_lang = lang(test_set["raw"][0])
    if set_lang == "cs":
        cs_test += 1
    elif set_lang == "zh":
        zh_test += 1
    elif set_lang == "en":
        en_test += 1

logging.info(f"{cs_test} CS sets, {zh_test} ZH sets, {en_test} EN sets")

logging.info("")
logging.info("Running Validation...")

avg_val_accuracy, validation_time, validation_stats, val_scores = evaluate_model(model, val_data)
logging.info("  ZH accuracy: {0:.2f}".format(100 * validation_stats["zh_correct"] / validation_stats["zh_total"]))
logging.info("  Total accuracy: {0:.2f}".format(avg_val_accuracy))
logging.info("  Validation took: {:}".format(validation_time))

with open("evaluate_model_val_scores.txt", 'w') as val_scores_f:
    for val_set, val_score in zip(val_data, val_scores):
        best_score = val_score[np.argmax(np.array(val_score))]
        for sent, score in zip(val_set["raw"], val_score):
            if score == best_score:
                val_scores_f.write(f"{sent}: {score}\t\tbest score\n")
            else:
                val_scores_f.write(f"{sent}: {score}\n")
        val_scores_f.write('-'*20 + "\n\n\n")
logging.info("  Written scores to evaluate_model_val_scores.txt")


logging.info("")
logging.info("Running Test...")

avg_test_accuracy, testing_time, testing_stats, test_scores = evaluate_model(model, test_data)
logging.info("  ZH accuracy: {0:.2f}".format(100 * testing_stats["zh_correct"] / testing_stats["zh_total"]))
logging.info("  Total accuracy: {0:.2f}".format(avg_test_accuracy))
logging.info("  Test took: {:}".format(testing_time))

with open("evaluate_model_test_scores.txt", 'w') as test_scores_f:
    for test_set, test_score in zip(test_data, test_scores):
        best_score = test_score[np.argmax(np.array(test_score))]
        for sent, score in zip(test_set["raw"], test_score):
            if score == best_score:
                test_scores_f.write(f"{sent}: {score}\t\tbest score\n")
            else:
                test_scores_f.write(f"{sent}: {score}\n")
        test_scores_f.write('-'*20 + "\n\n")
logging.info("  Written scores to evaluate_model_test_scores.txt")
