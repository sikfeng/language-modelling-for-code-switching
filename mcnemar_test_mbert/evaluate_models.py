import logging
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import torch.nn.functional as F
from transformers import BertForMaskedLM, AdamW, BertConfig, get_linear_schedule_with_warmup
import json
from tqdm import tqdm
import numpy as np
import pickle
import time
import datetime
import random
import math
import importlib

models_to_compare = ["w2_model", 
                     "w2_layer_norm_model", 
                     "w2_relu_model"]

model_pkgs = [importlib.import_module(name) for name in models_to_compare]
models = [getattr(model_pkg, "BertModelForCSModelling").from_pretrained("models/" + model_name) for model_pkg, model_name in zip(model_pkgs, models_to_compare)]


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

val_data = torch.load("data/val_data.pt")
test_data = torch.load("data/test_data.pt")


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
    t0 = time.time()
    model.eval()

    pred_score_eval = []
    for step, test_set in enumerate(eval_data):
      b_input_ids = test_set["input_ids"].to(device)
      b_attention_masks = test_set["attention"].to(device)

      with torch.no_grad():
        scores = model(b_input_ids, labels=b_input_ids, attention_mask=b_attention_masks).to('cpu').squeeze().tolist()

      pred_score_eval.append(scores)

    evaluation_time = format_time(time.time() - t0)
    return evaluation_time, pred_score_eval

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


val_model_scores = []
test_model_scores = []

for model, model_name in zip(models, models_to_compare):
    if torch.cuda.is_available():    
        model.cuda()
    model.eval()
    logging.info(f"Evaluating {model_name}...")
    logging.info("")
    val_time, val_scores = evaluate_model(model, val_data)
    val_model_scores.append(val_scores)
    logging.info("  Validation took: {:}".format(val_time))
    test_time, test_scores = evaluate_model(model, test_data)
    test_model_scores.append(test_scores)
    logging.info("  Test took: {:}".format(test_time))
    logging.info("")
    model.to("cpu")

val_truth_matrices = {'cs': [0] * 2**(len(models)), 'en': [0] * 2**(len(models)), 'zh': [0] * 2**(len(models))}

for val_idx, val_set in enumerate(val_data):
    matrix_idx = 0
    for model_idx in range(len(models)):
        scores = val_model_scores[model_idx][val_idx]
        if scores[scores[1:].index(max(scores[1:])) + 1] < scores[0]:
            matrix_idx += 2**model_idx

    set_lang = lang(val_set["raw"][0])
    val_truth_matrices[set_lang][matrix_idx] += 1

logging.info(val_truth_matrices)
logging.info("")

test_truth_matrices = {'cs': [0] * 2**(len(models)), 'en': [0] * 2**(len(models)), 'zh': [0] * 2**(len(models))}

for test_idx, test_set in enumerate(test_data):
    matrix_idx = 0
    for model_idx in range(len(models)):
        scores = test_model_scores[model_idx][test_idx]
        if scores[scores[1:].index(max(scores[1:])) + 1] < scores[0]:
            matrix_idx += 2**model_idx

    set_lang = lang(test_set["raw"][0])
    test_truth_matrices[set_lang][matrix_idx] += 1

logging.info(test_truth_matrices)
logging.info("")

