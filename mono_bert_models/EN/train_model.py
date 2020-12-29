import logging
import torch
from torch.nn import ReLU
from transformers import BertTokenizer, AdamW, BertConfig, get_linear_schedule_with_warmup
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
parser.add_argument("--model_output", default="model", help="name of model to write to")
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
        logging.FileHandler("train_model.log", mode='w'),
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

train_data = torch.load(args.pretrained_name.replace('/', '-') + "_data/train_data.pt")
val_data = torch.load(args.pretrained_name.replace('/', '-') + "_data/val_data.pt")
test_data = torch.load(args.pretrained_name.replace('/', '-') + "_data/test_data.pt")


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


model = BertModelForCSModelling.from_pretrained(args.pretrained_name)
relu = ReLU()
if torch.cuda.is_available():    
    model.cuda()

optimizer = AdamW(model.parameters(),
  lr = 2e-5,
  eps = 1e-8
)

epochs = 5
total_steps = len(train_data) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)


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
    return avg_eval_accuracy, evaluation_time, stats


total_t0 = time.time()

best_val_acc = 0.0

for epoch_i in range(0, epochs):
    logging.info("")
    logging.info('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    logging.info('Training...')

    t0 = time.time()
    model.train()

    random.shuffle(train_data)
    for step, batch in enumerate(train_data):

      if step % 10000 == 0 and (step != 0 or epoch_i == 0):

        logging.info("")
        logging.info("Running Validation...")

        avg_val_accuracy, validation_time, validation_stats = evaluate_model(model, val_data)
        logging.info("  EN accuracy: {0:.2f}".format(100 * validation_stats["en_correct"] / validation_stats["en_total"]))
        logging.info("  Total accuracy: {0:.2f}".format(avg_val_accuracy))
        logging.info("  Validation took: {:}".format(validation_time))

        logging.info("")
        logging.info("Running Test...")

        avg_test_accuracy, testing_time, testing_stats = evaluate_model(model, test_data)
        logging.info("  EN accuracy: {0:.2f}".format(100 * testing_stats["en_correct"] / testing_stats["en_total"]))
        logging.info("  Total accuracy: {0:.2f}".format(avg_test_accuracy))
        logging.info("  Test took: {:}".format(testing_time))
        model.train()

        if avg_val_accuracy > best_val_acc:
          model.save_pretrained("models/" + args.model_output)
          best_val_acc = avg_val_accuracy

      if step % 50 == 0:
          elapsed = format_time(time.time() - t0)
          logging.info('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_data), elapsed))

      model.zero_grad()

      b_input_ids = batch["input_ids"].to(device)
      b_labels = batch["labels"].to(device)
      b_attention_masks = batch["attention"].to(device)

      scores = model(b_input_ids, token_type_ids=None, labels=b_labels, attention_mask=b_attention_masks)

      batch_wer = torch.tensor(batch["wer"][1:], device=device)
      orig_score = scores[0].expand(1, len(batch['wer']) - 1).squeeze()
      batch_loss = torch.sum(relu(scores[1:].squeeze() + batch_wer - orig_score))
      batch_loss.backward()

      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()
      scheduler.step()
        
    training_time = format_time(time.time() - t0)

    logging.info("")
    logging.info("  Training epcoh took: {:}".format(training_time))

    logging.info("")
    logging.info("Running Validation...")

    avg_val_accuracy, validation_time, validation_stats = evaluate_model(model, val_data)
    logging.info("  EN accuracy: {0:.2f}".format(100 * validation_stats["en_correct"] / validation_stats["en_total"]))
    logging.info("  Total accuracy: {0:.2f}".format(avg_val_accuracy))
    logging.info("  Validation took: {:}".format(validation_time))

    logging.info("")
    logging.info("Running Test...")

    avg_test_accuracy, testing_time, testing_stats = evaluate_model(model, test_data)
    logging.info("  EN accuracy: {0:.2f}".format(100 * testing_stats["en_correct"] / testing_stats["en_total"]))
    logging.info("  Total accuracy: {0:.2f}".format(avg_test_accuracy))
    logging.info("  Test took: {:}".format(testing_time))
    model.train()

    if avg_val_accuracy > best_val_acc:
      model.save_pretrained("models/" + args.model_output)
      best_val_acc = avg_val_accuracy

logging.info("")
logging.info("Training complete!")

logging.info("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
