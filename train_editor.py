import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, AutoTokenizer, AutoConfig, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers.optimization import Adafactor, get_constant_schedule
from transformers.trainer_utils import set_seed
from sklearn.model_selection import train_test_split
from src import Masker
from src.datasets import (load_data, ParaphraseDataSet, EditorDataSet, T2TDataCollator,
                          preprocess_paraphrase_data, preprocess_editor_data)
from argparse import ArgumentParser

# All arguments that can be passed
parser = ArgumentParser()
parser.add_argument("-s", "--seed", type=int, default='42')  # pass list of seeds for experiments
parser.add_argument("-d", "--dataset", type=str, help="Name of the dataset (scientisbank, beetle, kn1)")
parser.add_argument("-n", "--name", type=str, help="Name under which the model should be saved")
parser.add_argument("-o", "--objective", type=str, help="Model mode")
parser.add_argument("-l", "--max_length", type=int, default='512', help="Model mode")

# Parse experimental arguments
args = parser.parse_args()
name = args.name
seed = args.seed
dataset = args.dataset
# num_labels = args.labels
objective = args.objective
max_length = args.max_length

set_seed(seed)

model = 't5-base'
config = AutoConfig.from_pretrained(model)
generator = T5ForConditionalGeneration.from_pretrained(model)
tokenizer = AutoTokenizer.from_pretrained(model)
masker = Masker()
print('Model loaded. Objective: ' + objective + ' Dataset: ' + dataset, flush=True)

train_data = load_data(dataset=dataset, split='train')

if objective == 'paraphrase':
    train_data = preprocess_paraphrase_data(data=train_data)
    train_set = ParaphraseDataSet(tokenizer=tokenizer, masker=masker, data=train_data, objective=objective,
                                  max_length=max_length)
else:
    train_data = preprocess_editor_data(data=train_data, dataset=dataset, objective=objective)
    label = True if objective == 'label_infill' else False
    train_set = EditorDataSet(tokenizer=tokenizer, masker=masker, data=train_data, label=label,
                              max_length=max_length)

print('Data loaded', flush=True)
print('Train examples: ' + str(len(train_data)), flush=True)

# 3. Setup training args
output_dir = './logs/{}/{}'.format(dataset, name)
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=10,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=16,
    eval_accumulation_steps=4,
    dataloader_num_workers=2,
    dataloader_drop_last=True,
    fp16=True,
    seed=seed,
    save_strategy="epoch",
    # evaluation_strategy="steps",
    adafactor=True,
    report_to='tensorboard',
    # do_predict=True
)
optimizer = Adafactor(generator.parameters(), lr=1e-3,
                      relative_step=False, warmup_init=False,
                      decay_rate=0.0, clip_threshold=1.0)

lr_scheduler = get_constant_schedule(optimizer)

# 4. train model
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
generator.to(device)

trainer = Seq2SeqTrainer(
    model=generator,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=test_set,
    data_collator=T2TDataCollator(),
    optimizers=(optimizer, lr_scheduler)
)

print('Start training model', flush=True)
# 4. train model
trainer.train()

print('Finished training model', flush=True)
save_dir = './new_trained_models/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
trainer.save_model(save_dir + '/{}/{}'.format(dataset, name))
print('Model saved', flush=True)
