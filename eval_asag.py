import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, Trainer, TrainingArguments
from transformers.trainer_utils import set_seed
from argparse import ArgumentParser
from src.datasets.dataset import load_data
from src.datasets import SemEvalDataSet
from src.util import compute_asag_metrics
from src.util.utils import load_asag_model

parser = ArgumentParser()
parser.add_argument("-s", "--seed", type=int, default='42')
parser.add_argument("-d", "--dataset", type=str, help="Name of the dataset (scientisbank, beetle, kn1)")
parser.add_argument("-m", "--model", type=str, help="Model name (Hugging Face model hub name)")
parser.add_argument("-n", "--name", type=str, help="Name under which the model should be saved")
parser.add_argument("-l", "--max_length", type=int, default='512', help="Max length of input sequences")

args = parser.parse_args()
seed = args.seed
name = args.name
model = args.model
dataset = args.dataset
max_length = args.max_length

set_seed(seed)

if dataset == 'kn1':
    config = AutoConfig.from_pretrained(model, num_labels=3,
                                        id2label={0: 'Incorrect', 1: 'Partially correct', 2: 'Correct'},
                                        label2idd={'Incorrect': 0, 'Partially correct': 1, 'Correct': 2})
else:
    config = AutoConfig.from_pretrained(model, num_labels=3,
                                        id2label={0: 'contradictory', 1: 'incorrect', 2: 'correct'},
                                        label2idd={'contradictory': 0, 'incorrect': 1, 'correct': 2})

asag_model = AutoModelForSequenceClassification.from_pretrained(model, config=config)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
asag_model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model)
print('Model loaded', flush=True)

train_data = load_data(dataset, 'train')
test_data_ua = load_data(dataset, 'ua')
test_data_uq = load_data(dataset, 'uq')
test_set_ua = SemEvalDataSet(sent_pairs=test_data_ua['sent_pairs'], scores=test_data_ua['scores'], tokenizer=tokenizer,
                             max_length=max_length)
test_set_uq = SemEvalDataSet(sent_pairs=test_data_uq['sent_pairs'], scores=test_data_uq['scores'], tokenizer=tokenizer,
                             max_length=max_length)

if dataset == 'scientsbank':
    test_data_ud = load_data(dataset, 'ud')
    test_set_ud = SemEvalDataSet(sent_pairs=test_data_ud['sent_pairs'], scores=test_data_ud['scores'],
                                 tokenizer=tokenizer,
                                 max_length=max_length)

print('Data loaded: ' + dataset, flush=True)

log_dir = './logs/{}/{}'.format(dataset, name)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

training_args = TrainingArguments(
    output_dir=log_dir,
    learning_rate=2e-05,
    num_train_epochs=24,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    dataloader_num_workers=2,
    dataloader_drop_last=True,
    fp16=True,
    seed=seed,
    warmup_steps=1024,
    weight_decay=0.01,
    save_strategy="epoch"
)

trainer = Trainer(
    model=asag_model,
    args=training_args,
    compute_metrics=compute_asag_metrics
)

# 5. eval model
print('Evaluation UA', flush=True)
metrics = trainer.evaluate(test_set_ua)
print(metrics, flush=True)

print('Evaluation UQ', flush=True)
metrics = trainer.evaluate(test_set_uq)
print(metrics, flush=True)

if dataset == 'scientsbank':
    print('Evaluation UD', flush=True)
    metrics = trainer.evaluate(test_set_ud)
    print(metrics, flush=True)
