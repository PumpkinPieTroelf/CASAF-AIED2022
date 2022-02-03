import os
import pandas as pd
import torch
from transformers import AutoTokenizer, BertForSequenceClassification
from os.path import isfile, join
from datasets import load_metric
import nltk
import spacy
from tqdm import tqdm
from argparse import ArgumentParser
from src.util.utils import load_asag_model

parser = ArgumentParser()
parser.add_argument("-s", "--seeds", nargs='+', type=int, default='42')  # pass list of seeds for experiments
parser.add_argument("-d", "--dataset", type=str, help="Name of the dataset (scientisbank, beetle, kn1)")
parser.add_argument("-l", "--max_length", type=int, help="Name of the dataset (scientisbank, beetle, kn1)")

args = parser.parse_args()

COUNTERFACTUALS = {
    'scientsbank': './evaluation_new/scientsbank/',
    'kn1': './evaluation_new/kn1/',
    'beetle': './evaluation_new/beetle/'
}


def get_prediction(ref, answer, model, tokenizer, device, max_length=512):
    inputs = tokenizer.encode_plus(ref, answer, add_special_tokens=True, max_length=max_length,
                                   return_token_type_ids=False, return_attention_mask=True, return_tensors="pt",
                                   padding="max_length", truncation=True)

    inputs = inputs.to(device)
    with torch.no_grad():
        prediction = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])[0]
    scores = torch.exp(prediction).cpu() / torch.exp(prediction).cpu().sum(-1, keepdims=True)
    idx = torch.argmax(scores).item()
    return {
        'label': model.config.id2label[idx].lower(),
        'score': torch.max(scores),
        'logits': prediction[0].tolist()
    }


def flip_rate(df):
    answers_not_pred_correct = len(df.loc[df['Orig Pred.'] != 'correct'])
    new_correct_cfs = len(df.loc[df['CF Pred.'] == 'correct'].loc[df['Orig Pred.'] != 'correct'])
    return round((new_correct_cfs / (answers_not_pred_correct)) * 100, 1), answers_not_pred_correct


def edit_distance(orig_sent, edited_sent, normalized=True):
    nlp = spacy.load('en_core_web_sm')
    tokenized_original = [t.text for t in nlp(orig_sent)]
    tokenized_edited = [t.text for t in nlp(edited_sent)]
    lev = nltk.edit_distance(tokenized_original, tokenized_edited)
    if normalized:
        return lev / len(tokenized_original)
    else:
        return lev


def compute_distance(df):
    norm_distance = 0
    for stud, cf in zip(df['stud'], df['cf']):
        if cf != "no counterfactual found":
            norm_distance += edit_distance(stud, cf)
    return norm_distance / len(df)


def score_counterfactuals(df, asag, tokenizer, device='cpu', max_length=512):
    predictions = list()
    for ref, stud, cf in zip(df['Ref.'], df['Stud.'], df['CF']):
        orig_pred = get_prediction(ref, stud, asag, tokenizer, device, max_length)
        cf_pred = get_prediction(ref, cf, asag, tokenizer, device, max_length)
        orig_label = orig_pred['label']
        cf_label = cf_pred['label']
        predictions.append((orig_label, orig_pred['logits'], cf_label, cf_pred['logits'], stud, cf))
    return predictions


def clean_cfs(df):
    df['CF'] = df['CF'].apply(lambda s: 'No counterfactual found' if 'No counterfactual found' in s else s)
    df['CF'] = df['CF'].apply(lambda x: ' '.join(x.split()).lower())
    df['Stud.'] = df['Stud.'].apply(lambda x: ' '.join(x.split()).lower())
    return df


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
asag = load_asag_model(args.dataset, device)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

path = COUNTERFACTUALS[args.dataset]
max_length = args.max_length

if args.dataset != 'kn1':
    path_0 = path + '/contradictory/'
    files_0 = [f for f in os.listdir(path + '/contradictory') if isfile(join(path + '/contradictory', f))]
    path_1 = path + '/incorrect/'
    files_1 = [f for f in os.listdir(path + '/incorrect') if isfile(join(path + '/incorrect', f))]
else:
    path_0 = path + '/incorrect/'
    files_0 = [f for f in os.listdir(path + '/incorrect') if isfile(join(path + '/incorrect', f))]
    path_1 = path + '/partially_correct/'
    files_1 = [f for f in os.listdir(path + '/partially_correct') if isfile(join(path + '/partially_correct', f))]

print('Compute metrics for dataset: ' + args.dataset, flush=True)
print('#' * 80, flush=True)


def compute_metrics(files, path):
    for csv_file in files:
        df = pd.read_csv(path + csv_file)
        df = clean_cfs(df)
        predictions = list()
        for ref, stud, cf in zip(df['Ref.'], df['Stud.'], df['CF']):
            orig_pred = get_prediction(ref, stud, asag, tokenizer, device, max_length)
            cf_pred = get_prediction(ref, cf, asag, tokenizer, device, max_length)
            orig_label = orig_pred['label']
            cf_label = cf_pred['label']
            predictions.append((orig_label, orig_pred['logits'], cf_label, cf_pred['logits'], stud, cf))
        df = pd.DataFrame(predictions, columns=['Orig Pred.', 'Orig logits', 'CF Pred.', 'CF logits', 'stud', 'cf'])
        df.to_csv(path + csv_file.split('.')[0] + '_predictions.csv', index=False)
        flip, answers_not_pred_correct = flip_rate(df)
        lev = round(compute_distance(df), 2)
        print('Model and Split: ' + csv_file, flush=True)
        print('Flip rate: {} '.format(flip), flush=True)
        print('Answers not previously/wrongly predicted correct: {}'.format(answers_not_pred_correct), flush=True)
        print('Levenshtein distance: {}'.format(lev), flush=True)
        print('-' * 80, flush=True)


if args.dataset != 'kn1':
    print('Contradictory answers', flush=True)
else:
    print('Incorrect answers', flush=True)
print('-' * 80, flush=True)
compute_metrics(files_0, path_0)

print('#' * 80, flush=True)
if args.dataset != 'kn1':
    print('Incorrect answers', flush=True)
else:
    print('Partially correct answers', flush=True)
print('-' * 80, flush=True)
compute_metrics(files_1, path_1)
