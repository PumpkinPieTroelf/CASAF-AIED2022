from typing import List
from argparse import ArgumentParser
import pandas as pd
import torch
from transformers import AutoTokenizer
from transformers.trainer_utils import set_seed
from src import ASAG, Editor, Masker
from src.explanation import Counterfactual
from src.datasets.dataset import load_data, filter_out_correct_answers
from src.util.utils import load_asag_model
from src.datasets.utils import semeval_keys, kn1_keys, split_in_answer_types
from tqdm import tqdm
from polyjuice import Polyjuice

parser = ArgumentParser()
parser.add_argument("-s", "--seed", type=int, default='42')
parser.add_argument("-d", "--dataset", type=str, help="Name of the dataset (scientisbank, beetle, kn1)")
parser.add_argument("-l", "--split", type=str, help="Name of the data split")
parser.add_argument("-a", "--asag_model", type=str, help="Name of the asag base model")
parser.add_argument("-e", "--editor_model", type=str, help="Name of the editor base model")
parser.add_argument("-i", "--iterations", type=int, default=4, help="Number of edit rounds")

# Get arguments
args = parser.parse_args()
dataset = args.dataset
asag_model_name = args.asag_model
editor_model_name = args.editor_model
split = args.split
edit_rounds = args.iterations
seed = args.seed

set_seed(seed)

# Load data split
data = load_data(dataset, split, add_questions=True)

# Filter out correct answers
data = filter_out_correct_answers(data, with_questions=True)
# print(data)

# Split data into answer types
ic_answers_sample, pc_answers_sample = split_in_answer_types(data, dataset=dataset)

if dataset != 'kn1':
    keys = semeval_keys
    label_0 = 'contradictory'
    label_1 = 'incorrect'
    target_label = 'correct'
else:
    keys = kn1_keys
    label_0 = 'Incorrect'
    label_1 = 'Partially correct'
    target_label = 'Correct'

asag_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
asag_model = load_asag_model(dataset, asag_device)
asag_tokenizer = AutoTokenizer.from_pretrained(asag_model_name)
asag = ASAG(tokenizer=asag_tokenizer, model=asag_model, device=asag_device, keys=keys)

model = 'polyjuice'
cf_generator = Counterfactual(asag, None, masker=None, attr_method=None)
ctrl_codes = ['resemantic', 'restructure', 'negation', 'insert', 'lexical', 'shuffle', 'quantifier', 'delete']
pj = Polyjuice(model_path="uw-hai/polyjuice", is_cuda=True)
print('Dataset: ' + dataset + ' ' + split + 'Model: ' + model, flush=True)


def find_counterfactual(stud, reference, label):
    target_label_id = asag.get_label_id(label)
    candidates = []
    for code in ctrl_codes:
        try:
            editor_output = pj.perturb(orig_sent=stud, ctrl_code=code, num_perturbations=3, num_beams=7)
        except:
            editor_output = []

    candidates += editor_output
    if not candidates:
        return 'no counterfactual found'
    logits = asag.batch_predict(candidates, reference)
    best_candidate, label = cf_generator._find_best_candidate(logits, candidates, target_label_id)
    return best_candidate


def generate_counterfactual(answers, label):
    counterfactuals = []
    times = []
    for index, pair in enumerate(tqdm(answers)):
        if index == 322:
            print(pair)
            continue
        question = pair[0]
        ref = pair[1]
        stud = pair[2]
        if stud == '?' or stud == '?/':
            counterfactual = 'no counterfactual found'
        else:
            counterfactual = find_counterfactual(stud=stud, reference=ref, label=label)
        counterfactuals.append((question, ref, stud, counterfactual))
    return counterfactuals, times


ic_res, ic_times = generate_counterfactual(ic_answers_sample, target_label)
print("IC counterfactuals created", flush=True)

ic_df = pd.DataFrame(ic_res, columns=['Question', 'Ref.', 'Stud.', 'CF'])

print('save ic', flush=True)
if dataset == 'kn1':
    ic_df.to_csv('./evaluation_new/{}/incorrect/{}_{}_ic_cfs.csv'.format(dataset, model, split), index=False)
else:
    ic_df.to_csv('./evaluation_new/{}/contradictory/{}_{}_co_cfs.csv'.format(dataset, model, split), index=False)

pc_res, pc_times = generate_counterfactual(pc_answers_sample, target_label)
print("PC counterfactuals created", flush=True)

pc_df = pd.DataFrame(pc_res, columns=['Question', 'Ref.', 'Stud.', 'CF'])

print('save pc', flush=True)
if dataset == 'kn1':
    pc_df.to_csv('./evaluation_new/{}/partially_correct/{}_{}_pc_cfs.csv'.format(dataset, model, split), index=False)
else:
    pc_df.to_csv('./evaluation_new/{}/incorrect/{}_{}_ic_cfs.csv'.format(dataset, model, split), index=False)
