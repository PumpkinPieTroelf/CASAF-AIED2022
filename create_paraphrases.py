from typing import List
from argparse import ArgumentParser
import pandas as pd
import torch
from timeit import default_timer as timer
from datetime import timedelta
from transformers import AutoTokenizer
from transformers.trainer_utils import set_seed
from src import ASAG, Editor, Masker
from src.explanation import Counterfactual, GradientAttribution
from src.datasets.dataset import load_data, filter_out_correct_answers
from src.util.utils import load_asag_model, load_editor_model, GENERATORS
from src.datasets.utils import semeval_keys, kn1_keys, split_in_answer_types
from src.util.utils import GENERATORS
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("-s", "--seed", type=int, default='42')  # pass list of seeds for experiments
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

# Initialize ASAG

asag_device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
asag_model = load_asag_model(dataset, asag_device)
asag_tokenizer = AutoTokenizer.from_pretrained(asag_model_name)
asag = ASAG(tokenizer=asag_tokenizer, model=asag_model, device=asag_device, keys=keys)

# Initialize Editor
editor_device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
editor_tokenizer = AutoTokenizer.from_pretrained(editor_model_name)

model = 'paraphrase'
editor_model = load_editor_model(GENERATORS[dataset]['paraphrase'], editor_device)
editor = Editor(tokenizer=editor_tokenizer, editor_model=editor_model, device=editor_device, editor_type=model)
cf_generator = Counterfactual(asag, editor, masker=None, attr_method=None)


def find_paraphrase(stud, reference, label):
    target_label_id = asag.get_label_id(label)
    candidates = editor.perturb_answer(masked_answers=[(stud, None)], reference=reference, target_label=label)
    logits = asag.batch_predict(candidates, reference)
    best_candidate, label = cf_generator._find_best_candidate(logits, candidates, target_label_id)
    return best_candidate


def generate_counterfactual(answers, label):
    counterfactuals = []
    times = []
    for _, pair in enumerate(tqdm(answers)):
        question = pair[0]
        ref = pair[1]
        stud = pair[2]
        start_time = timer()
        counterfactual = find_paraphrase(stud=stud, reference=ref, label=label)
        end_time = timer()
        counterfactuals.append((question, ref, stud, counterfactual))
        times.append((str(timedelta(seconds=end_time - start_time)), counterfactual))
    return counterfactuals, times


ic_res, ic_times = generate_counterfactual(ic_answers_sample, target_label)
print("IC counterfactuals created", flush=True)
pc_res, pc_times = generate_counterfactual(pc_answers_sample, target_label)
print("PC counterfactuals created", flush=True)

ic_df = pd.DataFrame(ic_res, columns=['Question', 'Ref.', 'Stud.', 'CF'])
ic_times_df = pd.DataFrame(ic_times, columns=['Time', 'CF'])
pc_df = pd.DataFrame(pc_res, columns=['Question', 'Ref.', 'Stud.', 'CF'])
pc_times_df = pd.DataFrame(pc_times, columns=['Time', 'CF'])

print('save examples', flush=True)
if dataset == 'kn1':
    ic_df.to_csv('./evaluation_new/{}/incorrect/{}_{}_ic_cfs.csv'.format(dataset, model, split), index=False)
    pc_df.to_csv('./evaluation_new/{}/partially_correct/{}_{}_pc_cfs.csv'.format(dataset, model, split), index=False)
    ic_times_df.to_csv('./evaluation_new/{}/incorrect/{}_{}_times_ic_cfs.csv'.format(dataset, model, split),
                       index=False)
    pc_times_df.to_csv('./evaluation_new/{}/partially_correct/{}_{}_times_pc_cfs.csv'.format(dataset, model, split),
                       index=False)
else:
    ic_df.to_csv('./evaluation_new/{}/contradictory/{}_{}_co_cfs.csv'.format(dataset, model, split), index=False)
    pc_df.to_csv('./evaluation_new/{}/incorrect/{}_{}_ic_cfs.csv'.format(dataset, model, split), index=False)
    ic_times_df.to_csv('./evaluation_new/{}/contradictory/{}_{}_times_co_cfs.csv'.format(dataset, model, split),
                       index=False)
    pc_times_df.to_csv('./evaluation_new/{}/incorrect/{}_{}_times_ic_cfs.csv'.format(dataset, model, split),
                       index=False)
