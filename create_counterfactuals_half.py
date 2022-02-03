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
parser.add_argument('--part', type=str)

# Get arguments
args = parser.parse_args()
dataset = args.dataset
asag_model_name = args.asag_model
editor_model_name = args.editor_model
split = args.split
edit_rounds = args.iterations
seed = args.seed
part= args.part

set_seed(seed)


# Load data split
data = load_data(dataset, split, add_questions=True)
#print(data)
#Filter out correct answers
data = filter_out_correct_answers(data, with_questions=True)
#print(data)

# Split data into answer types
ic_answers_sample, pc_answers_sample = split_in_answer_types(data, dataset=dataset)
x=int(len(ic_answers_sample)/4)
y=int(len(pc_answers_sample)/4)
if part=='one':
    ic_answers_sample = ic_answers_sample[:x]
    pc_answers_sample = pc_answers_sample[:y]
elif part=='two':
    ic_answers_sample = ic_answers_sample[x:x+x]
    pc_answers_sample = pc_answers_sample[y:y+y]
elif part=='three':
    ic_answers_sample = ic_answers_sample[x+x:x+x+x]
    pc_answers_sample = pc_answers_sample[y+y:y+y+y]
else:
    ic_answers_sample = ic_answers_sample[x+x+x:]
    pc_answers_sample = pc_answers_sample[y+y+y:]

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
editor_device = asag_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#asag_device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
asag_model = load_asag_model(dataset, asag_device)
asag_tokenizer = AutoTokenizer.from_pretrained(asag_model_name)
asag = ASAG(tokenizer=asag_tokenizer, model=asag_model, device=asag_device, keys=keys)

# Initialize Editor
#editor_device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
editor_tokenizer = AutoTokenizer.from_pretrained(editor_model_name)

# Initialize Masker
masker = Masker()

# Initialize Attribution Method
intGrad = GradientAttribution(asag_tokenizer, asag_device, asag_model)


def generate_counterfactual(generator, answers, label):
    counterfactuals = []
    times = []
    for _, pair in enumerate(tqdm(answers)):
        question = pair[0]
        ref = pair[1]
        stud = pair[2]
        counterfactual = generator.find(stud=stud, reference=ref, target_label=label, iterations=edit_rounds)
        counterfactuals.append((question, ref, stud, counterfactual))
    return counterfactuals, times


# Generate counterfactuals for each editor model
print("Dataset: " + dataset + " " + split, flush=True)
for model, path in GENERATORS[dataset].items():
    if model == 'paraphrase':
        continue
    print(model, flush=True)
    print(path, flush=True)

    editor_model = load_editor_model(path, editor_device)
    editor = Editor(tokenizer=editor_tokenizer, editor_model=editor_model, device=editor_device, editor_type=model)
    cf_generator = Counterfactual(asag, editor, masker=masker, attr_method=intGrad)
    
    ic_res, ic_times = generate_counterfactual(cf_generator, ic_answers_sample, target_label)
    print("IC counterfactuals created", flush=True)
    ic_df = pd.DataFrame(ic_res, columns=['Question', 'Ref.', 'Stud.', 'CF'])

    print('save examples', flush=True)
    if dataset == 'kn1':
        if part=='one':
            ic_df.to_csv('./evaluation_new/{}/incorrect/{}_{}_ic_cfs_1.csv'.format(dataset, model, split), index=False)
        elif part=='two':
            ic_df.to_csv('./evaluation_new/{}/incorrect/{}_{}_ic_cfs_2.csv'.format(dataset, model, split), index=False)
        elif part=='three':
            ic_df.to_csv('./evaluation_new/{}/incorrect/{}_{}_ic_cfs_3.csv'.format(dataset, model, split), index=False)
        else:
            ic_df.to_csv('./evaluation_new/{}/incorrect/{}_{}_ic_cfs_4.csv'.format(dataset, model, split), index=False)
    else:
        if part=='one':
            ic_df.to_csv('./evaluation_new/{}/contradictory/{}_{}_co_cfs_1.csv'.format(dataset, model, split), index=False)
        elif part=='two':
            ic_df.to_csv('./evaluation_new/{}/contradictory/{}_{}_co_cfs_2.csv'.format(dataset, model, split), index=False)
        elif part=='three':
            ic_df.to_csv('./evaluation_new/{}/contradictory/{}_{}_co_cfs_3.csv'.format(dataset, model, split), index=False)
        else:
            ic_df.to_csv('./evaluation_new/{}/contradictory/{}_{}_co_cfs_4.csv'.format(dataset, model, split), index=False)

    pc_res, pc_times = generate_counterfactual(cf_generator, pc_answers_sample, target_label)
    print("PC counterfactuals created", flush=True)
 
    pc_df = pd.DataFrame(pc_res, columns=['Question', 'Ref.', 'Stud.', 'CF'])

    print('save examples', flush=True)
    if dataset == 'kn1':
        if part=='one':
            pc_df.to_csv('./evaluation_new/{}/partially_correct/{}_{}_pc_cfs_1.csv'.format(dataset, model, split), index=False)
        elif part=='two':
            pc_df.to_csv('./evaluation_new/{}/partially_correct/{}_{}_pc_cfs_2.csv'.format(dataset, model, split), index=False)
        elif part=='three':
            pc_df.to_csv('./evaluation_new/{}/partially_correct/{}_{}_pc_cfs_3.csv'.format(dataset, model, split), index=False)
        else:
            pc_df.to_csv('./evaluation_new/{}/partially_correct/{}_{}_pc_cfs_4.csv'.format(dataset, model, split), index=False)
    else:
        if part=='one':
            pc_df.to_csv('./evaluation_new/{}/incorrect/{}_{}_ic_cfs_1.csv'.format(dataset, model, split), index=False)
        elif part=='two':
            pc_df.to_csv('./evaluation_new/{}/incorrect/{}_{}_ic_cfs_2.csv'.format(dataset, model, split), index=False)
        elif part=='three':
            pc_df.to_csv('./evaluation_new/{}/incorrect/{}_{}_ic_cfs_3.csv'.format(dataset, model, split), index=False)
        else:            
            pc_df.to_csv('./evaluation_new/{}/incorrect/{}_{}_ic_cfs_4.csv'.format(dataset, model, split), index=False)

