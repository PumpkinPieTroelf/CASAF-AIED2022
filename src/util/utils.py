import torch
from transformers import BertForSequenceClassification, T5ForConditionalGeneration

GENERATORS = {
    'kn1': {
        'paraphrase': './new_trained_models/kn1/athena_paraphrase',
        'label_infill': './new_trained_models/kn1/athena_label_infill',
        'infill': './new_trained_models/kn1/athena_infill'
    },
    'scientsbank': {
        'infill': './new_trained_models/scientsbank/athena_infill',
        'paraphrase': './new_trained_models/scientsbank/athena_paraphrase',
        'label_infill': './new_trained_models/scientsbank/athena_label_infill'
    },
    'beetle': {
        'infill': './new_trained_models/beetle/athena_infill',
        'label_infill': './new_trained_models/beetle/athena_label_infill',
        'paraphrase': './new_trained_models/beetle/athena_paraphrase'
    }
}
PREDICTORS = {
    'kn1': './new_trained_models/kn1/kn1_bert',
    'scientsbank': './new_trained_models/scientsbank/science_bert',
    'beetle': './new_trained_models/beetle/beetle_bert',
}


def load_asag_model(dataset: str, device):
    asag = BertForSequenceClassification.from_pretrained(PREDICTORS[dataset])
    asag.to(device)
    return asag


def load_editor_model(model: str, device):
    cf_editor = T5ForConditionalGeneration.from_pretrained(model)
    cf_editor.to(device)
    return cf_editor
