from random import randrange

import torch
from torch.utils.data import TensorDataset
from transformers import DataCollator


class ParaphraseDataSet(torch.utils.data.Dataset):
    """

    """

    def __init__(self, tokenizer, masker, data, objective='infill', max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.masker = masker
        self.task = objective
        self.answers, self.paraphrase = self.prepare_data(data)

    def __len__(self):
        return len(self.answers)

    def __getitem__(self, index):
        input_text = self.answers[index]
        label_text = self.paraphrase[index]

        source = self.tokenizer.batch_encode_plus([input_text],
                                                  truncation=True, max_length=self.max_length,
                                                  pad_to_max_length=True, return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([label_text],
                                                  truncation=True, max_length=self.max_length,
                                                  pad_to_max_length=True, return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'input_ids': source_ids.to(dtype=torch.long),
            'attention_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'target_mask': target_mask.to(dtype=torch.long)
        }

    def prepare_data(self, data):
        answers = []
        target_lst = []
        for index, _ in enumerate(data):
            stud_answer = ' '.join(str(data[index][0]).split()).lower()
            ref_answer = ' '.join(str(data[index][1]).split()).lower()
            formatted_answer = 'input: ' + stud_answer + ' </s>'
            formatted_targets = 'answer: ' + ref_answer + '</s>'

            answers.append(formatted_answer)
            target_lst.append(formatted_targets)
        return answers, target_lst
