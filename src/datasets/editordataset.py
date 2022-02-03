from random import randrange

import torch
from torch.utils.data import TensorDataset
from transformers import DataCollator
from src.datasets.dataset import clean_answer


class EditorDataSet(torch.utils.data.Dataset):
    """

    """

    def __init__(self, tokenizer, masker, data, label=False, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.masker = masker
        self.label = label
        self.masked_strings, self.targets = self.prepare_data(data)

    def __len__(self):
        return len(self.masked_strings)

    def __getitem__(self, index):
        input_text = self.masked_strings[index]
        label_text = self.targets[index]

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
        masked_strings = []
        target_lst = []
        for index, _ in enumerate(data):
            stud_answer = clean_answer(str(data[index][1]))
            ref_answer = clean_answer(str(data[index][0]))
            perc = randrange(20, 55) / 100

            masked_answer, targets = self.masker.mask(stud_answer, perc)
            if self.label:
                label = data[index][2]
                formatted_answer = 'label: ' + label + '. ' + 'input: ' + masked_answer + '</s>' + ref_answer
            else:
                formatted_answer = 'input: ' + masked_answer + '</s>' + ref_answer
            formatted_targets = 'answer: ' + targets + '</s>'

            masked_strings.append(formatted_answer)
            target_lst.append(formatted_targets)
        return masked_strings, target_lst
