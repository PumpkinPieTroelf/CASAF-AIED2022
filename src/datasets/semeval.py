import torch
from torch.utils.data import TensorDataset


class SemEvalDataSet(torch.utils.data.Dataset):
    """
    Datset for semeval style data
    """

    def __init__(self, sent_pairs, scores, tokenizer, max_length):
        self.sent_pairs = sent_pairs
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_len = tokenizer.model_max_length

    def __len__(self):
        return len(self.sent_pairs)

    def __getitem__(self, item):
        sent_pairs = self.sent_pairs[item]
        scores = self.scores[item]
        ref = ' '.join(sent_pairs[0].split()).lower()
        stud = ' '.join(sent_pairs[1].split()).lower()
        encoded_seq = self.tokenizer.encode_plus(
            ref,
            stud,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True

        )

        return {
            'input_ids': encoded_seq['input_ids'][0],
            'attention_mask': encoded_seq['attention_mask'][0],
            'labels': torch.tensor(scores, dtype=torch.long)
        }
