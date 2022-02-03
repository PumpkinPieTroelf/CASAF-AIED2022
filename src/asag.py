import torch


class ASAG:
    def __init__(self, model, tokenizer, device, keys):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.keys = keys

    def get_label(self, logits):
        return self.model.config.id2label[logits.argmax()]

    def get_label_id(self, label):
        return self.keys[label]

    def create_asag_input_ids(self, ref_answ, stud):
        input_ref_pair, ref_input_ids = self._construct_input_ref_pair(ref_answ, stud)
        input_ids = {}
        pos_ids = self._construct_input_pos_ids(input_ref_pair['input_ids'])
        input_ids['input_ids'] = input_ref_pair['input_ids'].to(self.device)
        input_ids['ref_input_ids'] = ref_input_ids.to(self.device)
        input_ids['token_type_ids'] = input_ref_pair['token_type_ids'].to(self.device)
        input_ids['attention_mask'] = input_ref_pair['attention_mask'].to(self.device)
        input_ids['position_ids'] = pos_ids.to(self.device)
        input_ids['sep_id'] = [
            index
            for index, x in enumerate(input_ref_pair['special_tokens_mask'].tolist()[0])
            if x == 1
        ]
        return input_ids

    def _construct_input_ref_pair(self, ref_answ, stud):
        """

        :param ref_answ:
        :param stud:
        :return:
        """
        encoded = self.tokenizer.encode_plus(ref_answ, stud, return_special_tokens_mask=True,
                                             pad_to_max_length=True, max_length=512,
                                             return_token_type_ids=True, return_tensors='pt')

        encoded['special_tokens_mask'][0][0] = self.tokenizer.cls_token_id
        # Get the indexes of nonzero values of the special tokens mask
        nonzeros = torch.nonzero(encoded['special_tokens_mask'][0], as_tuple=True)[0]
        encoded['special_tokens_mask'][0][nonzeros[1]] = self.tokenizer.sep_token_id
        encoded['special_tokens_mask'][0][nonzeros[2]] = self.tokenizer.sep_token_id
        if len(nonzeros) > 3:
            encoded['special_tokens_mask'][0][nonzeros[3]:] = 0
        return encoded, encoded['special_tokens_mask']

    def _construct_input_pos_ids(self, input_ids):
        """

        :param input_ids:
        :return:
        """
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=self.device)

        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        return position_ids

    def predict(self, answers, reference):
        """

        :param answers: List
        :param reference:
        :return:
        """
        if isinstance(answers, list):
            scores = []
            for answer in answers:
                ids, _ = self._construct_input_ref_pair(ref_answ=reference, stud=answer)
                with torch.no_grad():
                    logits = self.model(ids)[0]
                    scores.append(logits)
                del ids
        ids, _ = self._construct_input_ref_pair(ref_answ=reference, stud=answers)
        with torch.no_grad():
            logits = self.model(ids)[0]
        return logits

    def batch_predict(self, answers, reference):
        scores = []
        for answer in answers:
            ids, _ = self._construct_input_ref_pair(ref_answ=reference, stud=answer)
            ids = ids.to(self.device)
            with torch.no_grad():
                logits = self.model(ids['input_ids'], attention_mask=ids['attention_mask'])[0]
                scores.append(logits)
            del ids
        return scores
