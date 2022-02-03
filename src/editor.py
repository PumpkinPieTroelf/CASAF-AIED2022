import torch
import re


class Editor:
    def __init__(self, tokenizer, editor_model, device, editor_type='label_infill'):
        self.tokenizer = tokenizer
        self.editor_model = editor_model
        self.device = device
        self.editor_type = editor_type

    def preprocess_input(self, masked_answer, reference, label):
        if self.editor_type == 'label_infill':
            return 'label: ' + label + '. ' + 'input: ' + masked_answer + '</s>' + reference
        elif self.editor_type == 'paraphrase':
            return 'input: ' + masked_answer + ' </s>'
        return "input: " + masked_answer + '</s>' + reference

    def perturb_answer(self, masked_answers, reference, target_label):
        edits = []
        for masked_answer, bad_words in masked_answers:
            formated_answer = self.preprocess_input(masked_answer, reference, target_label)
            input_ids = self.tokenizer.encode_plus(formated_answer, truncation=True, max_length=512,
                                                   pad_to_max_length=True,
                                                   return_tensors='pt')
            if self.editor_type == 'paraphrase':
                bad_words_ids = None
            else:
                bad_words_ids = self.tokenizer(bad_words).input_ids

            editor_output = self.sample(input_ids, bad_words_ids)

            candidates = []
            for beam in editor_output:
                answer = self.post_process_edits(beam, masked_answer)
                candidates.append(answer)
            edits.extend(candidates)
        return edits

    def sample(self, formatted_answer, bad_words_ids):

        input_ids = formatted_answer.input_ids
        attention_mask = formatted_answer.attention_mask
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        with torch.no_grad():
            editor_output = self.editor_model.generate(input_ids=input_ids,
                                                       attention_mask=attention_mask,
                                                       bad_words_ids=[
                                                           bad_words_ids] if self.editor_type != 'paraphrase' else None,
                                                       no_repeat_ngram_size=2,
                                                       repetition_penalty=1.1,
                                                       length_penalty=1.1,
                                                       do_sample=True, max_length=512,
                                                       top_k=30,
                                                       top_p=0.95, early_stopping=True, num_return_sequences=7)

        return editor_output

    def post_process_edits(self, beams, masked_answer):

        labels = self.tokenizer.decode(beams, skip_special_tokens=False, clean_up_tokenization_spaces=True)
        labels = labels.replace('<pad>', '')
        labels = labels.replace('</s>', '')
        if self.editor_type == 'paraphrase':
            answer = labels.replace('<unk>', '')
            return ' '.join(answer.split()[1:])
        labels = re.split(r"<extra_id_\d+>", labels)[1:]
        answer = masked_answer
        for i, w in enumerate(labels):
            if i < len(labels):
                answer = answer.replace('<extra_id_' + str(i) + '>', w)
        return answer.replace('<unk>', '')
