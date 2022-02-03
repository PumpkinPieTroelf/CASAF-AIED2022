import operator
import torch


class Counterfactual:

    def __init__(self, asag, editor, masker, attr_method):
        self.asag = asag
        self.editor = editor
        self.masker = masker
        self.attr_method = attr_method

    def find(self, stud, reference, target_label='correct', iterations=10):
        """
        :param stud:
        :param reference:
        :param target_label:
        :param iterations:
        :return:
        """

        best_candidate = stud
        target_label_id = self.asag.get_label_id(target_label)

        for i in range(0, iterations - 1):
            # 1 Get importance scores
            importance_scores = self.get_importance_scores(ref=reference, stud=stud, target_label_id=target_label_id)
            # 2 Mask answer based on importance scores
            percentages = [0.15, 0.3, 0.45, 0.6]
            masked_answers = [
                self.masker.mask(stud, percentage, mask_type='gradient', importance_tuples=importance_scores)
                for percentage in percentages
            ]
            # 3 Perturb answers
            candidates = self.editor.perturb_answer(masked_answers, reference, target_label)
            #print(candidates)
            # 4 Grade candidate answers
            logits = self.asag.batch_predict(candidates, reference)
            # 5 Get best candidate
            best_candidate, label = self._find_best_candidate(logits, candidates, target_label_id)
            if label.lower() == 'correct':
                return best_candidate

        return best_candidate

    def _find_best_candidate(self, logits, candidates, target_label_id):
        """

        :param logits:
        :param candidates:
        :param target_label_id:
        :return:
        """
        # 1. Get logits for the target label
        target_scores = [score.detach().cpu().numpy()[0][target_label_id] for score in logits]
        # 2. Find the index of the highest target score
        index, _ = max(enumerate(target_scores), key=operator.itemgetter(1))
        # 3. Get all scores of candidate with the highest target score
        candidate_scores = logits[index].detach().cpu().numpy()
        # 4. Get the candidate with the highest target score
        highest_prob_example = candidates[index]

        return highest_prob_example, self.asag.get_label(candidate_scores)

    def get_importance_scores(self, ref, stud, target_label_id):

        input_ids = self.asag.create_asag_input_ids(ref_answ=ref, stud=stud)
        attribution_tensor = self.attr_method.get_gradient_attr(input_ids['input_ids'], input_ids['ref_input_ids'],
                                                                target_label_id)

        decoded_features = []
        for token in input_ids['input_ids'][0]:
            decoded_features.append(self.asag.tokenizer.decode(token))
        normalized_importance_scores = self.attr_method.normalize_importance_attributions(decoded_features,
                                                                                          attribution_tensor.tolist())

        return normalized_importance_scores
