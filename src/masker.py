import math
import random
import more_itertools as mit


class Masker:
    """

        """

    def __init__(self, token_max_num=512, blank_token_type='Sentinel'):
        self.token_max_num = token_max_num
        self.max_num_sentinels = 99
        self.blank_token_type = blank_token_type

    def get_random_blank_indexes(self, seq, mask_perc=0.3):
        """
        Returns a list of random indexs in the input sequence that are to be masked
        :param seq:
        :param mask_perc:
        :return:
        """
        num_tokens = min(self.token_max_num, len(seq.split()))
        return random.sample(range(num_tokens), math.ceil(mask_perc * num_tokens))

    def find_spans(self, blank_idxs):
        """
        Finds consecutive masked tokens
        :param blank_idxs:
        :return:
        """
        span_groups = [list(group) for group in mit.consecutive_groups(blank_idxs)]
        return span_groups[:self.max_num_sentinels]

    def get_gradient_blank_indexes(self, orig_seq, importance_tuples, mask_perc):
        """
        Returns a list of indexs in the input sequence that are to be masked, based on
        importance scores provided by a feature attribution method.

        :param orig_seq:
        :param importance_tuples:
        :param mask_perc:
        :return:
        """
        seq = orig_seq.split()
        # 1. Find n token with the highest importance scores
        idxs = {idx: tpl[1] for idx, tpl in enumerate(importance_tuples)}

        # 2. Find indexes of the n most important tokens from -1 to 1
        # lower negative importance scores indicate tokens that are a detriment for the target label
        n = math.ceil(mask_perc * len(seq))
        sorted_dic = dict(sorted(idxs.items(), key=lambda item: item[1]))
        return [k for k, _ in sorted_dic.items()][:n]

    def _get_masking_token(self, idx):
        """
        """
        if self.blank_token_type == 'BLANK':
            return '[BLANK]'
        return "<extra_id_" + str(idx) + ">"

    def mask(self, stud_answer, mask_perc, mask_type='random', importance_tuples=None):
        """
        Finds the tokens to be masked in the input sequence, either random or based on feature importance scores
        and replaces these tokens with a sentinel token. If multiple consecutive tokens, i.e. a span,
        are to be masked, we replace the whole span with a single sentinel token.

        We return the masked student answer and the masked out tokens.

        :param stud_answer:
        :param mask_perc:
        :param mask_type:
        :param importance_tuples:
        :return: tuple: String, String
        """
        if importance_tuples is None:
            importance_tuples = []
        if mask_type == 'gradient':
            blank_idxs = self.get_gradient_blank_indexes(stud_answer, importance_tuples, mask_perc)
        else:
            blank_idxs = self.get_random_blank_indexes(stud_answer, mask_perc)
        sorted_idxs = sorted(blank_idxs)
        span_groups = self.find_spans(sorted_idxs)

        orig = stud_answer.split()
        masked_stud_answer = []
        masked_parts = []

        sentinels = 0
        curr_span = 0
        skip_idx = -1

        for idx, _ in enumerate(orig):
            if idx < skip_idx:
                masked_parts.append(orig[idx])
                continue

            if curr_span < len(span_groups) and idx == span_groups[curr_span][0]:
                sentinel = self._get_masking_token(sentinels)
                masked_stud_answer.append(sentinel)
                sentinels += 1
                span = len(span_groups[curr_span])
                if span > 1:
                    skip_idx = idx + span
                masked_parts.append(sentinel)
                masked_parts.append(orig[idx])
                curr_span += 1
            else:
                masked_stud_answer.append(orig[idx])

        masked_parts.append(self._get_masking_token(sentinels))

        return ' '.join(masked_stud_answer), ' '.join(masked_parts)
