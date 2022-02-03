import torch
from captum.attr import LayerConductance, LayerIntegratedGradients


class GradientAttribution:
    """
    The class responsible for handling the computation of the importance scores.
    """

    def __init__(self, tokenizer, device, model):
        self.tokenizer = tokenizer
        self.ref_token_id = self.tokenizer.pad_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.device = device
        self.model = model

    def _summarize_attributions(self, attributions):
        """

        :param attributions:
        :return:
        """
        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        return attributions

    def get_gradient_attr(self, input_ids, ref_input_ids, target_label_id):
        """

        :param input_ids:
        :param ref_input_ids:
        :param token_type_ids:
        :param position_ids:
        :param attention_mas:
        :param target_label_id:
        :return:
        """
        lig = LayerIntegratedGradients(self.custom_forward, self.model.bert.embeddings)

        attributions = lig.attribute(inputs=input_ids, baselines=ref_input_ids, target=target_label_id,
                                     return_convergence_delta=False, n_steps=200,
                                     internal_batch_size=2)

        return self._summarize_attributions(attributions)

    def normalize_importance_attributions(self, features, importance_scores):
        """
        This method ensures that the importance scores which were computed on bert embeddings, are
        put together correctly for the normal words.
        :param features:
        :param importance_scores:
        :return:
        """
        normalized_features = []
        for idx, (f, v) in enumerate(zip(features, importance_scores)):
            f = f.strip('Ġ')
            if not f.startswith("##"):
                key, val = "", 0
            key += f.replace("#", "").strip()
            val += v
            if (idx == len(features) - 1 or (not features[idx + 1].startswith("##"))) and key != "":
                normalized_features.append((key, val))
        return normalized_features

    def custom_forward(self, inputs, token_type_ids=None, position_ids=None, attention_mask=None):
        """

        :param inputs:
        :param token_type_ids:
        :param position_ids:
        :param attention_mask:
        :return:
        """
        preds = self._predict(inputs, token_type_ids, position_ids, attention_mask)
        # decides which target_label we are looking at.
        # [0][0] => incorrect
        # [0][1] => contradiction
        # [0][2] => correct
        return torch.exp(preds) / torch.exp(preds).sum(-1, keepdims=True)

    def _predict(self, inputs, token_type_ids=None, position_ids=None, attention_mask=None):
        """

        :param inputs:
        :param token_type_ids:
        :param position_ids:
        :param attention_mask:
        :return:
        """
        logits = self.model(inputs)[0]
        return logits
