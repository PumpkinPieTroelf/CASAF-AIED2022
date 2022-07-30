# CASAF-AIED2022 
The repository for the paper "Towards Generating Counterfactual Examples as Automatic Short Answer Feedback" published at AIED 2022. This code may be used and modified. We would appreciate a citation of the following paper :) 

```
@InProceedings{10.1007/978-3-031-11644-5_17,
author="Filighera, Anna
and Tschesche, Joel
and Steuer, Tim
and Tregel, Thomas
and Wernet, Lisa",
editor="Rodrigo, Maria Mercedes
and Matsuda, Noburu
and Cristea, Alexandra I.
and Dimitrova, Vania",
title="Towards Generating Counterfactual Examples as Automatic Short Answer Feedback",
booktitle="Artificial Intelligence  in Education",
year="2022",
publisher="Springer International Publishing",
address="Cham",
pages="206--217",
abstract="Receiving response-specific, individual improvement suggestions is one of the most helpful forms of feedback for students, especially for short answer questions. However, it is also expensive to construct manually. For this reason, we investigate to which extent counterfactual explanation methods can be used to generate feedback from short answer grading models automatically. Given an incorrect student response, counterfactual models suggest small modifications that would have led the response to being graded as correct. Successful modifications can then be displayed to the learner as improvement suggestions formulated in their own words. As not every response can be corrected with only minor modifications, we investigate the percentage of correctable answers in the automatic short answer grading datasets SciEntsBank, Beetle and SAF. In total, we compare three counterfactual explanation models and a paraphrasing approach. On all datasets, roughly a quarter of incorrect responses can be modified to be classified as correct by an automatic grading model without straying too far from the initial response. However, an expert reevaluation of the modified responses shows that nearly all of them remain incorrect, only fooling the grading model into thinking them correct. While one of the counterfactual generation approaches improved student responses at least partially, the results highlight the general weakness of neural networks to adversarial examples. Thus, we recommend further research with more reliable grading models, for example, by including external knowledge sources or training adversarially.",
isbn="978-3-031-11644-5"
}
```
