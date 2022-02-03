from xml.dom import minidom
from .utils import *

SCIENTSBANK = {
    'train': './data/scientsbank/sciEntsBank_train.xml',
    'ua': './data/scientsbank/sciEntsBank_unseen_answers_3way.xml',
    'uq': './data/scientsbank/sciEntsBank_unseen_questions_3way.xml',
    'ud': './data/scientsbank/sciEntsBank_unseen_domains_3way.xml'
}
BEETLE = {
    'train': './data/beetle/beetle_training_3way.xml',
    'ua': './data/beetle/beetle_unseen_answers_3way.xml',
    'uq': './data/beetle/beetle_unseen_questions_3way.xml'
}

KN1 = {
    'train': './data/kn1/kn1_train.xml',
    'test': './data/kn1/kn1_ua.xml',
    'uq': './data/kn1/kn1_uq.xml',
    'ua': './data/kn1/kn1_ua.xml'

}

DATASETS = {
    'scientsbank': SCIENTSBANK,
    'beetle': BEETLE,
    'kn1': KN1
}


def load_semeval(dataset, keys, add_questions=False):
    """

        :param keys:
        :param dataset:
        :return:
        """

    file = minidom.parse(dataset)
    pairs = list()
    scores = list()
    for task in file.getElementsByTagName('task'):
        question = task.getElementsByTagName('question')
        for reference in task.getElementsByTagName('reference'):
            for answer in task.getElementsByTagName('answer'):
                if add_questions:
                    pairs.append((question[0].firstChild.data, reference.firstChild.data, answer.firstChild.data))
                else:
                    pairs.append((reference.firstChild.data, answer.firstChild.data))
                
                scores.append(keys[answer.attributes['accuracy'].value])

    return {
        'sent_pairs': pairs,
        'scores': scores
    }


def load_data(dataset: str, split='train', add_questions=False):
    if dataset == 'kn1':
        return load_semeval(DATASETS[dataset][split], kn1_keys, add_questions)
    return load_semeval(DATASETS[dataset][split], semeval_keys, add_questions)


def filter_out_correct_answers(dataset, with_questions=False):
    data = {
        'sent_pairs': [],
        'scores': []
    }
    for index, sent_pair in enumerate(dataset['sent_pairs']):
        if dataset['scores'][index] != 2:  # filter out all correct student answers
            if with_questions:
                question = ' '.join(sent_pair[0].split())
                ref = ' '.join(sent_pair[1].split())
                stud = ' '.join(sent_pair[2].split())
                data['sent_pairs'].append((question, ref, stud))
            else:
                ref = ' '.join(sent_pair[0].split()).lower()
                stud = ' '.join(sent_pair[1].split()).lower()
                data['sent_pairs'].append((ref, stud))
            data['scores'].append(dataset['scores'][index])
    return data


def score_to_label(score, dataset='semeval'):
    if dataset == 'kn1':
        return kn1_labels[score]
    return semeval_labels[score]


def clean_answer(answer):
    return ' '.join(answer.split()).lower()


def preprocess_paraphrase_data(data):
    paraphrases = []
    for index, sent_pair in enumerate(data['sent_pairs']):
        if data['scores'][index] == 2:  # filter out all incorrect student answers
            ref = clean_answer(sent_pair[0])
            stud = clean_answer(sent_pair[1])
            paraphrases.append((ref, stud))
            paraphrases.append((stud, ref))
    return paraphrases


def preprocess_editor_data(data, dataset, objective):
    lst = []
    for index, sent_pair in enumerate(data['sent_pairs']):
        if len(sent_pair[1].split()) <= 3:
            continue
        ref = clean_answer(sent_pair[0])
        stud = clean_answer(sent_pair[1])
        if objective == 'label_infill':
            label = score_to_label(data['scores'][index], dataset)
            lst.append((ref, stud, label))
            if data['scores'][index] == 2:
                lst.append((stud, ref, label))
        else:
            if data['scores'][index] == 2:
                lst.append((ref, stud))
                lst.append((stud, ref))

    return lst
