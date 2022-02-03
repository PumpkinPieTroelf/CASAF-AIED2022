semeval_keys = {
    'correct': 2,
    'incorrect': 1,
    'contradictory': 0
}

kn1_keys = {
    'Correct': 2,
    'Partially correct': 1,
    'Incorrect': 0
}
kn1_labels = {
    2: 'Correct',
    1: 'Partially correct',
    0: 'Incorrect'
}
semeval_labels = {
    2: 'correct',
    1: 'incorrect',
    0: 'contradictory'
}


def split_in_answer_types(data, dataset):
    if dataset != 'kn1':
        pc_answers_sample = [pair for pair, s in zip(data['sent_pairs'], data['scores']) if s == 1]
        ic_answers_sample = [pair for pair, s in zip(data['sent_pairs'], data['scores']) if s == 0]
    else:
        pc_answers_sample = [pair for pair, s in zip(data['sent_pairs'], data['scores'])
                             if s == 1 and len(pair[2].split()) > 3]
        ic_answers_sample = [pair for pair, s in zip(data['sent_pairs'], data['scores'])
                             if s == 0 and len(pair[2].split()) > 3]

    return ic_answers_sample, pc_answers_sample
