from nltk.tokenize import word_tokenize
from glob import glob

def load_dictionary(folder):
    print('loading dictionaries')
    files = glob('{}/*.txt'.format(folder))
    tokens = set()
    total_tokens = 0
    unique_tokens = 0
    for _file in files:
        with open(_file, 'r', encoding='utf-8') as f:
            text = f.read()
        text = text.lower()
        _tokens = text.split('\n')
        for _tok in _tokens:
            __tok = _tok.split(' ')
            total_tokens += len(__tok)
            for t in __tok:
                tokens.add(t)
    

    print('dictionaries loaded')
    unique_tokens = len(tokens)
    print('total tokens: ', total_tokens)
    print('unique tokens :', unique_tokens)

    tokens_dic = dict()
    for token in tokens:
        if len(token) == 0:
            continue
        key = token[0]
        if key not in tokens_dic:
            tokens_dic[key] = set([token])
        else:
            tokens_dic[key].add(token)

    return tokens_dic


def tokenize(sentence):
    tokens = word_tokenize(sentence)
    return tokens