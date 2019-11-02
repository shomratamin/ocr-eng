import distance
from time import time
import corpus
from difflib import SequenceMatcher

global dictionary
dictionary = corpus.load_dictionary('dictionaries')


def similar(a, b):
    a = a.lower()
    b = b.lower()
    a = a.strip()
    b = b.strip()
    return SequenceMatcher(None, a, b).ratio()


def get_candiate_words(word, limit = 20):
    global dictionary
    if word[0] not in dictionary:
        return []
    search_space = dictionary[word[0]]
    print('search space', len(search_space))
    words = sorted(distance.ifast_comp(word, search_space))[:limit]
    return words


def get_correct_word(word):
    candidates = get_candiate_words(word)
    print(candidates)
    if len(candidates) == 0:
        return word

    max_sim = 0
    max_sim_index = 0
    for i, c in enumerate(candidates):
        _word = c[1]
        similarity = similar(_word,word)
        if similarity > max_sim:
            max_sim = similarity
            max_sim_index = i

    return candidates[max_sim_index][1]



if __name__ == "__main__":
    t1 = time()
    sentence = 'Addess: S/O: Manou Ram Palel, House'
    tokens = corpus.tokenize(sentence)
    print(tokens)
    
    for word in tokens:
        word = word.lower()
        corrected = get_correct_word(word)
        print('original: ', word)
        print('corrected: ',corrected)
    
    t2 = time()
    print('time taken', t2 - t1)