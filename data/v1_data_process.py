import nltk
import json
import spacy
import numpy
import pickle
from spacy.tokens import Doc



sent_dic = {0: 'NEU', 1: 'POS', 2: 'NEG'}


class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split()
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)


nlp = spacy.load('en_core_web_sm')
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

def load_sentic_word():

    path = '../senticNet/senticnet_word.txt' 
    senticNet = {}

    fp = open(path, 'r')
    for line in fp:
        line = line.strip()
        if not line:
            continue
        word, sentic = line.split('\t')
        senticNet[word] = sentic
    fp.close()
    return senticNet

def word_to_char_indices(word, char_to_idx, max_char_len=20):
    indices = [char_to_idx.get(c, char_to_idx['<UNK>']) for c in word.lower()]
    return indices

def dependency_adj_matrix(text, senticNet):
    tokens = nlp(text)
    words = text.split()
    matrix = numpy.zeros((len(words), len(words))).astype('float32')
    assert len(words) == len(list(tokens))

    for token in tokens:
        if str(token) in senticNet:  
            sentic = float(senticNet[str(token)]) + 1
        else:  
            sentic = 0
        if token.i < len(words):
            matrix[token.i][token.i] = 1 * sentic
            for child in token.children:
                if child.i < len(words):
                    matrix[token.i][child.i] = 1 * sentic
                    matrix[child.i][token.i] = 1 * sentic

    return matrix


def POS_generate(text):
    tokens = nltk.tokenize.WhitespaceTokenizer().tokenize(text)  
    pos_tags = nltk.pos_tag(tokens) 
    pos = [0] * len(pos_tags)  
    for i, tags in enumerate(pos_tags):
        word, tag = tags  
        if tag.startswith('NN'):  
            pos[i] = 1
        elif tag.startswith('VB'): 
            pos[i] = 2
        elif tag.startswith('JJ'):  
            pos[i] = 3
        elif tag.startswith('RB'):  
            pos[i] = 4
    return pos  


def process(filename):
    senticNet = load_sentic_word()  
    with open(filename[:14] + filename[8:13] + '_pair/' + filename[14:-4] + '_pair.pkl', 'rb') as f:
        pair = pickle.load(f)

    with open(filename, encoding='utf-8') as f:
        text = f.readlines()  

    with open(filename[:14] + '/char2idx.json', 'r') as f:
        char_to_idx = json.load(f)

    assert len(pair) == len(text)  
    count = 0
    for sentence_pack in text:
        data = {'ID': count}  
        sentence = sentence_pack.strip().split('####')[0]
        sentence = sentence.strip()
        data['char_indices'] = [word_to_char_indices(word, char_to_idx) for word in sentence.split()]
        triplets = pair[count]
        data['sentence'] = sentence
        data['tokens'] = str(sentence.split())
        pos = POS_generate(sentence)
        data['pos_tags'] = str(pos)
        data['pairs'] = []
        for triplet in triplets:
            data['pairs'].append(
                [triplet[0][0], triplet[0][-1] + 1, triplet[1][0], triplet[1][-1] + 1, sent_dic[triplet[2]]])
        entities = []
        for entity in data['pairs']:
            tar = [sentence.split()[i] for i in range(entity[0], entity[1])]
            opn = [sentence.split()[i] for i in range(entity[2], entity[3])]
            entities.append(["target", entity[0], entity[1], str(tar), ' '.join(tar)])
            entities.append(["opinion", entity[2], entity[3], str(opn), ' '.join(opn)])
        data['entities'] = entities
        adj_matrix = dependency_adj_matrix(sentence, senticNet)
        adj_matrix = adj_matrix.tolist()  
        data['adj'] = adj_matrix
        all_data.append(data)  
        assert len(adj_matrix) == len(sentence.split())
        assert len(pos) == len(sentence.split())
        count += 1

   
    with open(filename[:-4] + '.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(all_data, ensure_ascii=False, indent=1))
    print(filename[:-4] + ' is ok')


if __name__ == '__main__':
    # V1 14lap
    process('data/V1/14lap/train.txt')
    process('data/V1/14lap/dev.txt')
    process('data/V1/14lap/test.txt')
    # V1 14res
    process('data/V1/14res/train.txt')
    process('data/V1/14res/dev.txt')
    process('data/V1/14res/test.txt')
    # V1 15res
    process('data/V1/15res/train.txt')
    process('data/V1/15res/dev.txt')
    process('data/V1/15res/test.txt')
    # V1 16res
    process('data/V1/16res/train.txt')
    process('data/V1/16res/dev.txt')
    process('data/V1/16res/test.txt')
