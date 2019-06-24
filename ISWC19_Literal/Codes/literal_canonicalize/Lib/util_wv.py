# This file includes functions that encode the triple by word vector models
import os
import re
import json
import numpy as np
from gensim.models import Word2Vec
from util_kb import URIParse
from pattern.text.en import tokenize

# preprocess phrase, and transform it into words
def to_lower_words(s):
    s = s.replace('_', ' ').replace('-', ' ').replace('.', ' ').replace('/', ' '). \
        replace('"', ' ').replace("'", ' ').replace('\\', ' ').replace('(', ' ').replace(')', ' ')
    tokenized_line = ' '.join(tokenize(s))
    return [word for word in tokenized_line.lower().split() if word.isalpha()]

# cutting or zero padding
def zero_padding(words, size):
    return words[0:size] if len(words) >= size else words + ['NaN'] * (size - len(words))

# encoder of triples in the form of (subject, property, literal)
class TripleEncoder(object):

    # property name is parsed by a mapping file -- p_split_file
    def __init__(self, wv_model_dir, seq_lens, p_split_file):
        self.sub_len, self.prop_len, self.obj_len = seq_lens
        self.wv_model = Word2Vec.load(os.path.join(wv_model_dir, 'word2vec_gensim'))
        self.wv_dim = self.wv_model.vector_size
        self.p_split = json.load(open(p_split_file))

    # parse a property
    def property_parse(self, property):
        if property in self.p_split:
            return self.p_split[property]
        else:
            return re.sub('(.)([A-Z][a-z]+)', r'\1 \2 ', URIParse(property))

    # Input: a list of triples
    # Output: a tensor with size of triple size * (sub_len + prop_len + obj_len) * wv_dim
    def encode(self, triples, triple_format='spol'):
        M = np.zeros((len(triples), self.sub_len + self.prop_len + self.obj_len, self.wv_dim))
        for i, triple in enumerate(triples):
            if triple_format == 'spol':
                s, p, l = triple[0], triple[1], triple[3]
            elif triple_format == 'spl':
                s, p, l = triple
            s_words = zero_padding(to_lower_words(URIParse(s)), self.sub_len)
            p_words = zero_padding(to_lower_words(self.property_parse(p)), self.prop_len)
            l_words = zero_padding(to_lower_words(l), self.obj_len)
            for j, w in enumerate(s_words + p_words + l_words):
                if w == 'NaN' or w not in self.wv_model.wv.vocab:
                    M[i, j, :] = np.zeros(self.wv_dim)
                else:
                    M[i, j, :] = self.wv_model.wv[w.lower()]
        return M
