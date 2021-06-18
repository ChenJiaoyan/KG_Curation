# This file includes functions that encode the triple by
# 1. labels with word vector
# 2. NodeFeat (occurrence of subject/object in domain/range of the property) + LinkFeat (path with length 1)
# 3. RDF2Vec
import os
import re
import json
import numpy as np
import pickle
import jieba
from gensim.models import Word2Vec, KeyedVectors
from Lib.util_kb import DBpedia_URI_Parse


# labels of (s, p, o) with word2vec
class TripleLabelEncoder(object):

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
            return re.sub('(.)([A-Z][a-z]+)', r'\1 \2 ', DBpedia_URI_Parse(property))

    # preprocess phrase, and transform it into words
    @staticmethod
    def to_lower_words(s):
        s = s.replace('_', ' ').replace('-', ' ').replace('.', ' ').replace('/', ' '). \
            replace('"', ' ').replace("'", ' ').replace('\\', ' ').replace('(', ' ').replace(')', ' ')
        return [word for word in re.split('\W+', s.lower()) if word.isalpha()]

    # cutting or zero padding
    @staticmethod
    def zero_padding(words, size):
        return words[0:size] if len(words) >= size else words + ['NaN'] * (size - len(words))

    # Input: a list of triples, each of which is [subject, property, object]
    # Output: a tensor with size of triple size * (sub_len + prop_len + obj_len) * wv_dim
    def encode(self, target_triples):
        M = np.zeros((len(target_triples), self.sub_len + self.prop_len + self.obj_len, self.wv_dim))
        for i, triple in enumerate(target_triples):
            s, p, o = triple[0], triple[1], triple[2]
            s_words = self.zero_padding(self.to_lower_words(DBpedia_URI_Parse(s)), self.sub_len)
            p_words = self.zero_padding(self.to_lower_words(self.property_parse(p)), self.prop_len)
            l_words = self.zero_padding(self.to_lower_words(DBpedia_URI_Parse(o)), self.obj_len)
            for j, w in enumerate(s_words + p_words + l_words):
                if w == 'NaN' or w not in self.wv_model.wv.vocab:
                    M[i, j, :] = np.zeros(self.wv_dim)
                else:
                    M[i, j, :] = self.wv_model.wv[w.lower()]
        return M


# labels of (s, p, o) with word2vec
class TripleLabelEncoderCN(object):

    # property name is parsed by a mapping file -- p_split_file
    def __init__(self, wv_model_file, seq_lens):
        self.sub_len, self.prop_len, self.obj_len = seq_lens
        self.wv_model = KeyedVectors.load_word2vec_format(wv_model_file, binary=True)
        self.wv_dim = self.wv_model.vector_size

    # cutting or zero padding
    @staticmethod
    def zero_padding(words, size):
        return words[0:size] if len(words) >= size else words + ['NaN'] * (size - len(words))

    # Input: a list of triples, each of which is [subject, property, object]
    # Output: a tensor with size of triple size * (sub_len + prop_len + obj_len) * wv_dim
    def encode(self, target_triples):
        M = np.zeros((len(target_triples), self.sub_len + self.prop_len + self.obj_len, self.wv_dim))
        for i, triple in enumerate(target_triples):
            s, p, o = triple[0], triple[1], triple[2]
            s_words = [token for token in jieba.cut(s)]
            s_words = self.zero_padding(s_words, self.sub_len)
            p_words = [token for token in jieba.cut(p)]
            p_words = self.zero_padding(p_words, self.prop_len)
            o_words = [token for token in jieba.cut(o)]
            o_words = self.zero_padding(o_words, self.obj_len)
            for j, w in enumerate(s_words + p_words + o_words):
                if w == 'NaN' or w not in self.wv_model.wv.vocab:
                    M[i, j, :] = np.zeros(self.wv_dim)
                else:
                    M[i, j, :] = self.wv_model.wv[w]
        return M


# Observed features: NodeFeat and LinkFeat, from the sub-KB
class TripleGraphEncoder(object):

    def __init__(self, properties, triples, triplesStrings, link_feat=True, entity_class_file=None):
        self.properties = properties
        self.LinkFeatDim = len(properties) * 2
        self.link_feat = link_feat

        self.triples = triples
        self.tripleStrings = triplesStrings
        self.pSub_n, self.pObj_n = dict(), dict()
        for t in triples:
            pSub, pObj = '"%s","%s"' % (t[1], t[0]), '"%s","%s"' % (t[1], t[2])
            self.pSub_n[pSub] = self.pSub_n[pSub] + 1 if pSub in self.pSub_n else 1
            self.pObj_n[pObj] = self.pObj_n[pObj] + 1 if pObj in self.pObj_n else 1

        if entity_class_file is not None:
            self.e_class = json.load(open(entity_class_file))
            self.classes = list()
            for e in self.e_class:
                for c in self.e_class[e]:
                    if c not in self.classes:
                        self.classes.append(c)
            self.ClassFeatDim = len(self.classes)*2 + len(self.properties)
        else:
            self.ClassFeatDim = 0


    def NodeFeat(self, t):
        pSub, pObj = '"%s","%s"' % (t[1], t[0]), '"%s","%s"' % (t[1], t[2])
        subV = 1.0 if pSub in self.pSub_n and self.pSub_n[pSub] > 1 else 0.0
        objV = 1.0 if pObj in self.pObj_n and self.pObj_n[pObj] > 1 else 0.0
        return subV, objV

    def LinkFeat(self, t):
        v = np.zeros(self.LinkFeatDim)
        for i, p in enumerate(self.properties):
            if not p == t[1]:
                tStr = '"%s","%s","%s"' % (t[0], p, t[2])
                if tStr in self.tripleStrings:
                    v[i] = 1.0
                tStr = '"%s","%s","%s"' % (t[2], p, t[0])
                if tStr in self.tripleStrings:
                    v[i + len(self.properties)] = 1.0
        return v

    def ClassFeat(self, t):
        s, p, o = t[0], t[1], t[2]
        v = np.zeros(self.ClassFeatDim)

        for s_c in self.e_class[s]:
            s_c_i = self.classes.index(s_c)
            v[s_c_i] = 1.0

        p_i = self.properties.index(p)
        v[len(self.classes) + p_i] = 1.0

        for o_c in self.e_class[o]:
            o_c_i = self.classes.index(o_c)
            v[len(self.classes) + len(self.properties) + o_c_i] = 1.0

        return v


    def encode(self, target_triples):
        target_n = len(target_triples)
        if self.link_feat:
            dim = self.LinkFeatDim + self.ClassFeatDim + 2
        else:
            dim = self.ClassFeatDim + 2

        M = np.zeros((target_n, dim))
        for i, t in enumerate(target_triples):

            if self.link_feat:
                vl = self.LinkFeat(t=t)
                M[i, 0:self.LinkFeatDim] = vl

            if self.ClassFeatDim > 0:
                vc = self.ClassFeat(t=t)
                M[i, self.LinkFeatDim:(self.LinkFeatDim + self.ClassFeatDim)] = vc

       #     subV, objV = self.NodeFeat(t=t)
       #     M[i, -2], M[i, -1] = subV, objV

        return M


# Observed features: NodeFeat and LinkFeat, paths from the whole KB
class TripleGraphEncoderWholeKB(object):
    def __init__(self, graph_link_file, triples):

        self.pSub_n, self.pObj_n = dict(), dict()
        for t in triples:
            pSub, pObj = '"%s","%s"' % (t[1], t[0]), '"%s","%s"' % (t[1], t[2])
            self.pSub_n[pSub] = self.pSub_n[pSub] + 1 if pSub in self.pSub_n else 1
            self.pObj_n[pObj] = self.pObj_n[pObj] + 1 if pObj in self.pObj_n else 1

        self.so_P = json.load(open(graph_link_file))
        self.properties = list()
        for so in self.so_P:
            for p in self.so_P[so]:
                if p not in self.properties:
                    self.properties.append(p)
        self.LinkFeatDim = len(self.properties) * 2


    def NodeFeat(self, t):
        pSub, pObj = '"%s","%s"' % (t[1], t[0]), '"%s","%s"' % (t[1], t[2])
        subV = 1.0 if pSub in self.pSub_n and self.pSub_n[pSub] > 1 else 0.0
        objV = 1.0 if pObj in self.pObj_n and self.pObj_n[pObj] > 1 else 0.0
        return subV, objV


    def LinkFeat(self, t):
        v = np.zeros(self.LinkFeatDim)
        so = '"%s","%s"' % (t[0], t[2])
        for p in self.so_P[so]:
            ind = self.properties.index(p)
            v[ind] = 1.0
        so = '"%s","%s"' % (t[2], t[0])
        for p in self.so_P[so]:
            ind = self.properties.index(p)
            v[len(self.properties) + ind] = 1.0
        return v


    def encode(self, target_triples):
        target_n = len(target_triples)
        dim = self.LinkFeatDim + 2
        M = np.zeros((target_n, dim))
        for i, t in enumerate(target_triples):
            if t[1] in self.properties:
                p_index = self.properties.index(t[1])
                M[i, p_index] = 1

            vl = self.LinkFeat(t=t)
            M[i, 0:self.LinkFeatDim] = vl

            subV, objV = self.NodeFeat(t=t)
            M[i, -2], M[i, -1] = subV, objV

        return M


# RDF2Vec features
class TripleRDF2VecEncoder(object):
    def __init__(self, rdf2vec_file, properties):
        self.rdf2vec_dim, self.e_rdf2vec = pickle.load(open(rdf2vec_file, 'rb'))
        self.properties = properties
        self.p_num = len(self.properties)

    def encode(self, target_triples):
        target_n = len(target_triples)
        dim = self.p_num + 2 * self.rdf2vec_dim
        M = np.zeros((target_n, dim))

        for i, t in enumerate(target_triples):
            s, p, o = t[0], t[1], t[2]
            if p in self.properties:
                p_index = self.properties.index(p)
                M[i, p_index] = 1

            M[i, self.p_num:(self.p_num + self.rdf2vec_dim)] = self.e_rdf2vec[s]
            M[i, (self.p_num + self.rdf2vec_dim):(self.p_num + 2 * self.rdf2vec_dim)] = self.e_rdf2vec[o]

        # return M.reshape((target_n, dim, 1))
        return M


# RDF2Vec features + Observed features: NodeFeat and LinkFeat, from the sub-KB
class TripleGraphRDF2VecEncoder(object):

    def __init__(self, properties, triples, triplesStrings, rdf2vec_file):
        self.rdf2vec_dim, self.e_rdf2vec = pickle.load(open(rdf2vec_file, 'rb'))
        self.properties = properties
        self.p_num = len(self.properties)
        self.LinkFeatDim = len(properties) * 2
        self.triples = triples
        self.tripleStrings = triplesStrings
        self.pSub_n, self.pObj_n = dict(), dict()
        for t in triples:
            pSub, pObj = '"%s","%s"' % (t[1], t[0]), '"%s","%s"' % (t[1], t[2])
            self.pSub_n[pSub] = self.pSub_n[pSub] + 1 if pSub in self.pSub_n else 1
            self.pObj_n[pObj] = self.pObj_n[pObj] + 1 if pObj in self.pObj_n else 1

    def NodeFeat(self, t):
        pSub, pObj = '"%s","%s"' % (t[1], t[0]), '"%s","%s"' % (t[1], t[2])
        subV = 1.0 if pSub in self.pSub_n and self.pSub_n[pSub] > 1 else 0.0
        objV = 1.0 if pObj in self.pObj_n and self.pObj_n[pObj] > 1 else 0.0
        return subV, objV

    def LinkFeat(self, t):
        v = np.zeros(self.LinkFeatDim)
        for i, p in enumerate(self.properties):
            if not p == t[1]:
                tStr = '"%s","%s","%s"' % (t[0], p, t[2])
                if tStr in self.tripleStrings:
                    v[i] = 1.0
                tStr = '"%s","%s","%s"' % (t[2], p, t[0])
                if tStr in self.tripleStrings:
                    v[i + len(self.properties)] = 1.0
        return v

    def encode(self, target_triples):
        target_n = len(target_triples)
        dim = len(self.properties) + self.LinkFeatDim + 2 + 2 * self.rdf2vec_dim
        M = np.zeros((target_n, dim))
        for i, t in enumerate(target_triples):
            s, p, o = t[0], t[1], t[2]
            p_index = self.properties.index(p)
            M[i, p_index] = 1

            vl = self.LinkFeat(t=t)
            M[i, self.p_num:(self.p_num + self.LinkFeatDim)] = vl
            subV, objV = self.NodeFeat(t=t)
            M[i, self.p_num + self.LinkFeatDim], M[i, self.p_num + self.LinkFeatDim + 1] = subV, objV

            tmp_l = self.p_num + self.LinkFeatDim + 2
            M[i, tmp_l:(tmp_l + self.rdf2vec_dim)] = self.e_rdf2vec[s]
            M[i, (tmp_l + self.rdf2vec_dim):(tmp_l + 2 * self.rdf2vec_dim)] = self.e_rdf2vec[o]


        #return M.reshape((target_n, dim, 1))
        return M