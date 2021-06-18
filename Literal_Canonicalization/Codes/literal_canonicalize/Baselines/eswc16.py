# this is to get the head of each literal
# run StanfordCoreNLP server:
#   cd stanford-corenlp-full-2018-10-05
#   export CLASSPATH="`find . -name '*.jar'`"
#   java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer
# Cache of DBO class and class labeled saved in ../Cache/class_label.json
import os
import json
import csv
import numpy as np
from nltk.tree import Tree
from pycorenlp import StanfordCoreNLP
from requests import get
from gensim.models import Word2Vec
from Lib.util_kb import lookup_entities_raw_with_sleep, URIParse, queryClassByEntity, queryAncestorByClass


def find_noun_phrases(tree):
    return [subtree for subtree in tree.subtrees(lambda t: t.label() == 'NP')]


def find_head_of_np(np):
    noun_tags = ['NN', 'NNS', 'NNP', 'NNPS']
    top_level_trees = [np[i] for i in range(len(np)) if type(np[i]) is Tree]
    top_level_nouns = [t for t in top_level_trees if t.label() in noun_tags]
    if len(top_level_nouns) > 0:
        return top_level_nouns[-1][0]
    else:
        top_level_nps = [t for t in top_level_trees if t.label() == 'NP']
        if len(top_level_nps) > 0:
            return find_head_of_np(top_level_nps[-1])
        else:
            nouns = [p[0] for p in np.pos() if p[1] in noun_tags]
            if len(nouns) > 0:
                return nouns[-1]
            else:
                return np.leaves()[-1]


def getTree(text):
    nlp = StanfordCoreNLP('http://localhost:9000')
    output = nlp.annotate(text, properties={
        'annotators': 'parse',
        'outputFormat': 'json'
    })
    return output['sentences'][0]['parse']


def getHead(tree):
    for np in find_noun_phrases(tree):
        # print "noun phrase:",
        # print " ".join(np.leaves())
        head = find_head_of_np(np)
        return head


def getNgrams(focus_term, text):
    s = text.replace('_', ' ').replace('-', ' ').replace('.', ' ').replace('/', ' '). \
        replace('"', ' ').replace("'", ' ').replace('\\', ' ').replace('(', ' ').replace(')', ' ')
    words = [word for word in s.lower().split() if word.isalpha()]
    ngrams = []
    if len(words) <= GRAM_NUM:
        ngrams.append(' '.join(words))
    else:
        for index, word in enumerate(words):
            if index + GRAM_NUM < len(words):
                ng = ' '.join(words[index:(index + GRAM_NUM + 1)])
                if focus_term.lower() in ng:
                    ngrams.append(ng)
    return ngrams


def getTypesFromLabel(focusTerm):
    label_classes = set()
    for c in c_label:
        if focusTerm.lower() == c_label[c].lower():
            label_classes.add(c)
    return label_classes


def getTypesFromNGrams(ngrams):
    ngram_classes = set()
    for ng in ngrams:
        for cls in c_label:
            if ng.lower() == c_label[cls].lower():
                ngram_classes.add(cls)

    if len(ngram_classes) == 0:
        for ng in ngrams:
            ents = lookup_entities_raw_with_sleep(text=ng, top_k=5)

            ents_filter = list()
            if ENTITY_MASK:
                for ent in ents:
                    if ent not in entities_mask:
                        ents_filter.append(ent)
            else:
                ents_filter = ents

            for ent in ents_filter:
                if ENTITY_LOOKUP_TYPE == 'EXACT':
                    if ng.lower() == URIParse(ent).replace('_', ' ').lower():
                        for cls in queryClassByEntity(e=ent):
                            ngram_classes.add(str(cls))
                else:
                    for cls in queryClassByEntity(e=ent):
                        ngram_classes.add(str(cls))

    return ngram_classes


def getUMBCSimilarity(s1, s2, umbc_type='relation', corpus='webbase'):
    sss_url = "http://swoogle.umbc.edu/SimService/GetSimilarity"
    try:
        response = get(sss_url, params={'operation': 'api', 'phrase1': s1, 'phrase2': s2,
                                        'type': umbc_type, 'corpus': corpus})
        return float(response.text.strip())
    except:
        print 'Error in getting similarity for %s: %s' % ((s1, s2), response)
        return 0.0


def getWord2VecSimilarity(s1, s2):
    d = wv_model.vector_size
    v1, v2 = np.zeros(d), np.zeros(d)
    n1, n2 = 0, 0
    for word in s1.split():
        if word.lower() in wv_model.wv.vocab:
            v1 += wv_model.wv[word.lower()]
            n1 += 1
    for word in s2.split():
        if word.lower() in wv_model.wv.vocab:
            v2 += wv_model.wv[word.lower()]
            n2 += 1
    if n1 > 0:
        v1 = v1 / n1
    if n2 > 0:
        v2 = v2 / n2
    cosine_similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)) if n1 > 0 and n2 > 0 else 0.0
    return cosine_similarity


def getMatchedType(focusTerm):
    c_selected, sim = '', 0
    for c in c_label:
        # umbc_s = getUMBCSimilarity(s1=c_label[c], s2=focusTerm)
        w2v_s = getWord2VecSimilarity(s1=c_label[c], s2=focusTerm)
        if w2v_s >= sim:
            c_selected, sim = c, w2v_s
    return set() if c_selected == '' else {c_selected}


# 'EXACT' or 'SOFT'
ENTITY_LOOKUP_TYPE = 'SOFT'

# 'Word2Vec' or 'UMBC'
TYPE_SIMILARITY_TYPE = 'Word2Vec'

# "SData" or "RData"
DATA_NAME = 'RData'

GRAM_NUM = 2

# mask the entities used for generating SData
ENTITY_MASK = True

eORt_gtType = json.load(open('../Data/%s_Type.json' % DATA_NAME))
c_label = json.load(open('../Cache/class_label.json'))
wv_model = Word2Vec.load(os.path.join('/Users/jiahen/Data/TableAnnotate/w2v_model/enwiki_model/', 'word2vec_gensim'))

c_ancestors = dict()
data_file = '../Data/RData_Clean.csv'
quads = list()
entities_mask = set()
with open(data_file) as f:
    for row in csv.reader(f, delimiter=',', quotechar='"'):
        if DATA_NAME == 'SData':
            quads.append([row[0], row[1], row[2], row[3]])
            entities_mask.add(row[3])
        else:
            quads.append([row[0], row[1], row[2]])

precision, recall, f1 = 0.0, 0.0, 0.0
for i, quad in enumerate(quads):
    l = quad[2]
    tree_s = getTree(l)
    tr = Tree.fromstring(tree_s)
    h = getHead(tr)

    if h is not None and len(h) > 0:
        classes = getTypesFromLabel(focusTerm=h)
        if len(classes) == 0:
            ngs = getNgrams(focus_term=h, text=l)
            classes = getTypesFromNGrams(ngrams=ngs)
            if len(classes) == 0:
                classes = getMatchedType(focusTerm=h)
    else:
        classes = set()

    classes_a = set()
    for c in classes:
        classes_a.add(c)
        if c in c_ancestors:
            ancestors = c_ancestors[c]
        else:
            ancestors = [str(a) for a in queryAncestorByClass(c)]
            c_ancestors[c] = ancestors
        for a in ancestors:
            classes_a.add(a)

    if DATA_NAME == 'SData':
        e = quad[3]
        classes_gt = set(eORt_gtType[e])
    else:
        triple_s = ' '.join(quad[0:3])
        classes_gt = set(eORt_gtType[triple_s])

    mP = float(len(classes_a.intersection(classes_gt))) / float(len(classes_a)) if len(classes_a) > 0 else 0.0
    mR = float(len(classes_a.intersection(classes_gt))) / float(len(classes_gt))
    mF1 = (2 * mP * mR) / (mP + mR) if mP + mR > 0 else 0

    precision += mP
    recall += mR
    f1 += mF1
    print('%d, %s done, %.4f, %.4f, %.4f' % (i, l, precision, recall, f1))

precision = precision / len(quads)
recall = recall / len(quads)
f1 = f1 / len(quads)

print('precision: %.4f, recall: %.4f, F1 score: %.4f' % (precision, recall, f1))
