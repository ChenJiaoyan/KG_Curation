# cache topK entities according to the entity label
import os
import re
import sys
import csv
import json
import argparse
import numpy as np
from gensim.models import Word2Vec

current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
parser = argparse.ArgumentParser()
parser.add_argument('--data_file', type=str, default=os.path.join(current_path, 'Data/Annotate.csv'))
parser.add_argument('--label_file', type=str, default='labels_en_small.ttl', help='DBpedia label dump')
parser.add_argument('--wv_model_dir', type=str, default='w2v_model/enwiki_model/', help='trained word2vec model')
parser.add_argument('--topK', type=int, default=30)
parser.add_argument('--start_index', type=int, default=0)
parser.add_argument('--end_index', type=int, default=1000)
parser.add_argument('--cache_file', type=str, default=os.path.join(current_path, 'Data/Word2Vec_Cache_100_0_1000.json'))

parser.add_argument('--merging', dest='merging', action='store_true')
parser.set_defaults(merging=True)
parser.add_argument('--word2vec_cache_dir', type=str, default=os.path.join(current_path, 'Data/Word2Vec_Cache'))
parser.add_argument('--word2vec_cache_file', type=str, default=os.path.join(current_path, 'Data/Word2Vec_Cache.json'))
FLAGS, unparsed = parser.parse_known_args()


if FLAGS.merging:
    cache_merge = dict()
    cache_files = os.listdir(FLAGS.word2vec_cache_dir)
    for cache_file in cache_files:
        cache = json.load(open(os.path.join(FLAGS.word2vec_cache_dir, cache_file)))
        for k in cache:
            cache_merge[k] = cache[k]
    with open(FLAGS.word2vec_cache_file, 'w') as out_f:
        json.dump(cache_merge, out_f)
    sys.exit(0)

lits = list()
with open(FLAGS.data_file) as f:
    for row in csv.reader(f, delimiter=',', quotechar='"'):
        l = row[2]
        if l not in lits:
            lits.append(l)
lits = lits[FLAGS.start_index:FLAGS.end_index]
print('start_index: %d, end_index: %d' % (FLAGS.start_index, FLAGS.end_index))

wv_model = Word2Vec.load(os.path.join(FLAGS.wv_model_dir, 'word2vec_gensim'))

def word2vec_encoder(s):
    v = np.zeros(wv_model.vector_size)
    s = s.replace('_', ' ').replace('-', ' ').replace('.', ' ').replace('/', ' '). \
        replace('"', ' ').replace("'", ' ').replace('\\', ' ').replace('(', ' ').replace(')', ' ')
    words = [word for word in re.split('\W+', s.lower()) if word.isalpha()]
    n = 0
    for w in words:
        if w in wv_model.wv.vocab:
            v += wv_model.wv[w]
            n += 1
    if n > 0:
        return v/n
    else:
        return v


def extract_ent_label(t_str):
    ent = t_str.split(' ')[0][1:-1]
    lab = t_str[(t_str.index('"')+1) : (t_str.rindex('"'))]
    return ent, lab

def insert(arr, ent, score):
    if len(arr) == 0:
        arr.append((ent, score))
    else:
        flag = False
        if arr[-1][1] < score:
            for j, item in enumerate(arr):
                if score > item[1]:
                    arr.insert(j,(ent,score))
                    flag = True
                    break
        if not flag:
            arr = arr + [(ent, score)]
    return arr[0:FLAGS.topK] if len(arr) > FLAGS.topK else arr

with open(FLAGS.label_file, 'r') as f:
    lines = f.readlines()
    lines = lines[1:-1]
    print('%d lines' % len(lines))

if os.path.exists(FLAGS.cache_file):
    lit_ents = json.load(FLAGS.cache_file)
else:
    lit_ents = dict()

for i, lit in enumerate(lits):
    if lit not in lit_ents:
        lit_ents[lit] = list()

        v_lit = word2vec_encoder(s=lit)
        if np.count_nonzero(v_lit) > 0:
            entScores = list()
            for line in lines:
                entity, label = extract_ent_label(t_str=line)
                v_label = word2vec_encoder(s=label)
                if np.count_nonzero(v_label) > 0:
                    sim = np.dot(v_label, v_lit) / (np.linalg.norm(v_label) * np.linalg.norm(v_lit))
                    entScores = insert(arr=entScores, ent=entity, score=sim)

            for entScore in entScores:
                lit_ents[lit].append(entScore[0])

        with open(FLAGS.cache_file, 'w') as out_f:
            json.dump(lit_ents, out_f)

    print('i=%d, done' % i)

