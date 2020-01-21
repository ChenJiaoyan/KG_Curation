# read RDF2vec for entities
import os
import sys
import argparse
import pickle
import numpy as np
from gensim.models import Word2Vec

current_path = os.path.dirname(os.path.realpath(sys.argv[0]))

parser = argparse.ArgumentParser()
parser.add_argument('--G_entity_file', type=str, default=os.path.join(current_path, 'Data/G_entities.csv'))
parser.add_argument('--RDF2Vec_Model_file', type=str, default=os.path.join(current_path, ''))
parser.add_argument('--RDF2Vec_Cache_file', type=str, default=os.path.join(current_path, 'Data/G_RDF2Vec.pkl'))
FLAGS, unparsed = parser.parse_known_args()

rdf2vec = Word2Vec.load(FLAGS.RDF2Vec_Model_file)
dim = rdf2vec.vector_size

E = [line.strip() for line in open(FLAGS.G_entity_file).readlines()]
print('E: %d' % len(E))

e_rdf2vec = dict()
num = 0
for i, e in enumerate(E):
    es = '<%s>' % e
    if es in rdf2vec.wv.vocab:
        v = rdf2vec.wv[es]
        num += 1
    else:
        v = np.zeros(dim)
    e_rdf2vec[e] = v
    if i % 10000 == 0:
        print('%.1f%% done' % (100 * float(i)/float(len(E))))

with open(FLAGS.RDF2Vec_Cache_file, 'wb') as f:
    pickle.dump([dim, e_rdf2vec], f)

print('%d (%.1f%%) entities have embeddings' % (num, (100 * float(num)/len(E))) )