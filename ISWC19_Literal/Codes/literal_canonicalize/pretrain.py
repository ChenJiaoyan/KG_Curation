# get general positive triples from the cache: class_triple.json
# get general negative triples according to a class' siblings
# encode triples into matrices
# pre-train a classifier for each class
import os
import sys
import csv
import json
import random
import argparse
import numpy as np
from Lib.util_kb import URIParse
from Lib.util_wv import TripleEncoder
from Lib.util_ml import rnn_train

current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
parser = argparse.ArgumentParser()
parser.add_argument('--class_file', type=str, default=os.path.join(current_path, 'Cache/class_triple.json'))
parser.add_argument('--class_joint', type=str, default=os.path.join(current_path, 'Cache/class_joint.json'))
parser.add_argument('--property_file', type=str, default=os.path.join(current_path, 'Data/property_split.json'))
parser.add_argument('--nn_type', type=str, default='AttBiRNN', help='AttBiRNN, BiRNN, MLP')
parser.add_argument('--nn_dir', type=str, default=os.path.join(current_path, 'AttBiRNN-50-RData'))
parser.add_argument('--entity_mask_type', type=str, default='NO', help='NO, YES, ALL')
parser.add_argument('--data_file', type=str, default=os.path.join(current_path, 'Data/RData_Clean.csv'))
parser.add_argument('--sequence_lens', type=str, default='12,4,15')
parser.add_argument('--rnn_hidden_size', type=int, default=200)
parser.add_argument('--attention_size', type=int, default=50)
parser.add_argument('--num_epochs', type=int, default=4)
parser.add_argument('--dropout_keep_prob', type=float, default=0.5)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--evaluate_every', type=int, default=100, help='Evaluate model after this many steps')
parser.add_argument('--wv_model_dir', type=str, default='xx/enwiki_model/')
FLAGS, unparsed = parser.parse_known_args()
FLAGS.sequence_lens = [int(i) for i in FLAGS.sequence_lens.split(',')]
print(FLAGS)
if not os.path.exists(FLAGS.nn_dir):
    os.mkdir(FLAGS.nn_dir)

# read entities mask; NO mask for RData; YES and ALL for SData
# In synthetic data, we assume (i) entities for constructing the literals do not exist (YES)
#                           or (ii) entities that are objects of the target properties all do not exist (ALL)
ents_mask = set()
if FLAGS.entity_mask_type == 'YES':
    with open(FLAGS.data_file) as f:
        for row in csv.reader(f, delimiter=',', quotechar='"'):
            ents_mask.add(row[3])

# read class_triple.json
c_triples = json.load(open(FLAGS.class_file))
c_joint = json.load(open(FLAGS.class_joint))

# load word vector encoder
t_encoder = TripleEncoder(wv_model_dir=FLAGS.wv_model_dir, seq_lens=FLAGS.sequence_lens,
                          p_split_file=FLAGS.property_file)

# pre-train a network for each classifier
classes = c_triples.keys()
print('%d classes to do' % len(classes))

for i, c in enumerate(classes):
    print('\n------- %s -------\n' % c)

    # positive and negative triples
    triples_pos = c_triples[c]
    n = len(triples_pos)
    if n == 0:
        print('No positive samples')
        continue
    triples_neg = list()
    c_disjoint = set(classes) - set(c_joint[c])
    for cc in c_disjoint:
        triples_neg += c_triples[cc]
    triples_neg = random.sample(triples_neg, n)

    # encoding
    X_pos = t_encoder.encode(triples=triples_pos)
    X_neg = t_encoder.encode(triples=triples_neg)
    X = np.concatenate((X_pos, X_neg))
    Y_pos, Y_neg = np.zeros((n, 2)), np.zeros((n, 2))
    Y_pos[:, 1], Y_neg[:, 0] = 1.0, 1.0
    Y = np.concatenate((Y_pos, Y_neg))
    shuffle_indices = np.random.permutation(np.arange(X.shape[0]))
    X, Y = X[shuffle_indices], Y[shuffle_indices]
    print('X shape: %s, Y shape: %s' % (str(X.shape), str(Y.shape)))

    # train
    c_nn_dir = os.path.join(FLAGS.nn_dir, URIParse(c))
    rnn_train(x_train=X, y_train=Y, PARAMETERS=FLAGS, rnn_dir=c_nn_dir)

print('All classes done!')
