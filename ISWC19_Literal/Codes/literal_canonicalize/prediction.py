# For each property:
#   fine tune the networks of candidate classes (optional)
#   predict the types of the literal of each triple: one score per candidate class
#   determine the annotation classes
# calculate the metrics -- mean precision, recall, F1
import os
import sys
import csv
import json
import argparse
import numpy as np
from Lib.util_wv import TripleEncoder, URIParse
from Lib.util_ml import rnn_predict

current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
parser = argparse.ArgumentParser()
parser.add_argument('--data_file', type=str, default=os.path.join(current_path, 'Data/RData_Clean.csv'))
parser.add_argument('--candidate_classes', type=str, default=os.path.join(current_path, 'RData_Classes.json'))
parser.add_argument('--out_score_file', type=str, default=os.path.join(current_path, 'RData_Scores_AttBiRNN_FTF.json'))
parser.add_argument('--need_finetune', type=str, default='Yes', help='Yes, No')
parser.add_argument('--particular_sample_file', type=str,
                    default=os.path.join(current_path, 'RData_PSamples_fixed.json'))
parser.add_argument('--class_descendant', type=str, default=os.path.join(current_path, 'Cache/class_descendant.json'))
parser.add_argument('--property_file', type=str, default=os.path.join(current_path, 'Data/property_split.json'))
parser.add_argument('--sequence_lens', type=str, default='12,4,15')
parser.add_argument('--nn_dir', type=str, default=os.path.join(current_path, 'AttBiRNN-50-RData'))
parser.add_argument('--wv_model_dir', type=str, default='xx/enwiki_model/')
parser.add_argument('--num_epochs', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--dropout_keep_prob', type=float, default=0.5)
FLAGS, unparsed = parser.parse_known_args()
FLAGS.sequence_lens = [int(i) for i in FLAGS.sequence_lens.split(',')]
FLAGS.need_finetune = True if FLAGS.need_finetune == 'Yes' else False
print('Need fine tuning: %s' % FLAGS.need_finetune)

# read properties and their triples
p_triples = dict()
with open(FLAGS.data_file) as f:
    for row in csv.reader(f, delimiter=',', quotechar='"'):
        p = row[1]
        triple = [row[0], row[1], row[2]]
        if p in p_triples:
            p_triples[p].append(triple)
        else:
            p_triples[p] = [triple]

# load candidate classes and particular samples
p_classes = json.load(open(FLAGS.candidate_classes))
p_PSamples = json.load(open(FLAGS.particular_sample_file))

# transform triples into a matrix
t_encoder = TripleEncoder(wv_model_dir=FLAGS.wv_model_dir, seq_lens=FLAGS.sequence_lens,
                          p_split_file=FLAGS.property_file)

# property by property
t_c_score = dict()
for p in p_triples:
    print('\n------- %s -------\n' % p)
    triples = p_triples[p]
    X = t_encoder.encode(triples=triples, triple_format='spl')
    classes = p_classes[p]['C']
    print('\t%d classes' % len(classes))

    c_scores = dict()
    for c in classes:
        rnn_dir = os.path.join(FLAGS.nn_dir, URIParse(c))
        print('\t%s' % c)

        if os.path.exists(rnn_dir):

            if '%s POS' % c not in p_PSamples[p] or len(p_PSamples[p]['%s POS' % c]) == 0:
                print('\t class ignored')
                continue

            # fine tune the network, then predict
            if FLAGS.need_finetune:
                triples_pos, triples_neg = p_PSamples[p]['%s POS' % c], p_PSamples[p]['%s NEG' % c]
                X_pos = t_encoder.encode(triples=triples_pos, triple_format='spl')
                X_neg = t_encoder.encode(triples=triples_neg, triple_format='spl')
                X_ft = np.concatenate((X_pos, X_neg))
                Y_pos, Y_neg = np.zeros((len(triples_pos), 2)), np.zeros((len(triples_neg), 2))
                Y_pos[:, 1], Y_neg[:, 0] = 1.0, 1.0
                Y_ft = np.concatenate((Y_pos, Y_neg))
                shuffle_indices = np.random.permutation(np.arange(X_ft.shape[0]))
                X_ft, Y_ft = X_ft[shuffle_indices], Y_ft[shuffle_indices]
                c_scores[c], alpha = rnn_predict(test_x=X, rnn_dir=rnn_dir, need_ft=True, x_ft=X_ft, y_ft=Y_ft,
                                                 batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs,
                                                 dropout_keep_prob=FLAGS.dropout_keep_prob)
            # directly predict
            else:
                c_scores[c], alpha = rnn_predict(test_x=X, rnn_dir=rnn_dir)

            # print('\n---attentions of %s---' % c)
            # print(alpha)

        else:
            print('\tno pre-trained network')

    for i, triple in enumerate(triples):
        triple_s = ' '.join(triple)
        c_score = dict()
        for c in c_scores:
            c_score[c] = float(c_scores[c][i][1])
        t_c_score[triple_s] = c_score

print('Prediction done!\n')

with open(FLAGS.out_score_file, 'w') as out_f:
    json.dump(t_c_score, out_f)
