# Link prediction utilizing labels, with Attentive Bidirectional RNNs
import os
import sys
import csv
import time
import json
import random
import argparse
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

sys.path.append('../')
from Lib.util_encode import TripleLabelEncoder, TripleGraphEncoder, TripleGraphEncoderWholeKB, TripleRDF2VecEncoder, TripleGraphRDF2VecEncoder
from Lib.util_nn import nn_train, nn_predict

current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
parser = argparse.ArgumentParser()
parser.add_argument('--data_file', type=str, default=os.path.join(current_path, 'Data/Annotate.csv'))
parser.add_argument('--G_entity_file', type=str, default=os.path.join(current_path, 'Data/G_entities.csv'))
parser.add_argument('--G_property_file', type=str, default=os.path.join(current_path, 'Data/G_properties.csv'))
parser.add_argument('--G_triple_file', type=str, default=os.path.join(current_path, 'Data/G_triples.csv'))
parser.add_argument('--lookup_cache_file', type=str, default=os.path.join(current_path, 'Data/Lookup_Cache_Split30.json'))
parser.add_argument('--PObject_cache_file', type=str, default=os.path.join(current_path, 'Data/PObject_Cache.json'))
parser.add_argument('--property_file', type=str, default=os.path.join(current_path, 'Data/property_split.json'))
parser.add_argument('--entity_class_file', type=str, default=os.path.join(current_path, 'Data/Entity_Class_Cache.json'))

parser.add_argument('--sampling', dest='sampling', action='store_true')
parser.add_argument('--sampling_mask', dest='sampling_mask', action='store_true')
parser.set_defaults(sampling=False)
parser.set_defaults(sampling_mask=False)
parser.add_argument('--T_sample_file', type=str, default=os.path.join(current_path, 'Data/T_sample.pkl'))
parser.add_argument('--negative_sample_cache', type=int, default=5)

parser.add_argument('--train', dest='train', action='store_true')
parser.set_defaults(train=True)
parser.add_argument('--feature_type', type=str, default='node_path', help='label, node_path, path_kb, rdf2vec, path_rdf2vec')
parser.add_argument('--graph_link_file', type=str, default=os.path.join(current_path, 'Data/graph_link.json'))
parser.add_argument('--graph_node_po_file', type=str, default=os.path.join(current_path, 'Data/graph_node_po.json'))
parser.add_argument('--graph_node_ps_file', type=str, default=os.path.join(current_path, 'Data/graph_node_ps.json'))
parser.add_argument('--RDF2Vec_Cache_file', type=str, default=os.path.join(current_path, 'Data/G_RDF2Vec.pkl'))
parser.add_argument('--nn_type', type=str, default='MLP', help='MLP, BiRNN, AttBiRNN, RF')
parser.add_argument('--nn_dir', type=str, default=os.path.join(current_path, 'Data/Model/NN'))
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--evaluate_every', type=int, default=0)
parser.add_argument('--dropout_keep_prob', type=float, default=0.5)
parser.add_argument('--rnn_hidden_size', type=int, default=200)
parser.add_argument('--attention_size', type=int, default=50)
parser.add_argument('--mlp_hidden_size', type=int, default=0)
parser.add_argument('--wv_model_dir', type=str, default='w2v_model/enwiki_model/', help='trained word2vec model')
parser.add_argument('--sequence_lens', type=str, default='12,4,15')

parser.add_argument('--predict', dest='predict', action='store_true')
parser.set_defaults(predict=True)
parser.add_argument('--predict_topk', type=int, default=30)
parser.add_argument('--predict_out_file', type=str, default=os.path.join(current_path, 'Data/Predicts_Tmp.csv'))

FLAGS, unparsed = parser.parse_known_args()
FLAGS.sequence_lens = [int(i) for i in FLAGS.sequence_lens.split(',')]

# load target triples
entity_cache = json.load(open(FLAGS.lookup_cache_file))
Es, Em, lit_Em, T_target = set(), set(), dict(), list()
with open(FLAGS.data_file) as f:
    for row in csv.reader(f, delimiter=',', quotechar='"'):
        s, p, l = row[0], row[1], row[2]
        T_target.append([s, p, l])
        Es.add(s)
        Em.update(set(entity_cache[l]))
        lit_Em[l] = set(entity_cache[l])

# load sub-KB
P = [line.strip() for line in open(FLAGS.G_property_file).readlines()]
E = [line.strip() for line in open(FLAGS.G_entity_file).readlines()]
T = [[r[0], r[1], r[2]] for r in csv.reader(open(FLAGS.G_triple_file), delimiter=',', quotechar='"')]

# For synthetic data, we assume the triples used for data construction are excluded
if FLAGS.sampling_mask:
    with open(FLAGS.data_file) as tf:
        for r in csv.reader(tf, delimiter=',', quotechar='"'):
            t = [r[0], r[1], r[3]]
            if t in T:
                T.remove(t)
TStrings = set()
for t in T:
    TStrings.add('"%s","%s","%s"' % (t[0], t[1], t[2]))
print('E: %d, P: %d, T: %d, Es #: %d, Em #: %d, T_target #: %d' %
      (len(E), len(P), len(T), len(Es), len(Em), len(T_target)))


def train_sampling():
    # positive samples, Tsp: triples whose subjects are among Es, Top: triples whose objects are among Em
    Tsp, Top = list(), list()
    for t in T:
        if t[0] in Es:
            Tsp.append(t)
        if t[2] in Em:
            Top.append(t)
    print('Tsp #: %d, Top #: %d' % (len(Tsp), len(Top)))
    Tspop = Tsp + Top

    # negative sample: replace the object of triples in Tsp, replace the subject of triple in Top
    T_syn = list()
    for t in Tsp:
        for e in random.sample(E, FLAGS.negative_sample_cache):
            t_syn = [t[0], t[1], e]
            if '"%s","%s","%s"' % (t_syn[0], t_syn[1], t_syn[2]) not in TStrings:
                T_syn.append(t_syn)
    for t in Top:
        for e in random.sample(E, FLAGS.negative_sample_cache):
            t_syn = [e, t[1], t[2]]
            if '"%s","%s","%s"' % (t_syn[0], t_syn[1], t_syn[2]) not in TStrings:
                T_syn.append(t_syn)
    if len(T_syn) < len(Tspop):
        raise Exception('Synthetic (negative) samples #: %d;'
                        'There should be more negative (synthetic) samples than positive samples; '
                        'Try to set NEG_CACHE larger.' % len(T_syn))
    T_syn = random.sample(T_syn, len(Tspop))

    return Tspop, T_syn


def predict_sampling():
    triple_Candidate = dict()
    for triple in T_target:
        sub, prop, lit = triple[0], triple[1], triple[2]
        candidate = [[sub, prop, e] for e in lit_Em[lit]]
        triple_key = '"%s","%s","%s"' % (sub, prop, lit)
        triple_Candidate[triple_key] = candidate
    return triple_Candidate


if FLAGS.feature_type.lower() == 'node_path':
    t_encoder = TripleGraphEncoder(properties=P, triples=T, triplesStrings=TStrings, link_feat=True)
elif FLAGS.feature_type.lower() == 'path_kb':
    t_encoder = TripleGraphEncoderWholeKB(graph_link_file=FLAGS.graph_link_file,triples=T)
elif FLAGS.feature_type.lower() == 'label':
    t_encoder = TripleLabelEncoder(wv_model_dir=FLAGS.wv_model_dir, seq_lens=FLAGS.sequence_lens,
                                   p_split_file=FLAGS.property_file)
elif FLAGS.feature_type.lower() == 'rdf2vec':
    t_encoder = TripleRDF2VecEncoder(rdf2vec_file=FLAGS.RDF2Vec_Cache_file, properties=P)
elif FLAGS.feature_type.lower() == 'path_rdf2vec':
    t_encoder = TripleGraphRDF2VecEncoder(rdf2vec_file=FLAGS.RDF2Vec_Cache_file,
                                          properties=P, triples=T, triplesStrings=TStrings)
else:
    raise Exception('feature type "%s" unimplemented' % FLAGS.feature_type)


if FLAGS.train:
    if FLAGS.sampling:
        print('sampling ...')
        start = time.time()
        T_pos, T_neg = train_sampling()
        elapsed = (time.time() - start)
        with open(FLAGS.T_sample_file, 'wb') as f:
            pickle.dump([T_pos, T_neg], f)
        print("%.1f minutes for sampling" % (elapsed/60.0))
    else:
        T_pos, T_neg = pickle.load(open(FLAGS.T_sample_file, 'rb'))

    print('encoding positive samples')
    X_pos = t_encoder.encode(target_triples=T_pos)
    print('encoding negative samples')
    X_neg = t_encoder.encode(target_triples=T_neg)
    X = np.concatenate((X_pos, X_neg))
    Y_pos, Y_neg = np.zeros((len(T_pos), 2)), np.zeros((len(T_neg), 2))
    Y_pos[:, 1], Y_neg[:, 0] = 1.0, 1.0
    Y = np.concatenate((Y_pos, Y_neg))
    shuffle_indices = np.random.permutation(np.arange(X.shape[0]))
    X, Y = X[shuffle_indices], Y[shuffle_indices]
    print('X shape: %s, Y shape: %s' % (str(X.shape), str(Y.shape)))

    print('training ...')
    start = time.time()
    if FLAGS.nn_type == 'RF':
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X, np.argmax(Y, axis=1))
    else:
        nn_train(x_train=X, y_train=Y, PARAMETERS=FLAGS)
    elapsed = (time.time() - start)
    print("%.1f minutes for training" % (elapsed/60.0))


if FLAGS.predict:
    if os.path.exists(FLAGS.predict_out_file):
        os.remove(FLAGS.predict_out_file)
    t_Candidate = predict_sampling()
    print('predicting ...')
    start = time.time()
    for index, t_key in enumerate(t_Candidate):
        candidates  = t_Candidate[t_key]
        random.shuffle(candidates)
        if len(candidates) > 0:
            X = t_encoder.encode(target_triples=candidates)
            if FLAGS.nn_type == 'RF':
                s = clf.predict_proba(X)[:, 1]
            else:
                s = nn_predict(test_x=X, PARAMETERS=FLAGS)[:, 1]
            topk = FLAGS.predict_topk if FLAGS.predict_topk <= len(s) else len(s)
            i_topk = np.argpartition(s, -topk)[-topk:]
            i_topk = i_topk[np.argsort(s[i_topk])][::-1]
            R_topk = ['%s %.4f' % (candidates[i][2],s[i]) for i in i_topk]
            res = R_topk + [''] * (FLAGS.predict_topk - topk)
        else:
            res = [''] * FLAGS.predict_topk
        with open(FLAGS.predict_out_file, 'a+') as f:
            f.write('%s,"%s"\n' % (t_key, '","'.join(res)))
        if index % 500 == 0:
            print('%.1f%% done' % (100 * float(index)/len(t_Candidate.keys())))

    elapsed = (time.time() - start)
    print("%.1f minutes for prediction" % (elapsed/60.0))
