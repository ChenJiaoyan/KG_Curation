# Evaluate the performance of KG embeddings by TransE/DistMult
import os
import sys
import csv
import json
import argparse
import random
import numpy as np

current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
parser = argparse.ArgumentParser()
parser.add_argument('--data_file', type=str, default=os.path.join(current_path, 'Data/Annotate.csv'))
parser.add_argument('--lookup_cache_file', type=str, default=os.path.join(current_path, 'Data/Lookup_Cache_Split30.json'))

parser.add_argument('--sampling', dest='sampling', action='store_true')
parser.add_argument('--sampling_mask', dest='sampling_mask', action='store_true')
parser.set_defaults(sampling=False)
parser.set_defaults(sampling_mask=False)
parser.add_argument('--G_triple_file', type=str, default=os.path.join(current_path, 'Data/G_triples.csv'))
parser.add_argument('--sample_file', type=str, default=os.path.join(current_path, 'Data/embedding_sample.txt'))

parser.add_argument('--predict_out_file', type=str, default=os.path.join(current_path, 'Data/Predicts_TransE.csv'))
parser.add_argument('--predict_topk', type=int, default=30)

parser.add_argument('--embedding_source', type=str, default='Erik', help='Erik or OpenKE')
parser.add_argument('--embedding_type', type=str, default='TransE', help='TransE, DistMult')
parser.add_argument('--R_embedding_file', type=str,
                    default=os.path.join(current_path, 'Data/TransE_Embeddings/testembedding_relation_embedding_embeddings.txt'))
parser.add_argument('--E_embedding_file', type=str,
                    default=os.path.join(current_path, 'Data/TransE_Embeddings/testembedding_entity_embedding_embeddings.txt'))
parser.add_argument('--R_mapping_file', type=str,
                    default=os.path.join(current_path, 'Data/TransE_Embeddings/dataRData_r_mapping.txt'))
parser.add_argument('--E_mapping_file', type=str,
                    default=os.path.join(current_path, 'Data/TransE_Embeddings/dataRData_e_mapping.txt'))

parser.add_argument('--OpenKE_Embedding_dir', type=str, default=os.path.join(current_path, 'Data/OpenKE_DistMult_AllKB'))

parser.add_argument('--embedding_dim', type=int, default=100)
FLAGS, unparsed = parser.parse_known_args()


# sample triples for the sub-graph
# input the sample into an external embedding program
# get the embeddings and use them for prediction
if FLAGS.sampling:
    T = ['%s\t%s\t%s' % (r[0], r[1], r[2]) for r in csv.reader(open(FLAGS.G_triple_file), delimiter=',', quotechar='"')]
    if FLAGS.sampling_mask:
        with open(FLAGS.data_file) as tf:
            for r in csv.reader(tf, delimiter=',', quotechar='"'):
                t = '%s\t%s\t%s' %  (r[0], r[1], r[3])
                if t in T:
                    T.remove(t)
    with open(FLAGS.sample_file, 'w') as sf:
        for t in T:
            sf.write('%s\n' % t)
    sys.exit(1)


# Read redirected entities and lookuped entities (masked)
with open(FLAGS.lookup_cache_file) as f:
    lookup_entities = json.load(f)


# Load embeddings of entities and properties
if FLAGS.embedding_source == 'Erik':
    P, E = np.loadtxt(FLAGS.R_embedding_file), np.loadtxt(FLAGS.E_embedding_file)
    P_v, E_v = dict(), dict()
    with open(FLAGS.R_mapping_file) as f:
        for i, line in enumerate(f.readlines()):
            r = line.strip().split()[0]
            P_v[r] = P[i]
    with open(FLAGS.E_mapping_file) as f:
        for i, line in enumerate(f.readlines()):
            e = line.strip().split()[0]
            E_v[e] = E[i]

elif FLAGS.embedding_source == 'OpenKE':
    e_id, p_id = dict(), dict()
    for line in open(os.path.join(FLAGS.OpenKE_Embedding_dir, 'entity2id.txt')).readlines()[1:]:
        tmp = line.strip().split()
        e_id[tmp[0]] = int(tmp[1])
    for line in open(os.path.join(FLAGS.OpenKE_Embedding_dir, 'relation2id.txt')).readlines()[1:]:
        tmp = line.strip().split()
        p_id[tmp[0]] = int(tmp[1])

    embeddings = json.load(open(os.path.join(FLAGS.OpenKE_Embedding_dir, 'embedding.vec.json')))
    Eembeddings = embeddings['ent_embeddings']
    Pembeddings = embeddings['rel_embeddings']
    E_v, P_v = dict(), dict()
    for e in e_id:
        E_v[e] = np.array(Eembeddings[e_id[e]])
    for p in p_id:
        P_v[p] = np.array(Pembeddings[p_id[p]])

else:
    raise NotImplementedError


''' calculate the score of a triple, according to the embeddings
'''
def triple_score(s_embed, p_embed, o_embed):
    if FLAGS.embedding_type == 'TransE':
        score = np.reciprocal(np.mean(abs(s_embed + p_embed - o_embed)))
    elif FLAGS.embedding_type == 'DistMult':
        score = np.sum(s_embed * p_embed * o_embed)
    else:
        raise NotImplementedError
    score = 1 / (1 + np.exp(-score))
    return score

# Rank related entities (lookup cache) according the score by embeddings
ranks = list()
total_RE_n, miss_RE_n = 0, 0
with open(FLAGS.data_file) as f:
    for row in csv.reader(f, delimiter=',', quotechar='"'):
        s, p, l = row[0], row[1], row[2]
        RE = lookup_entities[l]
        s_v = E_v[s] if s in E_v else np.zeros(FLAGS.embedding_dim)
        p_v = P_v[p] if p in P_v else np.zeros(FLAGS.embedding_dim)
        e_score = dict()
        random.shuffle(RE)
        for i,e in enumerate(RE):
            if i < FLAGS.predict_topk:
                total_RE_n += 1
                if e in E_v:
                    e_v = E_v[e]
                else:
                    miss_RE_n += 1
                    e_v = np.zeros(FLAGS.embedding_dim)
                e_score[e] = triple_score(s_embed=s_v, p_embed=p_v, o_embed=e_v)

        sorted_EScore = sorted(e_score.items(), key=lambda kv: kv[1], reverse=True)
        rank = list()
        for (e,score) in sorted_EScore:
            rank.append("%s %.4f" % (e, score))
        if len(rank) < FLAGS.predict_topk:
            rank += [''] * (FLAGS.predict_topk - len(rank))
        ranks.append([s, p, l] + rank)
print('object embedding missing rate: %.4f (%d/%d)' % (float(miss_RE_n)/float(total_RE_n),
                                                       miss_RE_n, total_RE_n))

with open(FLAGS.predict_out_file, 'w') as f:
    for rank in ranks:
        s = '"%s"' % ('","'.join(rank))
        f.write('%s\n' % s)




