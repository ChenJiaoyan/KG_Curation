# Evaluate the performance of KG embeddings by TransE/DistMult
import os
import sys
import csv
import json
import argparse
import random
import shutil
import numpy as np

current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
parser = argparse.ArgumentParser()
parser.add_argument('--data_file', type=str, default=os.path.join(current_path, 'RData/RData_Annotate.csv'))
parser.add_argument('--lookup_cache_file', type=str, default=os.path.join(current_path, 'RData/Lookup_Cache_Split30.json'))

parser.add_argument('--G_triple_file', type=str, default=os.path.join(current_path, 'RData/G_triples.csv'))
parser.add_argument('--sampling', dest='sampling', action='store_true')
parser.add_argument('--sampling_mask', dest='sampling_mask', action='store_true')
parser.set_defaults(sampling=True)
parser.set_defaults(sampling_mask=True)
parser.add_argument('--sample_file', type=str, default=os.path.join(current_path, 'RData/embedding_sample.txt'))
parser.add_argument('--benchmark_dir', type=str, default=os.path.join(current_path, '../OpenKE/benchmarks/DBP'))

parser.add_argument('--predict_out_file', type=str, default=os.path.join(current_path, 'RData/Predicts_TransE.csv'))
parser.add_argument('--predict_topk', type=int, default=30)
parser.add_argument('--embedding_type', type=str, default='TransE', help='TransE, DistMult')
parser.add_argument('--R_embedding_file', type=str,
                    default=os.path.join(current_path, 'RData/TransE_Embeddings/testembedding_relation_embedding_embeddings.txt'))
parser.add_argument('--E_embedding_file', type=str,
                    default=os.path.join(current_path, 'RData/TransE_Embeddings/testembedding_entity_embedding_embeddings.txt'))
parser.add_argument('--R_mapping_file', type=str,
                    default=os.path.join(current_path, 'RData/TransE_Embeddings/dataRData_r_mapping.txt'))
parser.add_argument('--E_mapping_file', type=str,
                    default=os.path.join(current_path, 'RData/TransE_Embeddings/dataRData_e_mapping.txt'))
parser.add_argument('--embedding_dim', type=int, default=128)
FLAGS, unparsed = parser.parse_known_args()


# sample triples for the sub-graph
# input the sample into an external embedding program
# get the embeddings and use them for prediction

def generate_openke_samples(valid_size, triples):

    entities, relations = list(), list()
    for triple in triples:
        [s, p, o] = triple.split('\t')
        entities.append(s)
        entities.append(o)
        relations.append(p)
    entities = list(set(entities))
    relations = list(set(relations))

    entity2id, relation2id = dict(), dict()
    with open(os.path.join(FLAGS.benchmark_dir, 'entity2id.txt'), 'w') as f:
        f.write('%d\n' % len(entities))
        for id, entity in enumerate(entities):
            entity2id[entity] = id
            f.write('%s\t%d\n' % (entity, id))
    with open(os.path.join(FLAGS.benchmark_dir, 'relation2id.txt'), 'w') as f:
        f.write('%d\n' % len(relations))
        for id, relation in enumerate(relations):
            relation2id[relation] = id
            f.write('%s\t%d\n' % (relation, id))
    print('%d entities, %d relations, written!' % (len(entities), len(relations)))

    def write_id_triples(file_name, save_triples):
        with open(os.path.join(FLAGS.benchmark_dir, file_name), 'w') as ff:
            ff.write('%d\n' % len(save_triples))
            for t in save_triples:
                [ts, tp, to] = t.split('\t')
                sid, oid, pid = entity2id[ts], entity2id[to], relation2id[tp]
                ff.write('%d %d %d\n' % (sid, oid, pid))

    random.shuffle(triples)
    valid_triples = triples[0:valid_size]
    write_id_triples(file_name='valid2id.txt', save_triples=valid_triples)
    shutil.copyfile(os.path.join(FLAGS.benchmark_dir, 'valid2id.txt'),
                    os.path.join(FLAGS.benchmark_dir, 'test2id.txt'))
    print('%d valid/test triples written' % len(valid_triples))

    write_id_triples(file_name='train2id.txt', save_triples=triples)
    print('%d train triples written' % len(triples))


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

    generate_openke_samples(valid_size=50, triples=T)

    sys.exit(1)


# The following are codes for predicting with the embedding,
# with triple scoring functions implemented for TransE and DistMult;
# It can supports external embeddings trained by OpenKE and others


# Read redirected entities and lookuped entities (masked)
with open(FLAGS.lookup_cache_file) as f:
    lookup_entities = json.load(f)


# Load embeddings of entities and properties
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




