import config
import models
import os
import csv
import random
import json
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data_file', type=str, default='../DBP-Lite/Data/RData_Annotate.csv')
parser.add_argument('--lookup_cache_file', type=str, default='../DBP-Lite/Data/Lookup_Cache_Split30.json')
parser.add_argument('--predict_topk', type=int, default=30)

parser.add_argument('--benchmark_dir', type=str, default='./benchmarks/DBP/')
parser.add_argument('--model_file', type=str, default='./res_transe_DBP/model.vec.tf')
parser.add_argument('--model_type', type=str, default='TransE', help='TransE, TransD, TransR, DistMult, RESCAL, TransH, ComplEx, HolE')

parser.add_argument('--predict_out_file', type=str, default='../DBP-Lite/Data/Predicts_TransE.csv')

FLAGS, unparsed = parser.parse_known_args()

os.environ['CUDA_VISIBLE_DEVICES']='7'
con = config.Config()
con.set_in_path(FLAGS.benchmark_dir)
con.set_test_triple_classification(True)
con.set_work_threads(5)
con.set_dimension(100)
con.set_import_files(FLAGS.model_file)
con.init()

if FLAGS.model_type == 'TransE':
    con.set_model(models.TransE)
elif FLAGS.model_type == 'TransD':
    con.set_model(models.TransD)
elif FLAGS.model_type == 'TransR':
    con.set_model(models.TransR)
elif FLAGS.model_type == 'DistMult':
    con.set_model(models.DistMult)
elif FLAGS.model_type == 'RESCAL':
    con.set_model(models.RESCAL)
elif FLAGS.model_type == 'TransH':
    con.set_model(models.TransH)
elif FLAGS.model_type == 'ComplEx':
    con.set_model(models.ComplEx)
elif FLAGS.model_type == 'HolE':
    con.set_model(models.HolE)
else:
    raise NotImplementedError

e_id, p_id, id_e = dict(), dict(), dict()
for line in open(os.path.join(FLAGS.benchmark_dir, 'entity2id.txt')).readlines()[1:]:
    tmp = line.strip().split()
    e_id[tmp[0]] = int(tmp[1])
    id_e[int(tmp[1])] = tmp[0]
for line in open(os.path.join(FLAGS.benchmark_dir, 'relation2id.txt')).readlines()[1:]:
    tmp = line.strip().split()
    p_id[tmp[0]] = int(tmp[1])


# Read redirected entities and lookuped entities (masked)
with open(FLAGS.lookup_cache_file) as f:
    lookup_entities = json.load(f)

# Rank related entities (lookup cache) according the score by embeddings
results = list()
with open(FLAGS.data_file) as f:
    for row in csv.reader(f, delimiter=',', quotechar='"'):
        s, p, l = row[0], row[1], row[2]
        RE = lookup_entities[l]
        random.shuffle(RE)
        result = '"%s","%s","%s"' % (s, p, l)
        ranked_RE = list()
        if s in e_id and p in p_id:
            si, pi = e_id[s], p_id[p]
            res, res_score = con.predict_tail_entity_score(h=si, r=pi, k=len(e_id.keys()))
            for i, item in enumerate(res):
                if id_e[item] in RE:
                    score = np.reciprocal(res_score[i])
                    score = 1 / (1 + np.exp(-score))
                    result = result + (',"%s %.4f"' % (id_e[item], score))
                    ranked_RE.append(id_e[item])
                if len(ranked_RE) >= len(RE):
                    break
        for e in set(RE) - set(ranked_RE):
            result = result + (',"%s 0.0"' % e)

        if len(RE) < FLAGS.predict_topk:
            for i in range(FLAGS.predict_topk - len(RE)):
                result = result + ',""'

        results.append(result)
        print('%s %s %s done' % (s, p, l))


with open(FLAGS.predict_out_file, 'w') as f:
    for result in results:
        f.write('%s\n' % result)
