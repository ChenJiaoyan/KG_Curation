# predict by lookup with the literal
# entities are ordered by lexical similarity
import os
import sys
import csv
import json
import argparse

current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
parser = argparse.ArgumentParser()
parser.add_argument('--data_file', type=str, default=os.path.join(current_path, 'Data/Annotate.csv'))
parser.add_argument('--predict_out_file', type=str, default=os.path.join(current_path, 'Data/Predicts_LP_Split.csv'))
parser.add_argument('--lookup_cache_file', type=str, default=os.path.join(current_path, 'Data/Lookup_Cache_Split30.json'))
parser.add_argument('--predict_topk', type=int, default=30)
FLAGS, unparsed = parser.parse_known_args()

t_literal = dict()
with open(FLAGS.data_file) as f:
    for row in csv.reader(f, delimiter=',', quotechar='"'):
        t = '"%s","%s","%s"' % (row[0], row[1], row[2])
        t_literal[t] = row[2]

lookup_cache = json.load(open(FLAGS.lookup_cache_file))

for i, t in enumerate(t_literal):
    l = t_literal[t]
    ents = lookup_cache[l][0:FLAGS.predict_topk]
    if len(ents) < FLAGS.predict_topk:
        ents = ents + [''] * (FLAGS.predict_topk - len(ents))
    with open(FLAGS.predict_out_file, 'a') as f:
        s = '%s,"%s"' % (t, '","'.join(ents))
        f.write('%s\n' % s)
    print('line %d done' % i)
