# measure recall for related entities, with ranging K settings

import os
import sys
import csv
import json
import argparse

current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
parser = argparse.ArgumentParser()
parser.add_argument('--data_file', type=str, default=os.path.join(current_path, 'Data/Annotate.csv'))
parser.add_argument('--redirect_cache_file', type=str, default=os.path.join(current_path, 'Data/Redirect_Cache.json'))
parser.add_argument('--related_entity_file', type=str, default=os.path.join(current_path, 'Data/Lookup_Cache_Raw50.json'))
parser.add_argument('--max_k', type=int, default=30)
parser.add_argument('--min_k', type=int, default=1)
parser.add_argument('--step', type=int, default=1)

FLAGS, unparsed = parser.parse_known_args()

# Read redirected entities and related entities
with open(FLAGS.redirect_cache_file) as f:
    e_redirect = json.load(f)

def isRightPrediction(predicted_ent, annotated_ent):
    for e2 in e_redirect[annotated_ent]:
        if e2.lower() == predicted_ent.lower():
            return True
    return False

with open(FLAGS.related_entity_file) as f:
    l_entities = json.load(f)

R = list()
for k in range(FLAGS.min_k, FLAGS.max_k + 1, FLAGS.step):
    with open(FLAGS.data_file) as f:
        n1, n2 = 0, 0
        for row in csv.reader(f, delimiter=',', quotechar='"'):
            s, p, l, e = row[0], row[1], row[2], row[3]
            if not e == '':
                n1 += 1

                related_entities = l_entities[l]
                if len(related_entities) > k:
                    related_entities = related_entities[:k]

                for re in related_entities:
                    if isRightPrediction(predicted_ent=re, annotated_ent=e):
                        n2 += 1
                        break
        r = float(n2)/float(n1)
        print('k: %d, recall: %.4f (%d/%d)' % (k, r, n2, n1 ))
        R.append(r)

for r in R:
    print('%.4f' % r)


c_hit_num, e_hit_num, right_num = 0, 0, 0
c_num, e_num, all_num = 0, 0, 0
with open(FLAGS.data_file) as f:
    for row in csv.reader(f, delimiter=',', quotechar='"'):
        l, e = row[2], row[3]
        all_num += 1
        related_entities = l_entities[l]
        if e == '':
            e_num += 1
            if len(related_entities) == 0:
                e_hit_num += 1
                right_num += 1
        else:
            c_num += 1
            if len(related_entities) > 0 and isRightPrediction(predicted_ent=related_entities[0], annotated_ent=e):
                c_hit_num += 1
                right_num += 1

c_rate = float(c_hit_num) / float(c_num)
e_rate = float(e_hit_num) / float(e_num)
accuracy = float(right_num) / float(all_num)
print('Correction rate: %.4f, elimination rate: %.4f, Accuracy: %.4f\n' % (c_rate, e_rate, accuracy))