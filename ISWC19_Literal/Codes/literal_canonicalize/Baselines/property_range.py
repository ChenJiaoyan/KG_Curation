# This file is to estimate the range of a property
# Candidate classes are extracted as the classes of E_p
# Each class is given a probability -- the percentage of entities that belong to this class
import os
import sys
import csv
import json
import argparse

current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
parser = argparse.ArgumentParser()
parser.add_argument('--data_file', type=str, default=os.path.join(current_path, '../Data/RData_Clean.csv'))
parser.add_argument('--cache_property_entity', type=str,
                    default=os.path.join(current_path, '../Cache/RData_property_entity.json'))
parser.add_argument('--cache_entity_class', type=str,
                    default=os.path.join(current_path, '../Cache/RData_entity_class.json'))
parser.add_argument('--out_score_file', type=str, default=os.path.join(current_path, 'RData_Scores_PropertyRange.json'))
FLAGS, unparsed = parser.parse_known_args()

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

# load cache
p_entity = json.load(open(FLAGS.cache_property_entity))
e_class = json.load(open(FLAGS.cache_entity_class))

# property by property
t_c_score = dict()
for p in p_triples:
    print('\n------- %s -------\n' % p)

    Ep = set(p_entity[p])
    c_eNum = dict()
    for e in Ep:
        for c in e_class[e]:
            if c in c_eNum:
                c_eNum[c] += 1
            else:
                c_eNum[c] = 1

    for triple in p_triples[p]:
        triple_s = ' '.join(triple)
        c_score = dict()
        for c in c_eNum:
            c_score[c] = float(c_eNum[c]) / float(len(Ep))
        t_c_score[triple_s] = c_score

print('Estimation done!\n')

with open(FLAGS.out_score_file, 'w') as out_f:
    json.dump(t_c_score, out_f)
