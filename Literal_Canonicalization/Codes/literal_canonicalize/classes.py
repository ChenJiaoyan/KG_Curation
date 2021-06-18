# Get candidate classes for each property
import os
import csv
import sys
import json
import argparse

current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type=str, default='RData', help='SData, RData')
parser.add_argument('--data_file', type=str, default=os.path.join(current_path, 'Data/RData_Clean.csv'))
parser.add_argument('--out_file', type=str, default=os.path.join(current_path, 'RData_Classes.json'))
parser.add_argument('--entity_mask_type', type=str, default='NO', help='NO, YES, ALL')
FLAGS, unparsed = parser.parse_known_args()

# read properties and their literals
p_literals = dict()
with open(FLAGS.data_file) as f:
    for row in csv.reader(f, delimiter=',', quotechar='"'):
        p, l = row[1], row[2]
        if p not in p_literals:
            p_literals[p] = {l}
        else:
            p_literals[p].add(l)

# read entities mask; NO mask for RData; YES and ALL for SData
ents_mask = set()
if FLAGS.entity_mask_type == 'YES':
    with open(FLAGS.data_file) as f:
        for row in csv.reader(f, delimiter=',', quotechar='"'):
            ents_mask.add(row[3])

# read cache
p_Ep = json.load(open(os.path.join(current_path, 'Cache/%s_property_entity.json' % FLAGS.data_name)))
l_E = json.load(open(os.path.join(current_path, 'Cache/%s_literal_entity.json' % FLAGS.data_name)))
e_C = json.load(open(os.path.join(current_path, 'Cache/%s_entity_class.json' % FLAGS.data_name)))

# read candidate classes
p_C = dict()
for p in p_literals:
    Cp, Cl = list(), list()
    for e in (set(p_Ep[p]) - ents_mask):
        Cp += e_C[e]
    for l in p_literals[p]:
        for e in (set(l_E[l]) - ents_mask):
            Cl += e_C[e]

    Cp, Cl, C = list(set(Cp)), list(set(Cl)), list(set(Cp + Cl))
    print('%s done, |Cp|: %d, |Cl|: %d, |C|: %d' % (p, len(Cp), len(Cl), len(C)))
    p_C[p] = {'Cp': Cp, 'Cl': Cl, 'C': C}

# save
with open(FLAGS.out_file, 'w') as out_f:
    json.dump(p_C, out_f)

