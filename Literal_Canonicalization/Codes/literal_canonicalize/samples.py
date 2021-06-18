# Get particular samples for each property
import os
import csv
import sys
import json
import random
import argparse

current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type=str, default='RData', help='SData, RData')
parser.add_argument('--data_file', type=str, default=os.path.join(current_path, 'Data/RData_Clean.csv'))
parser.add_argument('--entity_mask_type', type=str, default='NO', help='NO, YES, ALL')
parser.add_argument('--alpha', type=int, default=3, help='[0-5]')
parser.add_argument('--use_fixed_entity_class', type=str, default='YES', help='NO, YES')
parser.add_argument('--particular_sample_file', type=str,
                    default=os.path.join(current_path, 'RData_PSamples_fixed.json'))
FLAGS, unparsed = parser.parse_known_args()

# read properties and their literals
p_subLits = dict()
with open(FLAGS.data_file) as f:
    for row in csv.reader(f, delimiter=',', quotechar='"'):
        s, p, l = row[0], row[1], row[2]
        if p not in p_subLits:
            p_subLits[p] = [[s, l]]
        else:
            p_subLits[p].append([s, l])

# read candidate classes
p_classes = json.load(open(os.path.join(current_path, '%s_Classes.json' % FLAGS.data_name)))

# read the parse of property
p_splits = json.load(open(os.path.join(current_path, 'Data/property_split.json')))

# read cache of entity_class, literal_entity, entity_label
if FLAGS.use_fixed_entity_class == 'YES':
    e_classes = json.load(open(os.path.join(current_path, 'Cache/%s_entity_class_fixed.json' % FLAGS.data_name)))
else:
    e_classes = json.load(open(os.path.join(current_path, 'Cache/%s_entity_class.json' % FLAGS.data_name)))
l_entities = json.load(open(os.path.join(current_path, 'Cache/%s_literal_entity.json' % FLAGS.data_name)))
e_labels = json.load(open(os.path.join(current_path, 'Cache/%s_entity_label.json' % FLAGS.data_name)))
p_e_subs = json.load(open(os.path.join(current_path, 'Cache/%s_property_entity_subject.json' % FLAGS.data_name)))

# read entities mask; NO mask for RData; YES and ALL for SData
# In synthetic data, we assume (i) entities for constructing the literals do not exist (YES)
#                           or (ii) entities that are objects of the target properties all do not exist (ALL)
ents_mask = set()
if FLAGS.entity_mask_type == 'YES':
    with open(FLAGS.data_file) as f:
        for row in csv.reader(f, delimiter=',', quotechar='"'):
            ents_mask.add(row[3])

# sampling
p_PSamples = dict()
for p in p_subLits:
    print('\n%s\n' % p)
    classes = p_classes[p]['C']
    subLits = p_subLits[p]
    c_PSamples, c_GSamples = dict(), dict()
    for c in classes:

        # particular samples
        PS_pos, PS_neg = list(), list()

        for s, l in subLits:
            for e in (set(l_entities[l]) - ents_mask):
                if c in e_classes[e]:
                    for e_lab in e_labels[e]:
                        PS_pos.append([s, p, e_lab])
                if len(e_classes[e]) > 0 and c not in e_classes[e]:
                    for e_lab in e_labels[e]:
                        PS_neg.append(([s, p, e_lab]))

        for e in (set(p_e_subs[p]) - ents_mask):
            if c in e_classes[e]:
                for s in p_e_subs[p][e]:
                    for e_lab in e_labels[e]:
                        PS_pos.append([s, p, e_lab])
            if len(e_classes[e]) > 0 and c not in e_classes[e]:
                for s in p_e_subs[p][e]:
                    for e_lab in e_labels[e]:
                        PS_neg.append([s, p, e_lab])

#        NO Positive-Negative Balance
#        if len(PS_pos) <= len(PS_neg):
#            PS_neg = random.sample(PS_neg, len(PS_pos))
#        else:
#            n = len(PS_pos) / len(PS_neg)
#            PS_neg = PS_neg * n
#            PS_neg += random.sample(PS_neg, len(PS_pos) - len(PS_neg))

        if len(PS_pos) >= FLAGS.alpha:
            c_PSamples['%s POS' % c] = PS_pos
            c_PSamples['%s NEG' % c] = PS_neg
        print('     %s, %d, %d' % (c, len(PS_pos), len(PS_neg)))

    p_PSamples[p] = c_PSamples


with open(FLAGS.particular_sample_file, 'w') as out_f:
    json.dump(p_PSamples, out_f)
