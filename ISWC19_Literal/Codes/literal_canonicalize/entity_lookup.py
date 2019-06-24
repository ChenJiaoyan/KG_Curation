# This file is to annotate an entity to each literal (RData)
import os
import sys
import csv
import json
import argparse
from Lib.util_kb import queryClassByEntity

parser = argparse.ArgumentParser()
current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
parser.add_argument('--data_file', type=str, default=os.path.join(current_path, 'Data/RData_Clean.csv'))
parser.add_argument('--score_file', type=str, default=os.path.join(current_path, 'RData_Scores_AttBiRNN_FTF.json'))
parser.add_argument('--filter_by_types', type=str, default='Yes', help='Yes or No')
parser.add_argument('--threshold', type=float, default=0.01, help='0.15 or 0.01')
parser.add_argument('--cache_entity_file', type=str,
                    default=os.path.join(current_path, 'Cache/RData_literal_entity_raw.json'))
parser.add_argument('--cache_class_file', type=str,
                    default=os.path.join(current_path, 'Cache/RData_entity_class.json'))
parser.add_argument('--out_entity_file', type=str, default=os.path.join(current_patSh, 'RData_Entity_Lookup_Type01.csv'))
FLAGS, unparsed = parser.parse_known_args()
FLAGS.filter_by_types = True if FLAGS.filter_by_types == 'Yes' else False

l_entity = json.load(open(FLAGS.cache_entity_file))
e_class = json.load(open(FLAGS.cache_class_file))
t_c_score = json.load(open(FLAGS.score_file))

out_f = open(FLAGS.out_entity_file, 'w')

with open(FLAGS.data_file) as f:
    for i, row in enumerate(csv.reader(f, delimiter=',', quotechar='"')):
        l = row[2]
        triple_s = ' '.join(row[0:3])
        entities = l_entity[l]
        ent = ''

        if len(entities) > 0:

            if FLAGS.filter_by_types:
                types = set()
                c_score = t_c_score[triple_s]
                for c in c_score:
                    if c_score[c] >= FLAGS.threshold:
                        types.add(c)
                for e in entities:
                    classes = e_class[e] if e in e_class else queryClassByEntity(e=e)
                    classes = set([str(c) for c in classes])
                    if len(types.intersection(classes)) > 0:
                        ent = e
                        break

            else:
                ent = entities[0]

        try:
            line = '"%s","%s"\n' % (triple_s, ent)
        except UnicodeDecodeError:
            line = '"%s",""\n' % triple_s
            pass
        except UnicodeEncodeError:
            line = '"%s",""\n' % triple_s
            pass
        out_f.write(line)

        print('%d, %s done' % (i, l))

out_f.close()
