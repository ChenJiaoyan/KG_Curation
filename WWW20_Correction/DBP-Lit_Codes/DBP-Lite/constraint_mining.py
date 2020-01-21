# Mine property cardinality constraints and range constraints

import os
import sys
import csv
import json
import argparse

current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
parser = argparse.ArgumentParser()
parser.add_argument('--G_property_file', type=str, default=os.path.join(current_path, 'Data/G_properties.csv'))
parser.add_argument('--G_triple_file', type=str, default=os.path.join(current_path, 'Data/G_triples.csv'))
parser.add_argument('--class_ancestor_file', type=str, default=os.path.join(current_path, 'Data/Class_Ancestor_Cache.json'))
parser.add_argument('--entity_concrete_class_file', type=str, default=os.path.join(current_path, 'Data/Entity_ConClass_Cache.json'))
parser.add_argument('--data_file', type=str, default=os.path.join(current_path, 'Data/Annotate.csv'))

parser.add_argument('--cardinality', dest='cardinality', action='store_true')
parser.set_defaults(cardinality=True)
parser.add_argument('--cardinality_constraint_file', type=str, default=os.path.join(current_path, 'Data/Constraint_Cardinality.txt'))

parser.add_argument('--range', dest='range', action='store_true')
parser.set_defaults(range=True)
parser.add_argument('--concrete_range_constraint_file', type=str, default=os.path.join(current_path, 'Data/Constraint_ConRange.txt'))
parser.add_argument('--general_range_constraint_file', type=str, default=os.path.join(current_path, 'Data/Constraint_GenRange.txt'))
FLAGS, unparsed = parser.parse_known_args()

properties = list()
for s,p,l,e in csv.reader(open(FLAGS.data_file), delimiter=',', quotechar='"'):
    if p not in properties:
        properties.append(p)

p_T = dict()
for s,p,o in csv.reader(open(FLAGS.G_triple_file), delimiter=',', quotechar='"'):
    if p in properties:
        if p in p_T:
            p_T[p].append([s,p,o])
        else:
            p_T[p] = [[s,p,o]]

# calculate property cardinality
if FLAGS.cardinality:
    car_lines = list()
    for property in properties:
        sub_objNum = dict()
        if property in p_T:
            for s,p,o in p_T[property]:
                if s in sub_objNum:
                    sub_objNum[s] += 1
                else:
                    sub_objNum[s] = 1

        objNums = list(sub_objNum.values())
        objNums.sort()
        if len(objNums) > 0:
            objNum_max = max(objNums)
            car_rates = list()
            for n in set(objNums):
                car_rate = "%d;%.5f" % (n, float(objNums.count(n))/float(len(objNums)))
                car_rates.append(car_rate)
            car_line = '%s %d %s' % (property, objNum_max, ' '.join(car_rates))
        else:
            car_line = '%s 0' % property

        car_lines.append(car_line)
        print('%s done' % property)

    with open(FLAGS.cardinality_constraint_file, 'w') as f:
        for car_line in car_lines:
            f.write('%s\n' % car_line)

if FLAGS.range:
    e_conClass = json.load(open(FLAGS.entity_concrete_class_file))
    c_ancestor = json.load(open(FLAGS.class_ancestor_file))
    conRange_lines, genRange_lines = list(), list()
    for property in properties:
        if property in p_T:
            conC_objNum, genC_objNum, objNum = dict(), dict(), 0
            for _,_,o in p_T[property]:
                objNum += 1

                conClass = set(e_conClass[o])
                for c in conClass.copy():
                    conClass = conClass - set(c_ancestor[c])

                genClass = set()
                for c in conClass:
                    ancestor = set(c_ancestor[c])
                    genClass.update(ancestor)

                for c in conClass:
                    if c in conC_objNum:
                        conC_objNum[c] += 1
                    else:
                        conC_objNum[c] = 1
                for c in genClass:
                    if c in genC_objNum:
                        genC_objNum[c] += 1
                    else:
                        genC_objNum[c] = 1

            conC_rates = list()
            conC_objNum = dict(sorted(conC_objNum.items(), key=lambda kv: kv[1], reverse=True))
            for conC in conC_objNum:
                rate = float(conC_objNum[conC]) / objNum
                conC_rates.append('%s;%.5f' % (conC, rate))
            conRange_line = '%s %s' % (property, ' '.join(conC_rates)) if len(conC_rates) > 0 else property
            conRange_lines.append(conRange_line)

            genC_rates = list()
            genC_objNum = dict(sorted(genC_objNum.items(), key=lambda kv: kv[1], reverse=True))
            for genC in genC_objNum:
                rate = float(genC_objNum[genC]) / objNum
                genC_rates.append('%s;%.5f' % (genC, rate))
            genRange_line = '%s %s' % (property, ' '.join(genC_rates)) if len(genC_rates) > 0 else property
            genRange_lines.append(genRange_line)

        else:
            conRange_lines.append(property)
            genRange_lines.append(property)

        print('%s done' % property)

    with open(FLAGS.concrete_range_constraint_file, 'w') as f:
        for line in conRange_lines:
            f.write('%s\n' % line)
    with open(FLAGS.general_range_constraint_file, 'w') as f:
        for line in genRange_lines:
            f.write('%s\n' % line)
