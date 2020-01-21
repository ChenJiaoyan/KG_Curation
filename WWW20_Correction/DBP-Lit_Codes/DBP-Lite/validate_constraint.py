# validate candidate correction triples by property cardinality and range constraints
# calculate cardinality score and range score

import os
import sys
import csv
import json
import argparse

from Lib.util_kb import DBP_RESOURCE_NS

current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
parser = argparse.ArgumentParser()
parser.add_argument('--data_file', type=str, default=os.path.join(current_path, 'Data/Annotate.csv'))
parser.add_argument('--lookup_cache_file', type=str, default=os.path.join(current_path, 'Data/Lookup_Cache_Split30.json'))
parser.add_argument('--G_triple_file', type=str, default=os.path.join(current_path, 'Data/G_triples.csv'))
parser.add_argument('--predict_topk', type=int, default=30)
parser.add_argument('--class_ancestor_file', type=str, default=os.path.join(current_path, 'Data/Class_Ancestor_Cache.json'))
parser.add_argument('--entity_concrete_class_file', type=str, default=os.path.join(current_path, 'Data/Entity_ConClass_Cache.json'))

parser.add_argument('--cardinality', dest='cardinality', action='store_true')
parser.add_argument('--cardinality_constraint_file', type=str, default=os.path.join(current_path, 'Data/Constraint_Cardinality.txt'))
parser.add_argument('--cardinality_validation_file', type=str, default=os.path.join(current_path, 'Data/Validates_Cardinality.csv'))
parser.add_argument('--cardinality_threshold', type=float, default=1.0)
parser.set_defaults(cardinality=False)

parser.add_argument('--range', dest='range', action='store_true')
parser.add_argument('--concrete_range_constraint_file', type=str, default=os.path.join(current_path, 'Data/Constraint_ConRange.txt'))
parser.add_argument('--general_range_constraint_file', type=str, default=os.path.join(current_path, 'Data/Constraint_GenRange.txt'))
parser.add_argument('--range_validation_file', type=str, default=os.path.join(current_path, 'Data/Validates_Range.csv'))
parser.add_argument('--concrete_range_weight', type=float, default=0.8)
parser.add_argument('--general_range_weight', type=float, default=0.2)
parser.set_defaults(range=False)

parser.add_argument('--merge', dest='merge', action='store_true')
parser.set_defaults(merge=True)
parser.add_argument('--file1', type=str, default=os.path.join(current_path, 'Data/Validates_Constraint.csv'))
parser.add_argument('--file2', type=str, default=os.path.join(current_path, 'Data/Predicts_NodePath.csv'))
parser.add_argument('--merge_file', type=str, default=os.path.join(current_path, 'Data/Predicts_NodePathConstraint.csv'))

FLAGS, unparsed = parser.parse_known_args()


# Read redirected entities and lookuped entities (masked)
with open(FLAGS.lookup_cache_file) as f:
    lookup_entities = json.load(f)

if FLAGS.cardinality:
    # read cardinality
    p_objNumMax, p_car_prob = dict(), dict()
    with open(FLAGS.cardinality_constraint_file) as f:
        for line in f.readlines():
            tmp = line.strip().split()
            p = tmp[0]
            objNumMax = int(tmp[1])
            p_objNumMax[p] = objNumMax
            car_prob = dict()
            for item in tmp[2:]:
                tmp2 = item.split(';')
                car_prob[int(tmp2[0])] = float(tmp2[1])
            p_car_prob[p] = car_prob

    # load triples
    sp_objs = dict()
    for s, p, o in csv.reader(open(FLAGS.G_triple_file), delimiter=',', quotechar='"'):
        sp = '"%s","%s"' % (s, p)
        if o.startswith(DBP_RESOURCE_NS):
            if sp in sp_objs:
                sp_objs[sp].append(o)
            else:
                sp_objs[sp] = [o]

    # calculate the scores
    results = list()
    with open(FLAGS.data_file) as f:
        for row in csv.reader(f, delimiter=',', quotechar='"'):
            s, p, l = row[0], row[1], row[2]
            objNumMax = p_objNumMax[p]
            car_prob = p_car_prob[p]
            RE = lookup_entities[l]
            eScores = list()
            for i, e in enumerate(RE):
                if i < FLAGS.predict_topk:
                    if objNumMax <= 0:
                        score = 0.0
                    else:
                        sp = '"%s","%s"' % (s,p)
                        existing_objs = list() if sp not in sp_objs else sp_objs[sp].copy()
                        if e not in existing_objs:
                            existing_objs.append(e)
                        n = len(existing_objs)
                        r = float(n - objNumMax) / float(objNumMax)
                        if r >= FLAGS.cardinality_threshold:
                            score = 0.0
                        else:
                            if n == 1:
                                score = car_prob[1]
                            else:
                                probSum = 0
                                for car in car_prob:
                                    if car > 1:
                                        probSum += car_prob[car]
                                score = (1-r) * probSum if r > 0 else probSum
                    eScores.append('%s %.4f' % (e, score))
            result = '"%s","%s","%s","%s"' % (s, p, l, '","'.join(eScores))
            results.append(result)
            print('%s done' % p)

    with open(FLAGS.cardinality_validation_file, 'w') as f:
        for r in results:
            f.write('%s\n' % r)


if FLAGS.range:
    # load concrete range and general range
    def load_range(range_file):
        p_c_prob = dict()
        with open(range_file) as rf:
            for rline in rf.readlines():
                rtmp = rline.strip().split()
                rp = rtmp[0]
                p_c_prob[rp] = dict()
                for ritem in rtmp[1:]:
                    rtmp2 = ritem.split(';')
                    c, prob = rtmp2[0], float(rtmp2[1])
                    p_c_prob[rp][c] = prob
        return p_c_prob

    p_conC_prob = load_range(range_file=FLAGS.concrete_range_constraint_file)
    p_genC_prob = load_range(range_file=FLAGS.general_range_constraint_file)

    # load concrete classes and ancestors
    e_conC = json.load(open(FLAGS.entity_concrete_class_file))
    c_ancestor = json.load(open(FLAGS.class_ancestor_file))

    def range_score(c_prob, eClasses):
        # the candidate entity has no classes
        if len(eClasses) == 0 and len(c_prob) > 0:
            probs = list(c_prob.values())
            return sum(probs) / len(probs)

        matchedClasses = set()
        for c in eClasses:
            if c in c_prob:
                matchedClasses.add(c)
        if len(matchedClasses) == 0:
            return 0.0
        else:
            mul = 1.0
            for c in matchedClasses:
                mul *= (1 - c_prob[c])
            return 1 - mul

    # calculate the scores
    results = list()
    with open(FLAGS.data_file) as f:
        for row in csv.reader(f, delimiter=',', quotechar='"'):
            s, p, l = row[0], row[1], row[2]
            RE = lookup_entities[l]
            eScores = list()
            for i, e in enumerate(RE):
                if i < FLAGS.predict_topk:
                    conC = set(e_conC[e])
                    genC = set()
                    for c in conC:
                        for a in c_ancestor[c]:
                            genC.add(a)

                    conScore = range_score(c_prob=p_conC_prob[p], eClasses=conC)
                    genScore = range_score(c_prob=p_genC_prob[p], eClasses=genC)
                    score = FLAGS.concrete_range_weight*conScore + FLAGS.general_range_weight*genScore
                    eScores.append('%s %.4f' % (e, score))

            result = '"%s","%s","%s","%s"' % (s, p, l, '","'.join(eScores))
            results.append(result)
            print('%s done' % p)

    with open(FLAGS.range_validation_file, 'w') as f:
        for r in results:
            f.write('%s\n' % r)


if FLAGS.merge:

    def load_scores(file_name):
        t_e_score = dict()
        with open(file_name) as f:
            for row in csv.reader(f, delimiter=',', quotechar='"'):
                t = '"%s","%s","%s"' % (row[0], row[1], row[2])
                e_score = dict()
                for item in row[3:]:
                    if not item == '':
                        tmp = item.split()
                        e_score[tmp[0]] = float(tmp[1])
                t_e_score[t] = e_score
        return t_e_score

    t_e_score1 = load_scores(file_name=FLAGS.file1)
    t_e_score2 = load_scores(file_name=FLAGS.file2)

    with open(FLAGS.merge_file, 'w') as f:
        for t in t_e_score1:
            if len(t_e_score1[t]) == 0 or len(t_e_score2[t]) == 0:
                f.write('%s,""\n' % t)
            else:
                e_score1 = t_e_score1[t]
                e_score2 = t_e_score2[t]
                eScores = list()
                for e in e_score1:
                    if e in e_score2:
                        # score = e_score1[e] * e_score2[e]
                        score = (e_score1[e] + e_score2[e])/2
                    else:
                        score = 0.0
                    eScores.append('%s %.4f' % (e, score))
                f.write('%s,"%s"\n' % (t, '","'.join(eScores)))
