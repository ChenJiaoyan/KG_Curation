# Evaluate

# "link_prediction": calculate hit@1, hit@5 (and MRR), to compare the link prediction model
#          it's based on the triples whose GT entities are among the candidates (lookuped entities with topk mask)
#          it only evaluates the performance of the link prediction model
#          a better model should rank the correct candidate in the front

# "overall"
#       "correction rate": calculate hit@1 (correction rate) and hit@5,
#               based on triples with GT annotations
#               filter out those entities that lead to
#                   low triple score/rank (predicted)
#                   or semantic inconsistency (property constraints)
#               it evaluates lookup + filtering with link prediction and semantic consistency
#       "elimination rate"


import os
import sys
import csv
import json
import argparse
import numpy as np

current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
parser = argparse.ArgumentParser()
parser.add_argument('--data_file', type=str, default=os.path.join(current_path, 'RData/RData_Annotate.csv'))
parser.add_argument('--redirect_cache_file', type=str, default=os.path.join(current_path, 'RData/Redirect_Cache.json'))
parser.add_argument('--lookup_cache_file', type=str, default=os.path.join(current_path, 'RData/Lookup_Cache_Split30.json'))
parser.add_argument('--prediction_file', type=str, default=os.path.join(current_path, 'RData/Predicts_TransH.csv'))
parser.add_argument('--evaluate_type', type=str, default='link_prediction', help='link_prediction, overall')
parser.add_argument('--threshold_step', type=float, default=0.05)
parser.add_argument('--lookup_entity_mask_topk', type=int, default=30)
FLAGS, unparsed = parser.parse_known_args()

# Read redirected entities and lookuped entities (masked)
with open(FLAGS.redirect_cache_file) as f:
    e_redirect = json.load(f)
with open(FLAGS.lookup_cache_file) as f:
    lookup_entities = json.load(f)
    for l in lookup_entities:
        if len(lookup_entities[l]) > FLAGS.lookup_entity_mask_topk:
            lookup_entities[l] = lookup_entities[l][0:FLAGS.lookup_entity_mask_topk]

# Read GTs
all_t, t_Annotate, tGTinLookup, tHasNoAnnotate = list(), dict(), list(), list()
with open(FLAGS.data_file) as f:
    for row in csv.reader(f, delimiter=',', quotechar='"'):
        s, p, l, e = row[0], row[1], row[2], row[3]
        t = '"%s","%s","%s"' % (s, p, l)
        all_t.append(t)
        if e == '':
            tHasNoAnnotate.append(t)
        else:
            t_Annotate[t] = e
            if e in lookup_entities[l]:
                tGTinLookup.append(t)
print('%d: have annotations, %d: have NO annotations, %d: annotation are lookuped'
      % (len(t_Annotate.keys()), len(tHasNoAnnotate), len(tGTinLookup)))


''' Judge whether the predicted entity is among the annotated entities
'''
def isRightPrediction(predicted_ent, annotated_ent):
    for e2 in e_redirect[annotated_ent]:
        if e2.lower() == predicted_ent.lower():
            return True
    return False

if FLAGS.evaluate_type.lower() == 'link_prediction':
    hit1_num, hit5_num = 0, 0
    hit1_num_random, hit5_num_random = 0, 0
    reciprocal_rank = 0
    reciprocal_rank_random = 0
    with open(FLAGS.prediction_file) as f:
        for row in csv.reader(f, delimiter=',', quotechar='"'):
            s, p, l = row[0], row[1], row[2]
            t = '"%s","%s","%s"' % (s, p, l)
            if t in tGTinLookup:
                # get a rank of entities, ordered by their predicted scores
                predict_entities_rank = list()
                for item in row[3:]:
                    if not item == '':
                        e = item.split()[0]
                        if e in lookup_entities[l]:
                            predict_entities_rank.append(e)

                # calculate triples # with annotation hit by the prediction
                gtEnt = t_Annotate[t]
                if isRightPrediction(predicted_ent=predict_entities_rank[0], annotated_ent=gtEnt):
                    hit1_num += 1
                for i, e in enumerate(predict_entities_rank):
                    if i < 5 and isRightPrediction(predicted_ent=e, annotated_ent=gtEnt):
                        hit5_num += 1
                        break

                for ind, e in enumerate(predict_entities_rank):
                    if isRightPrediction(predicted_ent=e, annotated_ent=gtEnt):
                        reciprocal_rank += 1.0/(ind + 1)
                        break

                reciprocal_rank_random += 1.0/(len(lookup_entities[l])/2)
                hit1_num_random += (1.0/len(lookup_entities[l]))
                hit5_num_random += (5.0/len(lookup_entities[l])) if len(lookup_entities[l]) > 5 else 1.0

    hit1_random = float(hit1_num_random) / float(len(tGTinLookup))
    hit5_random = float(hit5_num_random) / float(len(tGTinLookup))
    mrr_random = reciprocal_rank_random / float(len(tGTinLookup))
    print('Random: hits@1 %.4f, hits@5 %.4f, MRR %.4f' % (hit1_random, hit5_random, mrr_random))
    hit1 = float(hit1_num) / float(len(tGTinLookup))
    hit5 = float(hit5_num) / float(len(tGTinLookup))
    mrr = reciprocal_rank / float(len(tGTinLookup))
    print('\nPrediction: hits@1 %.4f, hits@5 %.4f, MRR %.4f' % (hit1, hit5, mrr))


if FLAGS.evaluate_type.lower() == 'overall':
    scores = list()
    with open(FLAGS.prediction_file) as f:
        for row in csv.reader(f, delimiter=',', quotechar='"'):
            for item in row[3:]:
                if not item == '':
                    tmp = item.split()
                    scores.append(float(tmp[1]))
    min_score = min(scores) if min(scores) >= 0 else 0.0
    max_score = max(scores) if max(scores) <= 1 else 1.0
    thresholds = list()
    c_rates, e_rates, accs = list(),list(),list()

    for threshold in np.arange(0,1,FLAGS.threshold_step):
        c_hit1_num, c_hit5_num, e_hit_num, right_num = 0, 0, 0, 0

        with open(FLAGS.prediction_file) as f:
            for row in csv.reader(f, delimiter=',', quotechar='"'):
                t = '"%s","%s","%s"' % (row[0], row[1], row[2])

                # read predicted score
                e_score = dict()
                for rank, item in enumerate(row[3:]):
                    if not item == '':
                        tmp = item.split()
                        e_score[tmp[0]] = float(tmp[1])

                def match(literal, ent):
                    from Lib.util_kb import DBP_RESOURCE_NS
                    ent = ent.replace(DBP_RESOURCE_NS,'')
                    ent = ent.replace('_', ' ')
                    return True if literal.lower() == ent.lower() else False

                # entities by lookup
                entities = lookup_entities[row[2]].copy()
                for e in e_score:
                    normalized_score = (e_score[e] - min_score) / (max_score - min_score)
                    if normalized_score < threshold and not match(literal=row[2], ent=e):
                        entities.remove(e)

                if t in t_Annotate:
                    gtEnt = t_Annotate[t]
                    for i, e in enumerate(entities):
                        if i == 0 and isRightPrediction(predicted_ent=e, annotated_ent=gtEnt):
                            c_hit1_num += 1
                            right_num += 1
                    for i, e in enumerate(entities):
                        if i < 5 and isRightPrediction(predicted_ent=e, annotated_ent=gtEnt):
                            c_hit5_num += 1
                            break

                if t in tHasNoAnnotate and len(entities) == 0:
                    e_hit_num += 1
                    right_num += 1

        print('threshold: %.2f' % threshold)
        c_hit1 = float(c_hit1_num) / float(len(t_Annotate.keys()))
        c_hit5 = float(c_hit5_num) / float(len(t_Annotate.keys()))
        print('     Correction hit@1 (Recall): %.4f, hit@5: %.4f' % (c_hit1, c_hit5))
        e_rate = float(e_hit_num) / float(len(tHasNoAnnotate))
        print('     Elimination rate: %.4f' % e_rate)
        accuracy = float(right_num) / float(len(all_t))
        print('     Accuracy: %.4f\n' % accuracy)

        thresholds.append(threshold)
        c_rates.append(c_hit1)
        e_rates.append(e_rate)
        accs.append(accuracy)

    for t in thresholds:
        print('%.4f' % t)
    print('')

    for r in c_rates:
        print('%.4f' % r)
    print('')

    for r in e_rates:
        print('%.4f' % r)
    print('')

    for a in accs:
        print('%.4f' % a)
    print('')

