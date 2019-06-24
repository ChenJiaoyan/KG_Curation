# this file makes the decision for types and evaluate
import os
import sys
import csv
import json
import argparse
import numpy as np
current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type=str, default='RData', help='SData, RData')
parser.add_argument('--data_file', type=str, default=os.path.join(current_path, 'Data/RData_Clean.csv'))
parser.add_argument('--out_score_file', type=str,
                    default=os.path.join(current_path, 'RData_Scores_AttBiRNN_FTF.json'))
parser.add_argument('--type_gt', type=str, default=os.path.join(current_path, 'Data/RData_Type.json'))
parser.add_argument('--typing_method', type=str, default='I', help='I (independent) or (H) hierarchical')
parser.add_argument('--iota_range', type=str, default='0.0,1.0,0.01')
parser.add_argument('--kappa', type=float, default=0, help='[-0.1, 0.11]')
parser.add_argument('--class_descendant', type=str, default=os.path.join(current_path, 'Cache/class_descendant.json'))
FLAGS, unparsed = parser.parse_known_args()
print FLAGS
iota_start, iota_end, iota_step = [float(i) for i in FLAGS.iota_range.split(',')]

# read predicted scores
t_c_score = json.load(open(FLAGS.out_score_file))

# independently determine the type according to the score and the threshold iota
def independent_typing():
    mF1s = list()
    for iota in np.arange(iota_start, iota_end, iota_step):
        mP, mR, mF1, n = 0.0, 0.0, 0.0, 0
        with open(FLAGS.data_file) as f:
            for row in csv.reader(f, delimiter=',', quotechar='"'):
                triple_s = ' '.join([row[0], row[1], row[2]])
                if FLAGS.data_name == 'SData':
                    gt_types = set(eORt_types[row[3]])
                else:
                    gt_types = set(eORt_types[triple_s])
                c_score = t_c_score[triple_s]
                predict_classes = set()
                for c in c_score:
                    if c_score[c] >= iota:
                        predict_classes.add(c)
                P = float(len(predict_classes.intersection(gt_types))) / float(len(predict_classes)) \
                    if len(predict_classes) > 0 else 0
                R = float(len(predict_classes.intersection(gt_types))) / float(len(gt_types))
                F1 = 0 if P + R == 0 else 2*P*R/(P+R)
                mP, mR, mF1 = mP + P, mR + R, mF1 + F1
                n += 1
        mP, mR, mF1 = mP/n, mR/n, mF1/n
        mF1s.append(mF1)
        print('iota: %f,  mean P/R/F1: %.4f, %.4f, %.4f' % (iota, mP, mR, mF1))
    print('\n Avg-F1@all, %.4f, Avg-F1@top5, %.4f' % (np.average(mF1s), np.average(sorted(mF1s)[-5:])))


# re-calculate the score of a class as the maximum of itself and its descendant
def hierarchical_scoring():
    t_c_Hscore = dict()
    for t in t_c_score:
        c_score = t_c_score[t]
        c_Hscore = dict()
        for c in c_score:
            Hscore = c_score[c]
            for d in c_descendant[c]:
                if d in c_score and c_score[d] > Hscore:
                    Hscore = c_score[d]
            c_Hscore[c] = Hscore
        t_c_Hscore[t] = c_Hscore
    return t_c_Hscore


# hierarchically determine the type from general classes to concrete classes
def hierarchical_typing():
    mF1s = list()
    t_c_Hscore = hierarchical_scoring()
    for iota in np.arange(iota_start, iota_end, iota_step):
        mP, mR, mF1, n = 0.0, 0.0, 0.0, 0
        with open(FLAGS.data_file) as f:
            for row in csv.reader(f, delimiter=',', quotechar='"'):
                triple_s = ' '.join([row[0], row[1], row[2]])
                if FLAGS.data_name == 'SData':
                    gt_types = set(eORt_types[row[3]])
                else:
                    gt_types = set(eORt_types[triple_s])
                c_Hscore = t_c_Hscore[triple_s]
                classes = c_Hscore.keys()

                predict_classes = set()
                for c in classes:
                    if c_Hscore[c] >= iota:
                        max_disjoint_score = 0.0
                        for c_other in classes:
                            if c not in c_descendant[c_other] and c_other not in c_descendant[c] and \
                                    c_Hscore[c_other] > max_disjoint_score:
                                max_disjoint_score = c_Hscore[c_other]
                        if c_Hscore[c] - max_disjoint_score >= FLAGS.kappa:
                            predict_classes.add(c)

                P = float(len(predict_classes.intersection(gt_types))) / float(len(predict_classes)) \
                    if len(predict_classes) > 0 else 0
                R = float(len(predict_classes.intersection(gt_types))) / float(len(gt_types))
                F1 = 0 if P + R == 0 else 2*P*R/(P+R)
                mP, mR, mF1 = mP + P, mR + R, mF1 + F1
                n += 1
        mP, mR, mF1 = mP/n, mR/n, mF1/n
        mF1s.append(mF1)
        print('iota: %f,  mean P/R/F1: %.4f, %.4f, %.4f' % (iota, mP, mR, mF1))
    print('\n Avg-F1@all, %.4f, Avg-F1@top5, %.4f' % (np.average(mF1s), np.average(sorted(mF1s)[-5:])))


# if FLAGS.data_name == 'SData':
print('Evaluate...')
# read type ground truths
eORt_types = json.load(open(FLAGS.type_gt))

if FLAGS.typing_method == 'I':
    independent_typing()

else:
    c_descendant = json.load(open(FLAGS.class_descendant))
    hierarchical_typing()
