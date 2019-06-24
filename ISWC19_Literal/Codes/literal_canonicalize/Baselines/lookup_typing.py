# this file to vote the types of a literal by entities from lookup (using raw text)
import csv
import json
from Lib.util_kb import queryClassByEntity

TOPK = 3
ENTITY_MASK = True
data_file = '../Data/RData_Clean.csv'
l_entities = json.load(open('../Cache/RData_literal_entity_raw.json'))
eORt_gtTypes = json.load(open('../Data/RData_Type.json'))

# SData or RData
DATA_NAME = 'RData'

quads = []
entities_mask = set()
with open(data_file) as f:
    for row in csv.reader(f, delimiter=',', quotechar='"'):
        if DATA_NAME == 'SData':
            quads.append([row[0], row[1], row[2], row[3]])
            entities_mask.add(row[3])
        else:
            quads.append([row[0], row[1], row[2]])

precision, recall, f1 = 0.0, 0.0, 0.0
for i, quad in enumerate(quads):

    # lookup top-k entities
    l = quad[2]
    ents = l_entities[l]
    ents_filter = list()
    if ENTITY_MASK:
        for e in ents:
            if e not in entities_mask:
                ents_filter.append(e)
    else:
        ents_filter = ents
    if len(ents_filter) > TOPK:
        ents_filter = ents_filter[0:TOPK]

    # query the types
    classes = set()
    for e in ents_filter:
        for c in queryClassByEntity(e):
            classes.add(str(c))

    # calculate micro precision/recall/f1 score
    if DATA_NAME == 'SData':
        classes_gt = set(eORt_gtTypes[quad[3]])
    else:
        triple_s = ' '.join(quad[0:3])
        classes_gt = set(eORt_gtTypes[triple_s])

    mP = float(len(set(classes).intersection(classes_gt))) / float(len(classes)) if len(classes) > 0 else 0.0
    mR = float(len(set(classes).intersection(classes_gt))) / float(len(classes_gt))
    mF1 = (2 * mP * mR) / (mP + mR) if mP + mR > 0 else 0

    precision += mP
    recall += mR
    f1 += mF1
    print('%d, %s done' % (i, l))

precision = precision / len(quads)
recall = recall / len(quads)
f1 = f1 / len(quads)

print('precision: %.4f, recall: %.4f, F1 score: %.4f' % (precision, recall, f1))
