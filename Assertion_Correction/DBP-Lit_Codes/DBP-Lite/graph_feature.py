# This file is to extract observed features (LinkFeat and NodeFeat) from the whole DBpedia
import os
import sys
import json
import csv
import argparse
import pickle

sys.path.append('../')
from Lib.util_kb import Query_Properties, Query_Objects_Num, Query_Subjects_Num

current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
parser = argparse.ArgumentParser()
parser.add_argument('--data_file', type=str, default=os.path.join(current_path, 'Data/Annotate.csv'))
parser.add_argument('--lookup_cache_file', type=str, default=os.path.join(current_path, 'Data/Lookup_Cache.json'))
parser.add_argument('--T_sample_file', type=str, default=os.path.join(current_path, 'Data/T_sample.pkl'))
parser.set_defaults(sampling_mask=False)

parser.add_argument('--linkfeat', dest='linkfeat', action='store_true')
parser.set_defaults(linkfeat=True)
parser.add_argument('--graph_link_file', type=str, default=os.path.join(current_path, 'Data/graph_link.json'))

parser.add_argument('--nodefeatSub', dest='nodefeatSub', action='store_true')
parser.set_defaults(nodefeatSub=False)
parser.add_argument('--graph_node_ps_file', type=str, default=os.path.join(current_path, 'Data/graph_node_ps.json'))

parser.add_argument('--nodefeatObj', dest='nodefeatObj', action='store_true')
parser.set_defaults(nodefeatObj=False)
parser.add_argument('--graph_node_po_file', type=str, default=os.path.join(current_path, 'Data/graph_node_po.json'))
FLAGS, unparsed = parser.parse_known_args()

# samples for training
T_pos, T_neg = pickle.load(open(FLAGS.T_sample_file, 'rb'))
T_samples = T_pos + T_neg

# samples for prediction
entity_cache = json.load(open(FLAGS.lookup_cache_file))
soP_mask, psO_mask, poS_mask = dict(), dict(), dict()
with open(FLAGS.data_file) as f:
    for row in csv.reader(f, delimiter=',', quotechar='"'):
        s, p, l, o = row[0], row[1], row[2], row[3]
        for e in entity_cache[l]:
            T_samples.append([s, p, e])
        if FLAGS.sampling_mask:
            so = '"%s","%s"' % (s, o)
            if so in soP_mask:
                soP_mask[so].add(p)
            else:
                soP_mask[so] = {p}
            ps = '"%s","%s"' % (p,s)
            if ps in psO_mask:
                psO_mask[ps].add(o)
            else:
                psO_mask[ps] = {o}
            po = '"%s","%s"' % (p,o)
            if po in poS_mask:
                poS_mask[po].add(s)
            else:
                poS_mask[po] = {s}

def json_save(v_dict, filename):
    with open(filename, 'w') as ff:
        json.dump(v_dict, ff)

if FLAGS.linkfeat:
    print('Start: LinkFeat')
    properties = set()
    so_properties = json.load(open(FLAGS.graph_link_file)) if os.path.exists(FLAGS.graph_link_file) else dict()

    def update_properties(s, o):
        so = '"%s","%s"' % (s, o)
        if so not in so_properties:
            P = Query_Properties(s, o)
            if so in soP_mask:
                P = P - soP_mask[so]
                print('in mask: %d' % len(P))
            properties.update(P)
            so_properties[so] = list(P)

    for i, t in enumerate(T_samples):
        update_properties(s=t[0], o=t[2])
        update_properties(s=t[2], o=t[0])
        if i % 10000 == 0:
            print('%.1f%% done' % (100 * float(i+1)/float(len(T_samples)) ))
            json_save(v_dict=so_properties, filename=FLAGS.graph_link_file)

    json_save(v_dict=so_properties, filename=FLAGS.graph_link_file)
    print('LinkFeat done')

if FLAGS.nodefeatSub:
    print('Start: NodeFeat (property-subject)')
    ps_n = json.load(open(FLAGS.graph_node_ps_file)) if os.path.exists(FLAGS.graph_node_ps_file) else dict()
    for i, t in enumerate(T_samples):
        s, p = t[0], t[1]
        ps = '"%s","%s"' % (p, s)
        if ps not in ps_n:
            oN = Query_Objects_Num(s=s, p=p)
            if ps in psO_mask:
                oN -= len(psO_mask[ps])
            ps_n[ps] = oN
        if i % 10000 == 0:
            print('%.1f%% done' % (100 * float(i+1)/float(len(T_samples)) ))
            json_save(v_dict=ps_n, filename=FLAGS.graph_node_ps_file)

    json_save(v_dict=ps_n, filename=FLAGS.graph_node_ps_file)
    print('NodeFeat (property-subject) done')

if FLAGS.nodefeatObj:
    print('Start: NodeFeat (property-object)')
    po_n = json.load(open(FLAGS.graph_node_po_file)) if os.path.exists(FLAGS.graph_node_po_file) else dict()
    for i, t in enumerate(T_samples):
        p, o = t[1], t[2]
        po = '"%s","%s"' % (p, o)
        if po not in po_n:
            sN = Query_Subjects_Num(o=o, p=p)
            if po in poS_mask:
                sN -= len(poS_mask[po])
            po_n[po] = sN
        if i % 10000 == 0:
            print('%.1f%% done' % (100 * float(i+1)/float(len(T_samples)) ))
            json_save(v_dict=po_n, filename=FLAGS.graph_node_po_file)

    json_save(v_dict=po_n, filename=FLAGS.graph_node_po_file)
    print('NodeFeat (property-object) done')
