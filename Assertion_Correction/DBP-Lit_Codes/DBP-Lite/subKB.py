"""
This file is to generate the sub-KB for the context of the input triples
"""
import os
import sys
import csv
import json
import argparse

sys.path.append('../')
from Lib.util_kb import Query_Objects, Query_Subjects, Query_Subject_Triples, Query_Object_Triples

parser = argparse.ArgumentParser()
current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
parser.add_argument('--data_file', type=str, default=os.path.join(current_path, 'Data/RData_Annotate.csv'))
parser.add_argument('--lookup_cache_file', type=str, default=os.path.join(current_path, 'Data/Lookup_Cache_Split30.json'))
parser.add_argument('--PTriple_cache_file', type=str, default=os.path.join(current_path, 'Data/PTriple_Cache.json'))
parser.add_argument('--PObject_cache_file', type=str, default=os.path.join(current_path, 'Data/PObject_Cache.json'))

# Output
parser.add_argument('--G_entity_file', type=str, default=os.path.join(current_path, 'Data/G_entities.csv'))
parser.add_argument('--G_property_file', type=str, default=os.path.join(current_path, 'Data/G_properties.csv'))
parser.add_argument('--G_triple_file', type=str, default=os.path.join(current_path, 'Data/G_triples.csv'))
FLAGS, unparsed = parser.parse_known_args()

entity_cache = json.load(open(FLAGS.lookup_cache_file))
ptriple_cache = json.load(open(FLAGS.PTriple_cache_file))
pobject_cache = json.load(open(FLAGS.PObject_cache_file))

Es, Em, Ep, P = set(), set(), set(), set()
with open(FLAGS.data_file) as f:
    for row in csv.reader(f, delimiter=',', quotechar='"'):
        s, p, l = row[0], row[1], row[2]
        Es.add(s)
        P.add(p)
        Em = Em.union(set(entity_cache[l]))
        Ep = Ep.union(set(pobject_cache[p]))
E = Es.union(Em).union(Ep)

Ts = set()
for p in P:
    for t in ptriple_cache[p]:
        Ts.add('"%s","%s","%s"' % (t[0], t[1], t[2]))
'''
Cached triples associated with a property may be incomplete 
as DBpedia SPARQL Endpoint returns at most 10,000 triples
'''
print('Complement triples associated with (sub in Es, P) and (P, obj in Em)')
incomplete_P = set()
for p in ptriple_cache:
    if not len(ptriple_cache[p]) < 10000:
        incomplete_P.add(p)
print('%d properties to complement' % len(incomplete_P))
for p in incomplete_P:
    print('     %s start' % p)
    for s in Es:
        objs = Query_Objects(s, p)
        for o in objs:
            Ts.add('"%s","%s","%s"' % (s, p, o))
            E.add(o)
    print('     partially done' )
    for o in Em:
        subs = Query_Subjects(p, o)
        for s in subs:
            Ts.add('"%s","%s","%s"' % (s, p, o))
            E.add(s)
    print('     %s done' % p)

# save triples
with open(FLAGS.G_triple_file, 'w') as f:
    for ts in Ts:
        f.write('%s\n' % ts)
print('%d triples saved' % len(Ts))


print('''Get Em associated entities and properties, Em #: %d''' % len(Em))
avoid_P = ['http://dbpedia.org/ontology/wikiPageWikiLink']
T_num = 0
for i, e in enumerate(Em):
    for t in Query_Subject_Triples(s=e):
        if t[1] not in avoid_P:
            if t[2] in E or t[1] in P:
                Ts.add('"%s","%s","%s"' % (t[0], t[1], t[2]))
                T_num += 1
    for t in Query_Object_Triples(o=e):
        if t[1] not in avoid_P:
            if t[0] in E or t[1] in P:
                Ts.add('"%s","%s","%s"' % (t[0], t[1], t[2]))
                T_num += 1
    if i % 500 == 0:
        print('Em %d done' % i)
print('Em associated triples #: %d' % T_num) #Em associated triples #: 1175215

# save triples again (overwrite)
with open(FLAGS.G_triple_file, 'w') as f:
    for ts in Ts:
        f.write('%s\n' % ts)
print('%d triples saved' % len(Ts))

for t in csv.reader(open(FLAGS.G_triple_file), delimiter=',', quotechar='"'):
    E.add(t[0])
    E.add(t[2])
    P.add(t[1])

with open(FLAGS.G_entity_file, 'w') as f:
    for e in E:
        f.write('%s\n' % e)
print('%d entities saved\n' % len(E))

with open(FLAGS.G_property_file, 'w') as f:
    for p in P:
        f.write('%s\n' % p)
print('%d properties saved\n' % len(P))
