# Cache the entities of each literal by lookup
import os
import csv
import sys
import json
import argparse

sys.path.append('../')
from Lib.util_kb import lookup_entities_raw_with_sleep, lookup_entities_split_with_sleep
from Lib.util_kb import Query_Property_Triples, Query_Property_Objects
from Lib.util_kb import Query_WikiPage_Redirect, Query_Classes, Query_Concrete_Classes, Query_Ancestors

current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
parser = argparse.ArgumentParser()
parser.add_argument('--data_file', type=str, default=os.path.join(current_path, 'Data/Annotate.csv'))

parser.add_argument('--lookup_cache', dest='lookup_cache', action='store_true')
parser.add_argument('--lookup_cache_file', type=str, default=os.path.join(current_path, 'Data/Lookup_Cache_Split30.json'))
parser.set_defaults(lookup_cache=False)

parser.add_argument('--triple_cache', dest='triple_cache', action='store_true')
parser.add_argument('--triple_cache_file', type=str, default=os.path.join(current_path, 'Data/PTriple_Cache.json'))
parser.set_defaults(triple_cache=False)

parser.add_argument('--object_cache', dest='object_cache', action='store_true')
parser.add_argument('--object_cache_file', type=str, default=os.path.join(current_path, 'Data/PObject_Cache.json'))
parser.set_defaults(object_cache=False)

parser.add_argument('--redirect_cache', dest='redirect_cache', action='store_true')
parser.add_argument('--redirect_cache_file', type=str, default=os.path.join(current_path, 'Data/Redirect_Cache.json'))
parser.set_defaults(redirect_cache=False)

parser.add_argument('--entity_class_cache', dest='entity_class_cache', action='store_true')
parser.add_argument('--entity_class_type', type=str, default='all', help='all, concrete')
parser.add_argument('--entity_class_cache_file', type=str, default=os.path.join(current_path, 'Data/Entity_Class_Cache.json'))
parser.add_argument('--G_entity_file', type=str, default=os.path.join(current_path, 'Data/G_entities.csv'))
parser.set_defaults(entity_class_cache=True)

parser.add_argument('--class_ancestor_cache', dest='class_ancestor_cache', action='store_true')
parser.add_argument('--class_ancestor_cache_file', type=str, default=os.path.join(current_path, 'Data/Class_Ancestor_Cache.json'))
parser.set_defaults(class_ancestor_cache=False)

FLAGS, unparsed = parser.parse_known_args()


def load_cache(file_name):
    return json.load(open(file_name)) if os.path.exists(file_name) else dict()

def save_cache(c, file_name):
    with open(file_name, 'w') as out_f:
        json.dump(c, out_f)


if FLAGS.lookup_cache:

    # raw or preprocess
    #       raw: directly use the text for lookup;
    #       preprocess: split the text and use each token for lookup
    # LOOKUP_TYPE = 'raw'
    LOOKUP_TYPE = 'split'
    LOOKUP_TOPK = 30

    literals = set([row[2] for row in csv.reader(open(FLAGS.data_file), delimiter=',', quotechar='"')])
    print('%d literals to do' % len(literals))
    lit_entities = load_cache(file_name=FLAGS.lookup_cache_file)

    tmp_f = open('/tmp/tmp.txt', 'w')
    for i, lit in enumerate(literals):
        if lit not in lit_entities:
            if LOOKUP_TYPE.lower() == 'raw':
                ents = lookup_entities_raw_with_sleep(text=lit, top_k=LOOKUP_TOPK)
            elif LOOKUP_TYPE.lower() == 'split':
                ents = lookup_entities_split_with_sleep(text=lit, top_k=LOOKUP_TOPK)
            else:
                raise Exception('No text processing method specified. "raw" or "split"?')
            ents_filter = list()
            for e in ents:
                try:
                    tmp_f.write('%s\n' % e)
                    ents_filter.append(e)
                except UnicodeDecodeError or UnicodeEncodeError:
                    print('UnicodeDecodeError or UnicodeEncodeError detected')
                    pass
            lit_entities[lit] = ents_filter

            save_cache(c=lit_entities, file_name=FLAGS.lookup_cache_file)
            print('%d, %s cache added' % (i, lit))
    tmp_f.close()


if FLAGS.triple_cache:

    properties = set([row[1] for row in csv.reader(open(FLAGS.data_file), delimiter=',', quotechar='"')])
    print('%d properties to do' % len(properties))
    p_cache = load_cache(file_name=FLAGS.triple_cache_file)

    tmp_f = open('/tmp/tmp.txt', 'w')
    for i, p in enumerate(properties):
        if p not in p_cache:

            triples = Query_Property_Triples(p)

            triples2 = list()
            for t in triples:
                try:
                    tmp_f.write(' '.join(t))
                    triples2.append(t)
                except UnicodeDecodeError or UnicodeEncodeError:
                    print('UnicodeDecodeError or UnicodeEncodeError detected')
                    pass
            p_cache[p] = triples2

            save_cache(c=p_cache, file_name=FLAGS.triple_cache_file)
            print('%d, %s cache added' % (i, p))
    tmp_f.close()


if FLAGS.object_cache:
    properties = set([row[1] for row in csv.reader(open(FLAGS.data_file), delimiter=',', quotechar='"')])
    print('%d properties to do' % len(properties))
    p_cache = load_cache(file_name=FLAGS.object_cache_file)

    tmp_f = open('/tmp/tmp.txt', 'w')
    for i, p in enumerate(properties):
        if p not in p_cache:

            objects = Query_Property_Objects(p)
            objects2 = set()
            for o in objects:
                try:
                    tmp_f.write(o)
                    objects2.add(o)
                except UnicodeDecodeError or UnicodeEncodeError:
                    print('UnicodeDecodeError or UnicodeEncoderError detected')
                    pass
            p_cache[p] = list(objects2)

            save_cache(c=p_cache, file_name=FLAGS.object_cache_file)
            print('%d, %s cache added' % (i, p))
    tmp_f.close()


if FLAGS.redirect_cache:
    e_redirect = load_cache(file_name=FLAGS.redirect_cache_file)

    entities = set()
    with open(FLAGS.data_file, 'r') as f:
        for row in csv.reader(f, delimiter=',', quotechar='"'):
            ents = row[3]
            for e in row[3].strip().split():
                entities.add(e)
    print('%d entities to do' % len(entities))

    for i, e in enumerate(entities):
        if e not in e_redirect:
            equal_ents = Query_WikiPage_Redirect(e=e)
            e_redirect[e] = list(equal_ents)
        if i % 10 == 0:
            save_cache(c=e_redirect, file_name=FLAGS.redirect_cache_file)
            print('%d done' % i)

    save_cache(c=e_redirect, file_name=FLAGS.redirect_cache_file)


if FLAGS.entity_class_cache:
    e_class = load_cache(FLAGS.entity_class_cache_file)
    E = [line.strip() for line in open(FLAGS.G_entity_file).readlines()]
    for i, e in enumerate(E):
        if e not in e_class:
            if FLAGS.entity_class_type == 'all':
                classes = Query_Classes(e = e)
            elif FLAGS.entity_class_type == 'concrete':
                classes = Query_Concrete_Classes(e = e)
            else:
                raise NotImplementedError
            e_class[e] = list(classes)
        if i % 2000 == 0:
            save_cache(c=e_class, file_name=FLAGS.entity_class_cache_file)
            print('%.1f%% (%d) done' % (100*float(i)/len(E), i))
    print('all done, %d entities' % len(e_class.keys()))
    save_cache(c=e_class, file_name=FLAGS.entity_class_cache_file)


if FLAGS.class_ancestor_cache:
    e_conClass = load_cache(FLAGS.entity_class_cache_file)
    conClasses = set()
    for conClass in e_conClass.values():
        conClasses.update(set(conClass))

    c_ancestor = load_cache(FLAGS.class_ancestor_cache_file)
    for i, c in enumerate(conClasses):
        if c not in c_ancestor:
            ancestors = Query_Ancestors(c = c)
            c_ancestor[c] = list(ancestors)
        if i % 100 == 0:
            save_cache(c=c_ancestor, file_name=FLAGS.class_ancestor_cache_file)
            print('%.1f%% (%d) done' % (100*float(i)/len(conClasses), i))

    print('all done, %d classes' % len(conClasses))
    save_cache(c=c_ancestor, file_name=FLAGS.class_ancestor_cache_file)
