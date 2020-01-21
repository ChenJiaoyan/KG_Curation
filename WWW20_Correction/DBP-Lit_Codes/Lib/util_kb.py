# This file defines functions for KB Query Answering and Lookup
import requests
import time
import xml.etree.ElementTree as ET
from gensim.utils import tokenize
from SPARQLWrapper import SPARQLWrapper, JSON

DBP_END_POINT = 'http://dbpedia.org/sparql'
DBP_RESOURCE_NS = 'http://dbpedia.org/resource/' # please ensure '/' is added
DBP_ONTOLOGY_NS = 'http://dbpedia.org/ontology/'
DBP_PROPERTY_NS = 'http://dbpedia.org/property/'

ENDPOINT = 'http://dbpedia.org/sparql'


"""lookup top-k entities from DBPedia with each phrase token"""
def lookup_entities_split_with_sleep(text, top_k):
    text = text.strip().lower()
    entities = lookup_entities_raw_with_sleep(text=text, top_k=top_k)

    tokens = [t for t in tokenize(text)]
    token_len = len(tokens)

    # lookup with sub-phrases, from long to short
    if len(entities) < top_k and token_len >= 2:
        for l in range(token_len-1, 1, -1):
            for i in range(token_len - l + 1):
                sub_text = ' '.join(tokens[i: (i + l)])
                top_k_new = top_k/3 if top_k/3 >= 1 else 1
                tmp_entities = lookup_entities_raw_with_sleep(text=sub_text, top_k=top_k_new)
                for e in tmp_entities:
                    if e not in entities:
                        entities.append(e)
                        if len(entities) >= top_k:
                            return entities

    return entities


"""lookup top-k entities from DBPedia with raw text"""
def lookup_entities_raw_with_sleep(text, top_k):
    entities = list()
    try:
        lookup_url = 'http://lookup.dbpedia.org/api/search/KeywordSearch?MaxHits=%d&QueryString=%s' % (top_k, text)
        lookup_res = requests.get(lookup_url)
        if '400 Bad Request' not in lookup_res.text:
            root = ET.fromstring(lookup_res.text)
            for child in root:
                ent = child[1].text
                entities.append(ent)
        else:
            time.sleep(60 * 5)
            lookup_url = 'http://lookup.dbpedia.org/api/search/KeywordSearch?MaxHits=%d&QueryString=%s' % (top_k, text)
            lookup_res = requests.get(lookup_url)
            if '400 Bad Request' not in lookup_res.text:
                root = ET.fromstring(lookup_res.text)
                for child in root:
                    ent = child[1].text
                    entities.append(ent)
    except UnicodeDecodeError:
            pass
    if len(entities) > top_k:
        entities = entities[0:top_k]

    return entities


"""Get the name of a resource/property URI of DBpedia"""
def DBpedia_URI_Parse(uri):
    uri = uri.strip()
    if DBP_PROPERTY_NS in uri:
        return uri.split(DBP_PROPERTY_NS)[-1]
    elif DBP_ONTOLOGY_NS in uri:
        return uri.split(DBP_ONTOLOGY_NS)[-1]
    elif DBP_RESOURCE_NS in uri:
        return uri.split(DBP_RESOURCE_NS)[-1]
    else:
        return uri.split('/')[-1]


"""Get triples associated with a given property
    Note: at most 10,000 triples will be returned by DBpedia sparql endpoint
"""
def Query_Property_Triples(p):
    triples = list()
    q = 'select ?s ?o where {?s <%s> ?o. FILTER(strstarts(str(?o), "%s"))}' % (p, DBP_RESOURCE_NS)
    res = do_query(q)
    for r in res['results']['bindings']:
        s, o = r['s']['value'], r['o']['value']
        triples.append([s, p, o])
    return triples


"""Get objects (entities) associated with a given property
"""
def Query_Property_Objects(p):
    objects = set()
    q = 'select distinct ?o where {?s <%s> ?o. FILTER(strstarts(str(?o), "%s"))}' % (p, DBP_RESOURCE_NS)
    res = do_query(q)
    for r in res['results']['bindings']:
        o = r['o']['value']
        objects.add(o)
    return objects

"""Get objects (entities) associated with a given subject and property
"""
def Query_Objects(s, p):
    objects = set()
    q = 'select distinct ?o where {<%s> <%s> ?o. FILTER(strstarts(str(?o), "%s"))}' % (s, p, DBP_RESOURCE_NS)
    res = do_query(q)
    for r in res['results']['bindings']:
        o = r['o']['value']
        objects.add(o)
    return objects

"""Get triples, given an object entity
"""
def Query_Object_Triples(o):
    triples = list()
    q = 'select distinct ?s,?p where {?s ?p <%s>.}' % o
    res = do_query(q)
    for r in res['results']['bindings']:
        s, p = r['s']['value'], r['p']['value']
        triples.append([s,p,o])
    return triples


"""Get triples, given a subject entity
"""
def Query_Subject_Triples(s):
    triples = list()
    q = 'select distinct ?p,?o where {<%s> ?p ?o. FILTER(strstarts(str(?o), "%s"))}' % (s, DBP_RESOURCE_NS)
    res = do_query(q)
    for r in res['results']['bindings']:
        o, p = r['o']['value'], r['p']['value']
        triples.append([s,p,o])
    return triples


"""Get number of objects (entities) associated with a given subject and property
"""
def Query_Objects_Num(s, p):
    q = 'select count(distinct ?o) as ?n where {<%s> <%s> ?o. FILTER(strstarts(str(?o), "%s"))}' % (s, p, DBP_RESOURCE_NS)
    res = do_query(q)
    for r in res['results']['bindings']:
        n = r['n']['value']
        return int(n)


"""Get subjects (entities) associated with a given object and property
"""
def Query_Subjects(p, o):
    objects = set()
    q = 'select distinct ?s where {?s <%s> <%s>.}' % (p, o)
    res = do_query(q)
    for r in res['results']['bindings']:
        o = r['s']['value']
        objects.add(o)
    return objects


"""Get number of subjects (entities) associated with a given object and property
"""
def Query_Subjects_Num(p, o):
    q = 'select count(distinct ?s) as ?n where {?s <%s> <%s>.}' % (p, o)
    res = do_query(q)
    for r in res['results']['bindings']:
        n = r['n']['value']
        return int(n)


"""Query property given subject and object
"""
def Query_Properties(s, o):
    properties = set()
    q = 'select distinct ?p where {<%s> ?p <%s>.}' % (s, o)
    res = do_query(q)
    for r in res['results']['bindings']:
        p = r['p']['value']
        properties.add(p)
    return properties


"""Get all wikipage redirected entities of a given entity
"""
def Query_WikiPage_Redirect(e):
    entities = set()

    def getResult(query):
        results = do_query(q=query, attempts=3)
        result_set = set()
        if results is None:
            print("None results for", query)
            return result_set

        for result in results["results"]["bindings"]:
            uri_value = result["uri"]["value"]
            if uri_value.startswith('http://dbpedia.org/resource'):
                result_set.add(uri_value)

        return result_set

    q = "SELECT DISTINCT ?uri WHERE { <%s> <http://dbpedia.org/ontology/wikiPageRedirects> ?uri .}" % e
    entities.update(getResult(query=q))
    q = "SELECT DISTINCT ?uri WHERE { ?uri <http://dbpedia.org/ontology/wikiPageRedirects> <%s> . }" % e
    entities.update(getResult(query=q))

    new_entities = set()
    for e in entities:
        q = "SELECT DISTINCT ?uri WHERE { <%s> <http://dbpedia.org/ontology/wikiPageRedirects> ?uri .}" % e
        new_entities.update(getResult(query=q))
        q = "SELECT DISTINCT ?uri WHERE { ?uri <http://dbpedia.org/ontology/wikiPageRedirects> <%s> . }" % e
        new_entities.update(getResult(query=q))

    entities.add(e)
    entities.update(new_entities)
    return entities


"""Query classes of a given entity
"""
def Query_Classes(e):
    classes = set()
    q = 'select distinct ?cc where {<%s> rdf:type ?c. ?c rdfs:subClassOf* ?cc. ' \
        'FILTER (strstarts(str(?cc), "%s"))}' % (e, DBP_ONTOLOGY_NS)
    res = do_query(q)
    for r in res['results']['bindings']:
        c = r['cc']['value']
        classes.add(c.replace(DBP_ONTOLOGY_NS, ''))
    return classes


"""Query concrete classes of a given entity
"""
def Query_Concrete_Classes(e):
    classes = set()
    q = 'select distinct ?c where {<%s> rdf:type ?c. FILTER (strstarts(str(?c), "%s"))}' % (e, DBP_ONTOLOGY_NS)
    res = do_query(q)
    for r in res['results']['bindings']:
        c = r['c']['value']
        classes.add(c.replace(DBP_ONTOLOGY_NS, ''))
    return classes


"""Query concrete classes of a given entity
"""
def Query_Ancestors(c):
    ancestors = set()
    q = 'select distinct ?a where {<%s%s> rdfs:subClassOf* ?a. FILTER (strstarts(str(?a), "%s"))}' \
        % (DBP_ONTOLOGY_NS, c, DBP_ONTOLOGY_NS)
    res = do_query(q)
    for r in res['results']['bindings']:
        a = r['a']['value']
        ancestors.add(a.replace(DBP_ONTOLOGY_NS, ''))
    ancestors.remove(c)
    return ancestors


"""Query function with multiple tries and wait"""
def do_query(q, attempts = 3):
    try:
        endpoint = SPARQLWrapper(ENDPOINT)
        endpoint.setReturnFormat(JSON)
        endpoint.setQuery(q)
        return endpoint.query().convert()
    except:
        time.sleep(60 * 5)  # to avoid limit of calls, sleep 60s
        attempts -= 1
        if attempts > 0:
            return do_query(q, attempts)
        else:
            print('Error: failed to query "%s"' % q)
            return None
