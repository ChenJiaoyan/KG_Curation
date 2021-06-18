# This file defines functions for KB SPARQL query and inference
import re
import requests
import time
import sparql
import xml.etree.ElementTree as ET

SPARQL_END_POINT = 'http://dbpedia.org/sparql'
RESOURCE_NS = 'http://dbpedia.org/resource/'
ONTOLOGY_NS = 'http://dbpedia.org/ontology/'
PROPERTY_NS = 'http://dbpedia.org/property/'

"""Get [subject, literal]'s of a given property (URI)"""
"""DBPedia SPARQL end point returns at most 10,000 records"""
def querySubLiteralsByProperty(top_k, p, max_str_len):
    subs_lits = list()
    s = sparql.Service(SPARQL_END_POINT, "utf-8", "GET")
    statement = 'select str(?sub), str(?obj) where {?sub <%s> ?obj.  ' \
                'FILTER ( (regex(datatype(?obj), "langString") || regex(datatype(?obj), "string")) ' \
                '&& strlen(str(?obj))<%d )} ' \
                'ORDER BY RAND() LIMIT %d' % (p, max_str_len, top_k)
    result = s.query(statement)
    for row in result.fetchone():
        subs_lits.append([row[0], row[1]])
    return subs_lits


"""Get [subject, literal]'s of a given [property, str], with random selection"""
"""l_str may has '"' inside"""
def querySubLiteralsByPropertyStr(top_k, p, l_str, max_str_len):
    subs_lits = list()
    s = sparql.Service(SPARQL_END_POINT, "utf-8", "GET")
    l_str = l_str.replace("'", "\\'")
    statement = '''select str(?s),str(?o) where {?s <%s> ?o. FILTER ( contains(str(?o), '%s') 
                && (regex(datatype(?o), "langString") || regex(datatype(?o), "string")) && strlen(str(?o))<%d )} 
                ORDER BY RAND() LIMIT %d''' % (p, l_str, max_str_len, top_k)
    try:
        result = s.query(statement)
        for row in result.fetchone():
            subs_lits.append([row[0], row[1]])
    except UnicodeDecodeError:
        pass
    return subs_lits


"""Get [subject, object entity, object label]'s of a given property, with random selection"""
def querySubObjLabByProperty(top_k, p):
    subs_objs_labs = list()
    s = sparql.Service(SPARQL_END_POINT, "utf-8", "GET")
    statement = 'select distinct str(?s),str(?o),str(?l) where {?s <%s> ?o. ?o rdfs:label ?l. ' \
                'FILTER (strstarts(str(?o), "%s") ' \
                'and langMatches(lang(?l), "en"))} ORDER BY RAND() LIMIT %d' % (p, RESOURCE_NS, top_k)
    result = s.query(statement)
    for row in result.fetchone():
        subs_objs_labs.append([row[0], row[1], row[2]])
    return subs_objs_labs


"""Get types (classes) of a given entity"""
def queryClassByEntity(e):
    types = set()
    s = sparql.Service(SPARQL_END_POINT, "utf-8", "GET")
    statement = 'select distinct str(?s) where { <%s> rdf:type ?e. ?e rdfs:subClassOf* ?s. ' \
                'FILTER(strstarts(str(?s), "%s"))}' % (e, ONTOLOGY_NS)
    result = s.query(statement)
    for row in result.fetchone():
        types.add(row[0])

    statement = 'select distinct str(?ss) where { <%s>  dbo:wikiPageRedirects ?e. ?e rdf:type ?s. ' \
                '?s rdfs:subClassOf* ?ss. FILTER(strstarts(str(?ss), "%s"))}' % (e, ONTOLOGY_NS)
    result = s.query(statement)
    for row in result.fetchone():
        types.add(row[0])

    return types


"""Get object entities of a given property"""
def queryObjEntitiesByProperty(top_k, p):
    objs = set()
    s = sparql.Service(SPARQL_END_POINT, "utf-8", "GET")
    statement = 'select distinct str(?o) where {?s <%s> ?o. FILTER (strstarts(str(?o), "%s"))} ' \
                'ORDER BY RAND() limit %d' % (p, RESOURCE_NS, top_k)
    result = s.query(statement)
    for row in result.fetchone():
        objs.add(row[0])
    return objs


"""Get subject entities of a given pair of (property, object)"""
def querySubByPropertyAndObject(p, o, top_k):
    subs = set()
    s = sparql.Service(SPARQL_END_POINT, "utf-8", "GET")
    statement = 'select distinct str(?s) where {?s <%s> <%s> } limit %d' % (p, o, top_k)
    result = s.query(statement)
    for row in result.fetchone():
        subs.add(row[0])
    return subs


"""Get english labels of a given entity"""
def queryEngLabelByEntity(e):
    labels = set()
    s = sparql.Service(SPARQL_END_POINT, "utf-8", "GET")
    statement = 'select distinct str(?l) where {<%s> rdfs:label ?l. FILTER( langMatches(lang(?l), "en"))}' % e
    result = s.query(statement)
    for row in result.fetchone():
        labels.add(row[0])
    return labels


"""Get (subject, property, object, label)'s of a given class, such that object belongs to the class"""
def queryTripleByClass(top_k, c):
    triples = list()
    s = sparql.Service(SPARQL_END_POINT, "utf-8", "GET")
    statement = 'select distinct str(?s), str(?p), str(?o), str(?l) where {?s ?p ?o. ?o rdf:type <%s>. ' \
                '?o rdfs:label ?l. FILTER( langMatches(lang(?l), "en"))} ORDER BY RAND() limit %d' % (c, top_k)
    result = s.query(statement)
    for row in result.fetchone():
        triples.append([row[0], row[1], row[2], row[3]])
    return triples


"""Get siblings of a given class"""
def querySiblingByClass(c):
    siblings = set()
    s = sparql.Service(SPARQL_END_POINT, "utf-8", "GET")
    statement = 'select distinct str(?s) where {<%s> rdfs:subClassOf ?p. ?s rdfs:subClassOf ?p. ' \
                'FILTER(?s != <%s> && strstarts(str(?s), "%s"))}' % (c, c, ONTOLOGY_NS)
    result = s.query(statement)
    for row in result.fetchone():
        siblings.add(row[0])
    return siblings


"""Get ancestors of a given class"""
def queryAncestorByClass(c):
    ancestors = set()
    s = sparql.Service(SPARQL_END_POINT, "utf-8", "GET")
    statement = 'select distinct str(?a) where {<%s> rdfs:subClassOf* ?a. ' \
                'FILTER(strstarts(str(?a), "%s"))}' % (c, ONTOLOGY_NS)
    result = s.query(statement)
    for row in result.fetchone():
        ancestors.add(row[0])
    return ancestors


"""Get descendant of a given class"""
def queryDescendantByClass(c):
    descendants = set()
    s = sparql.Service(SPARQL_END_POINT, "utf-8", "GET")
    statement = 'select distinct str(?d) where {?d rdfs:subClassOf* <%s>. ' \
                'FILTER(strstarts(str(?d), "%s"))}' % (c, ONTOLOGY_NS)
    result = s.query(statement)
    for row in result.fetchone():
        descendants.add(row[0])
    return descendants


"""Get classes and their labels"""
def queryLabelByClass(class_text):
    s = sparql.Service(SPARQL_END_POINT, "utf-8", "GET")
    statement = 'select distinct str(?l) where {<%s%s> rdfs:label ?l. FILTER(langMatches(lang(?l), ' \
                '"en"))}' % (ONTOLOGY_NS, class_text)
    result = s.query(statement)
    for row in result.fetchone():
        return row[0]
    return None


"""lookup top-k entities from DBPedia with text cleaned and splitted"""
def lookup_entities_with_sleep(cell_text, top_k):
    entities = list()
    cell_items = list()
    cell_brackets = re.findall('\((.*?)\)', cell_text)
    for cell_bracket in cell_brackets:
        cell_text = cell_text.replace('(%s)' % cell_bracket, '')
    cell_text = cell_text.strip()
    if len(cell_text) > 2:
        cell_items.append(cell_text)
    for cell_bracket in cell_brackets:
        if len(cell_bracket) > 2:
            cell_items.append(cell_bracket.strip())
    for cell_item in cell_items:
        try:
            lookup_url = 'http://lookup.dbpedia.org/api/search/KeywordSearch?MaxHits=%d&QueryString=%s' \
                         % (top_k, cell_item)
            lookup_res = requests.get(lookup_url)
            if '400 Bad Request' not in lookup_res.content:
                root = ET.fromstring(lookup_res.content)
                for child in root:
                    ent = child[1].text
                    entities.append(ent)
            else:
                time.sleep(60*3)
                lookup_url = 'http://lookup.dbpedia.org/api/search/KeywordSearch?MaxHits=%d&QueryString=%s' \
                             % (top_k, cell_item)
                lookup_res = requests.get(lookup_url)
                if '400 Bad Request' not in lookup_res.content:
                    root = ET.fromstring(lookup_res.content)
                    for child in root:
                        ent = child[1].text
                        entities.append(ent)
        except UnicodeDecodeError:
            pass
    return entities


"""lookup top-k entities from DBPedia with raw text"""
def lookup_entities_raw_with_sleep(text, top_k):
    entities = list()
    try:
        lookup_url = 'http://lookup.dbpedia.org/api/search/KeywordSearch?MaxHits=%d&QueryString=%s' % (top_k, text)
        lookup_res = requests.get(lookup_url)
        if '400 Bad Request' not in lookup_res.content:
            root = ET.fromstring(lookup_res.content)
            for child in root:
                ent = child[1].text
                entities.append(ent)
        else:
            time.sleep(60 * 3)
            lookup_url = 'http://lookup.dbpedia.org/api/search/KeywordSearch?MaxHits=%d&QueryString=%s' % (top_k, text)
            lookup_res = requests.get(lookup_url)
            if '400 Bad Request' not in lookup_res.content:
                root = ET.fromstring(lookup_res.content)
                for child in root:
                    ent = child[1].text
                    entities.append(ent)
    except UnicodeDecodeError:
            pass
    return entities


"""Get the name of a resource/property URI"""
def URIParse(uri):
    uri = uri.strip()
    if PROPERTY_NS in uri:
        return uri.split(PROPERTY_NS)[-1]
    elif ONTOLOGY_NS in uri:
        return uri.split(ONTOLOGY_NS)[-1]
    elif RESOURCE_NS in uri:
        return uri.split(RESOURCE_NS)[-1]
    else:
        return uri.split('/')[-1]
