This folder contains cache from the DBPedia (by Lookup and SPARQL endpoint)

(R)SData_literal_entity.json
    -- key: literal, value: entities by lookup with the splitted terms of the literal 

(R)SData_literal_entity_raw.json
    -- key: literal, value: entities by lookup with the whole text phrase of the literal 

(R)SData_property_entity.json
    -- key: property, value: object entities 
    
(R)SData_entity_class.json
    -- key: entity, value: classes of the entity 

(R)SData_entity_class_fixed.json
    -- fixed version of (R)SData_entity.json, with Wikidata

(R)SData_property_entity_subject.json
    -- key: property, value: {key: object entity, value: subject entities}

(R)SData_entity_label.json
    -- key: entity, value: labels of the entity 

class_joint.json
    -- key: class, value: ancestors and descendants 
    
class_descendant.json
    -- key: class, values: descendant classes

class_label.json
    -- key: class, value: label of class (by cache_class_label.py) (used in ESWC16 baseline)

class_triple.json
    -- key: class, value: triples whose object belongs to the class (used in pretraining)

classes.txt
    -- all the DBPedia ontology classes from http://mappings.dbpedia.org/server/ontology/classes/
