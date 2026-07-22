import pandas as pd
import numpy as np
import pickle
import glob
import os
import ast
import random
from tqdm import tqdm
from print_color import print


miRNA_KG_relations = {
    'Gene - Disease': [
        'Gene - causes or contributes to condition - Disease',
        ],
    'miRNA - miRNA': [
        'miRNA - in similarity relationship with - miRNA',
        ],
    'miRNA - Disease': [
        'miRNA - causes or contributes to condition - Disease',
        'miRNA - under expressed in - Disease',
        'miRNA - over expressed in - Disease',
        'miRNA - is causal somatic mutation in - Disease',
        ],
    'Gene - Gene': [
        'Gene - genetically interacts with - Gene',
        ],
    'miRNA - Gene': [
        'miRNA - regulates activity of - Gene',
        ],
    'miRNA - GO': [
        'miRNA - participates in - GO',
        'miRNA - has function - GO',
        'miRNA - located in - GO',
        'miRNA - part of - GO',
        ]
}

PKT_KG_relations = {
    'Protein - Anatomy': [
        'Protein - located in - Anatomy'
        ],
    'Protein - Cell': [
        'Protein - located in - Cell'
        ],
    'Protein - Protein': [    
        'Protein - molecularly interacts with - Protein'
        ],
    'GO - GO': [
        'GO - negatively regulates - GO', 
        'GO - positively regulates - GO', 
        'GO - regulates - GO',
        ],
    'Chemical - GO': [
        'Chemical - participates in - GO',
        'Chemical - molecularly interacts with - GO',
        ],
    'Protein - GO': [
        'Protein - enables - GO',
        'Protein - located in - GO',
        ],
    'Chemical - Gene': [
        'Chemical - interacts with - Gene',
        ],
    'Chemical - Protein': [
        'Chemical - interacts with - Protein',
        'Chemical - molecularly interacts with - Protein'
        ],
    'Chemical - Disease': [
        'Chemical - is substance that treats - Disease'
        ],
    'Chemical - Pathway': [
        'Chemical - participates in - Pathway'
        ],
    'Protein - Pathway': [
        'Protein - participates in - Pathway'
        ],
    'Gene - Disease': [
        'Gene - causes or contributes to condition - Disease',
        ],
    'Gene - Gene': [
        'Gene - genetically interacts with - Gene'
        ],
    'Gene - Protein': [
        'Gene - interacts with - Protein'
        ],
    'Gene - Pathway': [
        'Gene - participates in - Pathway'
        ]
}

Hetionet_relations = {
    'Anatomy - Gene': [
        'Anatomy - downregulates - Gene',
        'Anatomy - expresses - Gene', 
        'Anatomy - upregulates - Gene',
        ],
    'Compund - Gene': [
        'Compound - binds - Gene',
        'Compound - downregulates - Gene',
        'Compound - upregulates - Gene',
        ],
    'Compound - Side_effect': [
        'Compound - causes - Side_effect', 
        ],
    'Compound - Compound': [
        'Compound - resembles - Compound',
        ],
    'Disease - Gene': [
        'Disease - associates - Gene',
        'Disease - downregulates - Gene',
        'Disease - upregulates - Gene',
        ],
    'Disease - Anatomy': [
        'Disease - localizes - Anatomy',
        ],
    'Disease - Symptom': [
        'Disease - presents - Symptom', 
        ],
    'Gene - Gene': [
        'Gene - covaries - Gene',
        'Gene - interacts - Gene', 
        'Gene - regulates - Gene', 
        ],
    'Gene - Biological_process': [
        'Gene - participates - Biological_process',
        ],
    'Gene - Cellular_component': [
        'Gene - participates - Cellular_component',
        ],
    'Gene - Molecular_function': [
        'Gene - participates - Molecular_function',
        ],
    'Gene - Pathway': [
        'Gene - participates - Pathway', 
        ],
    'Pharmacologic_class  - Compound': [
        'Pharmacologic_class - includes - Compound'
        ]
}

PrimeKG_relations = {
    'Anatomy - Gene_and_or_protein': [
        'Anatomy - expression absent - Gene_and_or_protein',
        'Anatomy - expression present - Gene_and_or_protein', 
        ],
    'Biological_process - Exposure': [
        'Biological_process - interacts with - Exposure', 
        ],
    'Biological_process - Gene_and_or_protein': [
        'Biological_process - interacts with - Gene_and_or_protein',
        ],
    'Cellular_component - Gene_and_or_protein' : [
        'Cellular_component - interacts with - Gene_and_or_protein',
        ],
    'Disease - Gene_and_or_protein' : [
        'Disease - associated with - Gene_and_or_protein',
        ],
    'Disease - Drug': [
        'Disease - contraindication - Drug',
        'Disease - indication - Drug',
        'Disease - off label use - Drug',
        ],
    'Disease - Exposure': [
        'Disease - linked to - Exposure',
        ],
    'Disease - Effect_and_or_phenotype': [
        'Disease - phenotype absent - Effect_and_or_phenotype',
        'Disease - phenotype present - Effect_and_or_phenotype', 
        ],
    'Drug - Gene_and_or_protein': [
        'Drug - enzyme - Gene_and_or_protein',
        'Drug - target - Gene_and_or_protein',
        'Drug - transporter - Gene_and_or_protein',
        ],
    'Drug - Effect_and_or_phenotype': [
        'Drug - side effect - Effect_and_or_phenotype',
        ],
    'Drug - Drug': [
        'Drug - synergistic interaction - Drug', 
        ],
    'Effect_and_or_phenotype - Gene_and_or_protein': [
        'Effect_and_or_phenotype - associated with - Gene_and_or_protein',
        ],
    'Exposure - Gene_and_or_protein': [
        'Exposure - interacts with - Gene_and_or_protein', 
        ],
    'Gene_and_or_protein - Molecular_function': [
        'Gene_and_or_protein - interacts with - Molecular_function',
        ],
    'Gene_and_or_protein - Pathway': [
        'Gene_and_or_protein - interacts with - Pathway',
        ],
    'Gene_and_or_protein - Gene_and_or_protein': [
        'Gene_and_or_protein - ppi - Gene_and_or_protein' 
        ],
}

OptimusKG_relations = {
    'Anatomy - Gene': [
        'Anatomy - EXPRESSION_ABSENT - Gene',               
        'Anatomy - EXPRESSION_PRESENT - Gene',              
        ],
    'Drug - Phenotype': [
        'Drug - ASSOCIATED_WITH - Phenotype',
        'Drug - CONTRAINDICATION - Phenotype',          
        'Drug - INDICATION - Phenotype',                
        ],         
    'Drug - Disease': [
        'Drug - OFF_LABEL_USE - Disease',               
        'Drug - CONTRAINDICATION - Disease',            
        'Drug - INDICATION - Disease',                  
        ],
    'Drug - Gene': [
        'Drug - TARGET - Gene',                         
        'Drug - TRANSPORTER - Gene',                    
        'Drug - ENZYME - Gene',                         
        ],
    'Biological_process - Gene': [
        'Biological_process - INTERACTS_WITH - Gene',
        ],
    'Cellular_component - Gene': [
        'Cellular_component - INTERACTS_WITH - Gene',   
        ],
    'Disease - Gene': [
        'Disease - ASSOCIATED_WITH - Gene',             
        ],
    'Pathway - Gene': [
        'Pathway - INTERACTS_WITH - Gene',              
        ],
    'Disease - Phenotype': [
        'Disease - PHENOTYPE_PRESENT - Phenotype',      
        ],
    'Drug - Drug': [
        'Drug - SYNERGISTIC_INTERACTION - Drug',            
        ],
    'Exposure - Biological_process': [
        'Exposure - INTERACTS_WITH - Biological_process'
        ],
    'Exposure - Gene': [
        'Exposure - INTERACTS_WITH - Gene',             
        ],
    'Exposure - Disease': [
        'Exposure - LINKED_TO - Disease',               
        ],
    'Phenotype - Gene': [   
        'Phenotype - ASSOCIATED_WITH - Gene',
        ],
    'Gene - Gene': [
        'Gene - INTERACTS_WITH - Gene',                 
        ],
    'Molecular_function - Gene': [ 
        'Molecular_function - INTERACTS_WITH - Gene'    
        ]
}


kg_schemas = {
    'miRNA-KG': miRNA_KG_relations,
    'PKT-KG': PKT_KG_relations,
    'Hetionet': Hetionet_relations,
    'PrimeKG': PrimeKG_relations,
    'OptimusKG': OptimusKG_relations,
}

def show_schema_facts(kg: str = None, seed: int = None):
    """
    Prints every schema fact for a given KG (or for all KGs if kg is not
    given), together with one randomly sampled node ID example for its source
    and target type 
    """
    rng = random.Random(seed)
    kgs = [kg] if kg else list(kg_schemas.keys())

    schema_fact_examples = {}
    for kg in kgs:
        nodes = pd.read_csv(f"data/nodes_{kg}.csv", usecols=['URI:ID', ':LABEL'], low_memory=False)
        ids_by_type = (
            nodes.assign(**{':LABEL': nodes[':LABEL'].fillna('').astype(str).str.split(';')})
            .explode(':LABEL')
            .groupby(':LABEL')['URI:ID']
            .apply(list)
            .to_dict()
        )

        print(f"=== {kg} ===", color='cyan')
        schema_fact_examples[kg] = {}
        for facts in kg_schemas[kg].values():
            for fact in facts:
                sub_type, _, obj_type = fact.split(' - ')
                source_id = rng.choice(ids_by_type[sub_type]) if sub_type in ids_by_type else None
                target_id = rng.choice(ids_by_type[obj_type]) if obj_type in ids_by_type else None
                schema_fact_examples[kg][fact] = (source_id, target_id)

                print(fact)
                print(f"  {sub_type} id example: {source_id}")
                print(f"  {obj_type} id example: {target_id}")

    return schema_fact_examples

def get_models(kg: str, emb_name: str, strategy: str, model: str, valid_rels: list):

    models = {}
    model_path = f"dumps_models/{kg}/{emb_name}/{strategy}/err_beta_1/{model}/*.pkl"

    for f in glob.glob(model_path):
        relation = f.split('/')[-1]
        start_pos = relation.find('_model_')
        relation = relation[start_pos + 7:]
        relation = relation[:-len('_fixed.pkl')] if relation.endswith('_fixed.pkl') else relation[:-len('.pkl')]
        if not relation in valid_rels: continue
        with open(f, "rb") as model:
            models[relation] = pickle.load(model)

    if not len(models):
        raise Exception("No model is obtained! Check the file path")

    return models

def get_input(kg: str, emb_name: str, strategy: str, needed_models: list, rels_pair: str, negative: bool):

    triples_dict = {}
    triples_cardinality = {}

    if not negative:
        dir_add = f"blind_test_occ/{kg}/*.csv"
    else:
        if kg == 'miRNA-KG' or kg == 'PKT-KG':
            dir_add = f"negative_samples/{kg}/{strategy}_1/*.csv"
        else:
            dir_add = f"negative_samples/{kg}/{strategy}/*.csv"
        # Get the number of blind sample triples for each relation
        blind_size = {}
        for f in glob.glob(f"blind_test_occ/{kg}/*.csv"):
            relation = f.split('/')[-1][:-4]
            if not relation in needed_models: continue
            blind_size[relation] = pd.read_csv(f).rename(columns={'subcject': 'subject'}).shape[0]
            

    for f in glob.glob(dir_add):
        relation = f.split('/')[-1][:-4]

        if not relation in needed_models: continue

        if not negative:
            triple_df = pd.read_csv(f).rename(columns={'subcject': 'subject'})
        else:
            # Get negative triples and sample them randomly with the corresponding blind sample size
            triple_df = pd.read_csv(f).rename(columns={'source': 'subject', 'target': 'object'})
            triple_df = triple_df.sample(n=min(blind_size[relation], len(triple_df)), random_state=5)

        if triple_df.shape[0] == 0:
            continue
        triples_dict[relation] = triple_df
        triples_cardinality[relation] = triples_dict[relation].shape[0]

    embedding = pd.read_csv(f"store_embeddings/{emb_name}/{kg}.csv")

    # Map triples to their embeddings
    mapped_emb = {}
    for rel in tqdm(triples_dict, desc="Mapping triples and create inputs"):
        mapped_emb[rel] = []
        for row in triples_dict[rel].iterrows():
            # Map URIs to embeddings
            uri = ''
            if emb_name == 'transe': uri = 'name'
            else: uri = 'node_id'

            # Find embedding vector from embedding list
            subject_emb = embedding[embedding[uri] == row[1]['subject']].squeeze().to_list()[1]
            object_emb = embedding[embedding[uri] == row[1]['object']].squeeze().to_list()[1]
            # Convert str to List
            subject_emb = ast.literal_eval(subject_emb)
            object_emb = ast.literal_eval(object_emb)
            # Compute model input
            mul_emb = np.array(subject_emb) * np.array(object_emb)
            mapped_emb[rel].append(mul_emb)

    if not negative:
        save_dir = f"formula_plausibility_files/{kg}/{kg}_blind_{rels_pair}.pkl"
    else:
        save_dir = f"formula_plausibility_files/{kg}/{kg}_{strategy}_{rels_pair}.pkl"
    print(save_dir)
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    with open(save_dir, "wb") as f:
        pickle.dump(mapped_emb, f)
    
    return mapped_emb

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def s_p(rels_all, rel_org, results):
    """
        Calculates the relation of size of a target fact to sum of other facts
    """
    pred_size = int(results.loc[results['relation'] == rel_org, 'edges'].iloc[0])
    total_size = 0
    for rel in rels_all:
        total_size += int(results.loc[results['relation'] == rel, 'edges'].iloc[0]) 
    
    return pred_size / total_size

def ba_p(rels_all, rel_org, results):
    """
        Calculates the relation of Balanced Accuracy of a target model to sum of other models
    """
    pred_ba = float(results.loc[results['relation'] == rel_org, 'balanced_accuracy'].values[0][:5]) 
    total_ba = 0
    for rel in rels_all:
        total_ba += float(results.loc[results['relation'] == rel, 'balanced_accuracy'].values[0][:5])
    
    return pred_ba / total_ba

def w_p(s_p, ba_p):
    beta = 0.5
    return (1 + beta**2) * (s_p * ba_p) / ((beta**2 * s_p) + ba_p)

def delta_w(score_main, score_alt, w_main, w_alt):
    score_alt_w = {}
    for rel in list(score_alt.keys()):
        score_alt_w[rel] = score_alt[rel] * w_alt[rel]

    all_alt_scores = np.stack(list(score_alt_w.values()))
    d_w = (score_main * w_main) - np.max(all_alt_scores, axis=0) 
    return d_w  

def p_base(model, samples):
    # probabilities for positive class
    pos_index = list(model.classes_).index(1)
    pos_probs = model.predict_proba(samples)[:, pos_index]
    return pos_probs

def p_max(rel_main, rels_alt, score_main, score_alt, results, lmbda):

    rels_all = []
    rels_all.append(rel_main)
    rels_all.extend(rels_alt)
    
    # Main relation size and balanced accuracy
    s_p_main = s_p(rels_all, rel_main, results)
    ba_p_main = ba_p(rels_all, rel_main, results)
    w_main = w_p(s_p_main, ba_p_main)
    if not len(rels_alt):
        return sigmoid((lmbda * (score_main * w_main)))


    # Alternative relations size and balanced accuracy
    s_p_alt, ba_p_alt, w_alt = {}, {}, {}
    for rel_a in rels_alt:
        s_p_alt[rel_a] = s_p(rels_all, rel_a, results)
        ba_p_alt[rel_a] = ba_p(rels_all, rel_a, results)
        w_alt[rel_a] = w_p(s_p_alt[rel_a], ba_p_alt[rel_a])
    
    d_w = delta_w(score_main, score_alt, w_main, w_alt)
    p_score = sigmoid(lmbda * d_w)

    return p_score  # p_score = {alt_rel1: [p_scores1], alt_rel2: [p_scores2], ...}

def p_comb(rel_main, rels_alt, score_main, score_alt, results, max_lmbda=10.0):
    p_max_scores = p_max(rel_main, rels_alt, score_main, score_alt, results, lmbda=max_lmbda)
    return (0.5 * score_main) + (0.5 * p_max_scores)

def p_softmax(rel_main, rels_alt, score_main, score_alt, results):
    lmbda = 10
    rels_all = [rel_main] + rels_alt

    s_p_main = s_p(rels_all, rel_main, results)
    ba_p_main = ba_p(rels_all, rel_main, results)
    w_main = w_p(s_p_main, ba_p_main)

    numerator = np.exp(lmbda * score_main * w_main)
    if len(rels_alt) == 0:
        return score_main * w_main

    s_p_alt, ba_p_alt, w_alt = {}, {}, {}
    for rel_a in rels_alt:
        s_p_alt[rel_a] = s_p(rels_all, rel_a, results)
        ba_p_alt[rel_a] = ba_p(rels_all, rel_a, results)
        w_alt[rel_a] = w_p(s_p_alt[rel_a], ba_p_alt[rel_a])

    if len(rels_alt) == 1:
        alt_score = list(score_alt.values())[0]
        alt_score_w = alt_score * w_alt[list(score_alt.keys())[0]]
        denominator = numerator + np.exp(lmbda * alt_score_w)
        return numerator / denominator

    alt_score_w = {rel_a: score_alt[rel_a] * w_alt[rel_a] for rel_a in rels_alt}
    stacked_alt_score_w = np.stack(list(alt_score_w.values()))
    score_best_alt_w = np.max(stacked_alt_score_w, axis=0)
    score_2nd_best_alt_w = np.partition(stacked_alt_score_w, -2, axis=0)[-2]
    denominator = numerator + np.exp(lmbda * score_best_alt_w) + np.exp(lmbda * score_2nd_best_alt_w)
    return numerator / denominator


def find_schema_fact(schema_fact: str):
    """
        Finds which KG a schema fact belongs to, along with the
        relations sharing the same subject/object types.
    """
    for kg, relations in kg_schemas.items():
        for rels_pair, facts in relations.items():
            if schema_fact in facts:
                return kg, rels_pair, facts

    raise Exception(f"Schema fact '{schema_fact}' was not found in any KG schema.")

def get_pair_embedding(kg: str, emb_name: str, source_id: str, target_id: str):
    """
        Looks up the embeddings of a single source/target ID pair and combines
        them the same way models were trained on (element-wise product).
    """
    embedding = pd.read_csv(f"store_embeddings/{emb_name}/{kg}.csv")

    source_row = embedding[embedding['name'] == source_id]
    target_row = embedding[embedding['name'] == target_id]
    if source_row.empty:
        raise Exception(f"Source id '{source_id}' not found in {kg} embeddings.")
    if target_row.empty:
        raise Exception(f"Target id '{target_id}' not found in {kg} embeddings.")

    source_emb = ast.literal_eval(source_row.squeeze().to_list()[1])
    target_emb = ast.literal_eval(target_row.squeeze().to_list()[1])
    mul_emb = np.array(source_emb) * np.array(target_emb)

    return mul_emb.reshape(1, -1)

def get_node_types(kg: str, node_ids: list):
    """
        Looks up the KG node type for each of the given node IDs.
    """
    nodes = pd.read_csv(f"data/nodes_{kg}.csv", usecols=['URI:ID', ':LABEL'], low_memory=False)

    types = {}
    for node_id in node_ids:
        row = nodes[nodes['URI:ID'] == node_id]
        if row.empty:
            raise Exception(f"Node id '{node_id}' not found in {kg} nodes.")
        types[node_id] = row[':LABEL'].iloc[0]

    return types

def check_types(schema_fact: str, source_id: str, target_id: str, kg: str):
    """
        Validates that source_id/target_id have the correct types
        declared by the schema fact.
    """
    sub_type, _, obj_type = schema_fact.split(' - ')
    node_types = get_node_types(kg, [source_id, target_id])

    source_types = str(node_types[source_id]).split(';')
    target_types = str(node_types[target_id]).split(';')

    if sub_type not in source_types:
        raise Exception(f"source_id '{source_id}' has type '{node_types[source_id]}', expected '{sub_type}' for schema fact '{schema_fact}'.")
    if obj_type not in target_types:
        raise Exception(f"target_id '{target_id}' has type '{node_types[target_id]}', expected '{obj_type}' for schema fact '{schema_fact}'.")

def get_medians(kg: str, schema_fact: str, formula: str = 'comb', max_lmbda: float = 10.0, strategy: str = 'c-b-n-s'):
    """
        Retrieves the median of the plausibility scores for positive (blind) and
        negative samples of the given schema fact, from the precomputed scores.
    """
    suffix = f'{formula}_{int(max_lmbda)}' if formula in ('gain', 'comb') else formula
    medians = {}
    for split in ('blind', 'neg'):
        path = f"plausibility_scores/{kg}/{strategy}/{kg}_{strategy}_{split}_{suffix}_scores.pkl"
        with open(path, 'rb') as f:
            scores = pickle.load(f)

        for facts in scores.values():
            if schema_fact in facts:
                medians[split] = float(np.median(facts[schema_fact]))
                break
        else:
            raise Exception(f"Schema fact '{schema_fact}' not found in {path}.")

    return medians['blind'], medians['neg']

def compute_plausibility_score(schema_fact: str, source_id: str, target_id: str, formula: str = 'comb', max_lmbda: float = 10.0):
    kg, rels_pair, valid_rels = find_schema_fact(schema_fact)
    check_types(schema_fact, source_id, target_id, kg)

    median_positive, median_negative = get_medians(kg, schema_fact, formula=formula, max_lmbda=max_lmbda)

    emb_name = 'transe'
    strategy = 'c-b-n-s'
    model_name = 'RF'

    results = pd.read_csv(f'experiments/{kg}/{emb_name}/RF_{strategy}.csv')[['relation', 'edges', 'balanced_accuracy']]

    models = get_models(kg=kg, emb_name=emb_name, strategy=strategy, model=model_name, valid_rels=valid_rels)
    if schema_fact not in models:
        raise Exception(f"No trained {model_name} model found for schema fact '{schema_fact}' in {kg}.")

    sample = get_pair_embedding(kg=kg, emb_name=emb_name, source_id=source_id, target_id=target_id)

    rels_alt = [rel for rel in valid_rels if rel != schema_fact and rel in models]

    score_main = p_base(models[schema_fact], sample)
    score_alt = {rel: p_base(models[rel], sample) for rel in rels_alt}

    if formula == 'base':
        score = score_main
    elif formula == 'gain':
        score = p_max(schema_fact, rels_alt, score_main, score_alt, results, max_lmbda)
    elif formula == 'softmax':
        score = p_softmax(schema_fact, rels_alt, score_main, score_alt, results)
    elif formula == 'comb':
        score = p_comb(schema_fact, rels_alt, score_main, score_alt, results, max_lmbda)
    else:
        raise Exception(f"Formula name does not exist: {formula!r}. Choose from 'base', 'gain', 'softmax', 'comb'.")

    if score[0] >= median_positive:
        print(f"Triple is in the Plausible region, Plausibility Score: {score[0]:.4f}", color='green')
        print(f"Implausible and Uncertain Region Threshold: {median_negative:.4f} | Uncertain and Plausible Region Threshold: {median_positive:.4f}")
    elif score[0] <= median_negative:
        print(f"Triple is in the Implausible region, Plausibility Score: {score[0]:.4f}", color='red')
        print(f"Implausible and Uncertain Region Threshold: {median_negative:.4f} | Uncertain and Plausible Region Threshold: {median_positive:.4f}")
    else:
        print(f"Triple is in the Uncertain region, Plausibility Score: {score[0]:.4f}", color='yellow')
        print(f"Implausible and Uncertain Region Threshold: {median_negative:.4f} | Uncertain and Plausible Region Threshold: {median_positive:.4f}")