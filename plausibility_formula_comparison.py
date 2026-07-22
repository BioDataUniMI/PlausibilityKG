import os
import pickle
import csv

import numpy as np
import pandas as pd
from tqdm import tqdm

from plausibility_score_computation import (
    miRNA_KG_relations, PKT_KG_relations, Hetionet_relations, PrimeKG_relations, OptimusKG_relations,
    s_p, ba_p, w_p, p_base, p_max, p_comb,
)

KGS = ['miRNA-KG', 'PKT-KG', 'Hetionet', 'PrimeKG', 'OptimusKG']
STRATEGIES = ['c-b-n-s', 'r-n-s']
FORMULAS = ['base', 'gain', 'softmax', 'comb']
FORMULAS_DIST = ['base_dist', 'gain_dist', 'softmax_dist', 'comb_dist']
MAX_LMBDA = 10

KG_RELATIONS = {
    'miRNA-KG': miRNA_KG_relations,
    'PKT-KG': PKT_KG_relations,
    'Hetionet': Hetionet_relations,
    'PrimeKG': PrimeKG_relations,
    'OptimusKG': OptimusKG_relations,
}


def p_base_dist(rel_main, score_main, score_alt):
    rels_all = [rel_main] + list(score_alt.keys())
    score_alt[rel_main] = score_main
    return {rel: score_alt[rel] for rel in rels_all}


def p_gain_dist(rel_main, rels_alt, score_main, score_alt, results, lmbda):
    rels_all = [rel_main] + rels_alt
    score_all = score_alt.copy()
    score_all[rel_main] = score_main

    ps_scores = {}
    for temp_main in rels_all:
        rels_alt_temp = [r for r in rels_all if r != temp_main]
        score_alt_temp = {k: score_all[k] for k in rels_alt_temp}
        ps_scores[temp_main] = p_max(temp_main, rels_alt_temp, score_all[temp_main], score_alt_temp, results, lmbda)
    return ps_scores


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


def p_softmax_dist(rel_main, rels_alt, score_main, score_alt, results):
    rels_all = [rel_main] + rels_alt
    score_all = score_alt.copy()
    score_all[rel_main] = score_main

    ps_scores = {}
    for temp_main in rels_all:
        rels_alt_temp = [r for r in rels_all if r != temp_main]
        score_alt_temp = {k: score_all[k] for k in rels_alt_temp}
        ps_scores[temp_main] = p_softmax(temp_main, rels_alt_temp, score_all[temp_main], score_alt_temp, results)
    return ps_scores


def p_comb_dist(rel_main, rels_alt, score_main, score_alt, results, max_lmbda=10.0):
    rels_all = [rel_main] + rels_alt
    score_all = score_alt.copy()
    score_all[rel_main] = score_main

    ps_scores = {}
    for temp_main in rels_all:
        rels_alt_temp = [r for r in rels_all if r != temp_main]
        score_alt_temp = {k: score_all[k] for k in rels_alt_temp}
        ps_scores[temp_main] = p_comb(temp_main, rels_alt_temp, score_all[temp_main], score_alt_temp, results, max_lmbda)
    return ps_scores


# --- model / input loading ---

def _experiments_csv(kg, emb, strategy, model='RF'):
    path = f'experiments/{kg}/{emb}/{model}_{strategy}.csv'
    if not os.path.exists(path):
        raise FileNotFoundError(f"No experiments csv found for {kg}/{emb}/{strategy}")
    return path


def get_models(kg: str, emb_name: str, strategy: str, model: str, valid_rels: list):
    import glob
    models = {}
    model_path = f"dumps_models/{kg}/{emb_name}/{strategy}/err_beta_1/{model}/*.pkl"

    for f in glob.glob(model_path):
        relation = f.split('/')[-1]
        start_pos = relation.find('_model_')
        relation = relation[start_pos + 7:]
        relation = relation[:-len('_fixed.pkl')] if relation.endswith('_fixed.pkl') else relation[:-len('.pkl')]
        if relation not in valid_rels:
            continue
        with open(f, "rb") as model_file:
            models[relation] = pickle.load(model_file)

    if not len(models):
        raise Exception("No model is obtained! Check the file path")
    return models


def get_input(kg: str, emb_name: str, strategy: str, needed_models: list, rels_pair: str, negative: bool):
    import ast
    import glob

    triples_dict = {}

    if not negative:
        dir_add = f"blind_test_occ/{kg}/*.csv"
    else:
        dir_add = f"negative_samples/{kg}/{strategy}/*.csv"
        blind_size = {}
        for f in glob.glob(f"blind_test_occ/{kg}/*.csv"):
            relation = f.split('/')[-1][:-4]
            if relation not in needed_models:
                continue
            blind_size[relation] = pd.read_csv(f).rename(columns={'subcject': 'subject'}).shape[0]

    for f in glob.glob(dir_add):
        relation = f.split('/')[-1][:-4]
        if relation not in needed_models:
            continue

        if not negative:
            triple_df = pd.read_csv(f).rename(columns={'subcject': 'subject'})
        else:
            triple_df = pd.read_csv(f).rename(columns={'source': 'subject', 'target': 'object'})
            triple_df = triple_df.sample(n=min(blind_size[relation], len(triple_df)), random_state=5)

        if triple_df.shape[0] == 0:
            continue
        triples_dict[relation] = triple_df

    embedding = pd.read_csv(f"store_embeddings/{emb_name}/{kg}.csv")

    mapped_emb = {}
    for rel in tqdm(triples_dict, desc=f"Mapping {kg}/{rels_pair}"):
        mapped_emb[rel] = []
        for row in triples_dict[rel].iterrows():
            uri = 'name' if emb_name == 'transe' else 'node_id'
            subject_emb = embedding[embedding[uri] == row[1]['subject']].squeeze().to_list()[1]
            object_emb = embedding[embedding[uri] == row[1]['object']].squeeze().to_list()[1]
            subject_emb = ast.literal_eval(subject_emb)
            object_emb = ast.literal_eval(object_emb)
            mul_emb = np.array(subject_emb) * np.array(object_emb)
            mapped_emb[rel].append(mul_emb)

    if not negative:
        save_dir = f"formula_plausibility_files/{kg}/{kg}_blind_{rels_pair}.pkl"
    else:
        save_dir = f"formula_plausibility_files/{kg}/{kg}_{strategy}_{rels_pair}.pkl"
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    with open(save_dir, "wb") as f:
        pickle.dump(mapped_emb, f)

    return mapped_emb


def get_score(kg: str, embeddings, models, formula, results, max_lmbda):
    relations = list(embeddings.keys())
    rel_parts = {}
    for rel in relations:
        sub, pred, obj = rel.split(' - ')
        rel_parts[rel] = {'subject': sub, 'object': obj, 'predicate': pred}

    plausibility = {}
    for rel in relations:
        sub, obj = rel_parts[rel]['subject'], rel_parts[rel]['object']
        preds_alt = [rel_parts[k]['predicate'] for k in rel_parts if rel_parts[k]['subject'] == sub and rel_parts[k]['object'] == obj]
        rels_alt = [f'{sub} - {p} - {obj}' for p in preds_alt]
        rels_alt.remove(rel)

        model_main = models[rel]
        embeddings_main = embeddings[rel]
        score_main = p_base(model_main, embeddings_main)

        score_alt = {}
        for rel_a in rels_alt:
            score_alt[rel_a] = p_base(models[rel_a], embeddings_main)

        if formula == 'gain':
            plausibility[rel] = p_max(rel, rels_alt, score_main, score_alt, results, max_lmbda)
        elif formula == 'gain_dist':
            plausibility[rel] = p_gain_dist(rel, rels_alt, score_main, score_alt, results, max_lmbda)
        elif formula == 'softmax':
            plausibility[rel] = p_softmax(rel, rels_alt, score_main, score_alt, results)
        elif formula == 'softmax_dist':
            plausibility[rel] = p_softmax_dist(rel, rels_alt, score_main, score_alt, results)
        elif formula == 'comb':
            plausibility[rel] = p_comb(rel, rels_alt, score_main, score_alt, results, max_lmbda)
        elif formula == 'comb_dist':
            plausibility[rel] = p_comb_dist(rel, rels_alt, score_main, score_alt, results, max_lmbda)
        elif formula == 'base':
            plausibility[rel] = score_main
        elif formula == 'base_dist':
            plausibility[rel] = p_base_dist(rel, score_main, score_alt)
        else:
            raise Exception(f"Formula name does not exist: {formula!r}")

    return plausibility


def _load_cached_pickle(path):
    """
    Load and returns the unpickled file of `path`
    """

    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except (EOFError, pickle.UnpicklingError):
        return None


def run(kg, emb_name, strategy, model, formula, results, rels_pair, valid_rels, negative, max_lmbda):
    models = get_models(kg=kg, emb_name=emb_name, strategy=strategy, model=model, valid_rels=valid_rels)
    models_name = list(models.keys())

    if not negative:
        cache_path = f"formula_plausibility_files/{kg}/{kg}_blind_{rels_pair}.pkl"
    else:
        cache_path = f"formula_plausibility_files/{kg}/{kg}_{strategy}_{rels_pair}.pkl"

    triple_embs = _load_cached_pickle(cache_path)
    if triple_embs is None:
        triple_embs = get_input(kg=kg, emb_name=emb_name, strategy=strategy, needed_models=models_name, rels_pair=rels_pair, negative=negative)

    relations = list(triple_embs.keys())
    models_subset = {k: models[k] for k in models if k in relations}
    triple_embs = {k: triple_embs[k] for k in relations if k in models_subset}

    return get_score(kg=kg, embeddings=triple_embs, models=models_subset, formula=formula, results=results, max_lmbda=max_lmbda)


def exe_ps(kg, rels_pair, strategy, formula, negative, max_lmbda):
    valid_rels = KG_RELATIONS[kg][rels_pair]
    emb = 'transe'
    results = pd.read_csv(_experiments_csv(kg, emb, strategy))[['relation', 'edges', 'balanced_accuracy']]
    return run(kg=kg, emb_name=emb, strategy=strategy, model='RF', formula=formula, results=results,
               rels_pair=rels_pair, valid_rels=valid_rels, negative=negative, max_lmbda=max_lmbda)


# --- evaluation metrics ---

def mean_distance_from_ideal(dict_list: list, negative: bool):
    """
    Mean/std distance from the "wrong" end of the [0, 1] plausibility range:
    distance from 0 for negative samples, distance from 1 for blind positives.
    Lower is better in both cases.
    """
    all_values = []
    all_values_keys = {}
    for scores_dict in dict_list:
        for rel in scores_dict.keys():
            if negative:
                all_values.append(np.abs(scores_dict[rel]))
                all_values_keys[rel] = np.abs(scores_dict[rel])
            else:
                all_values.append(np.abs(1 - scores_dict[rel]))
                all_values_keys[rel] = np.abs(1 - scores_dict[rel])

    all_values = np.concatenate(all_values)
    return np.mean(all_values), np.std(all_values), all_values_keys


def mean_distance_combined(pos_dict_list, neg_dict_list):
    """
    Mean/std of the combined blind-positive and negative distance errors
    (i.e. mean_distance_from_ideal for both splits pooled together).
    """
    all_values = []
    all_values_keys = {}
    for scores_dict in pos_dict_list:
        for rel in scores_dict.keys():
            all_values.append(np.abs(1 - scores_dict[rel]))
            all_values_keys[rel] = np.abs(1 - scores_dict[rel])

    for scores_dict in neg_dict_list:
        for rel in scores_dict.keys():
            all_values.append(np.abs(scores_dict[rel]))
            pos_err = all_values_keys[rel]
            neg_err = np.abs(scores_dict[rel])
            if len(pos_err) == len(neg_err):
                all_values_keys[rel] = np.stack([pos_err, neg_err])
            else:
                n = min(len(pos_err), len(neg_err))
                all_values_keys[rel] = np.mean(np.stack([pos_err[:n], neg_err[:n]]), axis=0).mean()

    all_values_keys = {k: np.mean(v, axis=0) if np.ndim(v) > 0 else v for k, v in all_values_keys.items()}
    all_values = np.concatenate(all_values)
    return np.mean(all_values), np.std(all_values), all_values_keys


def mean_separation_from_alternatives(dict_list):
    """
    Mean/std of how far a schema fact's plausibility score is from its
    competing alternative schema facts (same subject/object types, different
    predicate). Larger separation means the formula better distinguishes the
    target relation from its competitors.
    """
    all_values = []
    all_values_keys = {}
    for scores_dict in dict_list:
        for main_key, inner_dict in scores_dict.items():
            all_values_keys[main_key] = {}
            main_arr = inner_dict[main_key]
            for alt_key, alt_arr in inner_dict.items():
                if alt_key == main_key:
                    continue
                all_values.append(np.abs(main_arr - alt_arr))
                all_values_keys[main_key][alt_key] = np.abs(main_arr - alt_arr)

        if not all_values_keys[main_key]:
            del all_values_keys[main_key]

    all_values = np.concatenate(all_values)
    return np.mean(all_values), np.std(all_values), all_values_keys


def decision_agreement_rate(dict_list: list, alt: str, threshold: dict = None):
    """
    Agreement rate between the "correct" schema fact's plausibility score and
    the best (or worst) alternative competing schema fact, both thresholded
    into a binary plausible/implausible call. Used to check whether the
    correct schema fact would be selected over its top competitor.

    alt: 'best' or 'worst' - which competing alternative to compare against.
    threshold: optional per-relation threshold dict (defaults to 0.5).
    """
    all_signes = []
    all_values_keys = {}
    scores_used = {'P_correct': {}, 'P_best': {}, 'P_worst': {}, 'threshold_best': {}, 'threshold_worst': {}}
    scores_used_mean = {'P_correct': {}, 'P_best': {}, 'P_worst': {}, 'threshold_best': {}, 'threshold_worst': {}}
    for scores_dict in dict_list:
        for key, inner_dict in scores_dict.items():
            main_key = key
            filtered_inner_dict = dict(filter(lambda i: i[0] != main_key, inner_dict.items()))

            if not filtered_inner_dict:
                continue

            # get correct, best and worst scores
            scores_stack = np.stack(list(filtered_inner_dict.values()))
            scores_used['P_correct'][main_key] = inner_dict[main_key]
            scores_used['P_best'][main_key] = np.max(scores_stack, axis=0)
            scores_used['P_worst'][main_key] = np.min(scores_stack, axis=0)

            tsh = 0.5  # default threshold
            if threshold != None:
                tsh = threshold[main_key]
            # assign signs
            signed_main = [0 if s > tsh else 1 for s in scores_used['P_correct'][main_key]]
            if alt == 'best':
                signed_alt = [0 if s > tsh else 1 for s in scores_used['P_best'][main_key]]
            elif alt == 'worst':
                signed_alt = [0 if s > tsh else 1 for s in scores_used['P_worst'][main_key]]
            else:
                raise Exception("Alternative variable does not exist, choose between 'best' or 'worst'")

            # mean and std of scores to save
            scores_used_mean['P_correct'] = {k: (np.mean(v), np.std(v)) for k, v in scores_used['P_correct'].items()}
            scores_used_mean['P_best'] = {k: (np.mean(v), np.std(v)) for k, v in scores_used['P_best'].items()}
            scores_used_mean['P_worst'] = {k: (np.mean(v), np.std(v)) for k, v in scores_used['P_worst'].items()}
            scores_used_mean['threshold_best'] = {k: np.mean([scores_used_mean['P_correct'][k][0], scores_used_mean['P_best'][k][0]]) for k in scores_used['P_correct'].keys()}
            scores_used_mean['threshold_worst'] = {k: np.mean([scores_used_mean['P_correct'][k][0], scores_used_mean['P_worst'][k][0]]) for k in scores_used['P_correct'].keys()}

            # compute sign multiplication
            sign_xor = np.bitwise_xor(signed_main, signed_alt)
            all_signes.append(sign_xor)
            all_values_keys[main_key] = sign_xor

    all_signes = np.concatenate(all_signes)
    return np.mean(all_signes), np.std(all_signes), scores_used_mean, all_values_keys


# --- score computation and caching ---

def score_calculation(kg, strat, negatives, formulas, relation_pairs, max_lmbda, save=True):
    """Computes and caches the non-dist blind/neg scores used by NEG/POS/AVG."""
    for neg in negatives:
        for fr in formulas:
            temp_scores = {rp: exe_ps(kg, rp, strat, fr, neg, max_lmbda) for rp in relation_pairs}

            split = 'neg' if neg else 'blind'
            if fr in ('gain', 'comb'):
                score_dir = f"plausibility_scores/{kg}/{strat}/{kg}_{strat}_{split}_{fr}_{max_lmbda}_scores.pkl"
            else:
                score_dir = f"plausibility_scores/{kg}/{strat}/{kg}_{strat}_{split}_{fr}_scores.pkl"

            if save:
                os.makedirs(os.path.dirname(score_dir), exist_ok=True)
                with open(score_dir, "wb") as f:
                    pickle.dump(temp_scores, f)


def _load_or_compute_dist(kg, strat, fr, relation_pairs, max_lmbda):
    """Loads (or computes and caches) the per-relation-pair dist scores used by COM/BEST/WORST/etc."""
    base_fr = fr[:-len('_dist')]
    score_save_dir = f"p_score_log/{kg}/{base_fr}_scores_{kg}_{strat}.pkl"

    scores = _load_cached_pickle(score_save_dir)
    if scores is None:
        scores = {rp: exe_ps(kg, rp, strat, fr, False, max_lmbda) for rp in relation_pairs}
        os.makedirs(os.path.dirname(score_save_dir), exist_ok=True)
        with open(score_save_dir, "wb") as f:
            pickle.dump(scores, f)
    return scores


def save_results(scores, kg, name):
    os.makedirs(f'all_score_results/{kg}', exist_ok=True)
    with open(f'all_score_results/{kg}/{name}.pkl', 'wb') as f:
        pickle.dump(scores, f)


def compute_kg_metrics(kg, strategies, formulas, formulas_dist, relation_pairs, max_lmbda):
    """Computes NEG, POS, AVG, COM, BEST, WORST, BEST_balanced, WORST_balanced for one bioKG."""

    def blind_neg_path(strat, fr, split):
        if fr in ('gain', 'comb'):
            return f"plausibility_scores/{kg}/{strat}/{kg}_{strat}_{split}_{fr}_{max_lmbda}_scores.pkl"
        return f"plausibility_scores/{kg}/{strat}/{kg}_{strat}_{split}_{fr}_scores.pkl"

    NEG, POS, AVG = {}, {}, {}
    for strat in strategies:
        NEG[strat], POS[strat], AVG[strat] = {}, {}, {}
        for fr in formulas:
            with open(blind_neg_path(strat, fr, 'neg'), "rb") as f:
                scores_neg = pickle.load(f)
            with open(blind_neg_path(strat, fr, 'blind'), "rb") as f:
                scores_blind = pickle.load(f)

            _, _, NEG[strat][fr] = mean_distance_from_ideal(list(scores_neg.values()), negative=True)
            _, _, POS[strat][fr] = mean_distance_from_ideal(list(scores_blind.values()), negative=False)
            _, _, AVG[strat][fr] = mean_distance_combined(list(scores_blind.values()), list(scores_neg.values()))

    # su mirrors the notebook's own "su" - decision_agreement_rate()'s
    # scores_used_mean, carrying the per-relation threshold_best/threshold_worst
    # used by the balanced variants below. Same as the notebook, it's captured
    # from this same BEST/WORST loop rather than computed separately.
    COM, BEST, WORST, su = {}, {}, {}, {}
    for strat in strategies:
        COM[strat], BEST[strat], WORST[strat], su[strat] = {}, {}, {}, {}
        for fr in formulas_dist:
            scores = _load_or_compute_dist(kg, strat, fr, relation_pairs, max_lmbda)
            scores_list = list(scores.values())
            _, _, COM[strat][fr] = mean_separation_from_alternatives(scores_list)
            _, _, su[strat][fr[:-len('_dist')]], BEST[strat][fr] = decision_agreement_rate(scores_list, 'best')
            _, _, _, WORST[strat][fr] = decision_agreement_rate(scores_list, 'worst')

    # correct_best_worst/*.pkl caches "su" across runs, same as the notebook's
    # own save step - load it if already there, else save what was just computed.
    cbw_path = f"correct_best_worst/{kg}_correct_best_worst.pkl"
    cbw = _load_cached_pickle(cbw_path)
    if cbw is None:
        cbw = su
        os.makedirs(os.path.dirname(cbw_path), exist_ok=True)
        with open(cbw_path, "wb") as f:
            pickle.dump(cbw, f)

    BEST_balanced, WORST_balanced = {}, {}
    for strat in strategies:
        BEST_balanced[strat], WORST_balanced[strat] = {}, {}
        for fr in formulas_dist:
            scores = _load_or_compute_dist(kg, strat, fr, relation_pairs, max_lmbda)
            scores_list = list(scores.values())
            base_fr = fr[:-len('_dist')]
            _, _, _, BEST_balanced[strat][fr] = decision_agreement_rate(scores_list, 'best', cbw[strat][base_fr]['threshold_best'])
            _, _, _, WORST_balanced[strat][fr] = decision_agreement_rate(scores_list, 'worst', cbw[strat][base_fr]['threshold_worst'])

    return {
        'NEG': NEG, 'POS': POS, 'AVG': AVG, 'COM': COM,
        'BEST': BEST, 'WORST': WORST,
        'BEST_balanced': BEST_balanced, 'WORST_balanced': WORST_balanced,
    }


# --- CSV export ---

def _flatten_simple(vals):
    return np.concatenate([np.array([float(v)]) if np.ndim(v) == 0 else v for v in vals.values()])


def _flatten_avg(pos_vals, neg_vals):
    arrays = [np.atleast_1d(v) for v in pos_vals.values()] + [np.atleast_1d(v) for v in neg_vals.values()]
    return np.concatenate(arrays)


def _flatten_com(vals):
    arrays = [arr for inner in vals.values() for arr in inner.values()]
    return np.concatenate(arrays) if arrays else np.array([])


def build_csv_results(kgs=KGS, strategies=STRATEGIES, formulas=FORMULAS):
    os.makedirs('csv_results', exist_ok=True)

    for kg in kgs:
        base_dir = f'all_score_results/{kg}'
        if not os.path.exists(base_dir):
            print(f'Skipping {kg}: no results found')
            continue

        metrics = {}
        for name in ['NEG', 'POS', 'AVG', 'COM', 'BEST', 'WORST', 'BEST_balanced', 'WORST_balanced']:
            path = f'{base_dir}/{name}.pkl'
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    metrics[name] = pickle.load(f)
            else:
                metrics[name] = None

        rows = []
        for strat in strategies:
            for fr in formulas:
                fr_dist = fr + '_dist'
                row = {'view': kg, 'strategy': strat, 'formula': fr}

                for metric_name, flatten_fn, key in [
                    ('NEG', _flatten_simple, fr),
                    ('POS', _flatten_simple, fr),
                    ('COM', _flatten_com, fr_dist),
                    ('BEST', _flatten_simple, fr_dist),
                    ('WORST', _flatten_simple, fr_dist),
                    ('BEST_balanced', _flatten_simple, fr_dist),
                    ('WORST_balanced', _flatten_simple, fr_dist),
                ]:
                    data = metrics.get(metric_name)
                    try:
                        arr = flatten_fn(data[strat][key])
                        row[metric_name] = f"{round(float(np.mean(arr)), 4)} ± {round(float(np.std(arr)), 4)}"
                    except (KeyError, TypeError, ValueError):
                        row[metric_name] = None

                try:
                    arr = _flatten_avg(metrics["POS"][strat][fr], metrics["NEG"][strat][fr])
                    row["AVG"] = f"{round(float(np.mean(arr)), 4)} ± {round(float(np.std(arr)), 4)}"
                except (KeyError, TypeError, ValueError):
                    row["AVG"] = None

                rows.append(row)

        csv_path = f'csv_results/{kg}.csv'
        if rows:
            fieldnames = ['view', 'strategy', 'formula', 'NEG', 'POS', 'AVG', 'COM', 'BEST', 'WORST', 'BEST_balanced', 'WORST_balanced']
            with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(f'Saved {csv_path} ({len(rows)} rows)')


# --- main ---

def main(kgs=KGS, strategies=STRATEGIES, formulas=FORMULAS, formulas_dist=FORMULAS_DIST, max_lmbda=MAX_LMBDA):
    for kg in kgs:
        print(f"=== {kg} ===")
        relation_pairs = list(KG_RELATIONS[kg].keys())

        for strat in strategies:
            score_calculation(kg, strat, [True, False], formulas, relation_pairs, max_lmbda, save=True)

        results = compute_kg_metrics(kg, strategies, formulas, formulas_dist, relation_pairs, max_lmbda)
        for name, scores in results.items():
            save_results(scores, kg, name)

    build_csv_results(kgs, strategies, formulas)


if __name__ == "__main__":
    main()
