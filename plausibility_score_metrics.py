"""
Metrics used to evaluate the quality of plausibility scores 

These are used in to summarize how well a plausibility formula separates blind positive edges from negative
edges, and how a target schema fact's score compares to its alternatives.
"""

import numpy as np


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
