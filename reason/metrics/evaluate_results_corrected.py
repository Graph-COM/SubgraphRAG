import sys
sys.path.append('../')
from preprocess.prepare_data import get_data
from preprocess.prepare_prompts import unique_preserve_order

import argparse
import glob
import json
import os
import re
import string
import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from datasets import load_dataset


def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # remove <pad> token:
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split())
    return s


def match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)
    return s2 in s1


def remove_duplicates(input_list):
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def get_pred(prediction, split=None):
    if split is not None:
        return prediction.split(split)

    res = [p for p in prediction.split("\n") if 'ans:' in p and 'none' not in p.lower()]
    # if len(res) == 0:
    #     res = [p for p in prediction.split("\n") if '-' in p]
    # if len(res) == 0:
    #     res = [p for p in prediction.split("\n") if '*' in p]
    # if len(res) == 0:
    #     res = [prediction]
    if len(res) >= 1:
        res = [p for p in res if "ans: not available" not in p.lower() and "ans: no information available" not in p.lower()]
    return remove_duplicates(res)


def eval_recall(prediction, answer, double_check):
    prediction = deepcopy(prediction)
    prediction = sorted(prediction, key=len, reverse=True)
    matched = 0.
    for a in answer:
        for pred in prediction:
            if match(pred, a):
                matched += 1
                prediction.remove(pred)
                break
            elif double_check:
                if match(a, pred.split('ans:')[-1].strip()) or match(a, pred):
                    matched += 1
                    prediction.remove(pred)
                    break
    return matched / len(answer), matched, len(answer)


def eval_precision(prediction, answer, double_check):
    prediction = deepcopy(prediction)
    prediction = sorted(prediction, key=len, reverse=True)
    num_pred = len(prediction)
    if num_pred == 0:
        return 0, 0, 0
    matched = 0.
    for a in answer:
        for pred in prediction:
            if match(pred, a):
                matched += 1
                prediction.remove(pred)
                break
            elif double_check:
                if match(a, pred.split('ans:')[-1].strip()) or match(a, pred):
                    matched += 1
                    prediction.remove(pred)
                    break
    return matched / num_pred, matched, num_pred


def eval_f1(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)


def eval_hit(prediction, answer, double_check):
    if len(prediction) == 0:
        return 0
    for a in answer:
        if match(prediction[0], a):
            return 1
        elif double_check:
            if match(a, prediction[0].split('ans:')[-1].strip()):
                return 1
    return 0


def get_subgraph_dict(dataset_name):
    print(f"Loading subgraph dict for {dataset_name}...")
    data = load_dataset(os.path.join("rmanluo", f"RoG-{dataset_name}"), split='test')
    subgraph_dict = {}
    for each in tqdm(data):
        subgraph = each['graph']
        subgraph_dict[each['id']] = []
        for each_triplet in subgraph:
            subgraph_dict[each['id']].append(each_triplet[0])
            subgraph_dict[each['id']].append(each_triplet[2])
    return subgraph_dict


def get_all_retrieved_entities(triplet_list):
    all_ent = set()
    for triplet in triplet_list:
        all_ent.add(triplet[0])
        all_ent.add(triplet[2])
    return list(all_ent)


def eval_hal_score(prediction, answer, double_check, good_sample, no_ans, subgraph_ent, stats):
    answer = deepcopy(answer)
    score = 0
    stats['total_samples'] += 1
    if good_sample:
        stats['total_g_samples'] += 1
        if no_ans:
            stats['g_no_ans'] += 1
            return 0, stats

        for pred in prediction:
            stats['total_ans'] += 1
            stats['total_g_ans'] += 1
            no_match = True
            for a in answer:
                if match(pred, a) or (double_check and match(a, pred.split('ans:')[-1].strip())) or (double_check and match(a, pred)):
                    score += 1
                    stats['g_c'] += 1
                    no_match = False
                    answer.remove(a)

                    not_in_graph = True
                    for ent in subgraph_ent:
                        if pred.lower().split('ans:')[-1].strip() in ent.lower() or ent.lower() in pred.lower():
                            not_in_graph = False
                            stats['g_c_in_graph'] += 1
                            break
                    if not_in_graph:
                        stats['g_c_out_graph'] += 1
                    break

            if no_match:
                score += -1
                stats['g_w'] += 1

                not_in_graph = True
                for ent in subgraph_ent:
                    if pred.lower().split('ans:')[-1].strip() in ent.lower() or ent.lower() in pred.lower():
                        not_in_graph = False
                        stats['g_w_in_graph'] += 1
                        break
                if not_in_graph:
                    stats['g_w_out_graph'] += 1

        return score / len(prediction), stats

    else:
        stats['total_b_samples'] += 1
        if no_ans:
            stats['b_no_ans'] += 1
            return 1, stats

        else:
            for pred in prediction:
                stats['total_ans'] += 1
                stats['total_b_ans'] += 1
                no_match = True
                for ent in subgraph_ent:
                    if pred.lower().split('ans:')[-1].strip() in ent.lower() or ent.lower() in pred.lower():
                        score += -1
                        stats['b_in_graph'] += 1
                        no_match = False
                        break
                if no_match:
                    score += -1.5
                    no_match_ans = True
                    for a in answer:
                        if match(pred, a) or (double_check and match(a, pred.split('ans:')[-1].strip())) or (double_check and match(a, pred)):
                            stats['b_out_graph_c'] += 1
                            no_match_ans = False
                            answer.remove(a)
                            break
                    if no_match_ans:
                        stats['b_out_graph_w'] += 1

            return score / len(prediction), stats


def eval_results(predict_file, cal_f1=True, split=None, subset=False, bad_samples=False, eval_hops=-1):
    # only one of subset and bad_samples can be True
    assert not (subset and bad_samples)

    # predict_file = os.path.join(result_path, 'predictions.jsonl')
    if subset:
        eval_name = f'subset_hop{eval_hops}_detailed_eval_result_corrected.jsonl'
    elif bad_samples:
        eval_name = f'badSamples_hop{eval_hops}_detailed_eval_result_corrected.jsonl'
    else:
        eval_name = f'full_hop{eval_hops}_detailed_eval_result_corrected.jsonl'

    detailed_eval_file = predict_file.replace('predictions.jsonl', eval_name)
    # Load results
    acc_list = []
    hit_list = []
    f1_list = []
    precision_list = []
    recall_list = []
    total_pred = 0
    total_answer = 0
    total_match = 0
    hal_score_list = []

    if "webqsp" in predict_file:
        samples_to_eval_path = "./scored_triples/webqsp_240912_unidir_test.pth"
        dataset_name = "webqsp"
    elif "cwq" in predict_file:
        samples_to_eval_path = "./scored_triples/cwq_240907_unidir_test.pth"
        dataset_name = "cwq"
    else:
        raise NotImplementedError
    pred_file_path = f"./results/KGQA/{dataset_name}/RoG/test/results_gen_rule_path_RoG-{dataset_name}_RoG_test_predictions_3_False_jsonl/predictions.jsonl"
    prompt_mode = predict_file.split('/')[-1].split('-')[0]
    triplets = get_data(dataset_name, pred_file_path, samples_to_eval_path, 'test', prompt_mode)
    triplets_dict = {}
    for each in triplets:
        if split == '\n':
            # RoG
            triplets_dict[each['id']] = each['good_triplets_rog']
        else:
            input_triplets = [(triplet[0], triplet[1], triplet[2]) for triplet in each['scored_triplets']]
            triplets_dict[each['id']] = unique_preserve_order(input_triplets)[:int(prompt_mode.split('_')[-1])]

    samples_to_eval = torch.load(samples_to_eval_path, weights_only=False)

    # if not subset and not bad_samples:
    #     subgraph_dict = get_subgraph_dict(dataset_name)
    total_cnt = 0
    no_ans_cnt = 0
    stats = {'g_no_ans': 0, 'g_c': 0, 'g_w': 0, 'b_no_ans': 0, 'b_in_graph': 0, 'b_out_graph_c': 0, 'b_out_graph_w': 0,
             'total_ans': 0, 'total_g_samples': 0, 'total_b_samples': 0, 'total_samples': 0,
             'total_g_ans': 0, 'total_b_ans': 0,
             'g_c_out_graph': 0, "g_w_out_graph": 0, 'g_c_in_graph': 0, 'g_w_in_graph': 0}
    with open(predict_file, 'r') as f, open(detailed_eval_file, 'w') as f2:
        for line in tqdm(f):
            try:
                data = json.loads(line)
            except:
                print(line)
                continue
            id = data['id']
            if eval_hops > 0:
                if eval_hops == 3:
                    if samples_to_eval[id]['max_path_length'] is None or samples_to_eval[id]['max_path_length'] < 3:
                        continue
                elif samples_to_eval[id]['max_path_length'] != eval_hops:
                    continue

            # if subset and id not in samples_to_eval:
            #     continue
            # if bad_samples and id in samples_to_eval:
            #     continue
            if subset and not samples_to_eval[id]['a_entity_in_graph']:
                continue
            if bad_samples and samples_to_eval[id]['a_entity_in_graph']:
                continue
            prediction = data['prediction']
            # answer = data['ground_truth']
            answer = sorted(remove_duplicates(data['ground_truth']), key=len, reverse=True)
            if 'when' in data['question'].lower() or 'what year' in data['question'].lower():
                for idx in range(len(answer)):
                    if '-' in answer[idx] and answer[idx].split('-')[0].isdigit():
                        answer[idx] = answer[idx].split('-')[0]

            question = data['question']
            double_check = any([keyword in question.lower() for keyword in ['when', 'what year', 'which year', 'where', 'sport', "what countr", "language", 'nba finals', 'world series']])
            # double_check = any([keyword in question.lower() for keyword in ['when', 'what year', 'sport']])
            if cal_f1:
                prediction = get_pred(prediction, split)
                total_cnt += 1
                no_ans_flag = False
                if split == '\n':
                    # RoG
                    if len(prediction) == 0:
                        no_ans_cnt += 1
                        no_ans_flag = True
                else:
                    if len(prediction) == 0 or 'ans:' not in data['prediction'] or "ans: not available" in data['prediction'].lower() or "ans: no information available" in data['prediction'].lower():
                        no_ans_cnt += 1
                        no_ans_flag = True

                precision_score, matched_1, num_pred = eval_precision(prediction, answer, double_check)
                recall_score, matched_2, num_answer = eval_recall(prediction, answer, double_check)
                f1_score = eval_f1(precision_score, recall_score)
                hit = eval_hit(prediction, answer, double_check)

                if not subset and not bad_samples:
                    subgraph_ent = get_all_retrieved_entities(triplets_dict[id])
                    hal_score, stats = eval_hal_score(prediction, answer, double_check, samples_to_eval[id]['a_entity_in_graph'], no_ans_flag, subgraph_ent, stats)
                else:
                    hal_score = 0
                    stats = None

                assert matched_1 == matched_2
                total_pred += num_pred
                total_answer += num_answer
                total_match += matched_1

                hal_score_list.append(hal_score)
                f1_list.append(f1_score)
                precision_list.append(precision_score)
                recall_list.append(recall_score)
                hit_list.append(hit)
                acc_list.append(recall_score)
                f2.write(json.dumps({'id': id, 'prediction': prediction, 'ground_truth': answer, 'hit': hit, 'f1': f1_score, 'precision': precision_score, 'recall': recall_score, 'hal_score': hal_score}) + '\n')
            else:
                raise NotImplementedError
    if len(hit_list) == 0:
        null_out = [0] * 13
        null_out.append(None)
        return null_out

    avg_hit = sum(hit_list) * 100 / len(hit_list)
    avg_f1 = sum(f1_list) * 100 / len(f1_list)
    avg_precision = sum(precision_list) * 100 / len(precision_list)
    avg_recall = sum(recall_list) * 100 / len(recall_list)
    avg_hal_score = sum(hal_score_list) / len(hal_score_list)
    avg_hal_score = (avg_hal_score + 1.5) / (1 + 1.5) * 100

    num_exact_match = (np.array(f1_list) == 1).sum() / len(f1_list) * 100
    num_totally_wrong = (np.array(recall_list) == 0).sum() / len(recall_list) * 100

    micro_precision = total_match / total_pred
    micro_recall = total_match / total_answer
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)

    result_str = f"Hit: {avg_hit}, F1: {avg_f1}, precision: {avg_precision}, Recall: {avg_recall},  Exact Match: {num_exact_match}, Totally Wrong: {num_totally_wrong}, Hal Score: {avg_hal_score}"
    print(result_str)
    print(f"Micro F1: {micro_f1}, Micro precision: {micro_precision}, Micro Recall: {micro_recall}")
    print(f"Total number of samples: {total_cnt}, no answer samples: {no_ans_cnt}, ratio: {no_ans_cnt / total_cnt}")

    result_name = 'eval_result_corrected.txt'
    eval_result_path = predict_file.replace('predictions.jsonl', result_name)
    with open(eval_result_path, 'w') as f:
        f.write(result_str)
    return avg_hit, avg_f1, avg_precision, avg_recall, num_exact_match, num_totally_wrong, micro_f1, micro_precision, micro_recall, total_cnt, no_ans_cnt, no_ans_cnt / total_cnt, avg_hal_score, stats
