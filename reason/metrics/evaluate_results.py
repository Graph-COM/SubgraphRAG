import argparse
import glob
import json
import os
import re
import string
import torch
from .evaluate_results_corrected import get_pred


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

def eval_acc(prediction, answer):
    matched = 0.
    for a in answer:
        if match(prediction, a):
            matched += 1
    return matched / len(answer)

def eval_hit(prediction, answer, double_check):
    for a in answer:
        if "ans:" in prediction:
            all_pred = get_pred(prediction)
            for each_pred in all_pred:
                if match(each_pred, a):
                    return 1
                elif double_check and match(a, each_pred.split('ans:')[-1].strip()):
                    return 1
        else:
            if match(prediction, a):
                return 1
            elif double_check:
                all_pred = prediction.split("\n")
                for each_pred in all_pred:
                    if match(a, each_pred):
                        return 1

    return 0

def eval_f1(prediction, answer, double_check):
    if len(prediction) == 0:
        return 0, 0, 0
    matched = 0
    prediction_str = '\n'.join(prediction)
    all_pred = get_pred(prediction_str)
    for a in answer:
        if match(prediction_str, a):
            matched += 1
        elif double_check:
            for each_pred in all_pred:
                if match(a, each_pred.split('ans:')[-1].strip()):
                    matched += 1
                    all_pred.remove(each_pred)
                    break
    precision = matched / len(prediction)
    recall = matched / len(answer)
    if precision + recall == 0:
        return 0, precision, recall
    else:
        return 2 * precision * recall / (precision + recall), precision, recall

def extract_topk_prediction(prediction, k=-1):
    results = {}
    for p in prediction:
        if p in results:
            results[p] += 1
        else:
            results[p] = 1
    if k > len(results) or k < 0:
        k = len(results)
    results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    return [r[0] for r in results[:k]]

def eval_results(predict_file, cal_f1=True, topk = -1, subset=False, bad_samples=False, eval_hops=-1):
    assert not (subset and bad_samples)

    # predict_file = os.path.join(result_path, 'predictions.jsonl')
    eval_name = "detailed_eval_result_top_{topk}.jsonl" if topk > 0 else 'detailed_eval_result.jsonl'
    detailed_eval_file = predict_file.replace('predictions.jsonl', eval_name)
    # Load results
    acc_list = []
    hit_list = []
    f1_list = []
    precission_list = []
    recall_list = []
    if "webqsp" in predict_file:
        samples_to_eval_path = "./scored_triples/webqsp_240912_unidir_test.pth"
    elif "cwq" in predict_file:
        samples_to_eval_path = "./scored_triples/cwq_240907_unidir_test.pth"
    else:
        raise NotImplementedError
    samples_to_eval = torch.load(samples_to_eval_path, weights_only=False)

    with open(predict_file, 'r') as f, open(detailed_eval_file, 'w') as f2:
        for line in f:
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
            # if 'gpt' in predict_file:
            #     start_point = prediction.find("ans")
            #     prediction = prediction[start_point:]
            # answer = data['ground_truth']
            answer = sorted(data['ground_truth'], key=len, reverse=True)
            if 'when' in data['question'].lower() or 'what year' in data['question'].lower():
                for idx in range(len(answer)):
                    if '-' in answer[idx] and answer[idx].split('-')[0].isdigit():
                        answer[idx] = answer[idx].split('-')[0]
            question = data['question']
            double_check = any([keyword in question.lower() for keyword in ['when', 'what year', 'which year', 'where', 'sport', "what countr", "language", 'nba finals', 'world series']])
            if cal_f1:
                if not isinstance(prediction, list):
                    prediction = prediction.split("\n")
                else:
                    prediction = extract_topk_prediction(prediction, topk)
                f1_score, precision_score, recall_score = eval_f1(prediction, answer, double_check)
                f1_list.append(f1_score)
                precission_list.append(precision_score)
                recall_list.append(recall_score)
                prediction_str = '\n'.join(prediction)
                acc = eval_acc(prediction_str, answer)
                hit = eval_hit(prediction_str, answer, double_check)
                acc_list.append(acc)
                hit_list.append(hit)
                f2.write(json.dumps({'id': id, 'prediction': prediction, 'ground_truth': answer, 'acc': acc, 'hit': hit, 'f1': f1_score, 'precission': precision_score, 'recall': recall_score}) + '\n')
            else:
                acc = eval_acc(prediction, answer)
                hit = eval_hit(prediction, answer, double_check)
                acc_list.append(acc)
                hit_list.append(hit)
                f2.write(json.dumps({'id': id, 'prediction': prediction, 'ground_truth': answer, 'acc': acc, 'hit': hit}) + '\n')
    if len(hit_list) == 0:
        return [0] * 4

    avg_hit = sum(hit_list) * 100 / len(hit_list)
    avg_f1 = sum(f1_list) * 100 / len(f1_list)
    avg_precission = sum(precission_list) * 100 / len(precission_list)
    avg_recall = sum(recall_list) * 100 / len(recall_list)
    result_str = f"Hit: {avg_hit}, F1: {avg_f1}, Precission: {avg_precission}, Recall: {avg_recall}"
    print(result_str)

    result_name = "eval_result_top_{topk}.txt" if topk > 0 else 'eval_result.txt'
    eval_result_path = predict_file.replace('predictions.jsonl', result_name)
    with open(eval_result_path, 'w') as f:
        f.write(result_str)
    return avg_hit, avg_f1, avg_precission, avg_recall
