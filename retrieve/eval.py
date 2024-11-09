import numpy as np
import pandas as pd
import torch

def main(args):
    pred_dict = torch.load(args.path)
    gpt_triple_dict = torch.load(f'data_files/{args.dataset}/gpt_triples.pth')
    k_list = [int(k) for k in args.k_list.split(',')]
    
    metric_dict = dict()
    for k in k_list:
        metric_dict[f'ans_recall@{k}'] = []
        metric_dict[f'shortest_path_triple_recall@{k}'] = []
        metric_dict[f'gpt_triple_recall@{k}'] = []
    
    for sample_id in pred_dict:
        if len(pred_dict[sample_id]['scored_triples']) == 0:
            continue
        
        h_list, r_list, t_list, _ = zip(*pred_dict[sample_id]['scored_triples'])
        
        a_entity_in_graph = set(pred_dict[sample_id]['a_entity_in_graph'])
        if len(a_entity_in_graph) > 0:
            for k in k_list:
                entities_k = set(h_list[:k] + t_list[:k])
                metric_dict[f'ans_recall@{k}'].append(
                    len(a_entity_in_graph & entities_k) / len(a_entity_in_graph)
                )
        
        triples = list(zip(h_list, r_list, t_list))
        shortest_path_triples = set(pred_dict[sample_id]['target_relevant_triples'])
        if len(shortest_path_triples) > 0:
            for k in k_list:
                triples_k = set(triples[:k])
                metric_dict[f'shortest_path_triple_recall@{k}'].append(
                    len(shortest_path_triples & triples_k) / len(shortest_path_triples)
                )
        
        gpt_triples = set(gpt_triple_dict.get(sample_id, []))
        if len(gpt_triples) > 0:
            for k in k_list:
                triples_k = set(triples[:k])
                metric_dict[f'gpt_triple_recall@{k}'].append(
                    len(gpt_triples & triples_k) / len(gpt_triples)
                )

    for metric, val in metric_dict.items():
        metric_dict[metric] = np.mean(val)
    
    table_dict = {
        'K': k_list,
        'ans_recall': [
            round(metric_dict[f'ans_recall@{k}'], 3) for k in k_list
        ],
        'shortest_path_triple_recall': [
            round(metric_dict[f'shortest_path_triple_recall@{k}'], 3) for k in k_list
        ],
        'gpt_triple_recall': [
            round(metric_dict[f'gpt_triple_recall@{k}'], 3) for k in k_list
        ]
    }
    df = pd.DataFrame(table_dict)
    print(df.to_string(index=False))

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, required=True, 
                        choices=['webqsp', 'cwq'], help='Dataset name')
    parser.add_argument('-p', '--path', type=str, required=True,
                        help='Path to retrieval result')
    parser.add_argument('--k_list', type=str, default='50,100,200,400',
                        help='Comma-separated list of K values for top-K recall evaluation')
    args = parser.parse_args()
    
    main(args)
