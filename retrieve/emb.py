import os
import torch

from datasets import load_dataset

from src.config.emb import load_yaml

def main(args):
    # Modify the config file for advanced settings and extensions.
    config_file = f'configs/emb/gte-large-en-v1.5/{args.dataset}.yaml'
    config = load_yaml(config_file)
    
    torch.set_num_threads(config['env']['num_threads'])

    if args.dataset == 'cwq':
        input_file = os.path.join('rmanluo', 'RoG-cwq')
    else:
        input_file = os.path.join('ml1996', 'webqsp')

if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser('Text Embedding Pre-Computation for Retrieval')
    parser.add_argument('-d', '--dataset', type=str, required=True, 
                        choices=['webqsp', 'cwq'], help='Dataset name')
    args = parser.parse_args()
    
    main(args)
