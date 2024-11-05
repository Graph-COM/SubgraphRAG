import torch

from src.config.emb import load_yaml

def main(args):
    # Check the config file for advanced settings and extensions.
    config_file = f'configs/emb/gte-large-en-v1.5/{args.dataset}.yaml'

if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser('Text Embedding Pre-Computation for Retrieval')
    parser.add_argument('-d', '--dataset', type=str, required=True, 
                        choices=['webqsp', 'cwq'], help='Dataset name')
    args = parser.parse_args()
    
    main(args)
