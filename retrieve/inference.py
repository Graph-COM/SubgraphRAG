import torch

def main(args):
    pass

if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument('-p', '--path', type=str, required=True,
                        help='Path to a saved model checkpoint, e.g., webqsp_Nov08-01:14:47/cpt.pth')
    args = parser.parse_args()
    
    main(args)
