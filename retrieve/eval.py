def main(args):
    pass

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, required=True, 
                        choices=['webqsp', 'cwq'], help='Dataset name')
    parser.add_argument()
    # parser.add_argument()
    parser.add_argument()
    args = parser.parse_args()
    
    main(args)
