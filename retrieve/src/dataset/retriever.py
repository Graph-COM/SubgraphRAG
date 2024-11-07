import os
import pickle

class RetrieverDataset:
    def __init__(self, config, split, skip_no_path=True):
        # Load pre-processed data.
        dataset_name = config['dataset']['name']
        processed_dict_list = self._load_processed(dataset_name, split)

    def _load_processed(self, dataset_name, split):
        processed_file = os.path.join(
            f'data_files/{dataset_name}/processed/{split}.pkl')
        with open(processed_file, 'rb') as f:
            return pickle.load(f)
