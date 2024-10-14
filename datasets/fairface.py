"""Fair Face Dassl dataset registry"""
import os
import pandas as pd
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase


@DATASET_REGISTRY.register()
class FairFace(DatasetBase):
    """Fair Face Dassl dataset class"""
    dataset_dir = 'FairFace'

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        train_x = self.read_data('train')
        test = self.read_data('val')
        super().__init__(train_x=train_x, test=test)

    def read_data(self, mode):
        """Read data using mode.csv file and returns list of Datum"""
        rd = {
            "White": 0,
            "Black": 1,
            "Indian": 2,
            "Latino_Hispanic": 3,
            "Southeast Asian": 4,
            "East Asian": 5,
            "Middle Eastern": 6
        }

        data_df = pd.read_csv(f"{self.dataset_dir}/fface_{mode}.csv")
        data_list = []
        for _, row in data_df.iterrows():
            impath = f"{self.dataset_dir}/{row['file']}"
            classname = row['race']
            label = rd[classname]
            data_list.append(
                Datum(impath=impath, label=label, classname=classname))
        return data_list
