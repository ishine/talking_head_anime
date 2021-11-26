import os

from datasets.base import BaseDataset


class CustomDataset(BaseDataset):
    def __init__(self, conf):
        super(CustomDataset, self).__init__(conf)

        self.data = self.build_data()

        self.sample_data = {
            'model_path': './4490707391186690073.vrm'
        }

    def build_data(self):
        path_root = self.conf.path['root']
        model_dirs = sorted([
            os.path.join(path_root, file) for file in os.listdir(path_root)
        ])

    def getitem(self, idx):
        data = self.sample_data
        model_path = data['model_path']

