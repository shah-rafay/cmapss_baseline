import os
import torch
from torch.utils.data import Dataset

from data import utils


class CMAPSSDataset(Dataset):
    def __init__(self, 
                 root: str = None, 
                 subdataset: int = 1,
                 train: bool = True,
                 cluster: bool = False, 
                 rul_start: int = 125, 
                 window_length: int = 20, 
                 window_step: int = 1, 
                 scaling_params = None,
                 scaling_method: str = "-11",
                 batch_first: bool = True):
        '''
        Init of Raw CMAPSS Turbofan Engine Dataset.

        Args:
            root (str, optional): Path to CMAPSS dataset folder. Defaults to None.
            subdataset (int, optional): CMAPSS subdataset. Defaults to 1.
            train (bool, optional): Defines between train and test. Defaults to True.
            cluster (bool, optional): Specifies whether or not to use operating condition cluster. Defaults to False.
            rul_start (int, optional): Start value of RUL in Piecewise function. Defaults to 125.
            window_length (int, optional): Sequence length of windowed data. Defaults to 20.
            window_step (int, optional): Stride of windowing. Defaults to 1.
            scaling_params (scaling_params, optional): Scaling params for test dataset. Defaults to None.
            bsc (bool, optional): If True, dataset is shaped as S,B,C (new for transformer)
                                     Else, dataset is shaped as B, C, S
                                         where, B = Batchsize, S = Sequence length, C = No. of features. Defaults to True.

        Raises:
            AttributeError: Checks if CMAPSS dataset folder exists.
            
        '''
        self.subdataset = subdataset
        self.train = train
        self.cluster = cluster
        self.rul_start = rul_start
        self.window_length = window_length
        self.window_step = window_step
        if "CMAPSS" not in os.listdir(root):
            raise AttributeError(
                f"'CMAPSS'- Dataset folder not found in specified path: {root}")
        else: 
            root = os.path.join(root, "CMAPSS")

        filepath, rul_filepath = utils.check_subdataset(
                                                train = train, 
                                                subdataset = subdataset,
                                                cluster = cluster, 
                                                root = root, 
                                                window_length = window_length, 
                                                window_step = window_step
                                                        )
        scaled_data, self.engine_cycles, self.params = \
                                        utils.get_data_and_scale(
                                            filepath = filepath, 
                                            train = train,
                                            cluster = cluster, 
                                            scaling_method = scaling_method, 
                                            params = scaling_params
                                            )
        data = torch.split(
                        tensor = scaled_data, 
                        split_size_or_sections = self.engine_cycles, 
                        dim = 0
                            )
        data, labels, self.batch_lengths = \
                            utils.get_windowed_data_and_labels( 
                                            data = data,
                                            train = train,
                                            cluster = cluster, 
                                            rul_start = rul_start, 
                                            window_length = window_length,
                                            window_step = window_step,                                           
                                            rul_filepath = rul_filepath
                                            )

        self.x, self.y = data, labels
        print(self.x.shape)
        print(self.y.shape)
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.x)