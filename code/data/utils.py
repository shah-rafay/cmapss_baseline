import os
import pandas as pd
import sklearn.preprocessing as preprocessor
import torch

import config
from data import dataset


def get_datasets(dataset_path, 
                 sub_dataset):
    
    train_dataset = dataset.CMAPSSDataset(root = dataset_path,
                                          subdataset = sub_dataset,
                                          rul_start = config.rul_start,
                                          window_length = config.window_length[sub_dataset - 1],
                                          window_step = config.window_step,
                                          scaling_method = config.scaling_method.value,
                                          batch_first = config.batch_first,
                                          )
    test_dataset = dataset.CMAPSSDataset(root = dataset_path, 
                                         subdataset = sub_dataset,
                                         train = False,
                                         rul_start = config.rul_start,
                                         window_length = config.window_length[sub_dataset - 1], 
                                         window_step = config.window_step,
                                         scaling_params = train_dataset.params,
                                         scaling_method = config.scaling_method.value,
                                         batch_first = config.batch_first,
                                         )
    train_batch_lengths = train_dataset.batch_lengths
    test_batch_lengths = test_dataset.batch_lengths
    
    return (train_dataset, train_batch_lengths), (test_dataset, test_batch_lengths)


def check_subdataset(train: bool = True, 
                     subdataset: int = 1,
                     cluster: bool = False, 
                     root: str = None, 
                     window_length: int = 20, 
                     window_step: int = 1):
    '''
    Function to check CMAPSS dataset folder and files availability.
    Also checks if input window length is greater than the lowest number of cycles in the test sub-dataset to avoid engine data overlapping.

    Args:
        train (bool, optional): Defines between train and test. Defaults to True.
        subdataset (int, optional): CMAPSS subdataset. Defaults to 1.
        cluster (bool, optional): Specifies whether or not to use operating condition cluster. Defaults to False.
        root (str, optional): Path to CMAPSS dataset folder. Defaults to None.
        window_length (int, optional): Sequence length of windowed data. Defaults to 20.
        window_step (int, optional): Stride of windowing. Defaults to 1.
        
    Raises:
        AttributeError: Checks if cluster of operating conditions is provided in the dataset train and test files.
        AttributeError: Checks if dataset file exists in dataset folder.  
        AttributeError: Checks if RUL file exists in dataset folder.
        AttributeError: Checks if input window length is greater than lowest number of cycles in the overall corresponding test sub-dataset.
    
    Returns:
        filepath (str): Directory for sub-dataset file.
        rul_filepath (str): Directory for RUL file for test sub-dataset. NoneType for train sub-dataset.

    '''
    
    if root is None: 
        raise AttributeError("'root' must be not None and contain path to CMAPSS dataset folder")
        
    title = ["train" if train else "test"][0]
    
    filename = title + "_FD00" + str(subdataset)
    if cluster and subdataset in [1, 3]:
        raise AttributeError(
            "Operating System cluster is not available for subdataset FD00%s. Set cluster = False." % subdataset)
    filename = filename + "_cluster.csv" if cluster else filename + ".csv"
    filepath = os.path.join(root, filename)
    if not os.path.exists(filepath):
        raise AttributeError(
            "The following file does not exist: " + filepath)

    # Import test RUL file for test dataset
    rul_filename = "RUL_FD00" + str(subdataset)
    rul_filepath = os.path.join(root, rul_filename + '.csv')
    if not os.path.exists(rul_filepath):
        raise AttributeError(
            "The following file does not exist: " + rul_filepath)

    # Check if window length is greater than the lowest number of cycles in the test sub-dataset.
    test_data = pd.read_csv(os.path.join(
        root, "test_FD00" + str(subdataset) + ".csv"))
    test_engines = test_data['engine_id'].unique().tolist()
    test_engine_cycles = [
        test_data[test_data['engine_id'] == i]['cycle'].max() for i in test_engines]
    min_cycle = min(test_engine_cycles)
    if window_length > min_cycle:
        raise AttributeError("Window length (%d) must be equal to or less than the lowest cycles in the test dataset (%d)" % (
            window_length, min_cycle))
    return filepath, rul_filepath


def scale_data(data, 
               scaling_method: str = "-11", 
               params = None):
    '''
    Function to scale raw dataset.

    Args:
        data (pandas.DataFrame): Raw dataset to scale.
        scaling_method (str, optional): Normalizing method. Defaults to "-11". Options = ['-11', '01', 'None (Standard Scaler)']
        params (scaling_params, optional): Scaling params for test dataset. Defaults to None.

    Returns:
        scaled_data (torch.Tensor): Scaled dataset.
        params (scaling_params): Scaling params from train dataset.

    '''
    
    if scaling_method == "-11":
        scaler = preprocessor.MinMaxScaler(feature_range = (-1, 1))
    elif scaling_method == "01":
        scaler = preprocessor.MinMaxScaler(feature_range = (0, 1))
    else:
        scaler = preprocessor.StandardScaler()
    if params is None:
        params = scaler.fit(data)
    scaled_data = torch.FloatTensor(params.transform(data))
    return scaled_data, params

def get_data_and_scale(filepath: str = None,
                       train: bool = True,                       
                       cluster: bool = False, 
                       scaling_method: str = "-11",
                       params = None,
                       meta_data: list = [
                           'cycle', 'setting1', 'setting2', 'setting3'],
                       redundant_sensors: list = [
                           's1', 's5', 's6', 's10', 's16', 's18', 's19']):
    '''
    Function to import and scale / normalize raw dataset.

    Args:
        filepath (str): Path to raw dataset. Defaults to None.
        train (bool, optional): Defines between train and test. Defaults to True.
        cluster (bool, optional): Specifies whether or not to use operating condition cluster. Defaults to False.
        scaling_method (str, optional): Normalizing method. Defaults to "-11". Options = ['-11', '01', 'None (Standard Scaler)']
        params (scaling_params, optional): Scaling params for test dataset. Defaults to None.
        meta_data (list, optional): Meta information or raw dataset. Defaults to ['cycle', 'setting1', 'setting2', 'setting3'].
        redundant_sensors (list, optional): Sensors with no relevant information. Defaults to ['s1', 's5', 's6', 's10', 's16', 's18', 's19'].

    Returns:
        scaled_data (torch.Tensor): Scaled dataset.
        engine_cycles (list): List containing cycle lengths of all engines.
        params (scaling_params): Scaling params from train dataset.

    '''
    raw_data = pd.read_csv(filepath)
    if cluster:
        meta_data.append('operating_condition')
    raw_data = raw_data.drop(columns = meta_data + redundant_sensors)
    engine_cycles = []
    engine_ids = raw_data['engine_id'].unique().tolist()
    engine_cycles = [len(raw_data[raw_data["engine_id"] == i]) for i in engine_ids]
    raw_data = raw_data.drop(columns = ['engine_id'])
    scaled_data, params = scale_data(data = raw_data, 
                                     scaling_method = scaling_method, 
                                     params = params)
    return scaled_data, engine_cycles, params

def get_windowed_data_and_labels(data: list = None, 
                                 train: bool = True,
                                 cluster: bool = False, 
                                 rul_start: int = 125, 
                                 window_length: int = 20, 
                                 window_step: int = 1,  
                                 rul_filepath: str = None,
                                 meta_data: list = [
                                     'cycle', 'setting1', 'setting2', 'setting3'],
                                 redundant_sensors: list = [
                                     's1', 's5', 's6', 's10', 's16', 's18', 's19']):
    '''
    Function to create windowed engine data and corresponding labels.

    Args:
        data (list): List of batches of engine data in raw form.
        train (bool, optional): Defines between train and test. Defaults to True.
        cluster (bool, optional): Specifies whether or not to use operating condition cluster. Defaults to False.
        rul_start (int, optional): Start value of RUL in Piecewise function. Defaults to 125.
        window_length (int, optional): Sequence length of windowed data. Defaults to 20.
        window_step (int, optional): Stride of windowing. Defaults to 1.
        rul_filepath (str, optional): Directory of RUL points for test dataset. Defaults to None.
        meta_data (list, optional): Meta information or raw dataset. Defaults to ['cycle', 'setting1', 'setting2', 'setting3'].
        redundant_sensors (list, optional): Sensors with no relevant information. Defaults to ['s1', 's5', 's6', 's10', 's16', 's18', 's19'].

    Returns:
        batched_data (TYPE): DESCRIPTION.
        label (TYPE): DESCRIPTION.
        batch_lengths (TYPE): DESCRIPTION.

    '''

    window_data_list = []
    label = []
    batch_lengths = []
    if not train:
        test_ruls = pd.read_csv(rul_filepath).values
    for i in range(len(data)):
        engine_data = data[i]
        cycles = engine_data.shape[0]
        windowed_data = engine_data.unfold(0, window_length, window_step)
        window_data_list.append(windowed_data)
        batch_size = windowed_data.shape[0]
        batch_lengths.append(batch_size)
        """
        The RUL for each test engine is non-zero and provided in the corresponding RUL file (e.g., RUL_FD001.csv).
        """
        if train:
            test_rul = 0
        else:
            test_rul = test_ruls[i].item()
        """
        If starting point of RUL is x, then the engine should degrade linearly from x-1 to 0 for x cycles.  
        Hence, degrading cycles = {x-1, x-2, x-3, ..., 0} with x instances.
        Therefore, healthy cycles = total engine cycles - degrading cycles.
        - Total cycles for test dataset (RUL Start to 0) is incomplete and is fulfilled by:
            Provided total cycles in the test dataset + No. of trimmed cycles after failure of engine at a non-zero cycle.
        """
        
        healthy_cycles = cycles + test_rul - rul_start
        healthy_rul = rul_start * torch.ones(healthy_cycles)
        degrading_rul = torch.linspace(start = rul_start-1, 
                                       end = 0, 
                                       steps = rul_start)
        rul_labels = torch.cat((healthy_rul, degrading_rul), dim=0)[window_length-1:]

        """
        Since, test engines fail at a non-zero cycle, hence the RUL steps are trimmed at the provided end-point.
        """
        if not train:
            rul_labels = rul_labels[: -test_rul]

        label.append(rul_labels)        
        
    windowed_data = torch.cat(window_data_list, dim = 0)
    label = torch.cat(label, dim = 0).unsqueeze(-1)
    return windowed_data, label, batch_lengths



def scoring_function(labels, predictions, batch_lengths):
    
    scores = []
    true_ruls = []
    predicted_ruls = []
    
    true_labels_split = torch.split(labels, batch_lengths)
    pred_label_split = torch.split(predictions, batch_lengths)
    
    for i in range(len(true_labels_split)):
        
        true_rul = true_labels_split[i][-1]
        predicted_rul = pred_label_split[i][-1]
        
        true_ruls.append(true_rul.item())
        predicted_ruls.append(predicted_rul.item())
    
        d = predicted_rul - true_rul
        a = [(torch.exp(d/10) - 1) if d >= 0 else (torch.exp(-d/13) - 1)][0]
        scores.append(a.item())
    
    score_df = pd.DataFrame({"True RUL": true_ruls,
                             "Predicted RUL": predicted_ruls,
                             "Score": scores})
    
    return sum(scores), score_df