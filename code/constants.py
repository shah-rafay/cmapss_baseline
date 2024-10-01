from enum import Enum


class LossEnum(Enum):
    MSE = "MSELoss"
    MAE = "L1Loss"
    RMSE = "RMSELoss"
    

class ModelEnum(Enum):
    TRANSFORMER = "Transformer"
    CNN = "CNN"
    LSTM = "LSTM"
    MLP = "MLP"
    
    
class ScalerEnum(Enum):
    MinusOneOne = '-11'
    ZeroOne = '01'
    ZeroMeanStdOne = 'ZMS1'
    
    
class LRSchedulerEnum(Enum):
    CUSTOM = 'Custom'
    EXPONENTIAL = 'Exponential'    
    MULTISTEP = 'MultiStepLR'
    REDUCEONPLATEAU = 'ReduceLROnPlateau'
    
    