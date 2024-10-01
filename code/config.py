from constants import ModelEnum, ScalerEnum, LossEnum, LRSchedulerEnum
import torch

#Set Computation Device
DEVICE= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Dataset Config
sub_dataset= 2
in_features= 14
n_outputs= 1
rul_start= 120
batch_size = 1024
window_length= [30, 20, 32, 18]
window_step= 1
scaling_method= ScalerEnum.MinusOneOne
batch_first= True

#Training Config
trials= 150
epochs= 75
loss_fn= LossEnum.RMSE
scheduler_name= LRSchedulerEnum.CUSTOM
startup_trials= 50  
prune_warmup= 30 

#Model selection from ModelEnum
model_selection = ModelEnum.TRANSFORMER

#Transformer Positional Encoding
pos_encoding= False

#LSTM Configuration
bidirectional= False