import json
import math
import matplotlib.pyplot as plt
import os
import pandas as pd
import shutil
import torch
import torch.nn as nn
import torch.optim as optim

import config
from constants import LossEnum
import Networks
from hyperparameters import EXPONENT_HYPERPARAMS


def collect_grads(model, 
                  counter, 
                  grad_storage,):
    
    if model.use_pe.item():
        feature_modules = list(model.children())[:-1]
    else:
        feature_modules = list(model.children())[1:-1]   
            
    network = nn.Sequential(*(feature_modules))
    
    with torch.no_grad():
        for idx, (name, params) in enumerate(network.named_parameters()):
            if counter == 0:
                grad_storage[name] = params.grad
            else: 
                grad_storage[name] = torch.add(grad_storage.get(name), params.grad)
                    
    return grad_storage


def copy_grads_file(best_trial_path: str, 
                    best_trial_num: int,
                    results_path: str,):
    filename = f'Trial_{best_trial_num}_avg_grads.pkl'
    grads_store_path = os.path.join(results_path, 
                                    "Best Trial Grads averaged")
    
    if not os.path.exists(grads_store_path):
        os.makedirs(grads_store_path)
        
    shutil.copy(os.path.join(best_trial_path, filename), 
                os.path.join(grads_store_path, "avg_grads.pkl"))
    

def save_study_config(main_path, results_path):
    shutil.copyfile(os.path.join(main_path, "config.py"),
                    os.path.join(results_path, "config.py"))


def get_model_instance(model_name,
                       parameters):
    
    if model_name == 'CNN':
        return Networks.cnn.CNN(in_channels = config.in_features,
                                out_channels = parameters['Out_Channels'],
                                kernel_size = parameters['Kernel_Size'], 
                                n_outputs = config.n_outputs, 
                                latent_size = parameters['Latent_Size'], 
                                p_dropout = parameters["Dropout"])
    elif model_name == 'Transformer':
        return Networks.transformer.Transformer(
                                    input_size = config.in_features,
                                    n_outputs = config.n_outputs,
                                    p_dropout = parameters["Dropout"], 
                                    num_heads = parameters["Attention_Heads"],
                                    num_layers = parameters["Num_Layers"],
                                    d_model = parameters["d_model"], 
                                    hidden_size = parameters["Hidden_Size"], 
                                    use_pe = config.pos_encoding, 
                                    batch_first = config.batch_first)
    elif model_name == 'LSTM':
        return Networks.lstm.LSTM(input_size = config.in_features,
                                  hidden_size = parameters["Hidden_Size"], 
                                  num_layers = parameters["Num_Layers"],
                                  n_outputs = config.n_outputs, 
                                  latent_size = parameters['Latent_Size'], 
                                  p_dropout = parameters["Dropout"],
                                  bidirectional = config.bidirectional, 
                                  batch_first = config.batch_first)
    elif model_name == 'MLP':
        return Networks.mlp.MLP(input_size = config.in_features,
                                window_length = config.window_length[config.sub_dataset - 1], 
                                n_outputs = config.n_outputs,
                                latent_size = parameters['Latent_Size'], 
                                hidden_size = parameters["Hidden_Size"],
                                p_dropout = parameters["Dropout"])


def get_criterion():
    if config.loss_fn == LossEnum.MAE:
        return nn.L1Loss()
    elif config.loss_fn == LossEnum.MSE or config.loss_fn == LossEnum.RMSE:
        return nn.MSELoss()

    
def get_scheduler(scheduler_name, optimizer, train_len, epochs, hyperparameters):
    
    if scheduler_name == 'Custom':       
        d_model = [hyperparameters['d_model'] if 'd_model' in hyperparameters else hyperparameters['Latent_Size']][0]
        total_steps = train_len * epochs
        n_warmup_steps = math.ceil(total_steps * hyperparameters['LR_Warmup_Ratio'])
        return CustomScheduler(optimizer,
                               d_model,
                               n_warmup_steps,
                               total_steps)    
    elif scheduler_name == "Exponential":
        return optim.lr_scheduler.ExponentialLR(optimizer, 
                                                gamma = 0.9)
    elif scheduler_name == "ReduceLROnPlateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                    mode = 'max',
                                                    factor = 0.9, 
                                                    patience = 5, 
                                                    min_lr = 1e-5)
    elif scheduler_name == "MultiStepLR":
        return optim.lr_scheduler.MultiStepLR(optimizer, 
                                              milestones = [math.ceil(0.5*epochs), math.ceil(0.75*epochs)], 
                                              gamma = 0.75)
    else:
        return None
    

class CustomScheduler():
    def __init__(self, optimizer, d_model, n_warmup_steps, total_steps):
        self.optimizer = optimizer
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.total_steps = total_steps
        self.cur_step = 0
        self.cur_lr = None
        self.step()

    def step(self):
        self.cur_step += 1
        self.cur_lr=self._get_lr()
        for p in self.optimizer.param_groups:
            p['lr'] = self.cur_lr

    def _get_lr(self): 
        if self.cur_step <  self.n_warmup_steps:
            return 0.5 * self.d_model**(-0.5) * 0.25*(self.cur_step*self.n_warmup_steps**(-1.5))
        else:
            return 0.5 * self.d_model**(-0.5) * 0.25*(self.n_warmup_steps*self.n_warmup_steps**(-1.5))

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

    def zero_grad(self):
        self.optimizer.zero_grad()

    

def get_results_dir(trial_num, results_path):
    trial_results_path = os.path.join(results_path, 
                                f"Trial_{trial_num}")
    inference_path = os.path.join(results_path, 
                                 "Inference Plots")
    if not os.path.exists(trial_results_path):
        os.makedirs(trial_results_path)
    if not os.path.exists(inference_path):
        os.makedirs(inference_path)    
    return trial_results_path, inference_path




def batchwise_loss_plot(batchwise_train_losses: list, 
                        trial_num: int,
                        results_path: str):
    
    plt.figure(figsize = (8, 6))
    plt.plot(batchwise_train_losses, label = 'Batchwise Training Losses')
    plt.grid(True)
    plt.ylabel('Losses', size = 15)
    plt.xlabel('Iterations', size = 15)
    plt.title(f'Trial {trial_num} Batchwise Training Losses', size = 20)
    plt.savefig(os.path.join(results_path, 
                             f"Trial_{trial_num} Batchwise Training Losses.svg"), 
                dpi = 200)
    plt.savefig(os.path.join(results_path, 
                             f"Trial_{trial_num} Batchwise Training Losses.png"), 
                dpi = 200)
    plt.clf()
    plt.close()


def loss_score(train_losses: list, 
               val_losses: list, 
               val_scores: list, 
               trial_num: int,
               results_path: str):
    
    plt.figure(figsize = (8, 6))
    plt.plot(train_losses, label = 'Training Losses')
    plt.plot(val_losses, label = 'Validation Losses')
    plt.plot(val_scores, label = 'Validation Scores')
    plt.grid(True)
    plt.ylabel('Value', size = 15)
    plt.xlabel('Epochs', size = 15)
    plt.legend()
    plt.title(f'Trial {trial_num} Metrics', size = 20)
    plt.savefig(os.path.join(results_path, f"Trial_{trial_num} Metrics.svg"), 
                dpi = 200)
    plt.savefig(os.path.join(results_path, f"Trial_{trial_num} Metrics.png"), 
                dpi = 200)
    plt.clf()
    plt.close()
    
def save_plot_lr(trial_num, 
                 learning_rates, 
                 results_path):
    plt.plot(learning_rates)
    plt.title("Change of Learning Rate over Training Period")
    plt.xlabel("Iterations")
    plt.ylabel("Learning Rate")
    
    plt.savefig(os.path.join(results_path, "Learning_Rate.png"), 
                             dpi = 200)
    plt.savefig(os.path.join(results_path, "Learning_Rate.svg"), 
                             dpi = 200)
    plt.clf()   
    plt.close()
    train_lr = pd.DataFrame({"Learning Rate" : learning_rates})
    train_lr.index.name = "Iterations"
    train_lr.to_excel(os.path.join(results_path, f"Trial_{trial_num}_LR.xlsx"))
    
    
def save_params_count(trial_num, model, results_path):
    param_count_list = []
    children_names_list = []

    for n, c in model.named_children():
        children_names_list.append(n)
        count = sum(p.numel() for p in c.parameters() if p.requires_grad)
        param_count_list.append(count)

    param_count_df = pd.DataFrame({"Sub-Model" : children_names_list,
                                          "Total Model Params" : param_count_list,
                                          })
    param_count_df.index.name = f"Trial_{trial_num}"
    param_count_df.to_csv(os.path.join(results_path, 
                                         f"Trial_{trial_num}_model_params_count.csv")) 

def plot_metric(trial_num: int, 
                values: list, 
                results_path: str, 
                task: str = 'Train',
                metric: str = 'Loss'):
    
    plt.figure(figsize = (8, 6))
    plt.plot(values, label = f'{task} {metric}')
    plt.grid(True)
    plt.ylabel(f'{metric}', size = 15)
    plt.xlabel('Epochs', size = 15)
    plt.title(f'Trial {trial_num} {task} {metric}', size = 20)
    plt.savefig(os.path.join(results_path, f"Trial_{trial_num}_{task}_{metric}.svg"), 
                dpi = 200)
    plt.savefig(os.path.join(results_path, f"Trial_{trial_num}_{task}_{metric}.png"), 
                dpi = 200)
    plt.clf()
    plt.close()    


def save_and_plot_metrics(trial_num: int, 
                          losses: list,
                          val_scores: list = None,
                          results_path: str = None,
                          train: bool = True):
    
    task = ["Train" if train else "Validation"][0]

    if train:
        metrics = {f"{task} Losses": losses}
        plot_metric(trial_num, 
                    values = losses, 
                    results_path = results_path, 
                    task = task,
                    metric = "Loss")
    else:            
        metrics = {f"{task} Losses": losses,
                   "Scores": val_scores}
        plot_metric(trial_num, 
                    values = losses, 
                    results_path = results_path, 
                    task = task,
                    metric = "Loss")
        plot_metric(trial_num, 
                    values = val_scores, 
                    results_path = results_path, 
                    task = task,
                    metric = "Scores")        
        
    metrics_df = pd.DataFrame(metrics)
    metrics_df.index.name = "Epochs"
    metrics_df.to_excel(os.path.join(results_path, 
                                     f"Trial_{trial_num}_{task}_metrics.xlsx"))


def save_score_df(trial_num, score_df, results_path):
    score_df.index.name = "Engine"
    score_df.to_excel(os.path.join(results_path, 
                                   f"Trial_{trial_num} Score_breakdown.xlsx"))    


def plot_all_rul(sub_dataset: int, 
                 predictions, 
                 labels, 
                 batch_lengths: list, 
                 results_path: str, 
                 train: bool = True):
    
    preds_rul = torch.split(predictions, batch_lengths)
    labels_rul = torch.split(labels, batch_lengths)
    
    task = ['Train' if train else 'Test'][0]
    
    for i in range(len(batch_lengths)):

        plt.plot(preds_rul[i], color = 'g', label = "Predicted RUL")
        plt.plot(labels_rul[i], color = 'r', label = "Actual RUL")

        plt.xlabel("Cycles")
        plt.ylabel("RUL")
        plt.title(f"FD00{sub_dataset} - {task} Engine ID {i+1}")
        plt.legend(loc = "upper right")
        plt.grid()
        
        plt.savefig(os.path.join(results_path,
                                 f"{task}_FD00{sub_dataset}_ID{i+1}.png"), 
                                 dpi = 200)
        plt.savefig(os.path.join(results_path,
                                 f"{task}_FD00{sub_dataset}_ID{i+1}.svg"), 
                                 dpi = 200)
        plt.clf()   
        plt.close()


def summarize_study(study = None, 
                    results_path: str = None):
    if study is not None:
        params_dict = study.best_trial.params.copy()
        for param_name, value in params_dict.items():
            if param_name in EXPONENT_HYPERPARAMS:
                params_dict[param_name] = 2**value
                
        best_stats = {**params_dict,
                      "Best_trial" : study.best_trial.number, 
                      "Score" : study.best_trial.value,
                      "Loss" : study.best_trial.user_attrs['val_loss'],
                      "Epoch" : study.best_trial.user_attrs['Epoch']}
        
        with open(os.path.join(results_path, "Best_trial.txt"), 
                  "w") as fp:
            json.dump(best_stats, fp, indent = 2)

        summary_df = study.trials_dataframe()
        summary_df.to_excel(os.path.join(results_path,
                                         "Trials_summary.xlsx")
                            )
        summary_df = summary_df.drop(['datetime_start', 
                                      'datetime_complete', 
                                      'duration'], axis = 1)
        summary_df = summary_df[summary_df['state'] == "COMPLETE"]
        cols = summary_df.columns.tolist()
        for c in cols:
            if c[7:] in EXPONENT_HYPERPARAMS:
                summary_df[c] = summary_df[c].apply(lambda x: 2**x)
        pd.set_option('display.max_columns', None)
        cols[:] = [c.replace('params_', '') for c in cols]
        cols[:] = [c.replace('user_attrs_', '') for c in cols]
        cols = ['Trial' if c == 'number' else c for c in cols]
        cols = ['Score' if c == 'value' else c for c in cols]
        summary_df.columns = cols
        summary_df.sort_values(by = 'Score', 
                               inplace = True)
        summary_df.to_excel(os.path.join(results_path,
                                         "Trials_summary_v2.xlsx")
                            )        
        
        
        



