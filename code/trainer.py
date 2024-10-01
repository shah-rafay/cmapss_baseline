import copy
from dataclasses import dataclass, field
import numpy as np
import optuna
import os
import pickle
import torch
from tqdm import tqdm
from typing import List

import config
from constants import LossEnum
from config import DEVICE
import data
import utils


@dataclass
class TrainMetrics:
    loss: List[float] = field(default_factory=lambda: [])
    loss_batchwise: List[float] = field(default_factory=lambda: [])
    lr_batchwise: List[float] = field(default_factory=lambda: [])

@dataclass
class ValidationMetrics:
    loss: List[float] = field(default_factory=lambda: [])
    epochs: List[int] = field(default_factory=lambda: [])
    score: List[float] = field(default_factory=lambda: [])


class Trainer():
    
    def __init__(self, 
                 trial = None,
                 trial_num = None, 
                 sub_dataset: int = None,
                 train_data = None,
                 validation_data = None, 
                 test_data = None, 
                 train_batch_lengths: list = None,
                 validation_batch_lengths: list = None,
                 test_batch_lengths: list = None,
                 epochs: int = None,
                 model = None, 
                 optimizer = None, 
                 criterion = None, 
                 scheduler = None, 
                 scheduler_name: str = None, 
                 results_path: str = None):
        
        self.trial = trial
        self.sub_dataset = sub_dataset
        self.train_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data
        self.train_batch_lengths = train_batch_lengths
        self.validation_batch_lengths = validation_batch_lengths
        self.test_batch_lengths = test_batch_lengths        
        self.epochs = epochs
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.scheduler_name = scheduler_name
        self.trial_num = trial_num
        self.results_path = results_path
        self.counter = 0
        self.grad_storage = {}

        self.train_metrics = TrainMetrics()
        self.val_metrics = ValidationMetrics()

        
    def train_epoch(self):
        self.model.train()
        train_loss = []
        for (inputs, label) in tqdm(self.train_data, leave = False):
            self.optimizer.zero_grad(set_to_none = True)
            inputs, label = inputs.to(DEVICE), label.to(DEVICE)       
            preds = self.model(inputs)

            loss = self.criterion(preds, label) 
            loss = [torch.sqrt(loss) if config.loss_fn == LossEnum.RMSE else loss][0]
            loss.backward()
            
            self.grad_storage = utils.collect_grads(self.model, 
                                                    self.counter, 
                                                    self.grad_storage
                                                    )
            self.counter += 1 
            
            self.optimizer.step()
            if self.scheduler_name == 'Custom':
                self.scheduler.step()

            train_loss.append(loss.item())
            self.train_metrics.loss_batchwise.append(loss.item())
            self.train_metrics.lr_batchwise.append(
                                    self.optimizer.param_groups[0]['lr'])
        epoch_loss = round(np.mean(train_loss), 2)
        return epoch_loss
    
    
    def validation_epoch(self):
        self.model.eval()
        with torch.no_grad():
            validation_loss = []
            predictions = []
            labels = []
            for (inputs, label) in tqdm(self.validation_data, leave = False): 
                inputs, label = inputs.to(DEVICE), label.to(DEVICE)       
                preds = self.model(inputs)
                loss = self.criterion(preds, label)
                loss = [torch.sqrt(loss) if config.loss_fn == LossEnum.RMSE else loss][0]                
                validation_loss.append(loss.item())
                predictions.append(preds.detach().cpu())
                labels.append(label.cpu())

            predictions = torch.cat(predictions, dim = 0)
            labels = torch.cat(labels, dim = 0)
            score, score_df = data.utils.scoring_function(
                                                   labels, 
                                                   predictions, 
                                                   self.validation_batch_lengths)
            epoch_loss = round(np.mean(validation_loss), 2)
        return epoch_loss, round(score), score_df
    
    def testing_epoch(self, model, inference_path: str, train: bool = False):
        model.eval()
        data_split = [self.train_data if train else self.test_data][0]
        batch_lengths = [self.train_batch_lengths if train else self.test_batch_lengths][0]
        
        with torch.no_grad():
            testing_loss = []
            predictions = []
            labels = []
            
            for (inputs, label) in tqdm(data_split, leave = False): 
                inputs, label = inputs.to(DEVICE), label.to(DEVICE)       
                preds = model(inputs)
                
                loss = self.criterion(preds, label)
                loss = [torch.sqrt(loss) if config.loss_fn == LossEnum.RMSE else loss][0]                
 
                testing_loss.append(loss.item())
                predictions.append(preds.detach().cpu())
                labels.append(label.cpu())
    
            predictions = torch.cat(predictions)
            labels = torch.cat(labels)
            epoch_loss = np.mean(testing_loss)
                
            if train:
                score = 0.0
            else:
                score, _ = data.utils.scoring_function(labels, 
                                                       predictions, 
                                                       self.test_batch_lengths)                
            utils.plot_all_rul(self.sub_dataset, 
                               predictions, 
                               labels, 
                               batch_lengths, 
                               inference_path, 
                               train)
            
        return round(epoch_loss, 2), round(score)
    
    
    def fit(self):
        best_score = float('inf')
        best_loss = float('inf')
        for epoch in range(self.epochs):
            train_loss = self.train_epoch()
            validation_loss, validation_score, self.score_df = self.validation_epoch()
            
            if self.scheduler_name is not None:
                if self.scheduler_name == 'ReduceLROnPlateau':
                    self.scheduler.step(validation_score)
                else:
                    self.scheduler.step()
            
            print(f"Epoch: {epoch}\tTrain Loss: {train_loss}\tVal Loss: {validation_loss}\tScore: {validation_score}")
            
            self.train_metrics.loss.append(train_loss)
            self.val_metrics.epochs.append(epoch)
            self.val_metrics.loss.append(validation_loss)
            self.val_metrics.score.append(validation_score)
            
            if validation_score < best_score:
                best_score = validation_score
                best_loss = validation_loss
                best_epoch = epoch
                self.save_model(self.trial.number if self.trial is not None else 0, 
                                self.model)
            self.trial.report(validation_score, epoch)
            if self.trial.should_prune():
                raise optuna.TrialPruned()
                
        for k in self.grad_storage:
            self.grad_storage[k] /= self.counter

        self.save_avg_grads(self.trial.number if self.trial is not None else 0, 
                            self.grad_storage)
        
        self.trial.set_user_attr('val_loss', best_loss)
        self.trial.set_user_attr('Epoch', best_epoch)

        self.save_metrics()

        return best_score
    
    
    def save_avg_grads(self, 
                       trial_num: int, 
                       avg_grads):
        with open(os.path.join(self.results_path,
                               f'Trial_{trial_num}_avg_grads.pkl'), 'wb') as f:
            pickle.dump(avg_grads, f)
    
    def save_model(self, 
                   trial_num: int, 
                   model):
        statedict = copy.deepcopy(model.state_dict())
        torch.save(statedict, os.path.join(self.results_path, 
                                           f'Trial_{trial_num}_model_statedict.pt')) 
        ckpt = {'model': model}
        torch.save(ckpt, os.path.join(self.results_path, 
                                      f"Trial_{trial_num}_model.pt"))    
        
    def save_metrics(self):
        utils.save_score_df(trial_num = self.trial_num,
                            score_df = self.score_df, 
                            results_path = self.results_path)
        utils.loss_score(train_losses = self.train_metrics.loss, 
                         val_losses = self.val_metrics.loss, 
                         val_scores = self.val_metrics.score, 
                         trial_num = self.trial_num , 
                         results_path = self.results_path)
        utils.batchwise_loss_plot(
                    batchwise_train_losses = self.train_metrics.loss_batchwise, 
                    trial_num = self.trial_num, 
                    results_path = self.results_path)
        utils.save_plot_lr(trial_num = self.trial_num, 
                           learning_rates = self.train_metrics.lr_batchwise, 
                           results_path = self.results_path)
        utils.save_and_plot_metrics(trial_num = self.trial_num, 
                                    losses = self.train_metrics.loss, 
                                    results_path = self.results_path)
        utils.save_and_plot_metrics(trial_num = self.trial_num, 
                                    losses = self.val_metrics.loss, 
                                    val_scores = self.val_metrics.score,
                                    results_path = self.results_path, 
                                    train = False)        