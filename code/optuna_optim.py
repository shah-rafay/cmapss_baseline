import json
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import config
from config import DEVICE
import data.utils
from hyperparameters import Hyperparameters
import trainer
import utils


class OptunaOptimizer(): 
    def __init__(self, 
                 main_path: str = None,
                 dataset_path: str = None, 
                 results_path: str = None):
        
        self.main_path = main_path
        self.dataset_path = dataset_path
        self.results_path = results_path
        self.epochs = config.epochs
        self.model_selection = config.model_selection.value
        self.scheduler_name = config.scheduler_name.value
        
        self.src_subset = config.sub_dataset
        self.get_datasets()
        
    def get_datasets(self):
        train_split, test_split = data.utils.get_datasets(self.dataset_path, 
                                                          self.src_subset
                                                          )
        self.train_dataset, self.train_batch_lengths = train_split
        self.test_dataset, self.test_batch_lengths = test_split
        
    def get_hyperparams(self, trial):
        self.hyperparams = Hyperparameters(trial,
                                           self.model_selection).get_hyperparams()
    
    def objective(self, trial):
        trial_num = trial.number
        self.trial_results, self.inference_path = utils.get_results_dir(trial_num, 
                                                                        self.results_path
                                                                        )
        self.get_hyperparams(trial
                             )
        training_data = DataLoader(self.train_dataset, 
                                   batch_size = config.batch_size, 
                                   shuffle = False
                                   )
        testing_data = DataLoader(self.test_dataset, 
                                  batch_size = config.batch_size
                                  )
        model = utils.get_model_instance(model_name = self.model_selection, 
                                         parameters = self.hyperparams).to(DEVICE
                                                                           )
        utils.save_params_count(trial_num, model, self.trial_results
                                )
        criterion = utils.get_criterion(
                                        )
        optimizer = optim.Adam(model.parameters(), 
                               lr = self.hyperparams['Learning_Rate']
                               )
        scheduler = utils.get_scheduler(self.scheduler_name, 
                                        optimizer, 
                                        len(training_data), 
                                        self.epochs, 
                                        self.hyperparams
                                        )
        print('\n', self.hyperparams)
        self.trainer = trainer.Trainer(
                                trial = trial,
                                trial_num = trial_num,
                                sub_dataset = self.src_subset,
                                train_data = training_data,
                                validation_data = testing_data, 
                                test_data = testing_data, 
                                train_batch_lengths = self.train_batch_lengths, 
                                validation_batch_lengths = self.test_batch_lengths, 
                                test_batch_lengths = self.test_batch_lengths,
                                epochs = self.epochs,
                                model = model, 
                                optimizer = optimizer, 
                                criterion = criterion, 
                                scheduler = scheduler,
                                scheduler_name = self.scheduler_name,
                                results_path = self.trial_results, 
                                       )
        best_score = self.trainer.fit()
        torch.cuda.empty_cache()
    
        return best_score
    
    def optimize(self):
        self.study = optuna.create_study(
            sampler = TPESampler(n_startup_trials = config.startup_trials),
                                 pruner = MedianPruner(
                                         n_startup_trials = config.startup_trials,
                                         n_warmup_steps = config.prune_warmup),
                                 direction = 'minimize'
                                 )
        self.study.optimize(self.objective, 
                            config.trials, 
                            gc_after_trial = True
                            )
        utils.save_study_config(self.main_path, self.results_path
                                )
        utils.summarize_study(study = self.study, 
                              results_path = self.results_path)
        
    def run_single_test(self):
        best_trial = self.study.best_trial.number
        best_trial_path = os.path.join(self.results_path, 
                                        f"Trial_{best_trial}")        
        
        with open(os.path.join(self.results_path, 
                               'Best_trial.txt')) as f:
            data = f.read()
        params = json.loads(data)
        
        rem_keys = ('Best_trial', 'Score', 'Loss', 'Epoch')
        for k in rem_keys:
            params.pop(k, None)
        model = torch.load(os.path.join(best_trial_path, 
                                        f"Trial_{best_trial}_model.pt"))['model']

        utils.copy_grads_file(best_trial_path, 
                              best_trial,
                              self.results_path) 
        print("\n\nRunning inference on Test Dataset using best model")
        print("\nSaving all RUL Plots")
        test_loss, test_score = self.trainer.testing_epoch(model, 
                                                           self.inference_path)
        train_loss, _ = self.trainer.testing_epoch(model, 
                                                   self.inference_path, 
                                                   train = True)
        with open(os.path.join(self.results_path, 
                               f"Trial_{best_trial}_Inference.txt"), "w") as fp:
            json.dump({"Score": test_score, 
                       "Test Loss": test_loss, 
                       "Train Loss": train_loss}, fp, indent = 2)

        
