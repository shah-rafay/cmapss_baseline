import datetime
import optuna_optim
import os
import pathlib
import config

ROOT = pathlib.Path(__file__)

date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
main_path = ROOT.parents[0].__str__()
data_path = os.path.join(ROOT.parents[1].__str__(), "Datasets")
results_path = os.path.join(ROOT.parents[0].__str__(), 
                            "00_Results", 
                            f"FD00{config.sub_dataset}_{date}")

manager = optuna_optim.OptunaOptimizer(main_path = main_path, 
                                       dataset_path = data_path, 
                                       results_path = results_path)
manager.optimize()
manager.run_single_test()
