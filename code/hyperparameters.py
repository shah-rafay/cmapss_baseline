import config
import constants

EXPONENT_HYPERPARAMS = ['Out_Channels', 'Latent_Size', 
                        'd_model', 'Attention_Heads', 
                        'Hidden_Size']

class Hyperparameters():
    def __init__(self, 
                 trial = None,
                 model_name: str = "CNN", 
                 ):
        
        self.trial = trial
        self.model_name = model_name
        
    def training(self):
        if config.scheduler_name != constants.LRSchedulerEnum.CUSTOM:
            lr = self.trial.suggest_loguniform("Learning_Rate",
                                               low = 1e-5,
                                               high = 1e-2)
            warmup_ratio = 0.0
        else:
            lr = 0.0
            warmup_ratio = self.trial.suggest_float("LR_Warmup_Ratio", 
                                                    low = 0.1, 
                                                    high = 0.4, 
                                                    step = 0.05)            
            
        training_params = {"Learning_Rate": lr,
                           "LR_Warmup_Ratio": warmup_ratio, 
                          }
        return training_params            
    
    def CNN_encoder(self):
        out_channels = 2**self.trial.suggest_int("Out_Channels", 
                                                 low = 6, 
                                                 high = 8, 
                                                 step = 1)
        kernel_size = self.trial.suggest_int("Kernel_Size", 
                                                 low = 3, 
                                                 high = 5, 
                                                 step = 2)
        latent_size = 2**self.trial.suggest_int("Latent_Size", 
                                                 low = 5, 
                                                 high = 8, 
                                                 step = 1)
        dropout = self.trial.suggest_int("Dropout", 
                                         low = 0.0, 
                                         high = 0.4, 
                                         step = 0.1)
        
        cnn_encoder_params = {"Out_Channels": out_channels,
                              "Kernel_Size": kernel_size, 
                              "Latent_Size": latent_size, 
                              "Dropout": dropout}
        return cnn_encoder_params
    
    def Transformer_encoder(self):
        num_layers = self.trial.suggest_int("Num_Layers", 
                                            low = 1, 
                                            high = 2,
                                            step = 1)
        d_model = 2**self.trial.suggest_int("d_model", 
                                            low = 6, 
                                            high = 8, 
                                            step = 1)
        nhead = 2**self.trial.suggest_int("Attention_Heads", 
                                          low = 1, 
                                          high = 3, 
                                          step = 1)

        dropout = self.trial.suggest_int("Dropout", 
                                         low = 0.0, 
                                         high = 0.4, 
                                         step = 0.1)
        
        transformer_encoder_params = {"Num_Layers": num_layers,
                                      "d_model": d_model, 
                                      "Attention_Heads": nhead, 
                                      "Dropout": dropout}
        return transformer_encoder_params    
    
    def LSTM_encoder(self):
        num_layers = self.trial.suggest_int("Num_Layers", 
                                            low = 1, 
                                            high = 3, 
                                            step = 1)
        latent_size = 2**self.trial.suggest_int("Latent_Size", 
                                                 low = 5, 
                                                 high = 8, 
                                                 step = 1)
        dropout = self.trial.suggest_int("Dropout", 
                                         low = 0.0, 
                                         high = 0.4, 
                                         step = 0.1)
        
        lstm_encoder_params = {"Num_Layers": num_layers, 
                               "Latent_Size": latent_size, 
                               "Dropout": dropout}
        return lstm_encoder_params    
    
 
    def MLP_encoder(self):
        latent_size = 2**self.trial.suggest_int("Latent_Size", 
                                                 low = 5, 
                                                 high = 8, 
                                                 step = 1)
        dropout = self.trial.suggest_int("Dropout", 
                                         low = 0.0, 
                                         high = 0.4, 
                                         step = 0.1)
        
        mlp_encoder_params = {"Latent_Size": latent_size, 
                              "Dropout": dropout}
        return mlp_encoder_params       
    
    def decoder(self):
        hidden_size = 2**self.trial.suggest_int("Hidden_Size", 
                                                low = 5, 
                                                high = 7, 
                                                step = 1)
        return {"Hidden_Size": hidden_size}
    
    def get_hyperparams(self):
        encoder_params = getattr(self, f"{self.model_name}_encoder")()
        decoder_params = self.decoder()
        training_params = self.training()
        return {**encoder_params, **decoder_params, **training_params}
