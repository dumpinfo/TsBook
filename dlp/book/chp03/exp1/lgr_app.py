import numpy as np
import csv
from lgr_engine import Lgr_Engine

class Lgr_App:
    def __init__(self):
        pass
        
    def startup(self):
        lgr_engine = Lgr_Engine('datasets/linear_data_train.csv', 
                'datasets/linear_data_eval.csv', 2, 2, 100)
        #lgr_engine.train()
        x = [[0.27, 0.02]]
        lgr_engine.run(x)
        
if '__main__' == __name__:
    lgr_app = Lgr_App()
    lgr_app.startup()