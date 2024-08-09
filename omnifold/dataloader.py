import numpy as np
import sys, os



class DataLoader():
    def __init__(
            self,
            reco,            
            pass_reco = None,
            gen = None,
            pass_gen = None,
            weight = None,
            normalize=False,
            normalization_factor = 1_000_000,
    ):
        self.reco = reco
        self.pass_reco = pass_reco
        self.gen = gen
        self.pass_gen = pass_gen

        self.nmax = self.reco.shape[0]
        self.weight = weight
        
        if self.weight is None:
            print("INFO: Creating weights ...")
            self.weight = np.ones(reco.shape[0],dtype=np.float32)
            
        if self.pass_reco is None:
            print("INFO: Creating pass reco flag ...")
            self.pass_reco = np.ones(reco.shape[0],dtype=bool)
        else:
            #Make a boolean mask
            self.pass_reco = np.array(self.pass_reco) == 1
            
        if self.gen is not None:
            self.is_mc = True
        else:
            self.is_mc = False
            
        if self.is_mc and self.pass_gen is None:
            print("INFO: Creating pass gen flag ...")
            self.pass_gen = np.ones(gen.shape[0],dtype=bool)
        else:
            #Make a boolean mask
            self.pass_gen = np.array(self.pass_gen) == 1


        if normalize:
            print(f"INFO: Normalizing sum of weights to {normalization_factor} ...")
            sumw = np.sum(self.weight[np.array(self.pass_reco)==1])
            self.weight *= (normalization_factor/sumw).astype(np.float32)
