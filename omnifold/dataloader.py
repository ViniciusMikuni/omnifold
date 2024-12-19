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
            bootstrap = False,
            rank = 0,
            size = 1,
    ):
        """
        Initializes the DataLoader with the required datasets and parameters for handling 
        the training in OmniFold.

        Parameters:
        -----------
        reco : numpy.ndarray
            The detector-level (reconstructed) dataset.        
        pass_reco : numpy.ndarray, optional (default=None)
            A boolean array or mask that specifies a subset of the reconstructed data passing reco cuts.         
        gen : numpy.ndarray, optional (default=None)
            The truth-level (generated) dataset. This can be `None` for measured data.        
        pass_gen : numpy.ndarray, optional (default=None)
            A boolean array or mask for the truth-level data, specifying a subset of generated data to be used.        
        weight : numpy.ndarray, optional (default=None)
            An array of weights associated with the reconstructed or truth-level data. 
            These weights can be initial MC weights for the simulation        
        normalize : bool, optional (default=False)
            If `True`, the dataset will be normalized according to the provided `normalization_factor`.
            Normalization ensures that the total sum of weights equals the normalization factor.        
        normalization_factor : float, optional (default=1_000_000)
            The factor by which to normalize the dataset if `normalize` is set to `True`. 
            This value is applied such that the total sum of weights matches this factor.        
        bootstrap : bool, optional (default=False)
            If `True`, bootstrapping will be applied to resample the data. Bootstrapping involves random sampling 
            with replacement using Poisson weights.
        """

        if gen is not None:
            assert reco.shape[0] == gen.shape[0], "ERROR: Reco and Gen Events have different number of entries"
        self.rank = rank
        self.size = size
        self.nmax = reco.shape[0]
        self.reco = reco
        self.weight = weight
        self.gen = gen        
        self.pass_reco = pass_reco
        self.pass_gen = pass_gen
        self.bootstrap=bootstrap

        self.reco = self.reco[rank::size]
        if self.gen is not None:
            self.gen = self.gen[rank::size]

        if self.weight is None:
            if self.rank==0:print("INFO: Creating weights ...")
            self.weight = np.ones(self.reco.shape[0],dtype=np.float32)
        else:
            self.weight = self.weight[rank::size]
            
        if self.bootstrap:
            self.weight = np.random.poisson(1,self.weight.shape[0])*self.weight
            
        if self.pass_reco is None:
            if self.rank==0:print("INFO: Creating pass reco flag ...")
            self.pass_reco = np.ones(self.reco.shape[0],dtype=bool)
        else:
            #Make a boolean mask
            self.pass_reco = np.array(self.pass_reco) == 1
            #Distribute across GPUs
            self.pass_reco = self.pass_reco[rank::size]

        self.is_mc =  self.gen is not None

        if self.is_mc:            
            if  self.pass_gen is None:
                if self.rank==0:print("INFO: Creating pass gen flag ...")
                self.pass_gen = np.ones(self.gen.shape[0],dtype=bool)
            else:
                #Make a boolean mask
                self.pass_gen = np.array(self.pass_gen) == 1
                #Distribute across GPUs
                self.pass_gen = self.pass_gen[rank::size]
                

        if normalize:
            if self.rank==0:print(f"INFO: Normalizing sum of weights to {normalization_factor} ...")
            sumw = np.sum(self.weight[np.array(self.pass_reco)==1])
            self.weight *= (normalization_factor/sumw).astype(np.float32)
