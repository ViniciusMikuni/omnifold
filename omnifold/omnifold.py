import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint, ReduceLROnPlateau
import sys, os



from datetime import datetime
import gc
import pickle
from omnifold.net import weighted_binary_crossentropy

hvd_installed = False
try:
    import horovod.tensorflow.keras as hvd
    # Use the optional package features if available
    print("Horovod instalation found.")
    hvd_installed = True
    hvd.init()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
        
except ImportError:
    # Continue running the code without the optional package
    print("Horovod not found, will continue with single only GPUs.")


def expit(x):
    return 1. / (1. + np.exp(-x))

class MultiFold():
    def __init__(self,
                 name,
                 model_reco,
                 model_gen,
                 data,
                 mc,
                 weights_folder = 'weights',
                 log_folder = './',
                 strap_id = 0,
                 niter = 3,
                 n_ensemble = 1,
                 batch_size = 128,
                 epochs = 50,
                 lr = 1e-4,
                 early_stop = 10,
                 start = 0,
                 train_frac = 0.8,
                 size = 1,
                 rank = 0,
                 verbose=False):
        """
        Initializes the MultiFold class for unbinned unfolding.
        
        Parameters:
        -----------
        name : str
            A unique name or identifier for this instance of the unfolding process.
        model_reco : keras.Model or tensorflow.keras.Model
            The model architecture used for reweighting the detector-level (reco) data.        
        model_gen : keras.Model or tensorflow.keras.Model
            The model architecture used for reweighting the truth-level (gen) data.        
        data : a DataLoader instance containing the measured data and initial weights.
        mc : a DataLoader instance containing the simulation both at reco and gen level..
        weights_folder : str, optional (default='weights')
            The folder where the trained model weights will be saved after each iteration.
        log_folder : str, optional (default='./')
            The folder where logs and other output files will be stored.
        strap_id : int, optional (default=0)
            A unique identifier for the bootstrap resampling, used in ensemble methods or 
            for generating different training data sets through resampling.
        niter : int, optional (default=3)
            Number of iterations to run the unfolding algorithm.
        n_ensemble: int, optinal (default=3)
            Number of ensembles (full runs of omnifold procedure) to avareg over for robustness
        batch_size : int, optional (default=128)
            Batch size used during training of the neural network models.
        epochs : int, optional (default=50)
            The maximum number of epochs to train the neural network models for each iteration.
        lr : float, optional (default=1e-4)
            Learning rate for the optimizer used during neural network training.
        early_stop : int, optional (default=10)
            The number of epochs to wait for improvement before stopping the training early.
        start : int, optional (default=0)
            The iteration number to start from, useful when resuming a previously stopped process.
        train_frac : float, optional (default=0.8)
            The fraction of the data to be used for training, with the remaining data used for validation.
        size : int, optional (default=1)
            Total number of processes or jobs in a distributed training setup (parallel training). This is the total number of tasks that are part of a multi-process run.
        rank : int, optional (default=0)
            Rank or identifier for the current process in a distributed or parallel training setup.
        verbose : bool, optional (default=False)
            Whether to print detailed information during training and unfolding iterations. 
            If set to `True`, detailed logs and progress updates will be shown.
        """
         
        self.name=name
        self.niter = niter
        self.n_ensemble = n_ensemble
        self.data = data
        self.mc = mc
        self.strap_id = strap_id
        self.start = start
        self.train_frac = train_frac
        self.size = size
        self.rank = rank
        self.verbose = verbose*(self.rank==0)
        self.log_file =  open(os.path.join(log_folder,'log_{}.txt'.format(self.name)),'w')
        
        #Model specific parameters
        self.model1 = model_reco
        self.model2 = model_gen
        self.BATCH_SIZE=batch_size
        self.EPOCHS=epochs
        self.LR = lr
        self.patience = early_stop

        # Used for number of train or val steps, used in self.CompileModel
        self.num_steps_reco = int((self.mc.nmax + self.data.nmax))//self.size//self.BATCH_SIZE
        self.num_steps_gen = 2*self.mc.nmax//self.size//self.BATCH_SIZE
        if self.rank==0:
            self.log_string(f"{self.num_steps_reco} training steps at reco and {self.num_steps_gen} steps at gen")
                
        self.weights_folder = weights_folder
        if self.strap_id>0:
            self.weights_folder = f'{self.weights_folder}_strap'            
            if self.verbose: self.log_string(f"INFO: Running bootstrapping number {self.strap_id}")
            np.random.seed(self.strap_id)
            
        if not os.path.exists(self.weights_folder):
            os.makedirs(self.weights_folder)

        self.PrepareInputs()


    def Unfold(self):
        
        self.step1_models = []  # list for model1 ensembles
        self.step2_models = []  # list for model2 ensembles

        self.weights_pull = np.ones(self.mc.weight.shape[0],dtype=np.float32)
        if self.start>0:
            self.LoadStart()  # Loads 'self.start' iteration. Starts at ensemble=0
        else:
            self.weights_push = np.ones(self.mc.weight.shape[0],dtype=np.float32)


        self.CompileModels()
        for i in range(self.start,self.niter):
            if self.rank==0:
                self.log_string("ITERATION: {}".format(i + 1))
            self.RunStep1(i)        
            self.RunStep2(i)
            self.CompileModels(fixed=True)

    def RunStep1(self,i):
        '''Data versus reco MC reweighting'''
        if self.rank==0:
            self.log_string("RUNNING STEP 1")
        
        self.RunModel(
            np.concatenate((self.labels_mc,self.labels_data)),
            
            np.concatenate((self.weights_push*self.mc.weight*self.mc.pass_reco,
                            self.data.weight*self.data.pass_reco)),
            
            i,self.model1,stepn=1,
            NTRAIN = self.num_steps_reco*self.BATCH_SIZE,
            cached = i>self.start #after first training cache the training data
        )

        #Don't update weights where there's no reco events
        new_weights = np.ones_like(self.weights_pull)
        new_weights[self.mc.pass_reco] = self.reweight(self.mc.reco,self.model1,batch_size=1000)[self.mc.pass_reco]
        self.weights_pull = self.weights_push *new_weights

    def RunStep2(self,i):
        '''Gen to Gen reweighing'''        
        if self.rank==0:
            self.log_string("RUNNING STEP 2")
        
        self.RunModel(
            np.concatenate((self.labels_mc, self.labels_gen)),
            np.concatenate((self.mc.weight*self.mc.pass_gen, 
                            self.mc.weight*self.weights_pull*self.mc.pass_gen)),
            i,self.model2,stepn=2,
            NTRAIN = self.num_steps_gen*self.BATCH_SIZE,
            cached = i>self.start #after first training cache the training data
        )
        new_weights = np.ones_like(self.weights_push)
        new_weights[self.mc.pass_gen]=self.reweight(self.mc.gen,self.model2)[self.mc.pass_gen]
        self.weights_push = new_weights


    def RunModel(self,
                 labels,
                 weights,
                 iteration,
                 model,
                 stepn,
                 NTRAIN=1000,
                 cached = False,
                 ):

        test_frac = 1.-self.train_frac
        NTEST = int(test_frac*NTRAIN)
        train_data, test_data = self.cache(labels,weights,stepn,cached,NTRAIN-NTEST)
        
        if self.verbose:
            self.log_string(80*'#')
            self.log_string("Train events used: {}, Test events used: {}".format(NTRAIN,NTEST))
            self.log_string(80*'#')

        # used for number of trainig steps
        num_steps = self.num_steps_reco if stepn==1 else self.num_steps_gen

        # Loop over Ensembles. Averaging done in self.reweight()
        for e in range(self.n_ensemble):
            ''' ensembling implemented here, in RunModel. This reduces parallelization''' 
            ''' but results in overall less variance. Called 'step ensembling' since  '''
            ''' the ensembling is done within each step, before passing onto the next '''
            ''' step or iteration. Alternative would be to call a script and run the  '''
            ''' OmniFold procedure as a whole (for all iterations), [n_ensemble] times'''

            if self.rank == 0 and self.n_ensemble > 1:
                self.log_string("Ensemble: {} / {}".format(e + 1, self.n_ensemble))

            # callbacks    
            if hvd_installed:
                callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0),
                             hvd.callbacks.MetricAverageCallback(),]
            else:
                callbacks = []

            callbacks = callbacks + [ReduceLROnPlateau(patience=1000, min_lr=1e-7,
                                                       verbose=self.verbose,
                                                       monitor="val_loss"),
                                     EarlyStopping(patience=self.patience,
                                                   restore_best_weights=True,
                                                   monitor="val_loss"), ]

            if self.rank == 0:  # Model checkpoint name
                model_name = self.get_model_name(iteration,stepn,e)
                callbacks.append(ModelCheckpoint(model_name,
                                                 save_best_only=True,
                                                 mode='auto',
                                                 save_weights_only=True))
                
            # Instantiate new model, then load from previous iteration
            if iteration < 1:
                model_e = tf.keras.models.clone_model(model)

                if stepn == 1:
                    self.step1_models.append(model_e)
                if stepn == 2:
                    self.step2_models.append(model_e)

            else:
                model_e = self.step1_models[e] if stepn == 1 else self.step2_models[e]
                        
            # model_e = model  # TEST of processing iterations w/o ensemble


            self.CompileModel(model_e, num_steps)

            hist =  model_e.fit(
                train_data,
                epochs=self.EPOCHS,
                steps_per_epoch=int(self.train_frac*NTRAIN//self.BATCH_SIZE),
                validation_data= test_data,
                validation_steps=NTEST//self.BATCH_SIZE,
                verbose= self.verbose,
                callbacks=callbacks)
            
            self.log_string(f"Last val loss {hist.history['val_loss'][0]}")
            
            if self.rank ==0:
                self.log_string("INFO: Dumping training history ...")
                with open(model_name.replace(".weights.h5",".pkl"),"wb") as f:
                    pickle.dump(hist.history, f)
            
        del train_data, test_data
        gc.collect()

    def get_model_name(self,iteration,stepn,e):
        model_name = '{}/OmniFold_{}_iter{}_step{}'.format(
            self.weights_folder, self.name, iteration, stepn)
        if self.n_ensemble > 1: model_name += '_ensemble{}'.format(e)
        if self.strap_id > 0: model_name += '_strap{}'.format(self.strap_id)
        return model_name + '.weights.h5'
        
        
    def cache(self,
              label,
              weights,
              stepn,
              cached,
              NTRAIN
              ):


        if not cached:
            if self.verbose:
                self.log_string("Creating cached data from step {}".format(stepn))
                    
            if stepn ==1:
                self.idx_1 = np.arange(label.shape[0])
                np.random.shuffle(self.idx_1)
                self.tf_data1 = tf.data.Dataset.from_tensor_slices(
                    np.concatenate([self.mc.reco,self.data.reco],0)[self.idx_1])
            elif stepn ==2:
                self.idx_2 = np.arange(label.shape[0])
                np.random.shuffle(self.idx_2)
                self.tf_data2 = tf.data.Dataset.from_tensor_slices(
                    np.concatenate([self.mc.gen,self.mc.gen],0)[self.idx_2])
                
        idx = self.idx_1 if stepn==1 else self.idx_2
        labels = tf.data.Dataset.from_tensor_slices(np.stack((label[idx],weights[idx]),axis=1))
        
        if stepn ==1:
            data = tf.data.Dataset.zip((self.tf_data1,labels))
        elif stepn==2:
            data = tf.data.Dataset.zip((self.tf_data2,labels))
        else:
            logging.error("ERROR: STEPN not recognized")

                
        train_data = data.take(NTRAIN).shuffle(NTRAIN).repeat().batch(self.BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
        test_data  = data.skip(NTRAIN).repeat().batch(self.BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
        del data
        gc.collect()
        return train_data, test_data

    def get_optimizer(self,num_steps,fixed=False,min_learning_rate = 1e-5):
        opt = tf.keras.optimizers.Adam(learning_rate=min_learning_rate if fixed else self.LR)
        if hvd_installed:
            opt = hvd.DistributedOptimizer(opt)
        return opt
        

    def CompileModel(self,model,num_steps,fixed=False):

        opt = self.get_optimizer(int(self.train_frac*num_steps),fixed=fixed)
        model.compile(opt,loss = weighted_binary_crossentropy,
                      weighted_metrics=[])

    def CompileModels(self,fixed=False):

        self.CompileModel(self.model1,self.train_frac*self.num_steps_reco,fixed)
        self.CompileModel(self.model2,self.train_frac*self.num_steps_gen, fixed)

        if self.n_ensemble > 1 and len(self.step1_models) > 0:
            for model in self.step1_models:
                self.CompileModel(model,self.train_frac*self.num_steps_reco,fixed)
            for model in self.step2_models:
                self.CompileModel(model,self.train_frac*self.num_steps_gen, fixed)


    def LoadStart(self):

        if self.verbose: self.log_string(f"INFO: Continuing OmniFold training from Iteration {self.start}")
        if self.rank == 0:
            self.log_string("Loading step 2 weights from iteration {}".format(self.start-1))

        if self.n_ensemble > 1:

            for e in range(self.n_ensemble):
                model1_name = '{}/OmniFold_{}_iter{}_step1_ensemble{}.weights.h5'.format(
                    self.weights_folder,self.name,self.start-1, e)
                model2_name = '{}/OmniFold_{}_iter{}_step2_ensemble{}.weights.h5'.format(
                    self.weights_folder,self.name,self.start-1, e)

                temp1 = tf.keras.models.clone_model(self.model1)
                temp2 = tf.keras.models.clone_model(self.model2)
                temp1.load_weights(model1_name)  #better starting point for model 1
                temp2.load_weights(model2_name)
                self.step1_models.append(temp1) #FIXME: need to put this in ensemble loop
                self.step2_models.append(temp2)

        else:  # no ensembling
            model1_name = '{}/OmniFold_{}_iter{}_step1.weights.h5'.format(self.weights_folder,self.name,self.start-1)
            model2_name = '{}/OmniFold_{}_iter{}_step2.weights.h5'.format(self.weights_folder,self.name,self.start-1)

            temp1 = tf.keras.models.clone_model(self.model1)
            temp2 = tf.keras.models.clone_model(self.model2)
            temp1.load_weights(model1_name)  #better starting point for model 1
            temp2.load_weights(model2_name)
            self.step1_models.append(temp1) #FIXME: need to put this in ensemble loop
            self.step2_models.append(temp2)

        self.weights_push = self.reweight(self.mc.gen,self.model2,batch_size=1000)


    def PrepareInputs(self):
        self.labels_mc = np.zeros(len(self.mc.pass_reco),dtype=np.float32)
        self.labels_data = np.ones(len(self.data.pass_reco),dtype=np.float32)
        self.labels_gen = np.ones(len(self.mc.pass_gen),dtype=np.float32)


    def reweight(self,events,model,batch_size=None):
        if batch_size is None:
            batch_size =  self.BATCH_SIZE

        if self.n_ensemble > 1 and len(self.step1_models) > 0:  # need to take avg of ensembles
            self.log_string("Averaging over ensembles...")
        models = self.step1_models if model == self.model1 else self.step2_models

        avg_weights = np.zeros((len(events)))
        for model in models:
            f = expit(model.predict(events,batch_size=batch_size,verbose=self.verbose))
            weights = f / (1. - f)  # this is the crux of the reweight, approximates likelihood ratio
            weights = np.nan_to_num(weights[:,0],posinf=1)
            avg_weights += weights / len(models)
        return avg_weights


    def log_string(self,out_str):
        self.log_file.write(out_str+'\n')
        self.log_file.flush()
        print(out_str)



