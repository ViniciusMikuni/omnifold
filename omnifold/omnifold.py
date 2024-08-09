import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint, ReduceLROnPlateau
import sys, os
import horovod.tensorflow.keras as hvd

from datetime import datetime
import gc
import pickle



def weighted_binary_crossentropy(y_true, y_pred):
    weights = tf.gather(y_true, [1], axis=1) # event weights
    y_true = tf.gather(y_true, [0], axis=1) # actual y_true for loss

    t_loss = weights*tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    return tf.reduce_mean(t_loss)
    
    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = 1e-9
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    t_loss = -weights * ((y_true) * tf.math.log(y_pred) +
                         (1 - y_true) * tf.math.log(1 - y_pred))
    return tf.reduce_mean(t_loss)


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
                 nstrap=0,
                 niter = 3,
                 batch_size = 128,
                 epochs = 50,
                 lr = 1e-4,
                 early_stop = 10,
                 start = 0,
                 train_frac = 0.8,
                 size = 1,
                 rank = 0,
                 verbose=False):

        self.name=name
        self.niter = niter
        self.data = data
        self.mc = mc
        self.nstrap = nstrap
        self.start = start
        self.train_frac = train_frac
        self.size = size
        self.rank = rank
        self.verbose = verbose*(self.rank==0)
        self.log_file =  open('log_{}.txt'.format(self.name),'w')
        
        #Model specific parameters
        self.model1 = model_reco
        self.model2 = model_gen
        self.BATCH_SIZE=batch_size
        self.EPOCHS=epochs
        self.LR = lr
        self.patience = early_stop

        self.num_steps_reco = None
        self.num_steps_gen = None
                
        self.weights_folder = weights_folder
        if self.nstrap>0:
            self.weights_folder = f'{self.weights_folder}_strap'
            
        if not os.path.exists(self.weights_folder):
            os.makedirs(self.weights_folder)

        if not hvd.is_initialized():
            hvd.init()

    def Unfold(self):                                        
        self.weights_pull = np.ones(self.mc.weight.shape[0])
        if self.start>0:
            if self.verbose: self.log_string(f"INFO: Continuing OmniFold training from Iteration {self.start}")
            if self.rank == 0:
                self.log_string("Loading step 2 weights from iteration {}".format(self.start-1))
            model_name = '{}/OmniFold_{}_iter{}_step2/checkpoint'.format(self.weights_folder,self.name,self.start-1)
            self.model2.load_weights(model_name).expect_partial()
            self.weights_push = self.reweight(self.mc.gen,self.model2,batch_size=1000)
            #Also load model 1 to have a better starting point
            model_name = '{}/OmniFold_{}_iter{}_step1/checkpoint'.format(self.weights_folder,self.name,self.start-1)
            self.model1.load_weights(model_name).expect_partial()
        else:
            self.weights_push = np.ones(self.mc.weight.shape[0])

        self.CompileModel()
        for i in range(self.start,self.niter):
            if self.rank==0:
                self.log_string("ITERATION: {}".format(i + 1))
            self.RunStep1(i)        
            self.RunStep2(i)
            self.CompileModel(fixed=True)

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
            np.concatenate((self.mc.weight*self.mc.pass_gen, self.mc.weight*self.weights_pull*self.mc.pass_gen)),
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
            print(80*'#')
            self.log_string("Train events used: {}, Test events used: {}".format(NTRAIN,NTEST))
            print(80*'#')


        
        callbacks = [
            hvd.callbacks.BroadcastGlobalVariablesCallback(0),
            hvd.callbacks.MetricAverageCallback(),

            #ReduceLR only used to keep track of the LR during training
            ReduceLROnPlateau(patience=1000, min_lr=1e-7,
                              verbose=self.verbose,
                              monitor="val_loss"),
            EarlyStopping(patience=self.patience,
                          restore_best_weights=True,
                          monitor="val_loss"),
        ]
        
        
        if self.rank ==0:
            if self.nstrap>0:
                model_name = '{}/OmniFold_{}_iter{}_step{}_strap{}/checkpoint'.format(
                    self.weights_folder,self.name,iteration,stepn,self.nstrap)
            else:
                model_name = '{}/OmniFold_{}_iter{}_step{}/checkpoint'.format(
                    self.weights_folder,self.name,iteration,stepn)
                
            callbacks.append(ModelCheckpoint(model_name,
                                             save_best_only=True,
                                             mode='auto',period=1,
                                             save_weights_only=True))
                    
        hist =  model.fit(
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
            with open(model_name.replace("/checkpoint",".pkl"),"wb") as f:
                pickle.dump(hist.history, f)
        
        del train_data, test_data
        gc.collect()


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

    def Preprocessing(self):
        self.PrepareInputs()

    def get_optimizer(self,num_steps,fixed=False,min_learning_rate = 1e-5):
        opt = tf.keras.optimizers.Adam(learning_rate=min_learning_rate if fixed else self.LR)
        opt = hvd.DistributedOptimizer(opt)
        return opt
        

    def CompileModel(self,fixed=False):

        if self.num_steps_reco ==None:
            self.num_steps_reco = int((self.mc.nmax + self.data.nmax))//self.size//self.BATCH_SIZE
            self.num_steps_gen = 2*self.mc.nmax//self.size//self.BATCH_SIZE
            if self.rank==0:
                self.log_string(f"{self.num_steps_reco} training steps at reco and {self.num_steps_gen} steps at gen")


        opt1 = self.get_optimizer(int(self.train_frac*self.num_steps_reco),fixed=fixed)
        opt2 = self.get_optimizer(int(self.train_frac*self.num_steps_gen),fixed=fixed)
        

        self.model1.compile(opt1,experimental_run_tf_function=False,
                            loss = weighted_binary_crossentropy,
                            weighted_metrics=[])
        self.model2.compile(opt2,experimental_run_tf_function=False,
                            loss = weighted_binary_crossentropy,
                            weighted_metrics=[])


    def PrepareInputs(self):
        self.labels_mc = np.zeros(len(self.mc.pass_reco),dtype=np.float32)
        self.labels_data = np.ones(len(self.data.pass_reco),dtype=np.float32)
        self.labels_gen = np.ones(len(self.mc.pass_gen),dtype=np.float32)

    def reweight(self,events,model,batch_size=None):
        if batch_size is None:
           batch_size =  self.BATCH_SIZE

        f = expit(model.predict(events,batch_size=batch_size,verbose=self.verbose))
        weights = f / (1. - f)
        return np.nan_to_num(weights[:,0],posinf=1)

        
    def log_string(self,out_str):
        self.log_file.write(out_str+'\n')
        self.log_file.flush()
        print(out_str)



