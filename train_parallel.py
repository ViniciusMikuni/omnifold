from omnifold import DataLoader, MultiFold, MLP, PET, SetStyle, HistRoutine
import numpy as np
import horovod.tensorflow.keras as hvd
hvd.init()
import os
import h5py as h5

path = '/pscratch/sd/v/vmikuni/PET/OmniFold/'

reco_data = h5.File(os.path.join(path,'train_herwig.h5'))['reco']
reco_mc = h5.File(os.path.join(path,'train_pythia.h5'))['reco']
gen_mc = h5.File(os.path.join(path,'train_pythia.h5'))['gen']

data = DataLoader(reco = reco_data,normalize=True,
                  rank=hvd.rank(),
                  size=hvd.size(),)
mc = DataLoader(reco = reco_mc,gen = gen_mc,normalize=True,
                rank=hvd.rank(),
                size=hvd.size(),)


model1 = PET(num_feat=reco_mc.shape[-1],num_part=reco_mc.shape[1],
             num_transformer = 4,projection_dim=64,K=10)
model2 = PET(num_feat=gen_mc.shape[-1],num_part=gen_mc.shape[1],
             num_transformer = 4,projection_dim=64,local=False)

omnifold = MultiFold(
    "OmniFold",
    model1,
    model2,
    data,
    mc,
    batch_size = 256,
    verbose = True,
    niter = 5,
    epochs=100,
    early_stop=3,
    rank=hvd.rank(),
    size=hvd.size(),
)

omnifold.Unfold()

if hvd.rank()==0:
    unfolded_weights  = omnifold.reweight(h5.File(os.path.join(path,'test_pythia.h5'))['gen'][:],omnifold.model2,batch_size=1000)
    
    
    #Plotting
    SetStyle()
    data_dict = {
        'gen_data': h5.File(os.path.join(path,'test_herwig.h5'))['gen_subs'][:,0],
        'reco_data': h5.File(os.path.join(path,'test_herwig.h5'))['reco_subs'][:,0],
        'gen_mc': h5.File(os.path.join(path,'test_pythia.h5'))['gen_subs'][:,0],
        'reco_mc': h5.File(os.path.join(path,'test_pythia.h5'))['reco_subs'][:,0],
        'unfolded': h5.File(os.path.join(path,'test_pythia.h5'))['gen_subs'][:,0],
    }
    
    weight_dict = {
        'gen_data': np.ones_like(data_dict['gen_data']),
        'reco_data': np.ones_like(data_dict['reco_data']),
        'gen_mc': np.ones_like(data_dict['gen_mc']),
        'reco_mc': np.ones_like(data_dict['reco_mc']),
        'unfolded': unfolded_weights,
    }
    
    fig,_ = HistRoutine(data_dict,'widths',
                        reference_name = 'gen_data',
                        weights = weight_dict
                        )
    fig.savefig('test_omnifold.pdf')
    
