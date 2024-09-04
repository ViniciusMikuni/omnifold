from omnifold import DataLoader, MultiFold, MLP, PET, SetStyle, HistRoutine
import numpy as np


nevts = 1_000_00
npart = 4
ndim = 2

def detector(x,std = 0.5):
    return x + np.random.normal(size=x.shape)*std

gen_data = np.random.normal(size=(nevts,npart,ndim),loc=ndim*[0.0],scale=ndim*[1.0])
reco_data = detector(gen_data) 
gen_mc = np.random.normal(size=(nevts,npart,ndim),loc=ndim*[1.0],scale=ndim*[1.0])
reco_mc = detector(gen_mc) 

data = DataLoader(reco = reco_data,normalize=True)
mc = DataLoader(reco = reco_mc,gen = gen_mc,normalize=True)

model1 = PET(ndim,num_part=npart)
model2 = PET(ndim,num_part=npart)

omnifold = MultiFold(
    "test",
    model1,
    model2,
    data,
    mc,
    batch_size = 1024,
    verbose = True,
    # niter = 1,
    epochs=10,
)

omnifold.Preprocessing()
omnifold.Unfold()

unfolded_weights  = omnifold.reweight(mc.gen,omnifold.model2,batch_size=1000)


#Plotting
SetStyle()
data_dict = {
    'gen_data': gen_data[:,0,0],
    'reco_data': reco_data[:,0,0],
    'gen_mc': gen_mc[:,0,0],
    'reco_mc': reco_mc[:,0,0],
    'unfolded': gen_mc[:,0,0],
}

weight_dict = {
    'gen_data': data.weight,
    'reco_data': data.weight,
    'gen_mc': mc.weight,
    'reco_mc': mc.weight,
    'unfolded': unfolded_weights,
    }

fig,_ = HistRoutine(data_dict,'x',
                    reference_name = 'gen_data',
                    weights = weight_dict
                    )
fig.savefig('test_part.pdf')

