# OmniFold: A Method to Simultaneously Unfold All Observables

This repository contains the implementation and examples of the OmniFold algorithm originally described in [Phys. Rev. Lett. 124 (2020) 182001](https://dx.doi.org/10.1103/PhysRevLett.124.182001), [1911.09107 [hep-ph]](https://arxiv.org/abs/1911.09107).  The code for the original paper can be found at [this repository](https://github.com/ericmetodiev/OmniFold), which includes a [binder demo](https://mybinder.org/v2/gh/ericmetodiev/OmniFold/master).  This repository was created to maintain the pip installable version of OmniFold with additional functionality compared to the original package:

# Installation

```bash
pip install omnifold
```

# Getting Started
Examples for tabular data and for Point Cloud-like inputs are provided in the notebooks OmniFold_example.ipynb and OmniFold_example_pc.ipynb, respectively.
To unfold your own data you can follow these steps:

## Creating a DataLoader to hold your data

```python
from omnifold import DataLoader

# Load your own dataset, possibly with weights.
mock_dataset = np.zeros((100,3,4))
mock_dataloader = DataLoader(
		  reco = mock_dataset,
		  gen = mock_dataset,
		  normalize=True)

```

The DataLoader class will automatically normalize the weights if ```normalize``` is true. To estimate the statistical uncertainty using the Bootstrap method, you can use the optional flag ```bootstrap = True```.

## Creating your own Keras model to be used for Unfolding

In the MultiFold class, we provide simple neural network models that you can use. For a Multilayer Perceptron you can load

```python
from omnifold import MLP
ndim = 3 #The number of features present in your dataset
reco_model = MLP(ndim)
gen_model = MLP(ndim)
```

to create the models to be used at both reconstruction and generator level trainings of OmniFold. In case your data is better described by a point cloud, we also provide the implementation of the [Point-Edge Transformer (PET)](https://arxiv.org/abs/2404.16091) model that can be used similarly to the MLP implementation:


```python
from omnifold import PET
ndim = 3 #The number of features present in your dataset
npart = 5 #Maximum number of particles present in the dataset

reco_model = PET(ndim,num_part = npart)
gen_model = PET(ndim,num_part = npart)
```

You can also provide your own custom keras.Model to be used by OmniFold

## Creating the MultiFold Object

Now that we have the dataset and models, we can create the MultiFold object that performs the unfolding and reweighting of new datasets

```python
omnifold = MultiFold(
    "Name_of_experiment",
    reco_model,
    gen_model,
    data, # a dataloader instance containing the measured data
    mc , # a dataloader instance containing the simulation
)

```

The last step is to finally run the unfolding!

```python
omnifold.Unfold()
```

# Evaluating the Unfolded Results:

We can evaluate the reweighting function learned by OmniFold by using the reweight function

```python
unfolded_weights  = omnifold.reweight(validation_data,omnifold.model2,batch_size=1000) 
```

These weights can be applied directly to the simulation used during the unfolding to produce the unfolded results.

# Plotting the Results of the Unfolding

The omnifold package also provides a histogram functionality that you can use to plot histograms. You can use the plotting code as:

```python
from omnifold import SetStyle, HistRoutine
SetStyle()

#Create a dictionary containing the data to plot
data_dict = {
    'Distribution A': data_a, 
    'Distribution B': data_b,
}
HistRoutine(data_dict,'Name of the feature to plot', reference_name = 'Name of the dataset to calculate the ratio plot')
```

The function will create the histograms for the datasets used as part of the inputs. Specifc binnings for the histograms can be passed as numpy arrays in the binning argument, or calculated directly by the routine.

Weights can be added to the histograms by passing an additional dictionary with the same key entries and the weights to be used for each distribution. For example:

```python
weight_dict = {
    'Distribution A': weight_a, 
    'Distribution B': weight_b,
}
HistRoutine(data_dict,'Name of the feature to plot', reference_name = 'Name of the dataset to calculate the ratio plot', weights = weight_dict)
```