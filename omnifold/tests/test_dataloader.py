from  omnifold import DataLoader
import numpy as np

mock_dataset = np.zeros((100,3,4))
mock_dataloader = DataLoader(reco = mock_dataset,gen = mock_dataset,normalize=True)
pass_reco = [0 if i % 2 == 0 else 1 for i in range(mock_dataloader.reco.shape[0])]

mock_dataloader = DataLoader(reco = mock_dataset,pass_reco = pass_reco,
                             normalize=True)
print(mock_dataloader.weight)
