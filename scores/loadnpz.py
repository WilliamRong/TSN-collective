import numpy as np

data =np.load('./cad_rgb_bs128_split_1.npz')

print(data['scores'])
print(np.shape(data['scores']))
print(data['labels'])
