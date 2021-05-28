from utils import Positional_Encoding
import torch
import matplotlib.pyplot as plt
n, d = 2048, 512
pos_encoding = Positional_Encoding(d,n,'cpu')
print(pos_encoding.Positional.shape)
pos_encoding = pos_encoding.Positional[0]

# Juggle the dimensions for the plot
pos_encoding = torch.reshape(pos_encoding, (n, d//2, 2))
pos_encoding = pos_encoding.permute(2,1,0)
pos_encoding = torch.reshape(pos_encoding, (d, n))

plt.pcolormesh(pos_encoding, cmap='RdBu')
plt.ylabel('Depth')
plt.xlabel('Position')
plt.colorbar()
plt.show()