import torch 
from model import BRUnet_raw_Multi

input_signal = torch.rand(64,3,32*64)
inshape = (64,3,32*64)
model = BRUnet_raw_Multi(inshape)

model(input_signal)
