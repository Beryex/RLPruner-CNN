from utils import extract_layers_info, extract_layer_dependence
import torch
device = 'cuda'

from models.googlenet import GoogleNet
ret = GoogleNet(3, 100).to(device)

_, _, layers = extract_layers_info(ret)
x = torch.rand(1, 3, 32, 32).to(device)
next_layers = extract_layer_dependence(ret, x, layers)
for i in range (len(layers)):
    print("Cur Conv/Linear:", layers[i])
    print("Next layers:", next_layers[i])