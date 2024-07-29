# RLPruner: Structural Pruner based on Reinforcement Learning for CNN Compression and Acceleration

End-to-end structural pruning for CNNs based on reinforcement learning.
The current release supports:
- RLPruner compression for classification CNNs

## News
[2024/08] We implement end-to-end pruning for classification CNNs.


## Contents
- [RLPruner: A Structural Pruner based on Reinforcement Learning for CNN Compression and Acceleration](#rlpruner-a-structural-pruner-based-on-reinforcement-learning-for-cnn-compression-and-acceleration)
	- [News](#News)
	- [Contents](#Contents)
	- [Install](#Install)
	- [Usage](#Usage)
	- [Results](#Results)
## Install
1. Clone the repository and aavigate to the RLPruner working directory
```bash 
git clone https://github.com/Beryex/RLPruner-CNN.git
cd RLPruner-CNN
```
2. Set up environment
```bash 
conda create -n RLPruner python=3.10 -y
conda activate RLPruner
pip install -r requirements.txt
```
## Usage
RLPruner could auto detect the layer dependence inside the model and execute structural pruning. So it can accept any type of CNN without model-wise pruning code implementation. However, RLPruner assume that, in the model definition:
- the output layer is defined last
- Use self.act = nn.ReLU(inplace=True) and def forward(x): return self.act(layer1(x) + layer2(x)), rather than def forward(x): return nn.ReLU(inplace=True)(layer1(x) + layer2(x))
There is a example bash in [scripts](scripts) folder. Change the model and dataset to try your own model and you need to store your model with name {model_name}_{dataset_name}_original.pth at pretrained_model folder.
## Results
