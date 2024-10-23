# RLPruner: Structural Pruner based on Reinforcement Learning for CNN Compression and Acceleration

End-to-end structural pruning for CNNs based on reinforcement learning.
The current release supports:
- RLPruner compression for classification CNNs

## News
[2024/08] We implement RLPruner for classification CNNs, supporting models that use concatenation, residual connectoin and grouped convolution.


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
git clone https://github.com/Beryex/RLPruner-CNN.git --depth 1
cd RLPruner-CNN
```
2. Set up environment
```bash 
conda create -n RLPruner python=3.10 -y
conda activate RLPruner
pip install -r requirements.txt
```
## Usage
RLPruner could auto detect the layer dependence inside the model and execute structural pruning. So it can accept any type of CNN without model-wise pruning code implementation. However, RLPruner assume that:

- the output layer is defined last
- Define the layers to be used during initialization, rather than creating new layers within the 'forward' method. For example, use self.act = nn.ReLU(inplace=True) and def forward(x): return self.act(layer1(x) + layer2(x)), rather than def forward(x): return nn.ReLU(inplace=True)(layer1(x) + layer2(x))
- The layers defined during initialization should be used only once in the forward method, not multiple times. For example, use self.act1 = nn.ReLU(inplace=True), self.act2 = nn.ReLU(inplace=True) and def forward(x): return self.act2(layer2(self.act1(layer1(x)))), rather than self.act = nn.ReLU(inplace=True) and def forward(x): return self.act(layer2(self.act(layer1(x))))

RLPruner support grouped convolution, but it will only prune depthwise convolution. Grouped convolution layers that are not depthwise will be skipped during the pruning process.

There is a example bash in [scripts](scripts) folder. Change the model and dataset to try your own model and you need to store your model with name {model_name}_{dataset_name}_original.pth at pretrained_model folder.
## Results
