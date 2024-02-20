# ECE 397: Individual Study in ECE @ UIUC Spring 2024 by Bryan Wang
[Bryan Wang](https://github.com/Beryex) | ECE 397: Individual Study, Spring 2024  
Research Supervisor:  
[Kindratenko Volodymyr](https://cs.illinois.edu/about/people/faculty/kindrtnk) @ University of Illinois at Urbana-Champaign

# Method Description
![MyMethod](https://github.com/Beryex/UIUC-ECE397/blob/main/Figures%20for%20Visualization/Method.png)

## **To Do List**
- Refine the hyperparameters for VGG-16 Model
- Visualization of layers' feature capture in simplified LeNet-5 Model
- Try ResNet, UNet Model

## To run it
### Test LeNet-5
![LeNet-5](https://github.com/Beryex/UIUC-ECE397/blob/main/Figures%20for%20Visualization/LeNet-5.png)
- in one terminal
```
python -m visdom.server
```
Then go to that local host to see the visualization of result
- run train.py in another terminal
```
cd .\LeNet_5\DynamicVersion\
python train.py
```
### Test VGG-16
![VGG-16](https://github.com/Beryex/UIUC-ECE397/blob/main/Figures%20for%20Visualization/VGG-16.png)
```
cd .\VGG_16\DynamicVersion\
python train.py -net vgg16 -gpu
```

## **Current Version: 0.2.0**
### Update Content:
- Find simplified architecture for LeNet-5
- Insert the method into VGG-16 model, but result is not good
### **Version: 0.1.2**
### Update Content:
- Improve the algorithm to train architecture during training epochs on LeNet-5
### **Version: 0.0.75**
### Update Content:
- Implement basic algorithm to update architecture during training epochs
### **Version: 0.0.50**
### Update Content:
- Implement basic algorithm to test different architecture before training epochs
### **Version: 0.0.10**
### Update Content:
- Implement LeNet model for digits recognization
- Implement of visualization of results
### **Version: 0.0.05**
### Update Content:
- Build Pytorch environment using anaconda and link it to VSCode
- Link the local repository to the github remote repository
