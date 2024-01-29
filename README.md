# ECE 397: Individual Study in ECE @ UIUC Spring 2024 by Bryan Wang
[Bryan Wang](https://github.com/Beryex) | ECE 397: Individual Study, Spring 2024  
Research Supervisor:  
[Kindratenko Volodymyr](https://cs.illinois.edu/about/people/faculty/kindrtnk) @ University of Illinois at Urbana-Champaign

## **To Do List**
- try to find optimal architecture during training epochs
- try to find best initial value based on the score of all architecture

## To run it
- in one terminal
```
python -m visdom.server
```
Then go to that local host to see the visualization of result
- run train.py in another terminal
```
cd .\LeNet_Module_for_Digits_Recognition\
python train.py
```

## **Current Version: 0.0.2**
### Update Content:
- Implement basic version of finding optimal architecture
### **Version: 0.0.10**
### Update Content:
- Implement LeNet model for digits recognization
- Implement of visualization of results
### **Version: 0.0.05**
### Update Content:
- Build Pytorch environment using anaconda and link it to VSCode
- Link the local repository to the github remote repository
