# ECE 397: Individual Study in ECE @ UIUC Spring 2024 by Bryan Wang
[Bryan Wang](https://github.com/Beryex) | ECE 397: Individual Study, Spring 2024  
Research Supervisor:  
[Kindratenko Volodymyr](https://cs.illinois.edu/about/people/faculty/kindrtnk) @ University of Illinois at Urbana-Champaign

# Method Description
![MyMethod](https://github.com/Beryex/UIUC-ECE397/blob/main/Figures%20for%20Visualization/Algorithm.png)
## Hyperparameter
Hyperparameter are all stored in global_settings.py in folder conf under each specific model's directory.

## Usage
First move to the DevVersion directory of a specific model, then run
```
python train_original.py
```
Then replace the model in train_compressed.py with the reference model gained by train_original.py.
Run
```
usage: [your_script_name.py] [-h] [--criteria C] [--accuracy_threshold A] [--compression_threshold C] [--enable_adaptive_pruning]

Adaptive Compressing

optional arguments:
  -h, --help            show this help message and exit
  --criteria C, -c C    Compressed the model with accuracy_threshold or compression_threshold
  --accuracy_threshold A, -A A
                        The final accuracy the architecture will achieve
  --compression_threshold C, -C C
                        The final compression ratio the architecture will achieve
  --enable_adaptive_pruning, -eap
                        Enable the special feature if set
```
For example, if you want the compression is based on accuracy with threshold 0.7, run
```
python train_compressed.py -c accuracy -A 0.7
```
if you want the compression is based on compression ratio with threshold 0.3, run
```
python train_compressed.py -c compression -C 0.3
```
Then replace the model in test.py with the reference model gained by train_original.py and compressed model gained by train_commpressed.py, and run
```
python test.py
```
to see the compressed model's architecture, the compressed ratio and corresponding accuracy.

## Results
![CurrentResult](https://github.com/Beryex/UIUC-ECE397/blob/main/Figures%20for%20Visualization/Current%20Result.png)
