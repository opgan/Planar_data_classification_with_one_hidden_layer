[![Python Install/Lint with Github Actions](https://github.com/sktan888/Planar_data_classification_with_one_hidden_layer/actions/workflows/main.yml/badge.svg)](https://github.com/sktan888/Planar_data_classification_with_one_hidden_layer/actions/workflows/main.yml)

# Planar_data_classification_with_one_hidden_layer
Binary classification with one hidden layer Neural Network and compare performance with the traditional logistic regression.

## Set up working environment
* virtual environment: ```virtualenv ENV```
    - remove directory: ``` rm -r hENV```
    - add ```source ENV/bin/activate``` in .bashrc file
    - edit bashrc file ```vim ~/.bashrc``` and goes to the last line: ```shift + g``` 
* Makefile for make utility : ``` touch Makefile ```
    - format codes ``` make format ```
    - run lint ``` make lint ```
    - run main.py ``` make run```
* Requirements.txt for package installation : ``` touch Requirement.txt ```
    - find out package version: ```pip freeze | less```
    - install packages: ``` make install ```
    - ``` pip install <name> ```
* Project folders:
   - create directory ``` mkdir myLib ```
   - rename a file: ```mv oldfilename newfilename```
   - myLib folder contains images of plots
   - tests folder contains test scripts to assess the correctness of some functions
   - plots folder planar_utils provide various useful functions used in this assignment
* Logging
    - info.log file contains information of prediction results
* Running Main.py
  - inside ipython ```run main.py``` or fr command line ```python main.py```

## Steps
    - Injest the dataset
    - Implement a 2-class classification neural network with a single hidden layer
    - Use the non-linear tanh activation function for hidden layer
    - Compute the cross entropy loss
    - Implement forward and backward propagation

