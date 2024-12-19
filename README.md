[![Python Install/Lint with Github Actions](https://github.com/sktan888/Planar_data_classification_with_one_hidden_layer/actions/workflows/main.yml/badge.svg)](https://github.com/sktan888/Planar_data_classification_with_one_hidden_layer/actions/workflows/main.yml)

# Planar_data_classification_with_one_hidden_layer
Binary classification with one hidden layer Neural Network and compare performance with the traditional logistic regression.

## Set up working environment
* create virtual environment: ```virtualenv ENV```
    - remove directory: ``` rm -r hENV```
* add ```source ENV/bin/activate``` in .bashrc file
    edit bashrc file ```vim ~/.bashrc``` and goes to the last line: ```shift + g``` 
* create Makefile for make utility : ``` touch Makefile ```
    - rename a file: ```mv oldfilename newfilename```
* create Requirements.txt for package install : ``` touch Requirement.txt ```
    - find out package version: ```pip freeze | less```
    - install packages: ``` make install ```
        - [numpy](https://www.numpy.org/)
        - [scikit-learn](http://scikit-learn.org/stable/)    
        - [matplotlib](http://matplotlib.org) 
* create library folder: ``` mkdir myLib ```
    - testCases provides some test examples to assess the correctness of your functions
    - planar_utils provide various useful functions used in this assignment

## Steps
    - Implement a 2-class classification neural network with a single hidden layer
    - Use the non-linear tanh activation function for hidden layer
    - Compute the cross entropy loss
    - Implement forward and backward propagation

