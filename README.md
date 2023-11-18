# FYS-STK4155: Project 2:


## Project overview:
This is the code for reproducing our results in _Project 2_ of **FYS-STK4155** for the Autumn 2023 semester at UiO. The graphs from the plotting are stored in the _plots_ folder. 

## Installation instructions:
To install all the necessary packages, run this code:

```Python
pip install -r requirements.txt
```

where **requirements.txt** contains all the required packages to run the code for this repository.


## Datasets:
There are two dataset used, one for the regression problem and one for the classification problem.

### Regression problem:
We use a syntizhed dataset that is made using the _Franke function_. The function `def FrankeFunction` is defined in **main.py**.

### Classification problem:
The dataset is UCI ML Breast Cancer Wisconsin (Diagnostic). It can be loaded in by using the _Python_ module _Scikit-learn_ `sklearn.datasets.load_breast_cancer()`.
More information on the dataset can be found on:

[https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)



## Usage guide:
The file **gradient_descent.py** contains the code for our analysis of the gradient descents method. The methods for the gradient descent starts at `line 547`.
DISCLAIMER: This code was also used for developing and testing the methods so if you want to produce certain results you may need to comment out certain parts of the code. 
There are comments showing how to do this.

```Python
python3 gradient_descent.py 
```

the plots that this script generate are stored at the base directory and not in the _plots_ folder.

To generate the result for both the regression and classification problems that our neural network produces, run the script **main.py**:

```Python
python3 main.py
```

this will then generate two folders in the _plots_ directory. Either _regression_ and _classification_ or _regression_skl_ and _classification_skl_. Where _regression_ and 
_classification_ is the full grid search of parameters, and the _regression_skl_ and _classification_skl_ is the narrow parameter search with _Scikit-learn's_ 
methods.

If you want to run the full grid search set both `run_scikit_learn_regression` (`line 67`) and `run_scikit_learn_classification` (`line 148`) to `True`. By defualt they are set to `False`.


To run the logistic regression run the script **logistic_regression.py** 

```Python
python3 logistic_regression.py 
```

this will generate the folder _logistic_regression_ in _plots_. Where the _Scikit-learn_ generate results is at the top of the folder.
