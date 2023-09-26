1. csci544_hw4.py will generate the results on the trained model.  (Note: perl command for evaluation script is present in this file already for prediction files)
2. Run csci544_hw4_train.ipynb to train the model (approx time required to execute: 1.5 hrs for task1 and 1 hr for task2)

The 2 trained models blstm1.pt and blstm2.pt should be in the same folder as CSCI544_HW4.py file.
Also, make sure label.json file is in the same folder as csci544_hw4.py file.

Command to run the python file : python csci544_hw4.py

python notebook to train the models: csci544_hw4_train.ipynb
Python file to generate the required results and files: csci544_hw4.py
Task 1 model: blstm1.pt
Task 2 model: blstm2.pt
Label File: label.json

Dataset: train, dev and test files should be present in the data folder

Files Generated:

dev1.out, dev2.out => (index, word, prediction) on dev data
test1.out, test2.out => (index, word, prediction) on test data
pred1.out, pred2.out => (index, word, gold tag, prediction) in the format as specified to run the perl evaluation script


Name: Rutuja Bhandigani