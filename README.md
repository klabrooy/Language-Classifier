# Language-Classifier
Classifies languages in a Twitter dataset using Naïve Bayes’ algorithm

## Introduction
*__NOTE:__ train.json not added to repo due to large size*

A Python program that determines the natural language of instances of a Twitter post dataset using an optimised variation of Naïve Bayes’ algorithm. During the training phase (on train.json), a language library is created for each of the 20 predefined languages (set in the task). The library is a dictionary of n-grams, to be scanned and a similarity to the text to be measured. During the testing phase (on test.json), each text instance is compared against these libraries to determine the most probable language.

## Usage
Run the following on the command line:

      
      ./python naive_bayes.py
      
      
*__NOTE:__ there is no train.json in this repo therefore the program cannot execute*      
      


## Task
COMP30027: Machine Learning: Project 2
