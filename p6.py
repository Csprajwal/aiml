import numpy as np
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
# Load the dataset
heartDisease = pd.read_csv(r"C:\Users\Lenovo\OneDrive\Desktop\AIML datasets\p6 csv.csv")
heartDisease = heartDisease.replace('?', np.nan)
print('Few examples from the dataset are given below')
print(heartDisease.head())
# Define the Bayesian Model
model = DiscreteBayesianNetwork([
    ('age', 'Heartdisease'),
    ('gender', 'Heartdisease'),
    ('exang', 'Heartdisease'),
    ('cp', 'Heartdisease'),
    ('Heartdisease', 'restecg'),
    ('Heartdisease', 'chol')
])
print('\nLearning CPD using Maximum likelihood estimators')
model.fit(heartDisease, estimator=MaximumLikelihoodEstimator)
print('\nInferencing with Bayesian Network:')
HeartDiseasetest_infer = VariableElimination(model)
# Query 1
print('\n1. Probability of HeartDisease given evidence: age = 35')
q1 = HeartDiseasetest_infer.query(variables=['Heartdisease'], evidence={'age': 35})
print(q1)
# Query 2
print('\n2. Probability of HeartDisease given evidence: chol = 250')
q2 = HeartDiseasetest_infer.query(variables=['Heartdisease'], evidence={'chol': 250})
print(q2)
