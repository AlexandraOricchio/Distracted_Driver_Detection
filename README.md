# Distracted Driver Detection
### _Our Team: Youssef Mroue, David Nassif, Alexandra Oricchio, Vaishall Pradeepkumar_

## Repository Structure:
- **Data Folder**
  - **cleanData Folder:** Cleaned data split into training and testing sets
  - **rawData Folder:** Raw data from [StateFarm](https://www.kaggle.com/c/state-farm-distracted-driver-detection/data)
  - **_model_preprocessing.ipynb:_** Jupyter notebook with model preprocessing 
  - **_model_preprocessing.py:_** Python script for model preprocessing 
  - **_evaluating.py:_** Python script evaluating _48_model's_ generators
  - **_test.py:_** Python script outputing whether model is using CPU or GPU
  - **_48_model.h5:_** Model version 1
  - **_DDD_model.h5:_** Model version 2
  - **_final_model.h5_** Final Model
  
- **NHTSA Crash Data Folder:** 
  - Raw data from the [National Highway Traffic Saftey Administration](https://www.nhtsa.gov/node/97996/221) for 2016 - 2018
  - **_analysis.ipynb:_** Cleaning, transforming, and analyzing crash data
  
- **Static Folder:** CSS styling, HTML images 

- **Templates Folder:** HTML pages

- **Uploads Folder:** Storage folder for images submited by users

- **_app.py file:_** Flask application
