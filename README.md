# Distracted Driver Detection
### _Our Team: Youssef Mroue, David Nassif, Alexandra Oricchio, Vaishall Pradeepkumar_

## Repository Structure:
- **Data Folder**
  - **cleanData Folder:** Cleaned data split into training and testing sets
  - **rawData Folder:** Raw data from [StateFarm](https://www.kaggle.com/c/state-farm-distracted-driver-detection/data)
  - **_model_preprocessing.ipynb:_** Jupyter notebook with model preprocessing 
  - **_model_test.ipynb_** Jupyter notebook testing final model
  - **_model_preprocessing.py:_** Python script for model preprocessing 
  - **_evaluating.py:_** Python script evaluating _48_model's_ generators
  - **_test.py:_** Python script outputing whether model is using CPU or GPU
  - **_48_model.h5:_** Model version 1
  - **_DDD_model.h5:_** Model version 2
  - **_final_model.h5:_** Final Model
  
- **NHTSA Crash Data Folder:** 
  - Raw data from the [National Highway Traffic Saftey Administration](https://www.nhtsa.gov/node/97996/221) for 2016 - 2018
  - **_Results folder:_** Resulting cleaned and transformed data in CSV files 
  - **_data-cleaner.ipynb:_** Cleaning, transforming, and analyzing crash data
  - **_data_transformation.ipynb:_** Additional cleaning and transforming of data
  
- **Static Folder:** 
  - CSS styling
  - HTML images
  - JavaScript files
  - **Uploads Folder:** Storage folder for images submited by users

- **Templates Folder:** HTML pages

- **tensorkeras Folder:** Conda environment for Heroku

- **_app.py file:_** Flask application

- **_DDD_presentation.pptx:_** PowerPoint Presentation Slides
