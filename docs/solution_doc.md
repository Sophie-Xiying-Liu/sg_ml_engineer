# Solution Documentation

## Project setup

Given the requirement, this project is set up as the following steps shows, with the structure:

1. alf: the main project folder

- dev: development environment

    The major pipelines and files are:

    - data_pipeline: imports and cleans the data, and check data quality

        Goal is to create curated data based on the raw data. 

    - fe_pipeline: feature engineering pipeline which creates feature tables from the curated data.

    - model_training: taking the data from fe_pipeline and preparing the training datasets to feed into the model, model training with cross-validation and hyperparameter training, saving the results. 

        Goal is to build up a model training pipeline for continuous improvement and testing. 

    - params: a yaml file which provides the parameters for the scripts.

    - log_file: records the relevant information or errors collected in the process. 

    Some other folders are:
    - artifacts: saving files such as sklearn pipeline which is used for preparing the data

    - data: raw and curated data and feature tables
    
        Most likely these will be saved in cloud database in practice. 

    - models: save the models trained

        In practice this is done with MLFlow to track and register models.
    

- prod: production environment where model is served and used in operation

    - model_champion: the champion version of the model is moved from dev to prod after the evaluation in dev. (No result is saved here due to the long training time)

    - inference_service: 

        Goal is to pack the pipelines and the champion version of the model for inference service. But not able to build up due to the limited time and experiences. The first try was to use Fast API due to its ease of use and it's fast. 

        The requirement asked for wrapping the solution into an **asynchronous API framework**. However, after some research and based on this project setting, the **asynchronous** turned out to be less relevant. Because in this project, the steps run in a sequential order, i.e. from data_pipeline to fe_pipeline, and then to model (either training or serving). There might not be much value added to use the asynchronous framework.

        To serve the API, this folder also has the Prod version of data_pipeline and fe_pipeline for data processing before feeding into the model for inference.


2. docs: documentation folder

- solution_doc: explanation of the project structure and the steps to get close to the answers of the questions. 

- thoughts_ml_engineering: 

    The initial goal is to form a checklist to decide how and when to put a model in production. But as the research continues, this developed somehow to a study notes based on both documents and my experience which tries to summarize some best practices for ML from my point of view. Still, I think it could be used somehow as a checklist to work on ML projects, from designing and developing (more content) to deployment (less content due to less experience, but can be extended). 

3. venv and requirements.txt

Next, I will try to answer the three main questions raised in the challenge.

### 1. Checklist to bring the model into production

This is detailed in the other doc on thoughts about ML engineering.

Generally speaking it follows the steps:

-  Starting from a multi-step time series forecasting problem, confirmed a ML model can be a suitable solution, in comparison to statistical models.

- Review the available data, meanwhile dropped the temperature data due to the large missing data.

- Based the solution from the Hackathon Group 3, built up the pipelines for data, feature engineering and model training. 

### 2. API

- Decided to use Fast API without specifying the asynchronized framework (explained above).

- But didn't set up due to limited time and experience.

### 3. Containerize the components

- Same as above, didn't set up due to limited time and experience.

Data was split into train and test (the last year) sets for training and testing.

## Feedback

From my point of view, this challenge consists of two components: 

- Coding part which requires functioning pipelines for processing data, training model, and for serving the containerized ML model.

- Thinking part which is about ML engineering best practice with a focus on bringing the model into production. 

For me, addressing both parts within the required time is challenging. 

If the coding part can run on cloud with the available up-to-date tools, it might be more efficient and closer to the daily work. 
