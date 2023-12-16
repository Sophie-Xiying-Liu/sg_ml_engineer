# Thoughts of ML Engineering

This document is intended to record my thoughts learned from practical work on use cases and from reading on ML engineering (reference). These thoughts are organized along different ML project phases, some overlapping points might be inevitable, as these phases influence and interact with each other.

## Overview


## ML Phase 1: Decide and design

### #1 With or without ML?

It's always valuable to check other possibilities, i.e. statistical models for time series problems tend to have a good answer with less efforts, and it requires less data comparing to ML models. As said in Google's Rules of ML documents, "don't be afraid to launch a product without machine learning".

However, if there is sufficient data for model training, and the system is way too complex to be analyzed and predicted by human decision or the existing method cannot handle it, ML could be the next to try.

For the day-ahead active losses forecasting use case, ML is a good fit, as there are relatively large dataset to train the model, and because the losses can be affected by numerous power system and market factors, that neither human or conventional time series models can predict well. It's clear that ML would be the more cost-efficient and more accurate and reliable (if built up on the reliable MLOps) solution.

### #2 Object and metrics

Once decide to go with ML, an object needs to be defined to train models and to select the model to deploy. 

In the case of active losses forecast, **RMSE** is chosen because we do not only want to minimize the forecasting errors, but also to avoid *large* errors. Because the forecasting error will become the unbalanced position, and each unit of unbalanced position (unit: MWh/h) can be balanced in the wholesale electricity market at different prices, the larger these units accumulates, the higher each marginal unit needs to pay. 

In addition to object, there are also metrics which also need to be defined, calculated, tracked, recorded, and monitored. They are important for monitoring the models, and for continuous improvement. 

Back to the active losses forecast case, the metrics used include MAE (should also add MAPE), and the cost-impacts translated from MAE.

### #3 First version data pipeline chart

To the best of current knowledge, based on experts' opiniion, existing models, basic statistics analysis, etc., form a list of data needed to build up this ML project.

Then, write down:
- where do these data is stored (internally or externally)
- who is the data owner/manager (so that you can request for data access and clarify data-quality related details)
- what the data governance rules specifically apply to these data
- how is the data generated and how often it updates
- what the original data look like in the original database
- how to extract the data (internally or externally, via which interface, which stakeholders might be involved in the data pipeline from the source to your destination)

Keep this chart well and share it with the relevant stakeholders. Continuously work on it to keep it as much up-to-date as possible.

### #4 Serving infrastructure

Below are the points to be decided:

- Deployment environment: cloud or on-premises (generally speaking cloud, some cases on-premises for security requirements)

- Scalability considerations: how the system will handle increased load and demand? (cloud infrastructure side is not clear to me) 

    From in the operation side, that the duration of the pipeline is too long to meet the inference requirement, i.e. day-ahead losses forecast must complete before 10:40. 

- Inference pipeline: from input data pipeline to processing pipeline, to model and generate prediction, then serving the results to the destination. It also helps to draw a chart to show these upstream-downstream flows.

- Monitoring and logging: 
    - Log prediction requests -> alarm when important tasks fail + BCM
    - Track performance -> alarm when performance drops below the (pre-defined) criterium
    - Track health of the serving infrastructure
        (response time, resource utilisation, availability & uptime, errors, error rates, success rates of requests, overall availability of interfaces)

- Versioning: for model updates and rollbacks 

- Integration with other systems: especially to the operational systems where the results of the ML models will be used to operate or make decisions. The compability should be considered when the project starts. 

## ML Phase 2: First Pipeline

