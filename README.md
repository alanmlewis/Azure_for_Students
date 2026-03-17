# Overview

This repository contains the scripts used to create and access an [Azure endpoint](https://deployment-rcivj.spaincentral.inference.ml.azure.com/score). They are intended to serve as an example of the scripts needed for a simple deployment - [detailed instructions](https://docs.google.com/document/d/1AUWdgBZjRzARRuDprdD7jJUVr4SSwVwnAmoCHemw2pQ/edit?usp=sharing) are accessible to anyone at the University of York.

# Model Description

`model.joblib` is a simple Logistic Regression model trained on the [OpenML Mushroom dataset](https://www.openml.org/search?type=data&sort=version&status=any&order=asc&exact_name=mushroom&id=24), which uses only the categorical variables from that dataset OneHot encoded. `train_model.py` is the script which trains the model, and a scoring script and a request script for the Azure endpoint are also provided, lightly adapted from the Azure tutorials.

# Access the endpoint

The request script is missing an API key; UoY students can find the key [here](https://docs.google.com/document/d/1AUWdgBZjRzARRuDprdD7jJUVr4SSwVwnAmoCHemw2pQ/edit?tab=t.0#heading=h.pe57llik4v0d).
