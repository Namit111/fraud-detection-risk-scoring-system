# Fraud Detection & Risk Scoring System

A machine learning-based fraud detection system that predicts the probability of a transaction being fraudulent and converts the prediction into an interpretable risk score and risk level.

## Project Overview

This project uses a Random Forest classifier to identify potentially fraudulent credit card transactions.

The trained machine learning model is exposed through a FastAPI REST API and packaged for containerized deployment using Docker.

### Key Features

- Fraud probability prediction
- Risk score from 0–100
- Risk classification: LOW, MEDIUM, HIGH
- Custom fraud decision threshold
- FastAPI REST API
- Automated API tests using Pytest
- Dockerized application
- Trained Random Forest model

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- FastAPI
- Pydantic
- Joblib
- Pytest
- Docker

## Dataset

The project uses the Credit Card Fraud Detection dataset from Kaggle.

The original dataset contains anonymized transaction features including:

- Time
- V1–V28
- Amount
- Class

`Class = 1` represents fraud and `Class = 0` represents a legitimate transaction.

Due to the highly imbalanced nature of fraud detection datasets, stratified splitting and class-weighted Random Forest training were used.

The dataset itself is not included in this repository.

## Machine Learning Approach

### Model

Random Forest Classifier

### Training Strategy

- Stratified train/test splitting
- Class-weighted Random Forest
- Validation-based threshold tuning
- Final evaluation on an untouched test set

### Decision Threshold

The default probability threshold was tuned to:

```text
0.395