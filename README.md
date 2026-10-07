# 🌦️ End-to-End Weather Temperature Forecasting Pipeline

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/)
[![Database](https://img.shields.io/badge/Database-SQLite3-lightgrey.svg)](https://www.sqlite.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end machine learning time-series regression pipeline designed and implemented as part of a technical interview assessment. The system forecasts hourly ambient temperatures using chronological sensor records, rich domain-specific feature engineering, an ensemble tree regressor, and automated database persistence.

---

## 📑 Table of Contents
- [Project Overview](#-project-overview)
- [Repository Structure](#-repository-structure)
- [Dataset Overview](#-dataset-overview)
- [End-to-End Pipeline Breakdown](#-end-to-end-pipeline-breakdown)
  - [1. Data Cleaning & Sanitization](#1-data-cleaning--sanitization)
  - [2. Feature Engineering & Leakage Prevention](#2-feature-engineering--leakage-prevention)
  - [3. Chronological Train-Test Split](#3-chronological-train-test-split)
  - [4. Model Training & Evaluation](#4-model-training--evaluation)
  - [5. SQLite Database Integration](#5-sqlite-database-integration)
- [Setup & Installation](#-setup--installation)
- [Usage](#-usage)
- [SQL Schema & Querying](#-sql-schema--querying)
- [License](#-license)

---

## 📌 Project Overview

Predicting local atmospheric temperature requires modeling non-linear meteorological dynamics, cyclical diurnal/seasonal trends, and short-to-medium-term thermal inertia. 

This project was built to meet rigorous production-oriented interview criteria:
* Constructing an end-to-end pipeline from raw CSV ingestion to relational database logging.
* Implementing cyclical trigonometry and lagged variables without forward-looking data leakage.
* Achieving benchmark forecasting accuracy ($R^2$, $\text{MAE}$, $\text{RMSE}$) evaluated under an 80/20 chronological split.
* Querying forecast logs and performance metadata via SQL.

---

## 📂 Repository Structure

```text
.
├── weatherHistory.csv          # Source hourly weather dataset
├── transvision.py              # Main execution pipeline script
├── weather_forecast.db         # Auto-generated SQLite database (tables: forecasts, metrics)
├── temperature_forecast.csv    # Exported predictions and residuals
├── README.md                   # Complete documentation
└── LICENSE                     # Project open-source license
```

# 📊 Dataset Overview
  1.Source File: weatherHistory.csv
  2.Target Feature: Temperature (C)
  3.Raw Predictors: Formatted Date, Summary, Precip Type, Apparent Temperature (C), Humidity, Wind Speed (km/h), Wind Bearing (degrees), Visibility (km), Loud Cover, Pressure (millibars).

# 🔄 End-to-End Pipeline Breakdown
    1.𝐃𝐚𝐭𝐚 𝐂𝐥𝐞𝐚𝐧𝐢𝐧𝐠 & 𝐒𝐚𝐧𝐢𝐭𝐢𝐳𝐚𝐭𝐢𝐨𝐧 :  Identifies and drops repeated observations using .drop_duplicates() to avoid biased frequency counts.  
    2. 𝐓𝐞𝐦𝐩𝐨𝐫𝐚𝐥 𝐒𝐨𝐫𝐭𝐢𝐧𝐠: Parses Formatted Date to standardized UTC timestamps and sorts the dataset in ascending chronological order before index resetting. 
    3. 𝐂𝐚𝐭𝐞𝐠𝐨𝐫𝐢𝐜𝐚𝐥 𝐈𝐦𝐩𝐮𝐭𝐚𝐭𝐢𝐨𝐧: Replaces missing values in Precip Type with "Unknown".   
    4. 𝐍𝐮𝐦𝐞𝐫𝐢𝐜 𝐈𝐦𝐩𝐮𝐭𝐚𝐭𝐢𝐨𝐧: Imputes missing values across numerical atmospheric indicators (Humidity, Wind Speed, Wind Bearing, Visibility, Pressure) using median imputation.   
    5. 𝐓𝐚𝐫𝐠𝐞𝐭 𝐈𝐧𝐭𝐞𝐠𝐫𝐢𝐭𝐲: Drops any row containing missing target values (Temperature (C)) prior to feature extraction.   

## 2. Feature Engineering & Leakage Prevention 
  A total of 30+ engineered and direct features are constructed across multiple domains:   Temporal 
        Extraction:Extracts discrete units: Hour (0--23), Day, Month (1--12), Year, DayOfWeek (0--6), and DayOfYear.
  **Cyclical Trigonometric Transformations**:
              • Hour_Sin = sin(2 * π * Hour / 24)
              • Hour_Cos = cos(2 * π * Hour / 24)
              • Month_Sin = sin(2 * π * Month / 12)
              • Month_Cos = cos(2 * π * Month / 12)
              • DayOfWeek_Sin = sin(2 * π * DayOfWeek / 7)
              • DayOfWeek_Cos = cos(2 * π * DayOfWeek / 7)


## 3. Chronological Train-Test Split 
    Because weather is a continuous time-series process, random K-Fold shuffling causes data leakage from future observations into past training sets. The pipeline applies a strict forward temporal cut:
        Train Split: First 80% of chronological observations.   
        Test Split: Subsequent 20% out-of-time observations.

## 4. Model Training & EvaluationModel: 
        Ensemble RandomForestRegressor configured with:  
                n_estimators=500   
                max_features=0.8   
                min_samples_split=2   
                min_samples_leaf=1   
                n_jobs=-1 (multiprocessing enabled)   
                random_state=42 
## Metrics Calculated:  
    R^2 Score: Proportion of variance explained.   
    MAE: Mean Absolute Error in C. 
    RMSE: Root Mean Squared Error in C.   
    Validation Gate: Automates checking whether model predictive power meets the >= 98%  threshold target specified in interview parameters. 

## 5. SQLite Database Integration 
    Post-inference, model outputs are written to disk using SQLite via sqlite3 and pandas.DataFrame.to_sql:   
          𝐓𝐚𝐛𝐥𝐞 𝐭𝐞𝐦𝐩𝐞𝐫𝐚𝐭𝐮𝐫𝐞_𝐟𝐨𝐫𝐞𝐜𝐚𝐬𝐭𝐬: Stores timestamps, actual values, predicted values, pointwise error, and run-level metrics.   
          𝐓𝐚𝐛𝐥𝐞 𝐦𝐨𝐝𝐞𝐥_𝐦𝐞𝐭𝐫𝐢𝐜𝐬: Stores overall regression performance metrics ($R^2$, MAE, RMSE) for validation audits.  

## 💻 Setup & Installation
    1. PrerequisitesEnsure you have Python 3.8+ installed on your local machine or Google Colab environment.   
    2. Install Dependencies
            pip install pandas numpy scikit-learn matplotlib seaborn sqlalchemy pymysql
    3. Place Data File
            Ensure weatherHistory.csv is placed in the root directory alongside transvision.py

## ⚡ Usage
      Execute the entire data ingestion, feature generation, modeling, and SQL database loading script in one command:
            python transvision.py
      
      Upon execution, the script will:
        1. Clean data, perform exploratory correlation analysis, and engineer features.
        2. Train the Random Forest Regressor and evaluate against the out-of-time test set.
        3. Generate diagnostic visual plots (Actual vs Predicted trajectory, Residual scatter plot, Top 15 Feature Importances). 
        4. Export temperature_forecast.csv and instantiate weather_forecast.db.   
        5. Execute exploratory SQL queries across the created tables and output validation results to terminal.

    
    
## 🗄️ SQL Schema & Querying
The pipeline exposes a clean relational database schema for analytical consumption:

      1. Table: temperature_forecasts
          
                  |      Column Name          | Type   |         Description                         |
                  |        :---               | :---   |              :---                           |
                  | `forecast_datetime`       | TEXT   | Timestamp of test observation               |
                  | `actual_temperature`      | REAL   | Ground-truth recorded temperature (°C)      |
                  | `forecasted_temperature`  | REAL   | Machine-learning predicted temperature (°C) |
                  | `error`                   | REAL   | Residual (actual - predicted)               |
                  | `r2_score`                | REAL   | Aggregate model R² score                    |
                  | `mae`                     | REAL   | Aggregate model Mean Absolute Error         |
                  | `rmse`                    | REAL   | Aggregate model Root Mean Squared Error     |
                  
```
      2. Sample SQL Query
        -- Query the top 10 highest forecasted temperatures
SELECT 
    forecast_datetime,
    actual_temperature,
    forecasted_temperature,
    ROUND(error, 3) AS residual_error
FROM temperature_forecasts
ORDER BY forecasted_temperature DESC
LIMIT 10;
```


## 📄 License

Copyright © 2026 Godavarthi Naga Manoj Balaji. All Rights Reserved.

This project is proprietary and was developed as a company technical interview assignment.

The source code and associated materials are provided for demonstration, evaluation, and portfolio purposes only. No permission is granted to modify, distribute, publish, the project without prior written permission from the author.
