# bdf-Forecasting-Public-Anxiety-from-Twitter-Data

This project focuses on analyzing and forecasting public anxiety levels derived from Twitter posts. Using LIWC-based linguistic features and engagement metadata, we construct a composite Anxiety Index and apply statistical, machine learning, and deep learning models to predict its future trends.

Dataset Overview

We used a curated dataset of ~23,000 tweets containing:

Psychological features (LIWC): Affect, emo_pos, emo_neg, Positive, Negative, Total_Sentiment, Cognition, Clout, etc.

Engagement metrics: retweet_count, reply_count, like_count, quote_count, Followers, Buzz.

Metadata: State, Day (weekday/weekend), OpnHours (working/non-working), ContentType, Date.

The dataset was aggregated into daily time-series (2019–2022) with a derived Anxiety Index:

Anxiety Index = Affect + emo_neg + Negative – emo_pos

Data Processing & Feature Engineering

The notebook implements:

Cleaning & validation of raw tweet-level features

Aggregation into daily Anxiety Index with post counts

Feature engineering: temporal lags (1,7,30), rolling stats (mean, std), weekday/weekend, month indicators

Normalization & bias-aware preprocessing (residualization option)

Key Tools & Libraries

Python: Pandas, NumPy, Scikit-learn, Statsmodels

Visualization: Matplotlib, Seaborn

Forecasting libraries: XGBoost, LightGBM, CatBoost, Prophet, ARIMA/SARIMAX

Deep Learning: TensorFlow/Keras (BiLSTM)

Team Collaboration

This project was developed collaboratively as part of an academic initiative on forecasting psychological trends using social media data. While the final notebook is consolidated here, multiple steps (EDA, modeling, testing) were conducted in a shared environment.

Collaborators

@amarrr33

@krishnakoushik2792005

@Guru1509

Results

Best single models:

Ridge: RMSE ≈ 0.051, R² ≈ 0.998

SARIMAX: RMSE ≈ 0.052, R² ≈ 0.998

ElasticNet: RMSE ≈ 0.060, R² ≈ 0.997

Ensembles:

Simple (Ridge+SARIMAX+CatBoost): RMSE ≈ 0.095, R² ≈ 0.992

Weighted (val-opt, Ridge+SARIMAX+ElasticNet+XGB+LightGBM): RMSE ≈ 0.052, R² ≈ 0.998

Stacking (ElasticNet L2): RMSE ≈ 0.079, R² ≈ 0.995

Other models:

CatBoost / XGBoost / LightGBM: R² ≈ 0.91–0.96 (strong performers)

ARIMA / Prophet / BiLSTM: Weak, negative R², not recommended for final use

Residuals: White-noise-like, confirming model adequacy.

Purpose

The aim of this project is to:

Explore psychological forecasting using social media data

Evaluate how traditional statistical models compare with ML and DL approaches

Provide a reproducible framework for time-series forecasting of mental health indicators
