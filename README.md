# music_genre_classifier

This repository contains a Machine Learning project that analyzes audio files to classify them into different musical genres (like Rock, Jazz, Pop, etc.). It explores Digital Signal Processing (DSP) techniques to extract features from sound waves and compares various classification models.

## To run this project, clone the repository and ensure the kagglehub library is installed:
`pip install kagglehub`

The project uses the famous GTZAN Dataset. It consists of:
10 Genres (Blues, Classical, Country, Disco, Hiphop, Jazz, Metal, Pop, Reggae, Rock)
100 tracks per genre (30 seconds each)

Project Overview & Features
This notebook guides through the entire Data Science pipeline:

## 1. Audio Processing & Visualization (EDA)
Using Librosa, I visualized the audio data to understand what distinguishes one genre from another visually.

* Waveforms: Time-domain representation.
* Spectrograms (STFT & Mel): Visualizing frequency intensity over time.
* Chromagrams: Mapping audio to the 12 different pitch classes.

## 2. Feature Extraction
I extracted mathematical features to feed into the machine learning models, including:

* MFCCs (Mel-Frequency Cepstral Coefficients): Crucial for timber/texture analysis.
* Spectral Centroid: The "center of mass" of the spectrum (brightness of sound).
* Zero Crossing Rate: Rate at which the signal changes sign.
* BPM (Tempo): Beats per minute.

## 3. Model Comparison

I trained and evaluated 10 different classifiers to find the best performance:
* Naive Bayes
* Stochastic Gradient Descent
* K-Nearest Neighbors (KNN)
* Decission Trees
* Random Forest
* Logistic Regression
* Support Vector Machine (SVM)
* Neural Networks (MLP)
* XGBoost (Selected as the final model due to superior performance)

## 4. Recommender System (Bonus)
As an experiment, I implemented a simple content-based recommender system using Cosine Similarity. It suggests songs that are mathematically similar to a given track based on their audio features.

The XGBoost Classifier achieved an accuracy of approximately 90% on the test set. The Confusion Matrix in the notebook shows exactly which genres are easily confused (e.g., Rock vs. Country) versus those that are very distinct (e.g., Classical).

## Libraries Used
Python 3.x

Librosa (Audio Analysis)

Pandas & NumPy (Data Manipulation)

Matplotlib & Seaborn (Visualization)

Scikit-Learn (Modeling)

XGBoost (Gradient Boosting)



