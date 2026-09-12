# A_M_project

## 🎬 Movie Recommender Project with Mark and Anita!!

### Authors: Mark Sammartino & Anita Ganesan
### Project: Movie Recommender System

## Overview

This project develops a movie recommendation system using the MovieLens dataset. The system analyzes movie ratings and metadata to identify movies that are likely to appeal to a given user.

The project includes both a movie recommendation system and a predictive model that estimates how a user might rate a movie based on their characteristics and movie preferences.

The recommender was deployed in a user-friendly Streamlit app.

## Dataset

MovieLens dataset from GroupLens: https://grouplens.org/datasets/movielens/

## Movie Recommender

The current movie recommender uses movie ratings and movie preferences to generate recommendations.

### Model uses

- Age
- Gender
- Occupation
- Movie preferences for 10 highly rated seed movies

## "Will I Like It?" Predictor

The predictive model answers:

> Given what I tell you about myself and the movies I like, what rating would I give this movie?

The model uses user information and movie preferences to predict the rating a user might give to a movie.

## Project Files

If you're working on the current movie recommender → `app.py`.

If you're experimenting with recommendation algorithms → `Notebooks/movie_recommender.ipynb`.

If you're working on the "Will I Like It?" predictor → `Notebooks/ratings_decision_tree.ipynb`.

If the notebook's model is ready to become reusable application code → `ratings_decision_tree.py`.

If you're specifically working on deployment/cloud → `Archive/app_cloud.py`.

If you're looking at old code → `Archive/`.