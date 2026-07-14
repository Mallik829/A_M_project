import streamlit as st
import pandas as pd
import gdown
import pathlib

# Google drive file ID: 1LrJnjiM2Ifnztutkq0xAbQk584nz9A1c
# Load data


ratings = pd.read_csv(
    "ml-100k/u.data",
    sep="\t",
    names=[
        "user_id",
        "movie_id",
        "rating",
        "timestamp"
    ]
)

movies = pd.read_csv(
    "ml-100k/u.item",
    sep="|",
    encoding="latin-1",
    header=None,
    usecols=[0,1],
    names=[
        "movie_id",
        "title"
    ]
)

df = ratings.merge(
    movies,
    on="movie_id"
)


# Create matrix
user_movie_matrix = df.pivot_table(
    index="user_id",
    columns="title",
    values="rating"
)


def recommend_movies(movie_title, top_n=10):

    if movie_title not in user_movie_matrix.columns:
        return None

    movie_ratings = user_movie_matrix[movie_title]

    similar = user_movie_matrix.corrwith(
        movie_ratings
    )

    corr_df = pd.DataFrame(
        similar,
        columns=["correlation"]
    )

    corr_df.dropna(inplace=True)

    corr_df["num_ratings"] = (
        df.groupby("title")["rating"]
        .count()
    )

    corr_df = corr_df[
        corr_df["num_ratings"] >= 30
    ]

    corr_df = corr_df.drop(
        movie_title,
        errors="ignore"
    )

    return corr_df.sort_values(
        "correlation",
        ascending=False
    ).head(top_n)



# ---- APP ----

st.title("🎬 Movie Recommendation System")

movie = st.selectbox(
    "Choose a movie:",
    user_movie_matrix.columns
)


if st.button("Recommend"):

    results = recommend_movies(movie)

    st.write(
        "Movies similar to:",
        movie
    )

    for title in results.index:
        st.write("🍿", title)