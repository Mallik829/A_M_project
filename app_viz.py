import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


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
    usecols=[0,1,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],
    names=[
        "movie_id",
        "title",
        "unknown",
        "Action",
        "Adventure",
        "Animation",
        "Childrens",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "FilmNoir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "SciFi",
        "Thriller",
        "War",
        "Western"
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

    st.subheader("Rating Distribution by Movie")
    
    # Create columns for side-by-side layout
    cols = st.columns(2)
    col_idx = 0
    
    for title in results.index:
        col = cols[col_idx % 2]
        
        with col:
            st.write("🍿", title)
            
            # Get ratings for this specific movie
            movie_ratings = df[df["title"] == title]["rating"]
            
            # Create small histogram
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.hist(movie_ratings, bins=5, color="red", edgecolor="black", linewidth=2)
            ax.set_xlabel("Rating")
            ax.set_ylabel("Frequency")
            ax.set_title(f"Ratings ({len(movie_ratings)} votes)")
            ax.set_xticks([1, 2, 3, 4, 5])
            ax.set_xlim(0.5, 5.5)
            ax.set_facecolor('grey')
            
            st.pyplot(fig)
        
        col_idx += 1
    
    # Genre distribution chart
    st.subheader("Genre Distribution of Recommended Movies")
    
    # Get genres for recommended movies
    recommended_movie_ids = []
    for title in results.index:
        movie_id = df[df["title"] == title]["movie_id"].iloc[0]
        recommended_movie_ids.append(movie_id)
    
    genre_cols = [col for col in movies.columns if col not in ["movie_id", "title"]]
    recommended_movies_genres = movies[movies["movie_id"].isin(recommended_movie_ids)][genre_cols]
    
    # Sum genres across recommended movies
    genre_counts = recommended_movies_genres.sum().sort_values(ascending=True)
    genre_counts = genre_counts[genre_counts > 0]
    
    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(genre_counts.index, genre_counts.values, color="red", edgecolor="black")
    ax.set_facecolor('grey')
    ax.set_xlabel("Count")
    ax.set_title("Genres in Recommended Movies")
    ax.grid(axis="x", alpha=0.3)
    
    st.pyplot(fig)