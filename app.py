import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor


# -------------------------------------------------
# Load MovieLens data
# -------------------------------------------------

ratings = pd.read_csv(
    "../ml-100k/u.data",
    sep="\t",
    names=[
        "user_id",
        "movie_id",
        "rating",
        "timestamp"
    ]
)

movies = pd.read_csv(
    "../ml-100k/u.item",
    sep="|",
    encoding="latin-1",
    header=None,
    usecols=[0, 1],
    names=[
        "movie_id",
        "title"
    ]
)

users = pd.read_csv(
    "../ml-100k/u.user",
    sep="|",
    names=[
        "user_id",
        "age",
        "gender",
        "occupation",
        "zip_code"
    ]
)


# -------------------------------------------------
# Combine data
# -------------------------------------------------

df = ratings.merge(
    movies,
    on="movie_id"
)

df = df.merge(
    users,
    on="user_id"
)


# -------------------------------------------------
# Encode demographic information
# -------------------------------------------------

df["gender_encoded"] = df["gender"].map({
    "M": 0,
    "F": 1
})

occupation_codes, occupation_categories = pd.factorize(
    df["occupation"]
)

df["occupation_encoded"] = occupation_codes


# -------------------------------------------------
# Select 10 most-rated movies
# -------------------------------------------------

top_movies = (
    df.groupby(["movie_id", "title"])
      .size()
      .reset_index(name="rating_count")
      .sort_values("rating_count", ascending=False)
)

seed_movies = top_movies["title"].head(10).tolist()

seed_ids = [
    int(
        movies.loc[
            movies["title"] == title,
            "movie_id"
        ].iloc[0]
    )
    for title in seed_movies
]


# -------------------------------------------------
# Create user-rating matrix
# -------------------------------------------------

comparison = (
    df[df["movie_id"].isin(seed_ids)]
    .pivot_table(
        index="user_id",
        columns="movie_id",
        values="rating"
    )
)

comparison = comparison.reindex(
    columns=seed_ids
)


# Fill missing ratings with movie averages

movie_means = comparison.mean()

comparison = comparison.fillna(
    movie_means
)


# -------------------------------------------------
# Add demographic information
# -------------------------------------------------

user_features = (
    df[
        [
            "user_id",
            "age",
            "gender_encoded",
            "occupation_encoded"
        ]
    ]
    .drop_duplicates("user_id")
    .set_index("user_id")
)

comparison = comparison.join(
    user_features
)


# -------------------------------------------------
# Streamlit APP
# -------------------------------------------------

st.title("🎬 Movie Recommendation System")

st.write(
    "Predict how you might rate a movie based on "
    "your movie preferences and demographics."
)


# -------------------------------------------------
# User demographics
# -------------------------------------------------

st.header("👤 About You")

user_age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=25
)

user_gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

gender_encoded = {
    "Male": 0,
    "Female": 1
}[user_gender]


# -------------------------------------------------
# Occupation
# -------------------------------------------------

occupation_display = {
    "writer": "Writer",
    "executive": "Executive",
    "technician": "Technician",
    "educator": "Teacher / Educator",
    "engineer": "Engineer",
    "librarian": "Librarian",
    "programmer": "Programmer",
    "administrator": "Administrator",
    "student": "Student",
    "retired": "Retired",
    "other": "Other",
    "doctor": "Doctor",
    "marketing": "Marketing",
    "artist": "Artist",
    "lawyer": "Lawyer",
    "salesman": "Sales",
    "homemaker": "Homemaker",
    "healthcare": "Healthcare",
    "none": "None",
    "entertainment": "Entertainment",
    "scientist": "Scientist"
}

occupation = st.selectbox(
    "Occupation",
    list(occupation_display.keys()),
    format_func=lambda x: occupation_display[x]
)

occupation_encoded = occupation_categories.tolist().index(
    occupation
)


# -------------------------------------------------
# Rate seed movies
# -------------------------------------------------

st.header("⭐ Rate Some Movies")

st.write(
    "Rate each movie from 1–5. "
    "Use 0 if you haven't seen it."
)

user_ratings = {}

for movie_id, title in zip(seed_ids, seed_movies):

    user_ratings[movie_id] = st.slider(
        title,
        min_value=0,
        max_value=5,
        value=0
    )


# -------------------------------------------------
# Choose target movie
# -------------------------------------------------

st.header("🎯 Choose a Movie to Predict")

target_movie = st.selectbox(
    "Which movie would you like us to predict?",
    movies["title"].tolist()
)


# -------------------------------------------------
# Prediction
# -------------------------------------------------

if st.button("Predict My Rating"):

    target_movie_id = int(
        movies.loc[
            movies["title"] == target_movie,
            "movie_id"
        ].iloc[0]
    )

    target_ratings = df[
        df["movie_id"] == target_movie_id
    ][
        ["user_id", "rating"]
    ].rename(
        columns={
            "rating": "target_rating"
        }
    )

    train_data = comparison.merge(
        target_ratings,
        left_index=True,
        right_on="user_id"
    ).dropna()

    feature_columns = (
        [str(movie_id) for movie_id in seed_ids]
        + [
            "age",
            "gender_encoded",
            "occupation_encoded"
        ]
    )

    X_train = train_data[
        seed_ids
        + [
            "age",
            "gender_encoded",
            "occupation_encoded"
        ]
    ].copy()

    X_train.columns = feature_columns

    y_train = train_data[
        "target_rating"
    ]

    # -------------------------------------------------
    # Train Decision Tree
    # -------------------------------------------------

    tree = DecisionTreeRegressor(
        max_depth=15,
        random_state=42
    )

    if len(X_train) == 0:

        st.error(
            "Not enough training data for this movie."
        )

    else:

        tree.fit(
            X_train,
            y_train
        )

        # -------------------------------------------------
        # Create user's test vector
        # -------------------------------------------------

        user_vector = [
            user_ratings[movie_id]
            for movie_id in seed_ids
        ]

        user_vector.extend([
            user_age,
            gender_encoded,
            occupation_encoded
        ])

        X_test = pd.DataFrame(
            [user_vector],
            columns=feature_columns
        )

        prediction = tree.predict(
            X_test
        )[0]

        prediction = max(
            1,
            min(5, prediction)
        )

        st.success(
            f"Predicted rating for **{target_movie}**: "
            f"⭐ **{prediction:.1f} / 5**"
        )

        st.info(
            f"The model trained on "
            f"{len(X_train)} MovieLens users."
        )


# -------------------------------------------------
# Bulk predictions for comparison
# -------------------------------------------------

st.header("🔢 Bulk Predictions")

if st.button("Predict Top 20 Movies"):

    # choose top movies not in the seed set
    candidate_movies = (
        top_movies[~top_movies["movie_id"].isin(seed_ids)]
        .head(20)
    )

    # prepare user's test vector
    user_vector = [
        user_ratings.get(movie_id, 0)
        for movie_id in seed_ids
    ]

    user_vector.extend([
        user_age,
        gender_encoded,
        occupation_encoded
    ])

    feature_columns = (
        [str(movie_id) for movie_id in seed_ids]
        + [
            "age",
            "gender_encoded",
            "occupation_encoded"
        ]
    )

    X_test = pd.DataFrame(
        [user_vector],
        columns=feature_columns
    )

    results = []

    for _, row in candidate_movies.iterrows():

        mid = int(row["movie_id"])
        title = row["title"]

        target_ratings = df[
            df["movie_id"] == mid
        ][["user_id", "rating"]].rename(
            columns={"rating": "target_rating"}
        )

        train_data = comparison.merge(
            target_ratings,
            left_index=True,
            right_on="user_id"
        ).dropna()

        X_train = train_data[
            seed_ids
            + [
                "age",
                "gender_encoded",
                "occupation_encoded"
            ]
        ].copy()

        X_train.columns = feature_columns

        y_train = train_data["target_rating"]

        n_train = len(X_train)

        if n_train == 0:
            predicted = float("nan")

        else:
            model = DecisionTreeRegressor(
                max_depth=15,
                random_state=42
            )

            model.fit(X_train, y_train)

            predicted = model.predict(X_test)[0]

            predicted = max(1, min(5, predicted))

        results.append({
            "movie_id": mid,
            "title": title,
            "predicted_rating": predicted,
            "n_train_users": n_train
        })

    results_df = pd.DataFrame(results).sort_values(
        "predicted_rating",
        ascending=False
    ).reset_index(drop=True)

    st.write("Predicted ratings for top 20 movies (based on dataset popularity).")

    st.dataframe(results_df[["title", "predicted_rating", "n_train_users"]])

    if st.checkbox("Compare with my actual ratings for these movies"):

        # collect actual ratings from user for the predicted list
        actuals = []

        st.write("Enter your actual ratings (0 if you haven't seen it):")

        for i, r in results_df.iterrows():

            actual = st.slider(
                r["title"],
                min_value=0,
                max_value=5,
                value=0,
                key=f"actual_{r['movie_id']}"
            )

            actuals.append(actual)

        results_df["actual_rating"] = actuals

        # compute basic error metric
        results_df["abs_error"] = (
            results_df["predicted_rating"] - results_df["actual_rating"]
        ).abs()

        mae = results_df["abs_error"].dropna().mean()

        st.write(f"Mean absolute error (on movies you rated): {mae:.2f}")

        st.dataframe(
            results_df[["title", "predicted_rating", "actual_rating", "abs_error"]]
        )