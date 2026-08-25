import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error


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
    usecols=[0,1],
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

df = ratings.merge(
    movies,
    on="movie_id"
)

df = df.merge(
    users,
    on="user_id"
)

# Convert gender to a number
df["gender_encoded"] = df["gender"].map({
    "M": 0,
    "F": 1
})

# Convert occupation to numbers
occupation_codes, occupation_categories = pd.factorize(
    df["occupation"]
)

df["occupation_encoded"] = occupation_codes



# -------------------------------------------------
# Select 10 movies for the user to rate
# -------------------------------------------------

# -------------------------------------------------
# Select 10 movies with the most ratings
# -------------------------------------------------

top_movies = (
    df.groupby(["movie_id", "title"])
      .size()
      .reset_index(name="rating_count")
      .sort_values("rating_count", ascending=False)
)

seed_movies = top_movies["title"].head(10).tolist()

print("\nMovies you will rate:")
for title in seed_movies:
    print("-", title)


user_ratings = {}

print("\nRate these movies from 1-5")
print("Enter 0 if you haven't seen it\n")


for title in seed_movies:

    movie = df[
        df["title"] == title
    ]

    if len(movie) == 0:
        continue

    movie_id = movie["movie_id"].iloc[0]

    rating = float(
        input(f"{title}: ")
    )

    user_ratings[movie_id] = rating



# -------------------------------------------------
# Create only a 10 movie user-rating matrix
# -------------------------------------------------

seed_ids = list(user_ratings.keys())


# -------------------------------------------------
# Create user-rating matrix for seed movies
# -------------------------------------------------

comparison = (
    df[df["movie_id"].isin(seed_ids)]
    .pivot_table(
        index="user_id",
        columns="movie_id",
        values="rating"
    )
)

# Make sure the columns are in the same order as seed_ids
comparison = comparison.reindex(columns=seed_ids)

# -------------------------------------------------
# Fill missing ratings
# -------------------------------------------------

# Some users have not rated every seed movie.
# Instead of removing those users, fill missing
# ratings with the average rating for that movie.

movie_means = comparison.mean()

comparison = comparison.fillna(movie_means)

# -------------------------------------------------
# Add demographic information for each user
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

comparison = comparison.join(user_features)


# -------------------------------------------------
# Create user's rating vector
# -------------------------------------------------

# -------------------------------------------------
# Get new user's demographic information
# -------------------------------------------------

print("\nTell us about yourself:")

user_age = int(
    input("Age: ")
)

user_gender = input(
    "Gender (M/F): "
).upper()

while user_gender not in ["M", "F"]:
    user_gender = input(
        "Please enter M or F: "
    ).upper()

user_occupation = input(
    "Occupation: "
).lower().strip()

# Allow common occupation names
occupation_aliases = {
    "teacher": "educator",
    "teaching": "educator",
    "student": "student",
    "software engineer": "engineer",
    "doctor": "doctor",
    "nurse": "healthcare",
}

user_occupation = occupation_aliases.get(
    user_occupation,
    user_occupation
)

# Convert gender to the same numeric format used by the model
user_gender_encoded = {
    "M": 0,
    "F": 1
}[user_gender]

# Convert occupation to the same numeric coding used in the training data
occupation_lookup = dict(
    zip(
        occupation_categories,
        range(len(occupation_categories))
    )
)

while user_occupation not in occupation_lookup:

    print("\nOccupation not found.")
    print("Available occupations:")
    print(", ".join(occupation_categories))

    user_occupation = input(
        "\nPlease enter your occupation: "
    ).lower().strip()

    user_occupation = occupation_aliases.get(
        user_occupation,
        user_occupation
    )

user_occupation_encoded = occupation_lookup[
    user_occupation
]


# -------------------------------------------------
# Create user's feature vector
# -------------------------------------------------

user_vector = [
    user_ratings[movie_id]
    for movie_id in seed_ids
]

user_vector.extend([
    user_age,
    user_gender_encoded,
    user_occupation_encoded
])

user_vector = np.array(
    user_vector
).reshape(1, -1)



tree = DecisionTreeRegressor(
    max_depth=15,
    random_state=42
)



# -------------------------------------------------
# Predict a movie rating
# -------------------------------------------------

print("\nChoose a movie to predict:")

movie_options = movies["title"].drop_duplicates().head(20).reset_index(drop=True)

for i, title in enumerate(movie_options, start=1):
    print(f"{i}. {title}")

selection = int(
    input(f"\nEnter a number (1-{len(movie_options)}): ")
)

movie_choice = movie_options.iloc[selection - 1]

print(f"\nYou selected: {movie_choice}")


matches = movies[
    movies["title"]
    .str.contains(
        movie_choice,
        case=False,
        regex=False
    )
]


if len(matches) == 0:

    print("Movie not found")

else:

    print("\nPossible movies:")
    print(matches.head(10))


    matches = matches.reset_index(drop=True)

    if len(matches) == 1:
        movie_id = int(matches.loc[0, "movie_id"])
        print(f"\nSelected movie: {matches.loc[0, 'title']}")
    else:
        print("\nMultiple matches found:")
        for idx, row in matches.iterrows():
            print(f"{idx + 1}. {row['title']}")

        selection = int(
            input(
                f"\nEnter selection number (1-{len(matches)}): "
            )
        )
        movie_id = int(matches.loc[selection - 1, "movie_id"])

    target_ratings = df[
        df["movie_id"] == movie_id
    ][
        ["user_id", "rating"]
    ].rename(columns={"rating": "target_rating"})

    # -------------------------------------------------
    # Combine training data
    # -------------------------------------------------

    train_data = comparison.merge(
        target_ratings,
        left_index=True,
        right_on="user_id"
    ).dropna()

    # -------------------------------------------------
    # Prepare training features
    # -------------------------------------------------

    feature_columns = (
    [str(movie_id) for movie_id in seed_ids]
    + [
        "age",
        "gender_encoded",
        "occupation_encoded"
    ]
)

X_train = train_data[seed_ids + [
    "age",
    "gender_encoded",
    "occupation_encoded"
]].copy()

X_train.columns = feature_columns

y_train = train_data["target_rating"]

print(
        f"\nTraining data: {len(X_train)} users with all seed movie ratings and target movie rating"
    )

if len(X_train) == 0:

    print("Not enough training data for this movie.")

else:

        tree.fit(
            X_train,
            y_train
        )

        # -------------------------------------------------
        # Create test data for our new user
        # -------------------------------------------------

        X_test = pd.DataFrame(
            [user_vector[0]],
            columns=feature_columns
        )

        # -------------------------------------------------
        # Predict target movie rating
        # -------------------------------------------------

        prediction = tree.predict(X_test)

        print(
            "\nPredicted rating:",
            round(prediction[0], 2)
        )


# -------------------------------------------------
# Bulk predict top-20 movies (no Streamlit)
# -------------------------------------------------

print("\nRunning bulk predictions for top 20 popular movies (excluding your seed movies)...")

# Choose top 20 candidate movies excluding the seed set
candidate_movies = (
    top_movies[~top_movies["movie_id"].isin(seed_ids)]
    .head(20)
)

results = []

for _, row in candidate_movies.iterrows():

    mid = int(row["movie_id"])
    title = row["title"]

    target_ratings = df[
        df["movie_id"] == mid
    ][["user_id", "rating"]].rename(columns={"rating": "target_rating"})

    train_data = comparison.merge(
        target_ratings,
        left_index=True,
        right_on="user_id"
    ).dropna()

    X_train = train_data[seed_ids + [
        "age",
        "gender_encoded",
        "occupation_encoded"
    ]].copy()

    feature_columns = [str(movie_id) for movie_id in seed_ids] + [
        "age",
        "gender_encoded",
        "occupation_encoded"
    ]

    if len(X_train) > 0:
        X_train.columns = feature_columns
        y_train = train_data["target_rating"]

        model = DecisionTreeRegressor(max_depth=15, random_state=42)
        model.fit(X_train, y_train)

        X_test = pd.DataFrame([user_vector[0]], columns=feature_columns)
        pred = model.predict(X_test)[0]
        pred = max(1, min(5, pred))
        n_train = len(X_train)

    else:
        pred = float("nan")
        n_train = 0

    results.append({
        "movie_id": mid,
        "title": title,
        "predicted_rating": pred,
        "n_train_users": n_train
    })

results_df = pd.DataFrame(results).sort_values("predicted_rating", ascending=False).reset_index(drop=True)

print("\nPredicted ratings (top 20):")
print(results_df[["title", "predicted_rating", "n_train_users"]].to_string(index=False))

# Placeholder for your actual ratings for these 20 movies.
# Replace the None entries with integers 1-5 (or 0 if you haven't seen it).
# The list should be in the same order as the printed predictions above.
actual_ratings_20 = [None] * len(results_df)  # <-- fill this list with your actual ratings

# Compute MAE ignoring None or 0 (0 means not seen)
valid_mask = [
    (a is not None) and (a != 0)
    for a in actual_ratings_20
]

if any(valid_mask):
    preds = results_df.loc[valid_mask, "predicted_rating"].astype(float).values
    acts = np.array([a for a, v in zip(actual_ratings_20, valid_mask) if v], dtype=float)
    mae = mean_absolute_error(acts, preds)
    print(f"\nMean Absolute Error (on your provided ratings): {mae:.3f}")
else:
    print("\nNo actual ratings provided for MAE calculation. Fill `actual_ratings_20` to compute MAE.")