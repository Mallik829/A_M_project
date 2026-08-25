"""
Non-interactive script to predict ratings for the top 30 popular movies
using a DecisionTreeRegressor for each movie. Provide your demographics and
seed ratings as variables below; fill `ACTUAL_RATINGS_30` to compute MAE.

Usage: python predict_top30_noninteractive.py

Files expected (relative to this script):
    ../ml-100k/u.data
    ../ml-100k/u.item
    ../ml-100k/u.user

Edit the variables in the CONFIG section and run the script.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# -----------------------------
# CONFIG - edit these values
# -----------------------------
# Your demographics
USER_AGE = 25
USER_GENDER = "M"  # 'M' or 'F'
USER_OCCUPATION = "student"  # e.g. 'student', 'engineer', 'educator', etc.

# Seed ratings: map movie title -> rating (1-5, 0 if not seen).
# The script will use the top-10 most-rated movies as the seed set and
# will fill any missing title keys with 0. Update the ratings for the
# printed seed movies below before running.
USER_SEED_RATINGS = {'Star Wars (1977)': 3,
                     'Contact (1997)': 4,
                     'Fargo (1996)': 4,
                     'Return of the Jedi (1983)': 3,
                     'Liar Liar (1997)': 4,
                     'English Patient, The (1996)': 1,
                     'Scream (1996)': 4,
                     'Toy Story (1995)': 4,
                     'Air Force One (1997)': 3,
                     'Independence Day (ID4) (1996)': 4
    # example: "Star Wars (1977)": 5,
}

# Optional: your actual ratings for the predicted 30 movies (in the
# same order as the printed predictions). Use None or 0 for movies you
# haven't seen. Fill this list to compute MAE.
ACTUAL_RATINGS_30 = [5, 3, 4, 3, 2, 2, 2, 5, 2, 4, 4, 3, 5, 3, 3, 3, 2, 4, 4, 2, 2, 4, 4, 4, 4, 2, 3, 2, 2, 5]

# -----------------------------
# Data loading
# -----------------------------
ratings = pd.read_csv(
    "../ml-100k/u.data",
    sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"]
)

movies = pd.read_csv(
    "../ml-100k/u.item",
    sep="|",
    encoding="latin-1",
    header=None,
    usecols=[0, 1],
    names=["movie_id", "title"]
)

users = pd.read_csv(
    "../ml-100k/u.user",
    sep="|",
    names=["user_id", "age", "gender", "occupation", "zip_code"]
)

# Merge into a single dataframe
df = ratings.merge(movies, on="movie_id").merge(users, on="user_id")

# Encode gender
df["gender_encoded"] = df["gender"].map({"M": 0, "F": 1})

# Factorize occupations
occupation_codes, occupation_categories = pd.factorize(df["occupation"])
df["occupation_encoded"] = occupation_codes
occupation_list = occupation_categories.tolist()

# -----------------------------
# Seed movies (top 10 most-rated)
# -----------------------------
top_movies = (
    df.groupby(["movie_id", "title"]).size()
      .reset_index(name="rating_count")
      .sort_values("rating_count", ascending=False)
)

seed_movies = top_movies["title"].head(10).tolist()
seed_ids = [
    int(movies.loc[movies["title"] == title, "movie_id"].iloc[0])
    for title in seed_movies
]

print("Seed movies (top 10 popular) used as features:")
for t in seed_movies:
    print(" -", t)

# Ensure USER_SEED_RATINGS contains entries for all seed movies
for t in seed_movies:
    if t not in USER_SEED_RATINGS:
        USER_SEED_RATINGS[t] = 0

# -----------------------------
# Build user-rating matrix for seed movies
# -----------------------------
comparison = (
    df[df["movie_id"].isin(seed_ids)]
    .pivot_table(index="user_id", columns="movie_id", values="rating")
)

# Align columns to seed_ids order
comparison = comparison.reindex(columns=seed_ids)

# Fill missing ratings with movie averages
movie_means = comparison.mean()
comparison = comparison.fillna(movie_means)

# Add demographic features per user
user_features = (
    df[["user_id", "age", "gender_encoded", "occupation_encoded"]]
    .drop_duplicates("user_id")
    .set_index("user_id")
)
comparison = comparison.join(user_features)

# -----------------------------
# Encode user demographics for the new user
# -----------------------------
USER_GENDER = USER_GENDER.upper()
if USER_GENDER not in ("M", "F"):
    raise ValueError("USER_GENDER must be 'M' or 'F'")

user_gender_encoded = {"M": 0, "F": 1}[USER_GENDER]

# Normalize occupation key and lookup
occupation_aliases = {
    "teacher": "educator",
    "teaching": "educator",
    "software engineer": "engineer",
    "dev": "programmer",
}
user_occ = occupation_aliases.get(USER_OCCUPATION.lower(), USER_OCCUPATION.lower())

if user_occ not in occupation_list:
    raise ValueError(
        f"USER_OCCUPATION '{USER_OCCUPATION}' not found in dataset occupations.\n"
        f"Available occupations: {occupation_list}"
    )

user_occupation_encoded = occupation_list.index(user_occ)

# -----------------------------
# Create user's feature vector
# -----------------------------
user_vector = [
    USER_SEED_RATINGS[title]
    for title in seed_movies
]
user_vector.extend([USER_AGE, user_gender_encoded, user_occupation_encoded])

feature_columns = [str(mid) for mid in seed_ids] + ["age", "gender_encoded", "occupation_encoded"]

X_test = pd.DataFrame([user_vector], columns=feature_columns)

# -----------------------------
# Candidate movies: top 30 excluding seeds
# -----------------------------
candidate_movies = top_movies[~top_movies["movie_id"].isin(seed_ids)].head(30)

results = []

for _, row in candidate_movies.iterrows():
    mid = int(row["movie_id"])
    title = row["title"]

    target_ratings = df[df["movie_id"] == mid][["user_id", "rating"]].rename(columns={"rating": "target_rating"})

    train_data = comparison.merge(target_ratings, left_index=True, right_on="user_id").dropna()

    if len(train_data) == 0:
        pred = float("nan")
        n_train = 0
    else:
        X_train = train_data[seed_ids + ["age", "gender_encoded", "occupation_encoded"]].copy()
        X_train.columns = feature_columns
        y_train = train_data["target_rating"]

        model = DecisionTreeRegressor(max_depth=5, random_state=42, min_samples_split=20, min_samples_leaf=5, max_features=None)
        model.fit(X_train, y_train)

        pred = model.predict(X_test)[0]
        pred = max(1, min(5, pred))
        n_train = len(X_train)

    results.append({"movie_id": mid, "title": title, "predicted_rating": pred, "n_train_users": n_train})

results_df = pd.DataFrame(results).sort_values("predicted_rating", ascending=False).reset_index(drop=True)

print("\nPredicted ratings for top 30 movies:")
print(results_df[["title", "predicted_rating", "n_train_users"]].to_string(index=False))

# Save predictions
results_df.to_csv("predictions_top30.csv", index=False)
print("\nSaved predictions to predictions_top30.csv")

# -----------------------------
# Compute MAE if ACTUAL_RATINGS_30 provided
# -----------------------------
if len(ACTUAL_RATINGS_30) != len(results_df):
    print("\nWarning: ACTUAL_RATINGS_30 length does not match number of predictions.")

valid_mask = [(a is not None) and (a != 0) for a in ACTUAL_RATINGS_30[: len(results_df)]]

if any(valid_mask):
    preds = results_df.loc[valid_mask, "predicted_rating"].astype(float).values
    acts = np.array([a for a, v in zip(ACTUAL_RATINGS_30, valid_mask) if v], dtype=float)
    mae = mean_absolute_error(acts, preds)
    print(f"\nMean Absolute Error (on provided ratings): {mae:.3f}")
else:
    print("\nNo actual ratings provided for MAE calculation. Fill ACTUAL_RATINGS_30 in the CONFIG section and re-run the script.")
