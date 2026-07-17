import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
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


# -------------------------------------------------
# Select 10 movies for the user to rate
# -------------------------------------------------

seed_movies = [
    "Toy Story (1995)",
    "Jumanji (1995)",
    "Star Wars: Episode IV - A New Hope (1977)",
    "Forrest Gump (1994)",
    "The Matrix (1999)",
    "Jurassic Park (1993)",
    "Titanic (1997)",
    "The Shawshank Redemption (1994)",
    "The Dark Knight (2008)",
    "Pulp Fiction (1994)"
]


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


comparison = (
    df[df["movie_id"].isin(seed_ids)]
    .pivot_table(
        index="user_id",
        columns="movie_id",
        values="rating"
    )
    .fillna(0)
)


# -------------------------------------------------
# Create user's rating vector
# -------------------------------------------------

user_vector = np.array(
    [
        user_ratings[movie_id]
        for movie_id in seed_ids
    ]
).reshape(1,-1)



# -------------------------------------------------
# Find similar users
# -------------------------------------------------

similarities = cosine_similarity(
    user_vector,
    comparison
)[0]


comparison["similarity"] = similarities


nearest_users = (
    comparison
    .nlargest(100,"similarity")
    .index
)


print(
    f"\nFound {len(nearest_users)} similar users"
)



# -------------------------------------------------
# Train decision tree on similar users
# -------------------------------------------------

similar_data = df[
    df["user_id"].isin(nearest_users)
]


X_train = similar_data[
    [
        "user_id",
        "movie_id"
    ]
]


y_train = similar_data["rating"]



tree = DecisionTreeRegressor(
    max_depth=15,
    random_state=42
)


tree.fit(
    X_train,
    y_train
)



# -------------------------------------------------
# Predict a movie rating
# -------------------------------------------------

movie_choice = input(
    "\nEnter a movie title to predict: "
)


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


    movie_id = int(
        input(
            "\nEnter movie_id: "
        )
    )


    prediction = tree.predict(
        [
            [
                999999,
                movie_id
            ]
        ]
    )


    print(
        "\nPredicted rating:",
        round(prediction[0],2)
    )