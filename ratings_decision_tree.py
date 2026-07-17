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
)

comparison = comparison[seed_ids]


# -------------------------------------------------
# Create user's rating vector
# -------------------------------------------------

user_vector = np.array(
    [
        user_ratings[movie_id]
        for movie_id in seed_ids
    ]
).reshape(1,-1)



tree = DecisionTreeRegressor(
    max_depth=15,
    random_state=42
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

    train_data = comparison.merge(
        target_ratings,
        left_index=True,
        right_on="user_id"
    ).dropna()

    X_train = train_data[seed_ids]
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

        X_test = pd.DataFrame(
            user_vector,
            columns=seed_ids
        )

        prediction = tree.predict(X_test)

        print(
            "\nPredicted rating:",
            round(prediction[0], 2)
        )