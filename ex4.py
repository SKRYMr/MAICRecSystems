import numpy as np
import pandas as pd
import os
import pickle
import sys
from typing import Dict

RATINGS_DAT_FILE = "./data/ratings.dat"
MOVIES_DAT_FILE = "./data/movies.dat"
USERS_DAT_FILE = "./data/users.dat"
DB_PICKLE_PATH = "./data/database.pickle"

MIN_USER_RATING = 4  # movies with rating equal or higher than this number are used to create the user profile


class MovieLens:
    def __init__(self):
        try:
            self.ratings = pd.read_table(RATINGS_DAT_FILE, engine="python", sep="::", usecols=[0, 1, 2], names=["user_id", "movie_id", "rating"], dtype={"rating": np.float32})

            self.movies = pd.read_table(MOVIES_DAT_FILE, engine="python", sep="::", names=["movie_id", "title", "genres"], encoding="latin")
            self.movies["genres"] = self.movies["genres"].apply(lambda x: x.split("|"))  # convert genres to list

            self.users = set(pd.read_table(USERS_DAT_FILE, engine="python", sep="::", usecols=[0], names=["user_id"])["user_id"].unique())

        except FileNotFoundError as err:
            print("Some database files are missing.")
            sys.exit(0)

    def get_user_profile(self, user_id: int, normalized=False):
        user_ratings = self.ratings[(self.ratings["user_id"] == user_id) & (self.ratings["rating"] >= MIN_USER_RATING)]
        user_movies = self.movies[self.movies["movie_id"].isin(user_ratings["movie_id"])]

        user_genres_weighted = user_movies["genres"].explode().value_counts(normalize=normalized).to_dict()

        return user_genres_weighted, pd.merge(user_movies, user_ratings, on="movie_id", how="inner").drop(columns="user_id")


def check_user_id(user_id: int, users: set) -> bool:
    if user_id not in users:
        print("User not found, try again")
        return False
    return True


def show_user_profile(genres: Dict[str, int], movies: pd.DataFrame, n: int = 10):
    print("\nUser Profile =======================")

    print("\nBest movies")
    for i, (_, row) in enumerate(movies.sort_values(by="rating", ascending=False).iterrows()):
        print(f"{i+1}) {row['title']} - {', '.join(row['genres'])} - {row['rating']}")
        if (i+1) >= n:
            break

    print(f"\nGenre")
    for k, v in genres.items():
        print(f"{k} - {v}")

    print(f"\n===================================")


def keyword_similarity(user_genres: Dict[str, int], movies: pd.DataFrame):
    genres = set(user_genres.keys())
    n_user_genres = len(genres)
    movies["keyword_similarity"] = movies["genres"].apply(lambda x: (2 * len(set(x).intersection(genres))) / (n_user_genres + len(x)))
    return movies


def show_recommendations(movies: pd.DataFrame, n: int = 10):
    print("\nKeyword Recommendations ===================================\n")
    for i, (_, row) in enumerate(movies.sort_values("keyword_similarity", ascending=False).iterrows()):
        print(f"{i+1}) {row['title']} - {', '.join(row['genres'])} - {row['keyword_similarity']:.2f}")
        if (i+1) >= n:
            break
    print(f"\n===================================")


if __name__ == "__main__":
    if os.path.isfile(DB_PICKLE_PATH):
        with open(DB_PICKLE_PATH, "rb") as fin:
            database = pickle.load(fin)
    else:
        database = MovieLens()
        with open(DB_PICKLE_PATH, "wb") as fout:
            pickle.dump(database, fout)

    while True:
        try:
            user_id = int(input("Enter user id: "))
            if check_user_id(user_id, database.users):
                break
        except ValueError as err:
            print("User ID must be a number.")

    user_genres, user_movies = database.get_user_profile(user_id, False)
    show_user_profile(user_genres, user_movies)

    # Clearly with this strategy if the user has liked many different movies the spread of genres is such
    # that the best recommended movies will simply be the ones that have the most genres assigned to them.
    movies = keyword_similarity(user_genres, database.movies)
    show_recommendations(movies)