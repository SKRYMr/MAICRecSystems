import numpy as np
import pandas as pd
import sys
from typing import Dict

RATINGS_DAT_FILE = "./data/ratings.dat"
MOVIES_DAT_FILE = "./data/movies.dat"
USERS_DAT_FILE = "./data/users.dat"

MIN_USER_RATING = 3  # movies with rating equal or higher than this number are used to create the user profile


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


def check_user_id(user_id: int, users: set):
    if user_id not in users:
        print("User not found, try again")
        return False
    return True


def show_user_profile(genres: Dict[str, int], movies: pd.DataFrame, n=10):
    print("User Profile =======================")

    print("\nBest movies")
    for i, (_, row) in enumerate(movies.sort_values(by="rating", ascending=False).iterrows()):
        print(f"{i+1}) {row['title']} - {', '.join(row['genres'])} - {row['rating']}")
        if (i+1) >= n:
            break

    print(f"\nGenre")
    for k, v in genres.items():
        print(f"{k} - {v}")

    print(f"\n===================================")


if __name__ == "__main__":
    database = MovieLens()

    while True:
        try:
            user_id = int(input("Enter user id: "))
            if check_user_id(user_id, database.users):
                break
        except ValueError as err:
            print("User ID must be a number.")

    user_genres, user_movies = database.get_user_profile(user_id, True)
    show_user_profile(user_genres, user_movies)
