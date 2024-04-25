import numpy as np
import pandas as pd
import os
import pickle
import sys
from sklearn.model_selection import train_test_split
from typing import Set

from ex3 import find_k_nearest, get_top_movies


RATINGS_DAT_FILE = "./data/ratings.dat"
MOVIES_DAT_FILE = "./data/movies.dat"
USERS_DAT_FILE = "./data/users.dat"
DB_PICKLE_PATH = "./data/database.pickle"

NEIGHBOURHOOD_SIZE = 50


def get_movies_recommendations(user_id: int, users: Set[int], ratings: pd.DataFrame, movies: pd.DataFrame):
    neighbours = find_k_nearest(user_id, users, ratings, NEIGHBOURHOOD_SIZE)
    return get_top_movies(user_id, neighbours, ratings, movies)


def MAELoss(test_movies: pd.DataFrame, recommended_movies: pd.DataFrame):
    # MAE loss = 1/n * sum( abs(pred_rating - user_rating) )
    # n = 10 = num of rec movies

    loss = 0
    for _, rec_movie in recommended_movies.iterrows():
        test_movie = test_movies[test_movies["movie_id"] == rec_movie["movie_id"]]
        if test_movie.empty:
            # Penelty: assume user ratings of this movie is 0
            loss += rec_movie["rating"]
        else:
            # Mean only because if there are more than 1 rating for the same movie by the user
            loss += abs(rec_movie["rating"] - test_movie["rating"].mean())

    return loss / 10


class MovieLens:
    def __init__(self):
        try:
            self.ratings = pd.read_table(RATINGS_DAT_FILE, engine="python", sep="::", usecols=[0, 1, 2], names=["user_id", "movie_id", "rating"], dtype={"rating": np.float32})

            self.movies = pd.read_table(MOVIES_DAT_FILE, engine="python", sep="::", names=["movie_id", "title", "genres"], encoding="latin")
            self.movies["genres"] = self.movies["genres"].apply(lambda x: x.split("|"))  # convert genres to list

            self.users = set(pd.read_table(USERS_DAT_FILE, engine="python", sep="::", usecols=[0], names=["user_id"])["user_id"].unique())
        except FileNotFoundError:
            print("Some database files are missing.")
            sys.exit(0)

    def use_subset(self, size=0.5):
        # Size = what percentage of orignal data to use
        self.movies, _ = train_test_split(self.movies, train_size=size)
        self.users, _ = train_test_split(list(self.users), train_size=size)
        self.users = set(self.users)

        # Remove ratings of users and movies which are not in the subset
        self.ratings = self.ratings[(self.ratings["user_id"].isin(self.users)) & (self.ratings["movie_id"].isin(self.movies["movie_id"]))]

    def split_ratings(self, test_size=0.2):
        return train_test_split(self.ratings, test_size=test_size)


if __name__ == "__main__":
    if os.path.isfile(DB_PICKLE_PATH):
        with open(DB_PICKLE_PATH, "rb") as f:
            database = pickle.load(f)
    else:
        database = MovieLens()
        with open(DB_PICKLE_PATH, "wb") as f:
            pickle.dump(database, f)

    database.use_subset(0.5)

    train_ratings, test_ratings = database.split_ratings()

    MAE = []
    RMSE = []
    total_count = 0

    for i, user_id in enumerate(test_ratings["user_id"].unique()):
        user_test_movies = test_ratings[test_ratings["user_id"] == user_id]
        recommended_movies = get_movies_recommendations(user_id, database.users, train_ratings, database.movies)

        # Find loss MAE and RMSE loss
        mae_loss = MAELoss(user_test_movies, recommended_movies)

        MAE.append(mae_loss)
        total_count += 1

        if i >= 10:
            break
            
        # Question: do we take average of average(doing this now), or just a single average

    print(f"Total users: {total_count}")
    print("MAE Loss")
    print(f"    Total: {sum(MAE)}")
    print(f"    Average: {sum(MAE) / total_count}")
    print(f"    Max: {max(MAE)}")
    print(f"    Min: {min(MAE)}")

