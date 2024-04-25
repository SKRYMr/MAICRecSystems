import numpy as np
import pandas as pd
import os
import pickle
import sys
from sklearn.model_selection import train_test_split

RATINGS_DAT_FILE = "./data/ratings.dat"
MOVIES_DAT_FILE = "./data/movies.dat"
USERS_DAT_FILE = "./data/users.dat"
DB_PICKLE_PATH = "./data/database.pickle"


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
        
        # Remove ratings of users which are not in the subset
        self.ratings = self.ratings[~self.ratings["user_id"].isin(self.users)]

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

