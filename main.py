import numpy as np
import pandas as pd
import os
import pickle
import sys
from sklearn.model_selection import train_test_split
from typing import Set
import time

from ex3 import find_k_nearest, get_top_movies

RATINGS_DAT_FILE = "./data/ratings.dat"
MOVIES_DAT_FILE = "./data/movies.dat"
USERS_DAT_FILE = "./data/users.dat"
DB_PICKLE_PATH = "./data/database.pickle"

NEIGHBOURHOOD_SIZE = 100


def get_movies_recommendations(user_id: int, users: Set[int], ratings: pd.DataFrame, movies: pd.DataFrame):
    neighbours = find_k_nearest(user_id, users, ratings, NEIGHBOURHOOD_SIZE)
    return get_top_movies(user_id, neighbours, ratings, movies)

class MovieLens:
    def __init__(self):
        try:
            self.ratings = pd.read_table(RATINGS_DAT_FILE, engine="python", sep="::", usecols=[0, 1, 2],
                                         names=["user_id", "movie_id", "rating"], dtype={"rating": np.float32})

            self.movies = pd.read_table(MOVIES_DAT_FILE, engine="python", sep="::",
                                        names=["movie_id", "title", "genres"], encoding="latin")
            self.movies["genres"] = self.movies["genres"].apply(lambda x: x.split("|"))  # convert genres to list

            self.users = set(pd.read_table(USERS_DAT_FILE, engine="python", sep="::", usecols=[0], names=["user_id"])[
                                 "user_id"].unique())
        except FileNotFoundError:
            print("Some database files are missing.")
            sys.exit(0)

    def use_subset(self, size=0.5):
        # Size = what percentage of orignal data to use
        self.movies, _ = train_test_split(self.movies, train_size=size)
        self.users, _ = train_test_split(list(self.users), train_size=size)
        self.users = set(self.users)

        # Remove ratings of users and movies which are not in the subset
        self.ratings = self.ratings[
            (self.ratings["user_id"].isin(self.users)) & (self.ratings["movie_id"].isin(self.movies["movie_id"]))]

    def split_ratings(self, test_size=0.2):
        return train_test_split(self.ratings, test_size=test_size)
    
def get_predictions(test_ratings: pd.DataFrame, train_ratings: pd.DataFrame, users: set, movies: pd.DataFrame, num_users: int=10,fillNaValue: str="zero"):
    predictions = []
    
    for i, user_id in enumerate(test_ratings["user_id"].unique()):
        s = time.time()
        user_test_movies = test_ratings[test_ratings["user_id"] == user_id]
        recommended_movies = get_movies_recommendations(user_id, users, train_ratings, movies)

        df = pd.merge(user_test_movies, recommended_movies, on="movie_id", how="left") # merge on movie_id which are test set
        if fillNaValue == "zero":
            df["rating_y"] = df["rating_y"].fillna(0) # if the pred rating does not exist, use 0
        elif fillNaValue == "user_mean_rating":
            user_mean_rating = train_ratings[train_ratings['user_id'] == user_id]['rating'].mean()
            df["rating_y"] = df["rating_y"].fillna(user_mean_rating) # if the pred rating does not exist, use user average rating

        df["rating_y"] = df["rating_y"].apply(round)
        predictions += df[["rating_x", "rating_y"]].values.tolist()
        print(f"One iteration time: {time.time() - s}")
        if i == num_users:
            break
    return predictions

def calculate_error(predictions):
    # Predictions[:, 1] = rating_y = the prediction. Column 0 = rating_x = the rating from test set
    MAE = np.mean(np.abs(np.array(predictions)[:, 1] - np.array(predictions)[:, 0]))
    RMSE = np.sqrt(np.mean((np.array(predictions)[:, 1] - np.array(predictions)[:, 0]) ** 2))
    return MAE, RMSE 

def print_error(mae, rmse, rows_covered):
    print(f"Rows covered: {rows_covered}")
    print(f"Loss:")
    print(f"Mean Absolute Error (MAE): {mae:.3f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")

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

    # Predictions when filling "non-recommendations" with 0 rating.
    s = time.time()
    predictions = get_predictions(test_ratings, train_ratings, database.users, database.movies, fillNaValue="zero")
    print(f"Time to get predicitons: {time.time() - s}")
    MAE, RMSE = calculate_error(predictions)
    print_error(MAE, RMSE, len(predictions))
    print(predictions)

    
    # Predictions when filling "non-recommendations" with the users average rating.
    print("---------------------")
    s = time.time()
    predictions = get_predictions(test_ratings, train_ratings, database.users, database.movies, fillNaValue="user_mean_rating")
    print(f"Time to get predicitons: {time.time() - s}")
    MAE, RMSE = calculate_error(predictions)
    print_error(MAE, RMSE, len(predictions))
    print(predictions)
    
    
    
  
    
