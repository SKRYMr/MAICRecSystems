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

NEIGHBOURHOOD_SIZE = 50


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


def get_prediction(movie_id: int, user_id: int, neighbours: list, train_ratings: pd.DataFrame) -> float:
    neighbor_ratings = train_ratings[
        train_ratings['user_id'].isin(neighbours) & (train_ratings['movie_id'] == movie_id)]
    # Get the mean to be used as "penalty"-prediction if no neighbours has watched the given movie.
    user_mean_rating = train_ratings[train_ratings['user_id'] == user_id]['rating'].mean()

    if neighbor_ratings.empty:
        # If no neighbour has watched the rated movie. Assume that the user would rate it similar as his "average" rating
        predicted_rating = user_mean_rating
    else:
        predicted_rating = neighbor_ratings['rating'].mean()

    # Round it. Ratings are given by 1,2,3,4 or 5.
    return round(predicted_rating)


def calculate_errors(predictions: pd.Series, true_ratings: pd.Series) -> tuple:
    mae = abs(predictions - true_ratings).mean()
    rmse = np.sqrt(((predictions - true_ratings) ** 2).mean())

    return mae, rmse


def predict_by_neighbours(test_ratings: pd.DataFrame, train_ratings: pd.DataFrame, users, NEIGHBOURHOOD_SIZE,
                          max_iterations=10):
    predictions = []
    true_ratings = []

    for _, test_row in test_ratings.iterrows():
        user_id = test_row['user_id']
        movie_id = test_row['movie_id']
        user_rating = test_row['rating']

        neighbours = find_k_nearest(user_id, users, train_ratings, NEIGHBOURHOOD_SIZE)
        pred = get_prediction(movie_id, user_id, neighbours, train_ratings)
        predictions.append(pred)
        true_ratings.append(user_rating)

        if len(predictions) == max_iterations:
            break

    return predictions, true_ratings


def evaluate_predictions(predictions, true_ratings):
    mae, rmse = calculate_errors(pd.Series(predictions), pd.Series(true_ratings))

    print(f"Total predictions: {len(predictions)}")
    print("Loss")
    print(f"    MAE: {mae}")
    print(f"    RMSE: {rmse:.3f}")


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
    s = time.time()
    for i, user_id in enumerate(test_ratings["user_id"].unique()):
        user_test_movies = test_ratings[test_ratings["user_id"] == user_id]
        recommended_movies = get_movies_recommendations(user_id, database.users, train_ratings, database.movies)

        df = pd.merge(user_test_movies, recommended_movies, on="movie_id", how="left") # merge on movie_id which are test set
        df["rating_y"] = df["rating_y"].fillna(0) # if the pred rating does not exist, use 0

        MAE += abs(df["rating_y"] - df["rating_x"]).tolist()
        RMSE += (df["rating_y"] - df["rating_x"]).tolist()

        if i == 10:
            break
    print(time.time() - s)
    print(f"Rows covered: {len(MAE)}")
    print(f"MSE: {sum(MAE) / len(MAE)}")
    print(f"RMSE: {np.sqrt(sum(MAE) / len(MAE))}")

    s = time.time()
    # Predicts ratings based on the nearest neighbours average ratings on the movies.
    predictions, true_ratings = predict_by_neighbours(test_ratings, train_ratings, database.users, NEIGHBOURHOOD_SIZE)
    evaluate_predictions(predictions, true_ratings)
    print(time.time() - s)
    
    # aakash 10 user -> 80sec, rows covered = 75
    # filip 10 row -> 71sec
