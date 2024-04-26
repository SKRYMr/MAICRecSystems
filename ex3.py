import numpy as np
import pandas as pd
import sys
import time

RATINGS_DAT_FILE = "./data/ratings.dat"
MOVIES_DAT_FILE = "./data/movies.dat"
USERS_DAT_FILE = "./data/users.dat"

class MovieLens():
    def __init__(self):
        try:
            self.ratings = pd.read_table(RATINGS_DAT_FILE, engine="python", sep="::", usecols=[0, 1, 2],
                                         names=["user_id", "movie_id", "rating"], dtype={"rating": np.float32})
            self.movies = pd.read_table(MOVIES_DAT_FILE, engine="python", sep="::",
                                        names=["movie_id", "title", "genres"], encoding="latin")
            self.users = set(pd.read_table(USERS_DAT_FILE, engine="python", sep="::", usecols=[0], names=["user_id"])[
                                 "user_id"].unique())
        except FileNotFoundError as err:
            print("Some database files are missing.")
            sys.exit(0)

    def get_user_ratings(self, user_id: int) -> pd.DataFrame:
        user_ratings = self.ratings[self.ratings["user_id"] == user_id]
        return user_ratings

    def get_user_movies(self, user_id: int) -> pd.DataFrame:
        return pd.merge(self.get_user_ratings(user_id), database.movies, on="movie_id")

    def get_movie_info(self, movie_id: int) -> pd.DataFrame:
        movie_info = self.movie_data[self.movie_data["movie_id"] == movie_id]
        return movie_info

    def get_user_mean(self, user_id):
        user_ratings = self.get_user_ratings(user_id)['rating']
        if user_ratings.empty:
            return None
        return user_ratings.mean()


def show_movies(movies: pd.DataFrame, n_movies: int = 15, recommendation: bool = False):
    if n_movies > 15:
        n_movies = 15

    if recommendation:
        print(f"\n{n_movies if len(movies) > n_movies else len(movies)} Best Movies For User =========================")
    else:
        print(f"\n{n_movies if len(movies) > n_movies else len(movies)} Movies Rated By User =========================")
    for i, row in movies.iterrows():
        genres = row["genres"].replace("|", ", ")
        print(f"{i + 1}) {row['title']} - {genres} - {row['rating']:.3f}")
        n_movies -= 1
        if n_movies == 0:
            break
    print(f"================================================\n")


def check_user_id(user_id: int, users: set) -> bool:
    if user_id not in users:
        print("User not found.")
        print(f"Minimum user ID: {min(users)}")
        print(f"Maximum user ID: {max(users)}")
        return False
    return True


def pearson_correlation(col1, col2) -> float:
    if len(col1) == 0 or len(col1) < 5:
        return 0
    col1_mean = col1.mean()
    col2_mean = col2.mean()
    a = ((col1 - col1_mean) * (col2 - col2_mean)).sum()
    b = ((col1 - col1_mean) ** 2).sum()
    c = ((col2 - col2_mean) ** 2).sum()
    if b == 0 or c == 0:
        b = b + 0.02
        c = c + 0.02
    return a / ((b * c) ** 0.5)


def find_k_nearest(user_id: int, users: set, ratings: pd.DataFrame, k: int) -> list:
    s = time.time()
    similarities = {}
    user_ratings = ratings[ratings.user_id == int(user_id)]
    ratings_comparison = pd.merge(user_ratings, ratings, on="movie_id")
    for id in users - {user_id}:
        other_user_ratings = ratings_comparison[ratings_comparison.user_id_y == int(id)]
        similarities[id] = pearson_correlation(other_user_ratings["rating_x"], other_user_ratings["rating_y"])
    similarities_sorted = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    users = [user[0] for user in similarities_sorted[:k]]
    print(f"Time to get nearest neighbours: {time.time() - s}")

    return users


def get_top_movies(user_id: int, neighbours: list, ratings: pd.DataFrame, movies: pd.DataFrame):
    #user_ratings = database.get_user_ratings(user_id)
    s = time.time()
    user_ratings = ratings[ratings["user_id"] == user_id]
    neighbours_ratings = ratings[ratings.user_id.isin(neighbours)]
    neighbours_ratings = neighbours_ratings[~neighbours_ratings.movie_id.isin(user_ratings.movie_id)]
    neighbours_ratings = neighbours_ratings[neighbours_ratings["movie_id"].map(neighbours_ratings["movie_id"].value_counts()) >= 5]
    neighbours_ratings = neighbours_ratings.groupby("movie_id").rating.mean().reset_index()
    neighbours_ratings.sort_values("rating", ascending=False, inplace=True)
    # neighbours_ratings = neighbours_ratings.head(10)
    print(f"Time to get top movies: {time.time() - s}")
    return pd.merge(neighbours_ratings, movies, on="movie_id")
    #return show_movies(neighbours_movies, 10, recommendation=True)


def get_user_item_table(user_id: int, df_ratings: pd.DataFrame) -> pd.DataFrame:
    user_item_table = df_ratings.pivot_table(index="user_id", columns="movie_id", values="rating", fill_value=0)
    user_vector = user_item_table.iloc[user_id - 1]
    user_item_table.drop(index=user_id, inplace=True)

    # Calculate user similarity
    user_item_table["score"] = user_item_table.apply(lambda x: user_vector.corr(x), axis=1)

    return user_item_table


def get_best_users(user_item_table: pd.DataFrame, min_corr: float = 0.25) -> pd.DataFrame:
    best_users = user_item_table[user_item_table["score"] > min_corr]

    if best_users.empty:
        print("No users found, reduce min correlation")
        sys.exit(0)
    else:
        print(f"Computing recommendations on {len(best_users)} users...\n")

    return best_users


def rate_movies_by_best_users(best_users: pd.DataFrame, user_movies: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(best_users.apply(np.average, axis=0), columns=["avg_rating"]).sort_values("avg_rating",
                                                                                                  ascending=False).drop(
        user_movies["movie_id"])


if __name__ == "__main__":
    database = MovieLens()
    while True:
        try:
            user_id = int(input("User id: "))
        except ValueError as err:
            print("User ID must be a number.")
        if check_user_id(user_id, database.users):
            break

    nearest_neighbours = 50

    show_movies(database.get_user_movies(user_id).sort_values("rating", ascending=False, ignore_index=True),
                n_movies=15)
    neighbours = find_k_nearest(user_id, database.users, database.ratings, nearest_neighbours)
    get_top_movies(user_id, neighbours, database.ratings, database.movies)
