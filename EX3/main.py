import pandas as pd
import numpy as np
import sys


def get_users():
    # UserID::Gender::Age::Occupation::Zip-code
    return pd.read_csv(
        "MAICRecSystems/EX3/data/users.dat",
        delimiter="::",
        names=["user_id", "gender", "age", "occupation", "zipcode"],
        engine='python'
    )


def get_movies():
    # MovieID::Title::Genres
    df = pd.read_csv(
        "MAICRecSystems/EX3/data/movies.dat",
        delimiter="::",
        names=["movie_id", "title", "genres"],
        engine='python',
        encoding="iso-8859-1"
    )
    df["genres"] = df["genres"].apply(lambda x: x.split("|"))

    return df


def get_ratings():
    # UserID::MovieID::Rating::Timestamp
    return pd.read_csv(
        "MAICRecSystems/EX3/data/ratings.dat",
        delimiter="::",
        names=["user_id", "movie_id", "rating", "timestamp"],
        engine='python'
    )


def check_user_id(user_id: str, df_users: pd.DataFrame) -> int:
    try:
        user_id = int(user_id)

        user_df = df_users[df_users["user_id"] == user_id]
        
        if user_df.empty:
            print("User not found")
            sys.exit(0)

        return user_id
    except ValueError:
        print("user id must be a number")
    except Exception as e:
        print(e)

    sys.exit(0)


def get_movie_rated_by_user(user_id: str, df_movies: pd.DataFrame, df_ratings: pd.DataFrame):
    user_ratings = df_ratings[df_ratings["user_id"] == user_id].copy()
    user_movies = df_movies[df_movies["movie_id"].isin(user_ratings["movie_id"])].copy()
    
    user_movies["rating"] = user_movies.apply(lambda x: user_ratings[user_ratings["movie_id"] == x["movie_id"]]["rating"].mean(), axis=1)
    
    return user_movies


def get_user_item_table(user_id: str, df_ratings: pd.DataFrame):
    user_item_table = df_ratings.pivot_table(index='user_id', columns='movie_id', values='rating', fill_value=0)
    user_vector = user_item_table.iloc[user_id-1]
    user_item_table.drop(index=user_id, inplace=True)

    # Calculate user similarity
    user_item_table["score"] = user_item_table.apply(lambda x: user_vector.corr(x), axis=1)
    
    return user_item_table


def get_best_users(user_item_table: pd.DataFrame, min_corr: float=0.25):
    best_users =  user_item_table[user_item_table["score"] > min_corr]

    if best_users.empty:
        print("No users found, reduce min correlation")
        sys.exit(0)

    return best_users


def rate_movies_by_best_users(best_users: pd.DataFrame, user_movies: pd.DataFrame):
    return pd.DataFrame(best_users.apply(np.median, axis=0), columns=["avg_rating"]).sort_values("avg_rating", ascending=False).drop(user_movies["movie_id"])


if __name__ == "__main__":
    df_users = get_users()
    df_movies = get_movies()
    df_ratings = get_ratings()

    user_id = input("Enter user id: ")
    user_id = check_user_id(user_id, df_users)

    user_movies = get_movie_rated_by_user(user_id, df_movies, df_ratings)

    print(f"\nMovies rated by user =========================")
    for i in range(0, 10 if len(user_movies) > 10 else len(user_movies)):
        row = user_movies.iloc[i]
        print(f'{i + 1}) {row["title"]}, {", ".join(row["genres"])}')
    print(f"================================================\n")

    # Movie Recommentations
    user_item_table = get_user_item_table(user_id, df_ratings)
    k_best_users = get_best_users(user_item_table)

    movie_scores = rate_movies_by_best_users(k_best_users, user_movies)

    print(f"\nMovies recommendations")
    for i, m_id in enumerate(movie_scores.head(10).index.values):
        m = df_movies[df_movies["movie_id"] == m_id]
        print(f'{i + 1}) {m["title"].values[0]}, {", ".join(m["genres"].to_list()[0])}, rating={movie_scores.loc[m_id]["avg_rating"]:.3f}')
    
    print("\n")