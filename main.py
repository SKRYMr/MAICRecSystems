import pandas as pd
import os

def show_user_movies(user_movies, nmovies=15):

    if nmovies > 15:
        print("Showing maximum 15 movies")
        nmovies = 15
    else:
        print(f"Showing {nmovies} movies")
    print("Title - Genres")
    for _, row in user_movies.iterrows():
        genres = row['genres'].replace("|", ", ")
        print(f"{row['title']} - {genres}")
        nmovies -= 1
        if nmovies == 0:
            break
    return

def pearson_correlation(col1, col2):
    if len(col1) == 0 or len(col1) < 5:
        return 0
    col1_mean = col1.mean()
    col2_mean = col2.mean()
    a = ((col1-col1_mean)*(col2-col2_mean)).sum()
    b = ((col1-col1_mean)**2).sum()
    c = ((col2-col2_mean)**2).sum()
    #print(a, b, c)
    if b == 0 or c == 0:
        b = b + 0.02
        c = c + 0.02
    return abs(a/((b*c)**0.5))


def find_k_nearest(user_id, k):
    similarities = {}
    user_ids = ratings['user_id'].unique()
    for id in user_ids:
        if int(id) == int(user_id):
            continue
        if int(user_id) == 2000:
            break
        user_ratings1 = ratings[ratings.user_id == int(id)]
        user_ratings_ab = pd.merge(ratings[ratings.user_id == int(user_id)], user_ratings1, on="movie_id")
        similarities[id] = pearson_correlation(user_ratings_ab['rating_x'], user_ratings_ab['rating_y'])
    similarities_sorted = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    users = [user[0] for user in similarities_sorted[:k]]
    return users

def get_top_movies(user_id, neighbours):
    user_ratings = ratings[ratings.user_id == int(user_id)]
    neighbours_ratings = ratings[ratings.user_id.isin(neighbours)]
    neighbours_ratings = neighbours_ratings[~neighbours_ratings.movie_id.isin(user_ratings.movie_id)]
    neighbours_ratings = neighbours_ratings.groupby('movie_id').rating.mean().reset_index()
    neighbours_ratings.sort_values('rating', ascending=False, inplace=True)
    neighbours_ratings = neighbours_ratings.head(10)
    neighbours_movies = pd.merge(neighbours_ratings, movies, on="movie_id")
    print(f"Top 10 recommended movies for user {user_id}")
    return show_user_movies(neighbours_movies,10)


if __name__ == '__main__':
    user_id = input('User id: ')
    print( user_id)

    if os.path.exists('ratings.csv') and os.path.exists('movies.csv'):
        ratings = pd.read_csv('ratings.csv')
        movies = pd.read_csv('movies.csv')
    else:
        ratings = pd.read_table("./ml-1m/ratings.dat", engine="python",sep="::", usecols=[0,1,2],names=["user_id", "movie_id", "rating"])

        movies = pd.read_table("./ml-1m/movies.dat", engine="python",sep="::", names=["movie_id", "title", "genres"], encoding = 'latin')

        ratings.to_csv('ratings.csv', index=False)
        movies.to_csv('movies.csv', index=False)



    user_ratings = ratings[ratings.user_id == int(user_id)]
    user_ratings1 = ratings[ratings.user_id == 4]

    user_movies = pd.merge(user_ratings, movies, on="movie_id")

    show_user_movies(user_movies, nmovies=15)

    neighbours = find_k_nearest(user_id, 5)
    
    print(neighbours)

    get_top_movies(user_id, neighbours)

