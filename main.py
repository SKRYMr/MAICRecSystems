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

def find_k_nearest(user_id, k):
    return


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

    user_movies = pd.merge(user_ratings, movies, on="movie_id")

    show_user_movies(user_movies, nmovies=15)

    
    print(user_movies.head())

 