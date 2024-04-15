import pandas as pd
import numpy as np

class DataHandler():
    def __init__(self):
        self.rating_data = pd.read_csv('ml-1m/ratings.dat', delimiter="::", names=['userId', 'movieId', 'rating','timestamp'], engine='python')
        self.movie_data = pd.read_csv('ml-1m/movies.dat', sep='::', header=None, names=['movieId', 'title','genres'], engine='python', encoding='latin1')
    
    def getUserRatings(self, user_id):
        user_ratings = self.rating_data[self.rating_data['userId'] == user_id]
        return user_ratings

    def getMovieInfo(self, movie_id):
        movie_info = self.movie_data[self.movie_data['movieId'] == movie_id]
        return movie_info
    
    def getUserMean(self, user_id):
        user_ratings = self.getUserRatings(user_id)['rating']
        if user_ratings.empty:
            return None
        return user_ratings.mean()
    
    def getCommonRatings(self, user_id_1, user_id_2):
        ratings_user_1 = self.getUserRatings(user_id_1)
        ratings_user_2 = self.getUserRatings(user_id_2)
        
        #find common movies between user 1 and user 2
        common_movie_ids = set(ratings_user_1['movieId']).intersection(set(ratings_user_2['movieId']))
        
        common_ratings_user_1 = ratings_user_1[ratings_user_1['movieId'].isin(common_movie_ids)]['rating']
        common_ratings_user_2 = ratings_user_2[ratings_user_2['movieId'].isin(common_movie_ids)]['rating']
        
        return common_ratings_user_1.values, common_ratings_user_2.values
    
    def get_rated_movies_ids(self, user_id):
        return set(self.getUserRatings(user_id)['movieId'])
    
    def get_rated_movies(self, user_id):
        return pd.merge(self.getUserRatings(user_id), self.movie_data, on="movieId")
    
    def get_unique_users(self):
        return self.rating_data['userId'].unique()
    
    def get_neighbour_movies(self,n):
        return self.rating_data[self.rating_data.user_id.isin(n)]

def showRatedMovies(id, data_handler, n=15):
    
    rated_movies = data_handler.get_rated_movies(id)
    
    for i, (_, movie) in enumerate(rated_movies.iterrows(), 1):
        if(i <=n):
            print(f"Movie {i}:")
            print("Title:", movie['title'])
            print("Genre:", movie['genres'])
        else:
            break   

def read_user():
    while True:
            try:
                id = int(input("Enter the user ID: "))
            except ValueError:
                print("The ID must be an integer, try again please!")
                continue
            else:
                break
    return id

def calculateSimilarity(user_a,user_b, data_handler):
    a_mean = data_handler.getUserMean(user_a)
    b_mean = data_handler.getUserMean(user_b)
  
    common_ratings_a, common_ratings_b = data_handler.getCommonRatings(user_a, user_b)
    
    covariance = np.sum((common_ratings_a - a_mean) * (common_ratings_b - b_mean))
     
    std_a = np.sqrt(np.sum((common_ratings_a-a_mean)**2))
    std_b = np.sqrt(np.sum((common_ratings_b-b_mean)**2))
    
    #do not really know the correct way of handling zero in this case
    if std_a == 0 or std_b == 0:
        correlation = 0  #set it as 0 for now
    else:
        correlation = covariance / (std_a * std_b)
    
    return correlation

def find_k_nearest(id, data_handler, k=10):
    similarities = []
    users = data_handler.get_unique_users()
    
    for other_id in users: #just set 1000 for now for speed purposes
        if other_id != id and other_id < 1000:
            similarity = calculateSimilarity(id, other_id, data_handler)
            similarities.append((other_id, similarity))
    
    similarities.sort(key=lambda x: x[1], reverse=True)  #sort similarity in descending order
    top_k = similarities[:k]  #get the k top
    
    return top_k

def recommendMovies(data_handler, id, neighbours):
    user_ratings = data_handler.getUserRatings(id)
    
    #get ratings from every k neighbour
    neighbours_ratings = pd.DataFrame()
    for n_id,sim in neighbours:
        if n_id != id:
            neighbour_ratings = data_handler.getUserRatings(n_id)
            neighbours_ratings = pd.concat([neighbours_ratings, neighbour_ratings])
    
    #remove movies rated by user
    neighbours_ratings = neighbours_ratings[~neighbours_ratings['movieId'].isin(user_ratings['movieId'])]
    
    #get the mean ratings of the movies rated by the neighbours
    neighbours_ratings = neighbours_ratings.groupby('movieId')['rating'].mean().reset_index()
    
    # sort by rating and take the top 10
    neighbours_ratings.sort_values(by='rating', ascending=False, inplace=True)
    top_movies = neighbours_ratings.head(10)
    
    #retrieve the movie info
    top_movies_info = []
    for _, row in top_movies.iterrows():
        movie_info = data_handler.getMovieInfo(row['movieId'])
        top_movies_info.append(movie_info)
    
    print(f"Top {10} recommended movies for user {id}:")
    for i, movie_info in enumerate(top_movies_info, 1):
        print(f"Movie {i}:")
        print("Title:", movie_info['title'].values[0])
        print("Genre:", movie_info['genres'].values[0])
        print()



def main():
    data_handler = DataHandler()
    id = read_user()
    showRatedMovies(id, data_handler)
    neighbours = find_k_nearest(id, data_handler,10)
    recommendMovies(data_handler, id, neighbours)

if __name__ == "__main__":
    main()