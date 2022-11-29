# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 20:33:24 2022

@author: Anupam Gajbhiye
OBJECTIVE:The Entertainment Company, which is an online movie watching platform, wants
 to improve its collection of movies and showcase those that are highly rated
 and recommend those movies to its customer by their movie watching footprint.
 automate its process for effective recommendations.


"""
import pandas as pd
import numpy as np
df=pd.read_csv('Entertainment.csv')


# Some values are mistaken as 99, most probably they might be 9, so cjanging the values to 9
df['Reviews'].mask(df['Reviews'] == 99, 9, inplace=True)

df.head()
# changing the rating scale from -9 , +9 to 0 to 9
# def norm(x):
    # x

df.shape    
df.columns
# =============================================================================
# Tfidfvectorizer:
# =============================================================================
from sklearn.feature_extraction.text import TfidfVectorizer 


# Creating a Tfidf Vectorizer to remove all stop words
tfidf = TfidfVectorizer(stop_words = "english")    # taking stop words from tfid vectorizer 

# Replacing the NaN values in overview column with empty string
df["Category"].isnull().sum()

# Preparing the Tfidf matrix by fitting and transforming
tfidf_matrix = tfidf.fit_transform(df.Category)   #Transform a count matrix to a normalized tf or tf-idf representation
tfidf_matrix.shape #51,34

# From the above matrix we need to find the similarity score.

# =============================================================================
# Cosine Similarity Score:
# =============================================================================

# calculating the dot product using sklearn's linear_kernel()
from sklearn.metrics.pairwise import linear_kernel

# Computing the cosine similarity on Tfidf matrix
cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

# creating a mapping of anime name to index number 
movie_index = pd.Series(df.index, index = df['Titles']).drop_duplicates()

movie_id = movie_index["Sabrina (1995)"]
movie_id

def get_recommendations(Name, topN):    
    movie_id = movie_index[Name]
    
    cosine_scores = list(enumerate(cosine_sim_matrix[movie_id]))
    
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores = sorted(cosine_scores, key=lambda x:x[1], reverse = True)
    
    # Get the scores of top N most similar movies 
    cosine_scores_N = cosine_scores[0: topN+1]
    
    # Getting the movie index 
    movie_idx  =  [i[0] for i in cosine_scores_N]
    movie_scores =  [i[1] for i in cosine_scores_N]
    
    # Similar movies and scores
    similar_movie= pd.DataFrame(columns=["name", "Score"])
    similar_movie["name"] = df.loc[movie_idx, "Titles"]
    similar_movie["Score"] = movie_scores
    similar_movie.reset_index(inplace = True)  
    print (similar_movie)
 

    
# Enter your anime and number of anime's to be recommended
movie_index["Sabrina (1995)"]
get_recommendations("Sudden Death (1995)", topN = 10)
