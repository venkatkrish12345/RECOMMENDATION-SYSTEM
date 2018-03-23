
# coding: utf-8

# In[4]:


import pandas as pd


# In[5]:


import numpy as np


# In[6]:


dataframe=pd.read_csv("E:/Datasets/tmdb/tmdb_5000_credits/tmdb_5000_movies.csv")


# In[7]:


dataframe


# In[8]:


C = dataframe['vote_average'].mean()
print(C)


# In[9]:


m=dataframe['vote_count'].quantile(0.90)


# In[10]:


print(m)


# In[11]:


# Filter out all qualified movies into a new DataFrame
q_movies = dataframe.copy().loc[dataframe['vote_count'] >= m]
q_movies.shape


# In[12]:


# Function that computes the weighted rating of each movie
def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)


# In[13]:


q_movies['score'] = q_movies.apply(weighted_rating, axis=1)


# In[14]:


q_movies.head()


# In[15]:


q_movies = q_movies.sort_values('score', ascending=False)


# In[16]:


q_movies.head()


# In[17]:


#Print the top 15 movies
q_movies[['title', 'vote_count', 'vote_average', 'score']].head(15)


# In[18]:


#Content-Based Recommender in Python


# In[19]:


#Print plot overviews of the first 5 movies.
dataframe['overview'].head()


# In[20]:


import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer


# In[21]:


tfidf = TfidfVectorizer(stop_words='english')


# In[22]:


tfidf


# In[23]:


dataframe['overview'] = dataframe['overview'].fillna('')


# In[24]:


dataframe


# In[25]:


#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(dataframe['overview'])


# In[26]:


tfidf_matrix


# In[27]:


tfidf_matrix.shape


# In[28]:


from sklearn.metrics.pairwise import linear_kernel


# In[29]:


# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# In[30]:


cosine_sim


# In[31]:


#Construct a reverse map of indices and movie titles
indices = pd.Series(dataframe.index, index=dataframe['title']).drop_duplicates()


# In[32]:


indices


# In[33]:


# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return dataframe['title'].iloc[movie_indices]


# In[34]:


get_recommendations('The Dark Knight Rises')


# In[35]:


get_recommendations('The Godfather')


# In[36]:


get_recommendations('On The Downlow')


# In[37]:


#Credits, Genres and Keywords Based Recommender
credits=pd.read_csv("E:/Datasets/tmdb/tmdb_5000_credits/tmdb_5000_credits.csv")


# In[38]:


credits.head()


# In[49]:


meta=pd.merge(dataframe,credits,on='movie_id')


# In[51]:


meta.drop([1255,1256,1257,1258,1259,1260,1261,1262,1263,1264])


# In[52]:


from ast import literal_eval


# In[54]:


features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    meta[feature] = meta[feature].apply(literal_eval)

