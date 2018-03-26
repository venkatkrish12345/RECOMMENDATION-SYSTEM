
# coding: utf-8

# In[2]:


get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from surprise import Reader, Dataset, SVD, evaluate
import warnings; warnings.simplefilter('ignore')


# In[ ]:


#Simple Recommendation
md = pd. read_csv('E:/Datasets/the-movies-dataset (1)/movies_metadata.csv')
md.head()


# In[172]:


md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] 
                                                                   if isinstance(x, list) else [])


# In[ ]:


md


# In[ ]:


#building our overall Top 250 Chart and will define a function to build charts for a particular genre.
vote_counts = md[md['vote_count'].notnull()]['vote_count'].astype('int')
vote_averages = md[md['vote_average'].notnull()]['vote_average'].astype('int')
a=vote_counts.mean()
C = vote_averages.mean()
a,C


# In[ ]:


m = vote_counts.quantile(0.95)
m


# In[ ]:


md['year'] = pd.to_datetime(md['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)


# In[ ]:


md['year']


# In[ ]:


md


# In[173]:


qualified = md[(md['vote_count'] >= m) & (md['vote_count'].notnull()) & (md['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]
qualified['vote_count'] = qualified['vote_count'].astype('int')
qualified['vote_average'] = qualified['vote_average'].astype('int')
qualified.shape


# In[174]:


qualified


# In[175]:


#the average rating for a movie on TMDB is 5.244 on a scale of 10. 2274 Movies qualify to be on our chart.
def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)


# In[176]:


qualified['Scoring'] = qualified.apply(weighted_rating, axis=1)


# In[177]:


qualified


# In[178]:


qualified = qualified.sort_values('Scoring', ascending=False).head(250)


# In[179]:


qualified.head()


# In[180]:


#TOP Movie Ratings
qualified.head(15)


# In[181]:


s = md.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'genre'
gen_md = md.drop('genres', axis=1).join(s)


# In[182]:


gen_md.head()


# In[183]:


def build_chart(genre, percentile=0.85):
    df = gen_md[gen_md['genre'] == genre]
    vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(percentile)
    
    qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity']]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    
    qualified['Scoring'] = qualified.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C), axis=1)
    qualified = qualified.sort_values('Scoring', ascending=False).head(250)
    
    return qualified


# In[184]:


#Displaying the Top 15 Romance Movies 
build_chart('Romance').head(15)


# In[185]:


build_chart('Drama').head(15)


# In[186]:


build_chart('Adventure').head(15)


# In[187]:


#Content Based Recommender
links_small = pd.read_csv('E:/Datasets/the-movies-dataset (1)/links_small.csv')
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')


# In[188]:


links_small.head(15)


# In[189]:


md = md.drop([19730, 29503, 35587])


# In[190]:


md


# In[191]:


#Check EDA Notebook for how and why I got these indices.
md['id'] = md['id'].astype('int')


# In[192]:


md


# In[193]:


smd = md[md['id'].isin(links_small)]
smd


# In[194]:


#Movie Description Based Recommender
smd['tagline'] = smd['tagline'].fillna('')
smd['description'] = smd['overview'] + smd['tagline']
smd['description'] = smd['description'].fillna('')
smd['description'].head()


# In[195]:


##Construct the required TF-IDF matrix by fitting and transforming the data
tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(smd['description'])


# In[196]:


tfidf_matrix.shape


# In[197]:


#Cosine Similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# In[198]:


cosine_sim


# In[199]:


cosine_sim[0]


# In[200]:



smd


# In[201]:


indices = pd.Series(smd.index, index=smd['title']).drop_duplicates()


# In[202]:


def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return smd['title'].iloc[movie_indices]


# In[203]:


get_recommendations('Jumanji').head(10)


# In[204]:


#content Based Recommendations Based on Crew and Keyword
credits = pd.read_csv('E:/Datasets/the-movies-dataset (1)/credits.csv')
keywords = pd.read_csv('E:/Datasets/the-movies-dataset (1)/keywords.csv')


# In[205]:


credits.head()


# In[206]:


keywords.head()


# In[207]:


keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
md['id'] = md['id'].astype('int')


# In[208]:


md.shape


# In[209]:


md = md.merge(credits, on='id')
md = md.merge(keywords, on='id')


# In[210]:


smd = md[md['id'].isin(links_small)]
smd.shape


# In[211]:


#Cast And Crew
smd['cast'] = smd['cast'].apply(literal_eval)
smd['crew'] = smd['crew'].apply(literal_eval)
smd['keywords'] = smd['keywords'].apply(literal_eval)
smd['cast_size'] = smd['cast'].apply(lambda x: len(x))
smd['crew_size'] = smd['crew'].apply(lambda x: len(x))


# In[ ]:


def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


# In[ ]:


smd['director'] = smd['crew'].apply(get_director)


# In[ ]:


smd.head()


# In[ ]:


smd['cast'] = smd['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
smd['cast'] = smd['cast'].apply(lambda x: x[:3] if len(x) >=3 else x)


# In[ ]:


smd['cast'].head()


# In[115]:


smd['keywords'] = smd['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])


# In[119]:


smd['keywords'].head()


# In[116]:


smd['cast'] = smd['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])


# In[117]:


smd['cast'].head()


# In[120]:


smd['director'] = smd['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
smd['director'] = smd['director'].apply(lambda x: [x,x])


# In[121]:


smd['director'].head()


# In[122]:


#KeyWords
s = smd.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'keyword'


# In[123]:


s = s.value_counts()
s[:5]


# In[124]:


s = s[s > 1]


# In[129]:


s


# In[130]:


stemmer = SnowballStemmer('english')
stemmer.stem('dogs')


# In[131]:


def filter_keywords(x):
    words = []
    for i in x:
        if i in s:
            words.append(i)
    return words


# In[132]:


smd['keywords'] = smd['keywords'].apply(filter_keywords)
smd['keywords'] = smd['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
smd['keywords'] = smd['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])


# In[134]:


smd['keywords'].head()


# In[133]:


smd['soup'] = smd['keywords'] + smd['cast'] + smd['director'] + smd['genres']
smd['soup'] = smd['soup'].apply(lambda x: ' '.join(x))


# In[135]:


smd['soup'].head()


# In[136]:


#count_Matrix
count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
count_matrix = count.fit_transform(smd['soup'])


# In[137]:


#Cosine Similarity
cosine_sim = cosine_similarity(count_matrix, count_matrix)


# In[140]:


cosine_sim


# In[139]:


cosine_sim[0]


# In[141]:


smd = smd.reset_index()
titles = smd['title']
indices = pd.Series(smd.index, index=smd['title'])


# In[142]:


get_recommendations('The Dark Knight').head(10)


# In[143]:


get_recommendations('Pulp Fiction').head(10)


# In[145]:


#Popularity and Ranking Based Recommendation
def improved_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]
    
    movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year']]
    vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(0.60)
    qualified = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    qualified['Scoring'] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values('Scoring', ascending=False).head(10)
    return qualified


# In[146]:


improved_recommendations('The Dark Knight')


# In[147]:


improved_recommendations('Pulp Fiction')


# In[149]:


#Collabrative Filtering
ratings=pd.read_csv("E:/Datasets/the-movies-dataset (1)/ratings.csv")


# In[156]:


ratings.head()


# In[157]:


from sklearn.model_selection import KFold


# In[162]:


data = (ratings[['userId', 'movieId', 'rating']])
kf = KFold(n_splits=5)


# In[163]:


kf.get_n_splits(data)


# In[164]:


print(kf)

