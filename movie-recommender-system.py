#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[3]:


credits.head()


# In[4]:


credits.info()


# In[5]:


movies.head()


# In[6]:


movies.info()


# In[7]:


movies.merge(credits, on = 'title')


# In[8]:


movies = movies.merge(credits, on = 'title')


# In[9]:


movies.info()


# In[10]:


# we use tags that will be used for recommendation
# tags can be genres, id, title, keywords, overview, cast, crew
movies = movies[['movie_id', 'title', 'genres', 'overview', 'keywords', 'cast', 'crew']]


# In[11]:


movies.info()


# In[12]:


movies.head()


# In[13]:


# preprocessing - remove missing / duplicate data


# In[14]:


movies.isnull().sum()


# In[15]:


# dropping missing data present in overview
movies.dropna(inplace=True)


# In[16]:


movies.isnull().sum()


# In[17]:


movies.duplicated().sum()


# In[18]:


movies.iloc[0].genres # this is a dictionary but we want a list


# In[19]:


import ast
def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[20]:


movies['genres'].apply(convert)


# In[21]:


movies['genres'] = movies['genres'].apply(convert)


# In[22]:


movies.head()


# In[23]:


movies['keywords'].apply(convert)


# In[24]:


movies['keywords'] = movies['keywords'].apply(convert)


# In[25]:


movies.head()


# In[26]:


def convert3(obj):
    L=[]
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
        else:
            break
    return L


# In[27]:


movies['cast'].apply(convert3)


# In[28]:


movies['cast'] = movies['cast'].apply(convert3)


# In[29]:


movies.head()


# In[30]:


def fetch_director(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L


# In[31]:


movies['crew'].apply(fetch_director)


# In[32]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[33]:


movies.head()


# In[34]:


#  overviiew is a string
movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[35]:


movies.head()


# In[36]:


# on genres, keywords, cast, crew - apply transformation
# 'Sam Mendes' changed to 'SamMendes' without space
# else Sam Mendes becomes two entities, Sam one Mendes another
# if we remove the space, they're one entity
movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])


# In[37]:


movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])


# In[38]:


movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[39]:


movies.head()


# In[40]:


movies['tags'] = movies['overview'] + movies['genres']+ movies['keywords'] + movies['cast'] + movies['crew']


# this tag is created as a column that's a concentation of genres, overview, keywords, cast, crew

# In[41]:


movies.head()


# In[42]:


new_df = movies[['movie_id', 'title', 'tags']]


# In[43]:


new_df


# In[44]:


# converting list to string
new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))


# In[45]:


new_df.head()


# In[46]:


new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())


# In[47]:


new_df.head()


# In[48]:


# Data Preprocessing done, now Vectorization
# text needs to be converted to vectors - bag of words, tfidf, word2vec
# we are using bag of words text vectorization technique 
# s1 : all tags concatenated to form a large text
# s2 : calculate frequency of top 5000 words (less words, best performance)
# s3 : during vectorization, we won't use stop words: are, in, is, to, etc
# s4 : we'll use class sklearn.feature_extraction.text.CountVectorizer


# In[49]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english') # max = no of words we'll take


# In[50]:


cv.fit_transform(new_df['tags']).toarray()


# In[51]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[52]:


vectors


# In[53]:


vectors[0]


# In[54]:


cv.get_feature_names_out()


# In[55]:


# stemming we apply it on tags to convert words like loves, loved, love to love 


# In[56]:


get_ipython().system('pip install nltk')


# In[57]:


import nltk


# In[58]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[59]:


def stem(text):
    y = []
    
    for i in text.split():
        y.append(ps.stem(i))
    
    return " ".join(y)


# In[60]:


new_df['tags'].apply(stem)


# In[61]:


new_df['tags'] = new_df['tags'].apply(stem)


# In[62]:


# again we import sklearn and make vectors
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english') # max = no of words we'll take


# In[63]:


cv.fit_transform(new_df['tags']).toarray()


# In[64]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[65]:


vectors


# In[66]:


vectors[0]


# In[67]:


cv.get_feature_names_out()


# In[68]:


len(cv.get_feature_names_out())


# In[69]:


# we calculate cosine distance bw movies (dist more, similarilities lesser)
# if angle less then similar
# euclidian dist is not a good measure for high dimensional data


# In[70]:


from sklearn.metrics.pairwise import cosine_similarity


# In[71]:


cosine_similarity(vectors).shape


# In[72]:


similarity = cosine_similarity(vectors)


# In[73]:


similarity


# In[74]:


similarity[1]


# In[75]:


sorted(list(enumerate(similarity[0])), reverse = True, key = lambda x:x[1])[1:6]


# In[76]:


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse = True, key = lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# In[77]:


recommend('Batman Begins')


# In[78]:


recommend('Avatar')


# In[79]:


recommend('King Kong')


# In[80]:


recommend('Titanic')


# In[82]:


recommend('My All American')


# In[83]:


recommend('My All American')

