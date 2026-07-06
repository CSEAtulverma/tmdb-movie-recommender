import streamlit as st
import pickle as pkl
import pandas as pd

movies_dict = pkl.load(open('E:\Movie Recommender System\movies_dict.pkl','rb'))

movies = pd.DataFrame(movies_dict)

similarity = pkl.load(open('E:\Movie Recommender System\similarity.pkl','rb'))

def recommend(movie):
    movie_index = movie[movie['title'] == movie].index[0]
    distance = similarity[movie_index]
    movies_list = sorted(list(enumerate(distance)),reverse=True , key = lambda x:x[1])[1:6]

    recommended_movies = []
    for i in movies_list:
        recommended_movies.append(movies.iloc[i[0]].title)
    return recommended_movies


st.title('Movie Recommender System')

selected_movie_name = st.selectbox(
    'Select a movie to get recommendations: ',
    movies['title'].values
    )

if st.button('Recommend'):

    recommendations = recommend(selected_movie_name)
    for i in recommendations:
        st.write(i)