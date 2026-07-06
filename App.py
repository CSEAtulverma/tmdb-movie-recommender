# Import Streamlit library to create the web app interface
import streamlit as st

# Import pickle to load our saved dictionary file
import pickle as pkl

# Import pandas to create and manipulate dataframes (tables of data)
import pandas as pd

# Import tools from scikit-learn to calculate movie similarities on the fly
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------------------
# STEP 1: LOAD DATA & CALCULATE SIMILARITY 
# -------------------------------------------------------------

# Load the movies dictionary from the small .pkl file you created in Jupyter
movies_dict = pkl.load(open('movies_dict.pkl','rb'))

# Convert that dictionary back into a Pandas DataFrame (a 2D table)
movies = pd.DataFrame(movies_dict)

# Set up the CountVectorizer to extract the top 5000 words from our movie tags, ignoring common English stop words
cv = CountVectorizer(max_features=5000, stop_words='english')

# Convert the 'tags' column of our dataframe into a mathematical matrix of word counts
# NOTE: If your text column is named something else in your notebook, change 'tags' below to match it!
vector = cv.fit_transform(movies['tags']).toarray()

# Calculate the cosine similarity (the mathematical distance) between every movie vector
# This single line entirely replaces your need for the 180MB similarity.pkl file!
similarity = cosine_similarity(vector)

# -------------------------------------------------------------
# STEP 2: RECOMMENDATION FUNCTION
# -------------------------------------------------------------

# Define a function that takes a movie name and returns 5 similar movies
def recommend(movie):
    
    # Find the index (row number) of the selected movie inside our 'movies' dataframe
    movie_index = movies[movies['title'] == movie].index[0]
    
    # Get the similarity scores for this specific movie against all other movies
    distance = similarity[movie_index]
    
    # Enumerate keeps track of the original index numbers. 
    # We sort them in descending order (reverse=True) based on their similarity score.
    # [1:6] fetches the top 5 most similar movies (skipping index 0, which is the exact same movie you searched for)
    movies_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1:6]

    # Create an empty list to store the final recommended movie names
    recommended_movies = []
    
    # Loop through the top 5 list we just generated
    for i in movies_list:
        # i[0] is the index number. We use it to find the movie title in our dataframe and add it to our list
        recommended_movies.append(movies.iloc[i[0]].title)
        
    # Return the final list of 5 movie names
    return recommended_movies

# -------------------------------------------------------------
# STEP 3: STREAMLIT WEB APP UI
# -------------------------------------------------------------

# Set the main title of your web page
st.title('Movie Recommender System')

# Create a dropdown box on the web page containing all movie titles
selected_movie_name = st.selectbox(
    'Select a movie to get recommendations: ',
    movies['title'].values
)

# Create a clickable button that says 'Recommend'
if st.button('Recommend'):
    
    # If the button is clicked, run our recommend function using the movie from the dropdown
    recommendations = recommend(selected_movie_name)
    
    # Loop through the 5 recommended movies returned by our function
    for i in recommendations:
        # Display each movie name safely on the web page
        st.write(i)