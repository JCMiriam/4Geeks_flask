import requests
import pickle
import pandas as pd
import json

from flask import Flask, request, render_template, jsonify, send_from_directory
from rapidfuzz.fuzz import ratio
from rapidfuzz import process

knn_url = "https://drive.google.com/uc?id=1DPXpcXTQXOiH0LneeStjWTVeKivPHGX3"
similarity_url = "https://drive.google.com/uc?id=1fNePvWCvxMoaD6QaZvBzIOTE7QUoQmmm"

def download_model(url, filename):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Error on download file: {response.status_code}")
    
    if response.text.startswith("<html>"):
        raise Exception("HTML file downloaded, posible error on the URL.")
    
    with open(filename, 'wb') as f:
        f.write(response.content)

download_model(knn_url, "knn_model.pkl")
download_model(similarity_url, "similarity_model.pkl")

with open("knn_model.pkl", "rb") as f:
    knn = pickle.load(f)

with open("similarity_model.pkl", "rb") as f:
    similarity = pickle.load(f)

movies_df = pd.read_csv('../data/raw/tmdb_5000_movies.csv')
processed_movies_df = pd.read_csv("../data/processed/processed_movies.csv")

app = Flask(__name__)

def get_movie_info_by_id(movie_id):
    movie_row = movies_df[movies_df["id"] == movie_id]
    if movie_row.empty:
        return None
    
    movie_data = movie_row.iloc[0]

    genres = json.loads(movie_data["genres"])
    genres_list = [genre["name"] for genre in genres]

    return {
        "title": movie_data["title"],
        "genres": ", ".join(genres_list),
        "overview": movie_data["overview"],
        "release_date": movie_data["release_date"],
        "vote_average": movie_data["vote_average"],
        "runtime": movie_data.get("runtime", "No disponible"),
    }

def get_movie_info_by_title(title):
    best_match = process.extractOne(
        title, 
        movies_df["title"].values, 
        scorer=ratio,
        score_cutoff=70
    )
    
    if not best_match:
        return None

    best_title = best_match[0]
    movie_row = movies_df[movies_df["title"] == best_title]
    
    movie_data = movie_row.iloc[0]
    genres = json.loads(movie_data["genres"])
    genres_list = [genre["name"] for genre in genres]
    
    return {
        "title": movie_data["title"],
        "genres": ", ".join(genres_list),
        "overview": movie_data["overview"],
        "release_date": movie_data["release_date"],
        "vote_average": movie_data["vote_average"],
        "runtime": movie_data.get("runtime", "No disponible"),
    }

def recommend(movie_title, n_recommendations=5):
    best_match = process.extractOne(
        movie_title,
        processed_movies_df["title"].values,
        scorer=ratio,
        score_cutoff=70
    )

    if not best_match:
        return [{"error": "No se encontró una película similar en la base de datos."}]

    best_title = best_match[0]
    movie_index = processed_movies_df[processed_movies_df["title"] == best_title].index[0]
    
    distances = similarity[movie_index]
    
    similar_movies = sorted(
        list(enumerate(distances)), reverse=True, key=lambda x: x[1]
    )[1:n_recommendations + 1]
    
    recommended_movie_ids = [processed_movies_df.iloc[i[0]]['movie_id'] for i in similar_movies]
    recommendations = [get_movie_info_by_id(movie_id) for movie_id in recommended_movie_ids]
    
    return recommendations

@app.route("/", methods=["GET", "POST"])
def index():
    movie_info = None
    recommendations = None

    if request.method == "POST":
        movie_title = request.form.get("movie_title")
        if movie_title:
            movie_info = get_movie_info_by_title(movie_title)

            if movie_info:
                recommendations = recommend(movie_title)
    
    return render_template("index.html", movie_info=movie_info, recommendations=recommendations)

@app.route('/static/js/<path:filename>')
def serve_js(filename):
    return send_from_directory('static/js', filename, mimetype='application/javascript')

if __name__ == '__main__':
    app.run(debug=True)

if __name__ == "__main__":
    app.run(debug=True)
