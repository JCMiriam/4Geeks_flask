import requests
import pickle
import pandas as pd
import json

from flask import Flask, request, render_template, jsonify, send_from_directory
from rapidfuzz.fuzz import ratio
from rapidfuzz import process

def download_file_from_google_drive(url, destination):
    file_id = url.split('/d/')[1].split('/')[0]
    download_url = f"https://drive.google.com/uc?id={file_id}&export=download"
    
    response = requests.get(download_url, stream=True)
    if response.status_code == 200:
        with open(destination, "wb") as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
    else:
        raise ValueError(f"No se pudo descargar el archivo desde {url}")

download_file_from_google_drive("https://drive.google.com/file/d/1DPXpcXTQXOiH0LneeStjWTVeKivPHGX3/view?usp=sharing", "knn_model.pkl")
download_file_from_google_drive("https://drive.google.com/file/d/1fNePvWCvxMoaD6QaZvBzIOTE7QUoQmmm/view?usp=sharing", "similarity_model.pkl")

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
