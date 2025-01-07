import json
import pandas as pd

from flask import Flask, request, render_template, send_from_directory
from rapidfuzz.fuzz import ratio
from rapidfuzz import process
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load datasets 
movies_df = pd.read_csv('../data/raw/movies_data.csv')
processed_movies_df = pd.read_csv("../data/processed/processed_movies.csv")

# Generate models
vec_model = CountVectorizer(max_features=5000, stop_words="english")
vectors = vec_model.fit_transform(processed_movies_df["tags"])
similarity = cosine_similarity(vectors)

# Recommend functions
def get_movie_info_by_id(movie_id):
    movie_row = movies_df[movies_df["id"] == movie_id]
    if movie_row.empty:
        return None

    movie_data = movie_row.iloc[0]
    genres = json.loads(movie_data["genres"])
    genres_list = [genre["name"] for genre in genres]

    print('EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE')
    print(", ".join(genres_list))
    

    return {
        "title": movie_data["title"],
        "genres": ", ".join(genres_list),
        "overview": movie_data["overview"],
        "release_date": movie_data.get("release_date", "Unknown release date"),
        "vote_average": movie_data["vote_average"],
        "runtime": movie_data.get("runtime", "Not available"),
        "poster_url": movie_data["poster"],
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

    print('EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE')
    print(", ".join(genres_list))
    
    return {
        "title": movie_data["title"],
        "genres": ", ".join(genres_list),
        "overview": movie_data["overview"],
        "release_date": movie_data.get("release_date", "Unknown release date"),
        "vote_average": movie_data["vote_average"],
        "runtime": movie_data.get("runtime", "Not available"),
        "poster_url": movie_data["poster"],
    }


def recommend(movie_title, n_recommendations=5):
    best_match = process.extractOne(
        movie_title,
        processed_movies_df["title"].values,
        scorer=ratio,
        score_cutoff=70
    )

    if not best_match:
        return [{"error": "Not movie found in the database"}]

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