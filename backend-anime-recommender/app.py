from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import html
import pickle
import requests
from fuzzywuzzy import process
import os
import re

app = Flask(__name__)
CORS(app)

print("Loading ML model & data...")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

try:
    with open(os.path.join(MODEL_DIR, "similarity_matrix.pkl"), "rb") as f:
        similarity_matrix = pickle.load(f)

    with open(os.path.join(MODEL_DIR, "Tfidf_vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)

    df = pd.read_pickle(os.path.join(MODEL_DIR, "anime_data.pkl"))

    all_titles = df["Name"].tolist()
    print("Successfully loaded all files")

except FileNotFoundError:
    print("Error: ML model files not found. Please run train_model.py first.")
    exit()


#!HELPER FUNCTIONS 

def clean_title(title):
    """
    Applies the same normalization as data_cleaning.py so incoming
    MAL/AniList titles match the format stored in our database.
    """
    title = html.unescape(title)
    title = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', title)
    title = title.strip().lower()
    return title


def find_indices_fuzzy(titles):                          
    """
    Matches a list of external titles (from MAL/AniList) to our local database
    using fuzzy string matching. Returns a list of unique positional indices.
    """
    matched_indices = []
    for title in titles:
        cleaned = clean_title(title)
        result = process.extractOne(cleaned, all_titles)
        if result is None:                               
            continue
        match, score = result[0], result[1]
        if score >= 85:
            idx = all_titles.index(match)
            matched_indices.append(idx)
    return list(set(matched_indices))


def get_recommendations_from_indices(watched_indices, n=10):  
    """
    Core recommendation engine.
    Calculates the average similarity score across all watched anime,
    then returns the top N anime the user hasn't seen yet.
    """
    if not watched_indices:
        return []

    avg_sim = similarity_matrix[watched_indices].mean(axis=0)
    sim_scores = sorted(enumerate(avg_sim), key=lambda x: x[1], reverse=True)

    recs = []
    for idx, score in sim_scores:
        if idx not in watched_indices:
            anime = df.iloc[idx].to_dict()
            anime["similarity_score"] = round(float(score), 4)
            recs.append(anime)
        if len(recs) >= n:                               
            break

    return recs


def fetch_all_mal_pages(username):
    """
    Fetches ALL completed anime from a MAL user by paginating through
    the Jikan v4 API (each page returns up to 300 entries).
    """
    titles = []
    page = 1
    while True:
        url = f"https://api.jikan.moe/v4/users/{username}/animelist?status=completed&page={page}"  
        res = requests.get(url, timeout=10)              
        res.raise_for_status()
        data = res.json()

        entries = data.get("data", [])
        if not entries:
            break

        for item in entries:
            titles.append(item["anime"]["title"])

        pagination = data.get("pagination", {})
        if not pagination.get("has_next_page", False):
            break

        page += 1

    return titles


# !API ENDPOINTS 

@app.route('/api/test', methods=['GET'])

def test():
    return jsonify({"message": "Backend is working!"})


@app.route('/api/recommend', methods=['GET'])
def recommend_by_title():
    """
    GET /api/recommend?title=Naruto&n=10
    Returns recommendations for a single anime title.
    """
    title = request.args.get('title', '').strip()
    n = int(request.args.get('n', 10))

    if not title:
        return jsonify({"error": "Query parameter 'title' is required"}), 400  

    result = process.extractOne(clean_title(title), all_titles)  
    if result is None or result[1] < 85:
        return jsonify({"error": f"Could not find '{title}' in the database"}), 404

    matched_title, score = result[0], result[1]
    idx = all_titles.index(matched_title)

    recs = get_recommendations_from_indices([idx], n=n)
    return jsonify({
        "matched_title": matched_title,
        "match_score": score,
        "recommendations": recs
    })


@app.route('/api/anilist', methods=['GET'])
def sync_anilist():
    """
    GET /api/anilist?username=yourname&n=10
    Syncs a user's completed anime from AniList and returns recommendations.
    """
    username = request.args.get('username', '').strip()
    n = int(request.args.get('n', 10))

    if not username:
        return jsonify({"error": "Query parameter 'username' is required"}), 400

    query = '''
    query ($name: String) {
    MediaListCollection(userName: $name, type: ANIME, status: COMPLETED) {
        lists {
        entries {
            media {
            title {
                romaji
                english
            }
            }
        }
    }
    }
    }
    '''
    try:
        response = requests.post(                        
            'https://graphql.anilist.co',                
            json={'query': query, 'variables': {'name': username}},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        media_collection = (data.get("data") or {}).get("MediaListCollection")
        if not media_collection:
            return jsonify({"error": f"AniList user '{username}' not found or list is private"}), 404

        external_titles = []
        for list_obj in media_collection["lists"]:
            for entry in list_obj["entries"]:
                t = entry["media"]["title"]
                title = t.get("romaji") or t.get("english")
                if title:
                    external_titles.append(title)

        if not external_titles:
            return jsonify({"error": "No completed anime found for this user"}), 404

        indices = find_indices_fuzzy(external_titles)

        if not indices:
            return jsonify({"error": "None of your watched anime matched our database"}), 404

        return jsonify({
            "titles_fetched": len(external_titles),
            "titles_matched": len(indices),
            "recommendations": get_recommendations_from_indices(indices, n=n)
        })

    except requests.exceptions.Timeout:
        return jsonify({"error": "AniList API timed out. Try again later."}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500          # FIX 11: duplicate except blocks removed, anilist exceptions were after mal's


@app.route('/api/mal', methods=['GET'])
def sync_mal():
    username = request.args.get('username', '').strip()
    n = int(request.args.get('n', 10))

    if not username:
        return jsonify({"error": "Query parameter 'username' is required"}), 400

    try:
        external_titles = fetch_all_mal_pages(username)

        if not external_titles:
            return jsonify({"error": f"No completed anime found for MAL user '{username}'"}), 404

        indices = find_indices_fuzzy(external_titles)

        if not indices:
            return jsonify({"error": "None of your watched anime matched our database"}), 404

        return jsonify({
            "titles_fetched": len(external_titles),
            "titles_matched": len(indices),
            "recommendations": get_recommendations_from_indices(indices, n=n)
        })

    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return jsonify({"error": f"MAL user '{username}' not found"}), 404
        return jsonify({"error": str(e)}), 500
    except requests.exceptions.Timeout:
        return jsonify({"error": "Jikan API timed out. Try again later."}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)