from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import re
import numpy as np
import base64
import requests
from functools import lru_cache
from typing import List

from flask_cors import CORS
app = Flask(__name__)
CORS(app)
CLIENT_ID = "5d8a545881c448a5a2dc01f11b8ad1c9"  
CLIENT_SECRET = "f19a4876084141ddafe1fcc2fdb3c472" 


# Firebase setup
cred = credentials.Certificate("./firebase.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

API_KEY = 'AIzaSyAcxRCnVwccCV9oMsmJjM85UWLq2JUmOb0'
EXCLUDED_KEYWORDS = ['mix', 'playlist', 'compilation', 'non-stop']

def get_favorite_songs(user_id):
    try:
        favorite_ref = db.collection("favorite").document(user_id)
        favorite_doc = favorite_ref.get()

        if favorite_doc.exists:
            return favorite_doc.to_dict()
        else:
            return {}
    except Exception as e:
        print("Lỗi khi lấy bài hát yêu thích:", e)
        return {}

def preprocess_favorites(favorite_data):
    return list(favorite_data.values())

def extract_keywords(favorite_songs, max_keywords=20):
    text_data = " ".join(
        [
            f"{' '.join(song.get('genres', []))}"
            for song in favorite_songs
        ]
    )
    words = [
        word.lower()
        for word in re.findall(r'\b\w+\b', text_data)
        if word.lower() not in ENGLISH_STOP_WORDS and len(word) > 2
    ]
    word_counts = Counter(words)
    keywords = [word for word, _ in word_counts.most_common(max_keywords)]
    return keywords


def parse_duration_iso8601(duration: str) -> int:
    match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration)
    if not match:
        return 0

    hours = int(match.group(1) or '0')
    minutes = int(match.group(2) or '0')
    seconds = int(match.group(3) or '0')

    return hours * 3600 + minutes * 60 + seconds

def fetch_videos_by_keyword(keyword: str, max_items: int = 50):
    search_url = f'https://www.googleapis.com/youtube/v3/search?part=snippet&q={requests.utils.quote(keyword)}&type=video&maxResults=50&key={API_KEY}'
    all_items = []
    next_page_token = None

    try:
        while True:
            if next_page_token:
                search_url = f'https://www.googleapis.com/youtube/v3/search?part=snippet&q={requests.utils.quote(keyword)}&type=video&maxResults=50&pageToken={next_page_token}&key={API_KEY}'
            response = requests.get(search_url)
            response.raise_for_status()
            data = response.json()
            items = data.get('items', [])
            all_items.extend(items)
            next_page_token = data.get('nextPageToken')
            if not next_page_token or len(all_items) >= max_items:
                break
        return all_items[:max_items]
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        return []

# get genres
def get_spotify_access_token(client_id: str, client_secret: str) -> str:
    url = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization": "Basic " + base64.b64encode(f"{client_id}:{client_secret}".encode()).decode(),
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data = {"grant_type": "client_credentials"}
    response = requests.post(url, headers=headers, data=data)
    response.raise_for_status()
    return response.json()["access_token"]

def retry_request(func, retries=3):
    for _ in range(retries):
        try:
            return func()
        except requests.RequestException as e:
            error = e
    raise error

@lru_cache(maxsize=100)  
def get_song_genre(song_name: str, client_id: str, client_secret: str) -> List[str]:
    access_token = retry_request(lambda: get_spotify_access_token(client_id, client_secret))

    search_url = "https://api.spotify.com/v1/search"
    search_params = {"q": song_name, "type": "track", "limit": 1}
    search_headers = {"Authorization": f"Bearer {access_token}"}

    search_response = retry_request(lambda: requests.get(search_url, headers=search_headers, params=search_params))
    search_response.raise_for_status()
    search_data = search_response.json()

    if search_data["tracks"]["items"]:
        track = search_data["tracks"]["items"][0]
        artist_id = track["artists"][0]["id"]

        artist_url = f"https://api.spotify.com/v1/artists/{artist_id}"
        artist_headers = {"Authorization": f"Bearer {access_token}"}

        artist_response = retry_request(lambda: requests.get(artist_url, headers=artist_headers))
        artist_response.raise_for_status()
        artist_data = artist_response.json()

        return artist_data.get("genres", ["Unknown Genre"])

    return ["Unknown Genre"]

def filter_music_videos(videos):
    video_ids = [video['id']['videoId'] for video in videos if video['id']['kind'] == 'youtube#video']
    details_url = f'https://www.googleapis.com/youtube/v3/videos?part=contentDetails,snippet&id={",".join(video_ids)}&key={API_KEY}'
    
    try:
        response = requests.get(details_url)
        response.raise_for_status()
        video_data = response.json()
        filtered_videos = []
        excluded_keywords = ['mix', 'playlist', 'compilation', 'non-stop', 'audio', 'remix', 'track']
        
        for item in video_data.get('items', []):
            duration = parse_duration_iso8601(item['contentDetails']['duration'])
            title = item['snippet']['title']
            description = item['snippet']['description']
            
            if 120 <= duration <= 420 and not any(keyword.lower() in title.lower() or keyword.lower() in description.lower() for keyword in excluded_keywords):
                filtered_videos.append({
                    'name': title,
                    'artists': item['snippet']['channelTitle'],
                    'videoUrl': f'https://music.youtube.com/watch?v={item["id"]}',
                    'image': item['snippet']['thumbnails']['high']['url'],
                    'id': item['id'],
                    'genres': get_song_genre(item['snippet']['channelTitle'], CLIENT_ID, CLIENT_SECRET)
                })
                
        return filtered_videos
    except requests.RequestException as e:
        print(f"Error fetching video details: {e}")
        return []


def get_music_list_by_keyword(keyword: str, max_items: int = 50):
    try:
        search_data = fetch_videos_by_keyword(keyword, max_items)
        if not search_data:
            return []
        return filter_music_videos(search_data)
    except Exception as e:
        print(f"Error getting music list: {e}")
        return []

def create_feature_string(song):
    return f"{song['artists']} {song['genres'] if 'genres' in song else ''}"

def get_song_recommendations(favorite_songs, all_songs, top_n=20):
    all_songs_features = [create_feature_string(song) for song in all_songs]
    vectorizer = TfidfVectorizer(stop_words='english')
    song_vectors = vectorizer.fit_transform(all_songs_features)
    favorite_songs_features = [create_feature_string(song) for song in favorite_songs]
    favorite_song_vectors = vectorizer.transform(favorite_songs_features)
    similarities = cosine_similarity(favorite_song_vectors, song_vectors)
    average_similarities = np.mean(similarities, axis=0)
    similar_song_indices = np.argsort(average_similarities)[::-1][:top_n]
    all_songs = list(all_songs)
    recommended_songs = [all_songs[i] for i in similar_song_indices]
    return recommended_songs

@app.route('/recommend_songs', methods=['POST'])
def recommend_songs():
    print("Yêu cầu đã được nhận")
    print(request)
    try:
        user_id = request.json.get('userId')
        if not user_id:
            return jsonify({"error": "UserId is required"}), 400

        favorite_data = get_favorite_songs(user_id)
        if not favorite_data:
            return jsonify({"message": "No favorite songs found"}), 404

        favorite_songs = preprocess_favorites(favorite_data)
        keywords = extract_keywords(favorite_songs)

        print(keywords)

        all_songs_from_keywords = []
        for keyword in keywords:
            songs = get_music_list_by_keyword(keyword, max_items=50)
            if songs:
                all_songs_from_keywords.extend(songs)

        unique_songs = {song['id']: song for song in all_songs_from_keywords}.values()
        recommended_songs = get_song_recommendations(favorite_songs, unique_songs, top_n=20)
        print(recommended_songs)
        return jsonify(recommended_songs)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
