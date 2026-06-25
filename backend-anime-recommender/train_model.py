import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

#* Convert text to numbers using TF-IDF
''' TF-IDF (Term Frequency-Inverse Document Frequency) is a statistical measure that evaluates
how relevant a word is to a document in a collection of documents. It helps filter out
common words (like 'the', 'a') and highlight unique, important words.
stop_words="english" removes common English words.
min_df=3 ignores words that appear in fewer than 3 documents, helping to reduce noise.'''

print("Training the data it will take some time.....")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

data_path = os.path.join(BASE_DIR, "data", "anime-data-cleaned.csv")

df = pd.read_csv(data_path)

vectorizer = TfidfVectorizer(stop_words="english",min_df=3)
feature_matrix = vectorizer.fit_transform(df["combined_features"])

#* Calculate Similarity using Cosine Similarity
# Cosine Similarity measures the cosine of the angle between two non-zero vectors.
# The closer the cosine value is to 1, the more similar the items (anime) are.
# This matrix tells us how similar every anime is to every other anime

similarity_matrix = cosine_similarity(feature_matrix)

print("saving all the model components")

'''We use 'pickle' to serialize (convert Python objects into a byte stream) the trained model components.
This will allows my Flask API to load them quickly without re-training every time it starts.'''

with open(os.path.join(MODEL_DIR,"similarity_matrix.pkl"),"wb") as f:
    pickle.dump(similarity_matrix,f)

with open(os.path.join(MODEL_DIR,"Tfidf_vectorizer.pkl"),"wb") as f:
    pickle.dump(vectorizer,f)
    
df[["anime_id", "Name", "Genres", "Synopsis", "Score", "Image_URL"]].to_pickle(os.path.join(MODEL_DIR,"anime_data.pkl"))

print("Done! Model training and saving complete.")
    









