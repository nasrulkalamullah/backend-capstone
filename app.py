import csv
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

nltk.download('stopwords')

app = FastAPI()

#loaded_function = pickle.load(open('C:.\model\results.pkl', 'rb'))
with open('./model/results.pkl', 'rb') as file:
    loaded_object = pickle.load(file)

with open('./model/cosine_similarity.pkl', 'rb') as file:
    cosine_sim = pickle.load(file)

with open('./model/vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Load the dataframe
df = pd.read_csv('./dataset/Place Detail (Scored + Keyword 1 & 2 Extracted  + Additional Feature (longlang, contact etc)) + (finished Vectorized).csv')

embeddings_index = {}
with open('glove.6B.100d.txt', 'r', errors="ignore") as f:
    for line in f:
        values = line.split(" ")
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

embedding_dim = 100  # Dimensionality of the word embeddings
embedding_matrix = np.zeros((len(embeddings_index), embedding_dim))
vocab = []

for i, word in enumerate(embeddings_index):
    embedding_vector = embeddings_index[word]
    embedding_matrix[i] = embedding_vector
    vocab.append(word)

search_data = []
for word in vocab:
    search_data.append(embeddings_index[word])

search_data = np.array(search_data)

class SearchRequest(BaseModel):
    user_input: str

def perform_search(user_input):
    def search(query, top_k=10):
        top_words_list = []
        for query_word in query:
            query_tokens = query_word.split()
            query_embedding = np.mean([embeddings_index[token] for token in query_tokens if token in embeddings_index], axis=0)
            similarity_scores = cosine_similarity([query_embedding], search_data)
            similarity_scores = similarity_scores.reshape(-1)
            top_indices = similarity_scores.argsort()[-top_k:][::-1]
            top_words = [vocab[i] for i in top_indices]
            top_words_list.append(top_words)
        return top_words_list

    def input_keyword(user_input):
        stop_words = set(stopwords.words('english'))

        additional_keywords = ["caffe", "place", "coffee", "nan", "cafe"]

        words = user_input.split()

        search_keywords = re.findall(r'\b\w+\b', user_input.lower())

        search_keywords = [word for word in search_keywords if word not in stop_words and word not in additional_keywords]

        return search_keywords

    result = input_keyword(user_input)

    top_words_list = search(result, top_k=10)

    list_of_words = []
    for i, query_word in enumerate(result):
        list_of_words.append(top_words_list[i])
    list_of_words = [word for sublist in list_of_words for word in sublist]

    def search_keywords(csv_file, keywords, column):
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = [row for row in reader]

        row_numbers = []
        for row in rows:
            for keyword in keywords:
                if keyword in row[column]:
                    row_numbers.append(rows.index(row))

        return row_numbers

    csv_file = './dataset/Place Detail (Scored + Keyword 1 & 2 Extracted  + Additional Feature (longlang, contact etc)) (1).csv'
    keywords = list_of_words
    column = 13

    row_numbers = search_keywords(csv_file, keywords, column)
    unique_list = list(set(row_numbers))
    sorted_list = sorted(unique_list)
    Place_list = sorted_list[:20]

    def caffe_result(Place_list):
        columns_to_extract = [0, 2, 4, 5, 14]
        output = []

        with open(csv_file, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row

            def get_data(row_numbers, column_numbers):
                data = []
                for i, row in enumerate(reader):
                    if i + 1 in row_numbers:
                        row_data = [row[col] for col in column_numbers]
                        data.append(row_data)
                return data

            data = get_data(Place_list, columns_to_extract)
            for row in data:
                output.append({
                    "name": row[0],
                    "address": row[1],
                    "rating": float(row[2]),
                    "total_review": int(row[3]),
                    "url_photo": row[4]
                })

        return output

    return caffe_result(Place_list)

@app.get("/caffe")
def data():
   return loaded_object

# API route to get recommendations
@app.get("/recommendations/{item_title}")
# Function to get recommendations
def get_recommendations(item_title: str) -> List[dict]:
    top_n: int = 10
    item_index = df[df['Name'] == item_title].index[0]
    similarity_scores = list(enumerate(cosine_sim[item_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_items = similarity_scores[1:top_n + 1]
    top_item_indices = [i[0] for i in top_items]
    top_item_data = df.iloc[top_item_indices].reset_index(drop=True)

    recommendations = []
    for _, row in top_item_data.iterrows():
        recommendation = {
            'name': row['Name'],
            'address': row['Formatted Address'],
            'rating': row['rating'],
            'total_review': row['total_reviews'],
            'url_photo': row.get('Photo URL', 'N/A')
        }
        recommendations.append(recommendation)
    return recommendations


@app.post('/search')
def search(request: SearchRequest):
    user_input = request.user_input
    # Perform the search
    results = perform_search(user_input)
    return results


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)