import openai
import ast
import re
import pandas as pd
import json
import pickle

from nltk.corpus import stopwords
from tqdm import tqdm

df = pd.read_csv('sample30.csv',sep=",")

def decontracted(phrase):
    contractions = {
        "won't": "will not", "can't": "can not", "n't": " not", "'re": " are",
        "'s": " is", "'d": " would", "'ll": " will", "'ve": " have", "'m": " am"
    }
    for contraction, expansion in contractions.items():
        phrase = re.sub(re.escape(contraction), expansion, phrase)
    return phrase

# Function to clean up text
def clean_text(text):
    text = decontracted(text)  # Decontract
    text = text.replace('\\r', ' ').replace('\\"', ' ').replace('\\n', ' ')  # Remove escape characters
    text = re.sub('[^A-Za-z0-9]+', ' ', text)  # Remove non-alphanumeric characters
    return text

# Use NLTK stopwords
stopwords_set = set(stopwords.words('english'))  # Get stopwords for English

# Function to preprocess essays
def preprocess_essays(essays):
    preprocessed_essays = []
    for sentence in tqdm(essays):
        sentence = clean_text(sentence)  # Clean the text
        # Remove stopwords
        sentence = ' '.join(word for word in sentence.split() if word.lower() not in stopwords_set)
        preprocessed_essays.append(sentence.lower().strip())
    return preprocessed_essays

def textpreprocessing(text):
    if isinstance(text, str):
        return preprocess_essays([text])[0]  # wrap in list, return first item
    return preprocess_essays(text)

tfidf_transformer = pickle.load(open('tfidf.pkl','rb')) # TFIDF Transformer
model1 = pickle.load(open('model_LR.pkl','rb'))                          # Classification Model
recommend_matrix = pickle.load(open('user_final_rating.pkl','rb'))

def model2(text):
    tfidf_vector = tfidf_transformer.transform(text)
    output = model1.predict(tfidf_vector)
    return output

def recommendtop20_products(userid):
    recommend_matrix = pickle.load(open('user_final_rating.pkl', 'rb'))
    product_list = pd.DataFrame(recommend_matrix.loc[userid].sort_values(ascending=False)[0:20])
    product_frame = df[df.name.isin(product_list.index.tolist())]
    output_df = product_frame[['name', 'reviews_text']]
    output_df['processed_text'] = output_df['reviews_text'].map(lambda text: textpreprocessing(text))
    print("test")
    output_df['predicted_sentiment'] = model2(output_df['processed_text'].tolist())
    print(output_df)
    return output_df



def recommendtop5_products(top20records):
    output_products=''
    total_product = df.groupby(['name']).agg('count')
    rec_df = df.groupby(['name', 'predicted_sentiment']).agg('count')
    rec_df = rec_df.reset_index()
    merge_df = pd.merge(rec_df, total_product['reviews_text'], on='name')
    merge_df['%percentage'] = (merge_df['reviews_text_x'] / merge_df['reviews_text_y']) * 100
    merge_df = merge_df.sort_values(ascending=False, by='%percentage')
    output_products = pd.DataFrame(merge_df['name'][merge_df['predicted_sentiment'] == 1][:5])
    return output_products
