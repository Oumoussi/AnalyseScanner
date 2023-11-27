# Importation des bibliothèques nécessaires
import streamlit as st
import os
from io import BytesIO
from PIL import Image
import streamlit as st
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import pickle
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB7, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import math

# Chargement du tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Chargement des embeddings GloVe dans un dictionnaire
glove_vectors = {}
with open('glove.6B.300d.txt', 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        glove_vectors[word] = vector

# Création de la matrice d'embedding
vocab_size = len(tokenizer.word_index) + 1
embedding_matrix = np.zeros((vocab_size, 300))

for word, i in tokenizer.word_index.items():
    vec = glove_vectors.get(word)
    if vec is not None:
        embedding_matrix[i] = vec
    else:
        continue

embedding_dim = 300
units = 256
vocab_size = 1611
max_doc_length_x = 138

# Définition de la classe d'attention de Bahdanau
class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def __call__(self, features, hidden):

        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # La forme du vecteur contexte après la somme == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

# Définition de la classe de l'encodeur
class Encoder(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(Encoder, self).__init__()
        self.fc = Dense(embedding_dim, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=45),
                  name="encoder_output_layer")

    def call(self, x):  # Changer __call__ en call
        encoder_concat = Concatenate()([x[:, 0], x[:, 1]])
        x = self.fc(encoder_concat)
        x = tf.nn.relu(x)
        return x

# Création de l'objet Encoder
embedding_dim = 300
encoder_loaded = Encoder(embedding_dim)
dummy_encoder_input = tf.random.uniform((2, 32, 80))
encoder_loaded(dummy_encoder_input)
encoder_loaded.load_weights("encoder_weights.h5")

# Définition de la classe du décodeur
class Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(Decoder, self).__init__()
        self.units = units
        self.embedding = Embedding(vocab_size, output_dim=300, mask_zero=True, weights=[embedding_matrix])
        self.lstm = LSTM(self.units, activation='tanh', recurrent_activation='sigmoid', use_bias=True,
                         return_sequences=True, return_state=True,
                         recurrent_initializer=tf.keras.initializers.glorot_uniform(seed=45))
        self.dense1 = Dense(units, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=45))
        self.dense2 = Dense(vocab_size, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=45))
        self.attention = BahdanauAttention(self.units)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.int32),
                                  tf.TensorSpec(shape=[None, 32, 300], dtype=tf.float32),
                                  tf.TensorSpec(shape=[None, 256], dtype=tf.float32)])
    def call(self, inputs, features, hidden):
        context_vector, attention_weights = self.attention(features, hidden)

        x1 = self.embedding(inputs)

        x1 = tf.concat([x1, tf.expand_dims(context_vector, 1)], axis=-1)
        output, state, _ = self.lstm(x1)
        x1 = self.dense1(output)
        x1 = tf.reshape(x1, (-1, x1.shape[2]))
        x1 = self.dense2(output)
        return x1, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))

# Création de l'objet Decoder
decoder_loaded = Decoder(embedding_dim, units, vocab_size)
decoder_input_shapes_and_types = [
    (1, 1, tf.int32),
    (1, 32, 300, tf.float32),
    (1, 256, tf.float32)
]

dummy_dec_input = tf.random.uniform((1, 1), minval=0, maxval=vocab_size, dtype=tf.int32)
dummy_img_features = tf.random.uniform((1, 32, 300))
dummy_hidden_state = tf.random.uniform((1, 256))

dummy_inputs = [dummy_dec_input, dummy_img_features, dummy_hidden_state]
decoder_loaded(*dummy_inputs)
decoder_loaded.load_weights("decoder_weights.h5")

# Fonction pour convertir une image en tenseur
def convert_to_tensor_ofimage(image_path, model_image):
    image_vec = tf.io.read_file(image_path)
    image_vec = tf.image.decode_jpeg(image_vec, channels=3)
    image_vec = tf.image.resize(image_vec, (299, 299))
    image_vec = preprocess_input(image_vec)
    features_image = features_image_model(tf.constant(image_vec)[None, :])
    return features_image

# Modèle d'image EfficientNetB7
model_image = EfficientNetB7(include_top=False, weights='imagenet', classifier_activation='softmax', pooling='avg')
effiencient_in_layer = model_image.input
efficient_out_layer = model_image.layers[-1].output
features_image_model = Model(effiencient_in_layer, efficient_out_layer)

# Fonction de recherche de faisceau
def beam_search(tensor, beam_width=3, top_captions=5):
    hidden = tf.zeros((1, units))
    k1 = tensor[0]
    k2 = tensor[1]
    k1 = tf.reshape(k1, [32, 80])
    k2 = tf.reshape(k2, [32, 80])
    img_tensor = tf.convert_to_tensor([k1, k2])

    image_features = tf.constant(img_tensor)[None, :]
    features_val = encoder_loaded(image_features)
    start = [tokenizer.word_index["<start>"]]
    dec_word = [[start, 0.0]]
    finished_cap = []

    # Recherche de faisceau
    while len(dec_word[0][0]) < max_doc_length_x:
        temp = []
        for s in dec_word:
            predictions, hidden, _ = decoder_loaded(tf.cast(tf.expand_dims([s[0][-1]], 0), tf.int32), features_val, hidden)
            predictions = tf.reshape(predictions, [predictions.shape[0], predictions.shape[2]])
            word_preds = np.argsort(predictions[0])[-beam_width:]

            for w in word_preds:
                cap, score = s
                candidates = [cap + [w], score - math.log(predictions[0][w])]
                temp.append(candidates)

        dec_word = sorted(temp, key=lambda l: l[1])[:beam_width]

        # Vérification des légendes terminées
        new_cap = []
        for cap, score in dec_word:
            if cap[-1] == tokenizer.word_index['<end>']:
                finished_cap.append([cap, score / len(cap)])
            else:
                new_cap.append([cap, score])
        dec_word = new_cap

        if not dec_word:
            break

    # Sélection des légendes supérieures
    finished_cap.sort(key=lambda l: l[1], reverse=True)
    top_captions = finished_cap[:top_captions]

    # Conversion des indices en mots
    result = []
    for cap, _ in top_captions:
        sentence = [tokenizer.index_word[word] for word in cap if word not in [tokenizer.word_index['<start>'], tokenizer.word_index['<end>']]]
        result.append(' '.join(sentence))

    return result

# Fonction pour générer des prédictions avec recherche de faisceau
def beam_predictions(img1, img2, num_captions):
    tensor1 = convert_to_tensor_ofimage(img1, features_image_model)
    tensor2 = convert_to_tensor_ofimage(img2, features_image_model)
    tensor = [tensor1, tensor2]
    width = 3
    captions = beam_search(tensor, width, num_captions)
    return captions

# Fonction pour sauvegarder le fichier téléchargé
def save_uploaded_file(directory, img, img_name):
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(os.path.join(directory, img_name), "wb") as f:
        f.write(img.getbuffer())
    return os.path.join(directory, img_name)

# Fonction principale
def main():
    st.title("Medical analyse")

    # Télécharger les images
    st.header("Téléchargez vos images")
    image1 = st.file_uploader(" Téléchargez l'image frontale ", type=["png", "jpg", "jpeg"])
    image2 = st.file_uploader(" Téléchargez l'image latérale", type=["png", "jpg", "jpeg"])

    # Sélectionner le nombre de légendes
    num_captions = st.slider(" Sélectionnez le nombre de légendes à générer", 1, 5, 1)

    # Bouton pour générer des légendes
    if st.button(" Générer des légendes"):
        if image1 is not None and image2 is not None:
            with st.spinner('Traitement des images...'):
                # Sauvegardez les fichiers téléchargés et obtenez leurs chemins
                path1 = save_uploaded_file("uploaded_images", image1, image1.name)
                path2 = save_uploaded_file("uploaded_images", image2, image2.name)

                captions = beam_predictions(path1, path2, num_captions)
                st.write(" Légendes générées:")
                for caption in captions:
                    st.markdown(f"* {caption}")
                st.balloons()
        else:
            st.error("Veuillez télécharger les deux images avant de générer des légendes.")

if __name__ == "__main__":
    main()
