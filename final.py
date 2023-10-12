import json
import pickle
import random

import Levenshtein
import nltk
import numpy as np
import re
import torch
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker
from transformers import BertTokenizer, BertModel

nltk.download('punkt')
nltk.download('wordnet')


# Charger les données JSON
def load_json(file):
    with open(file) as bot_responses:
        print(f"Chargé '{file}' avec succès !")
        return json.load(bot_responses)


response_data = load_json("intents.json")

# Fonction pour effectuer la correction orthographique
''' def correct_spelling(text):
    # Séparer le texte en mots
    words = text.split()

    # Correction orthographique pour chaque mot
    corrected_words = [spell.correction(word) for word in words]

    # Supprimer les éléments vides de la liste
    corrected_words = [word for word in corrected_words if word is not None and word != ""]

    # Rejoindre les mots corrigés pour former le texte corrigé
    corrected_text = " ".join(corrected_words)
    return corrected_text  


def get_responsee(input_phrase):
    # Convert input to lowercase for case-insensitive matching
    input_phrase = input_phrase.lower()

    # Initialize variables to store the closest match and its Levenshtein distance
    closest_match = None
    closest_distance = float('inf')

    # Iterate through intents in the data
    for intent in response_data['intents']:
        for pattern in intent['patterns']:
            # Calculate the Levenshtein distance between the input and the pattern
            distance = Levenshtein.distance(input_phrase, pattern.lower())
            if distance < closest_distance:
                closest_distance = distance
                closest_match = intent['responses']

    # If a close match is found, return a random response from the matched intent
    if closest_match and closest_distance <= 2:  # Set a threshold for considering a match
        response = random.choice(closest_match)
    else:
        response = "I'm sorry, I don't understand."
    print(closest_distance)
    return response

def get_response_for_gui(input_string):
    corrected_input = correct_spelling(input_string)
    print(corrected_input)
    response_generated = False
    bot_response = "I'm sorry, I don't have a suitable response for that."

    for response in response_data['intents']:
        if any(word in corrected_input for word in response["patterns"]):
            response_generated = True
            responses = response["responses"]
            bot_response = random.choice(responses)
            break

    if not response_generated:
        bot_response = "I'm sorry, I don't have a suitable response for that."

    return bot_response




def get_similar_responses(query, choices, threshold=0.8):
    query_tokens = tokenizer.tokenize(query)
    query_inputs = tokenizer.encode_plus(query, return_tensors='pt', padding=True, truncation=True)

    results = []

    for choice in choices:
        choice_tokens = tokenizer.tokenize(choice)
        choice_inputs = tokenizer.encode_plus(choice, return_tensors='pt', padding=True, truncation=True)

        with torch.no_grad():
            query_outputs = model(**query_inputs)
            choice_outputs = model(**choice_inputs)

        encoded_query = query_outputs.last_hidden_state[0][0]
        encoded_choice = choice_outputs.last_hidden_state[0][0]

        similarity_score = torch.cosine_similarity(encoded_query, encoded_choice, dim=0)

        if similarity_score.item() >= threshold:
            results.append(choice)

    return results

# Fonction pour obtenir la réponse
def get_response(input_string):
    corrected_input = correct_spelling(input_string)
    split_message = re.split(r'\s+|[,;?!.-]\s*', corrected_input.lower())
    print(split_message)
    print(corrected_input)
    score_list = []

    for response in response_data:
        response_score = 0
        required_score = 0
        required_words = response["required_words"]
        correct_responses = response["user_input"]

        if required_words:
            for word in split_message:
                if word in required_words:
                    required_score += 1

        if required_score == len(required_words):
            for word in split_message:
                if word in correct_responses:
                    print(word ,"\t", correct_responses)
                    response_score += 1
        score_list.append(response_score)
    print(score_list)
    best_response = max(score_list)
    response_index = score_list.index(best_response)

    if input_string == "":
        return "Veuillez écrire quelque chose pour que nous puissions discuter :("

    if best_response != 0:
        if corrected_input != input_string:
            correct_response = response_data[response_index]["bot_response"]
            return f"Did you mean: '{corrected_input}'? {correct_response}"
        else:
            return response_data[response_index]["bot_response"]

    return("I'm sorry, I don't have a suitable response for that.")          '''

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
print(classes)
model = load_model('best_chatbot_model.h5')


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list


def get_response(message):
    intents_list = predict_class(message)
    print(intents_list)
    tag = intents_list[0]['intent']
    list_of_intents = response_data['intents']
    result = ''
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    print("TEST:------------", intents_list)
    if (float(intents_list[0]['probability']) < 0.7):
        return ("I'm sorry, I don't have a suitable response for that.")
    return result
