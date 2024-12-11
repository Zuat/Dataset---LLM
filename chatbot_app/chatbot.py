from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import torch
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import random

# Verificar si las stopwords y el lematizador ya están descargados
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')

# Inicializar lematizador y stopwords en inglés
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Cargar el archivo JSON con las intenciones del chatbot
with open('./datasetPT.json', 'r', encoding='utf-8') as file:
    dataset = json.load(file)

# Inicializar Sentence-BERT
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  

# Preprocesamiento de texto con lematización
def preprocess_text(text):
    text = text.lower()
    words = text.split()
    filtered_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Lematizamos y filtramos stopwords
    return ' '.join(filtered_words)

# Función para convertir texto en embeddings usando Sentence-BERT
def get_embeddings(text):
    return model.encode([text], convert_to_tensor=True) 

# Precalcular los embeddings de las intenciones
corpus_embeddings = []
for intent in dataset['Intents']:
    patterns_embeddings = [get_embeddings(preprocess_text(pattern)) for pattern in intent['patterns']]
    corpus_embeddings.append(patterns_embeddings)

# Función para generar respuestas
def generate_response(prompt):
    prompt = preprocess_text(prompt)
    user_embedding = get_embeddings(prompt)

    # Calcular la similitud del coseno entre los embeddings del usuario y los precalculados
    similarities = []
    for intent_embeddings in corpus_embeddings:
        pattern_similarities = [cosine_similarity(user_embedding.cpu(), pattern_emb.cpu())[0][0] for pattern_emb in intent_embeddings]
        similarities.append(max(pattern_similarities))

    # Seleccionar la intención más similar
    best_intent_index = similarities.index(max(similarities))
    best_intent = dataset['Intents'][best_intent_index]

    # Ajustar el umbral dinámico según los resultados de la similitud
    similarity_threshold = 0.55  # Aumentamos el umbral para asegurar mayor precisión

    if similarities[best_intent_index] < similarity_threshold:
        response = "Lo siento, no entiendo tu pregunta. ¿Puedes reformularla de manera más clara?"
    else:
        response = random.choice(best_intent['responses'])

    return response

# Ejemplo de uso con pregunta
if __name__ == "__main__":
    while True:
        user_input = input("Tú: ")
        if user_input.lower() == 'exit':
            break
        print("Chatbot:", generate_response(user_input))
