from transformers import AutoModel, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from nltk.corpus import stopwords
import nltk
import torch
import random

# Verificar si las stopwords ya están descargadas
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    # Descargar las stopwords si no están presentes
    nltk.download('stopwords')

# Crear una lista de stopwords en inglés
stop_words = set(stopwords.words('english'))

# Cargar el archivo JSON
with open('datasetPT.json', 'r', encoding='utf-8') as file:
    dataset = json.load(file)

# Inicializar BERT
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Preprocesamiento de texto
def preprocess_text(text):
    text = text.lower()
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    text = ' '.join(filtered_words)
    return text

# Función para convertir texto en embeddings usando BERT
def get_embeddings(text):
    # Tokenizar el texto
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    # Pasar los tokens por el modelo BERT
    with torch.no_grad():
        outputs = model(**inputs)
    # Usar las últimas representaciones ocultas como embeddings
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

# Construir el corpus de documentos (intenciones) y calcular embeddings
corpus_embeddings = [get_embeddings(' '.join(intent['patterns'])) for intent in dataset['Intents']]

# Función para generar respuestas
def generate_response(prompt):
    prompt = preprocess_text(prompt)
    user_embedding = get_embeddings(prompt)
    
    # Calcular similitud del coseno entre la pregunta del usuario y las intenciones
    cosine_similarities = [cosine_similarity(user_embedding, corpus_emb)[0][0] for corpus_emb in corpus_embeddings]

    # Seleccionar la intención más similar
    best_intent_index = cosine_similarities.index(max(cosine_similarities))
    best_intent = dataset['Intents'][best_intent_index]

    # Establecer un umbral de similitud
    similarity_threshold = 0.2

    # Verificar si la similitud es menor que el umbral
    if cosine_similarities[best_intent_index] < similarity_threshold:
        response = "Lo siento, no entiendo tu pregunta. ¿Puedes reformularla de manera más clara?"
    else:
        response = random.choice(best_intent['responses'])
        
    return response


# # Función para manejar la entrada del usuario
# def send_message(event=None):
#     # Obtener la entrada del usuario desde el widget de entrada
#     user_input = user_input_entry.get()
#     user_input_entry.delete(0, tk.END)

#     # Generar una respuesta del chatbot utilizando Transformers
#     chatbot_response = generate_response(user_input)

#     # Agregar la entrada del usuario y la respuesta del chatbot al cuadro de chat
#     chatbox.config(state=tk.NORMAL)
#     chatbox.insert(tk.END, "Tu: " + user_input + "\n\n")
#     chatbox.insert(tk.END, "Chatbot: " + chatbot_response + "\n\n")
#     chatbox.config(state=tk.DISABLED)
#     chatbox.see(tk.END)

# # Set up GUI
# root = tk.Tk()
# root.title("Chatbot")
# root.configure(bg="#ff6c3a")

# # Add chatbox
# chatbox = tk.Text(root, height=20, width=60, state=tk.DISABLED)
# chatbox.pack(padx=10, pady=10)

# # Add user input entry widget
# user_input_entry = tk.Entry(root, width=50)
# user_input_entry.pack(padx=10, pady=10)
# user_input_entry.bind("<Return>", send_message)

# # Add send button
# send_button = tk.Button(root, text="Enviar", command=send_message)
# send_button.pack(padx=10, pady=10)

# # Start GUI main loop
# root.mainloop()
