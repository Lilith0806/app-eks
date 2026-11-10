from flask import Flask, render_template, request, jsonify
import re
import unicodedata
from collections import Counter
import math

app = Flask(__name__)

# Lista básica de Stop Words en español
STOP_WORDS = set([
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por", 
    "un", "para", "con", "no", "una", "su", "al", "lo", "como", "más", "pero", 
    "sus", "le", "ya", "o", "este", "sí", "porque", "esta", "entre", "cuando", 
    "muy", "sin", "sobre", "también", "me", "hasta", "hay", "donde", "quien"
])

def remover_acentos_ascii(texto):
    """2. Diccionario ASCII: Normaliza y remueve tildes/caracteres especiales."""
    nfkd = unicodedata.normalize('NFKD', texto)
    return "".join([c for c in nfkd if not unicodedata.combining(c)])

def preprocesar(texto):
    """1. Preprocesamiento: Minúsculas y limpieza de puntuación."""
    texto_limpio = remover_acentos_ascii(texto.lower())
    palabras = re.findall(r'\b[a-z0-9]+\b', texto_limpio)
    return palabras

def extraer_ner(texto_original):
    """5. NER (Reconocimiento de Entidades Nombradas básico): 
    Detecta palabras que inician en mayúscula dentro del texto."""
    palabras = re.findall(r'\b[A-Z][a-z0-9áéíóúÁÉÍÓÚñÑ]+\b', texto_original)
    # Filtrar si está al inicio de oración o muy común
    entidades = list(set(palabras))
    return entidades if entidades else ["No se detectaron entidades explícitas"]

def calcular_tfidf(palabras_filtradas):
    """4. TF-IDF Simplificado: Calcula la relevancia relativa de cada término."""
    total_palabras = len(palabras_filtradas)
    if total_palabras == 0:
        return {}
    
    conteo = Counter(palabras_filtradas)
    tfidf = {}
    for palabra, freq in conteo.items():
        tf = freq / total_palabras
        # IDF simulado para un documento único
        idf = math.log(1 + (1 / freq))
        tfidf[palabra] = round(tf * idf, 4)
    return dict(sorted(tfidf.items(), key=lambda x: x[1], reverse=True))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/procesar', methods=['POST'])
def procesar_texto():
    datos = request.get_json()
    texto_original = datos.get('texto', '')

    if not texto_original.strip():
        return jsonify({'error': 'El texto está vacío'}), 400

    # 1. Preprocesamiento & 2. Diccionario ASCII
    palabras_preprocesadas = preprocesar(texto_original)

    # 3. Stop Words
    palabras_sin_stopwords = [p for p in palabras_preprocesadas if p not in STOP_WORDS]

    # 4. TF-IDF
    resultado_tfidf = calcular_tfidf(palabras_sin_stopwords)

    # 5. NER
    entidades_ner = extraer_ner(texto_original)

    # 6. Histograma (Frecuencia de palabras)
    histograma = dict(Counter(palabras_sin_stopwords).most_common(10))

    return jsonify({
        'original': texto_original,
        'preprocesado': " ".join(palabras_preprocesadas),
        'sin_stopwords': " ".join(palabras_sin_stopwords),
        'tfidf': resultado_tfidf,
        'ner': entidades_ner,
        'histograma': histograma
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)