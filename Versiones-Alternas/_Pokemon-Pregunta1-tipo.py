import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter
import re

pokemon = pd.read_csv("../Code to expose/smogon.csv")

TIPOS_VALIDOS = {
    "Normal", "Fire", "Water", "Electric", "Grass", "Ice", "Fighting", "Poison",
    "Ground", "Flying", "Psychic", "Bug", "Rock", "Ghost", "Dragon",
    "Dark", "Steel", "Fairy"
}

# Verificar el número de filas para ajustar los clusters
print(f"Número de Pokémon en el dataset: {len(pokemon)}")
print(f"Columnas disponibles: {list(pokemon.columns)}")

def extraer_tipo(texto):
    texto = texto or ""
    match = re.search(r"(?:BaseType|MegaType|Type)([A-Z][a-z]+)([A-Z][a-z]+)?", texto)
    if match:
        tipo1 = match.group(1)
        tipo2 = match.group(2)

        # Validar si ambos son tipos reales
        if tipo1 in TIPOS_VALIDOS and tipo2 in TIPOS_VALIDOS:
            return f"{tipo1}/{tipo2}"
        elif tipo1 in TIPOS_VALIDOS:
            return tipo1
    return "Desconocido"

# Aplicar al DataFrame
pokemon["Tipo"] = pokemon["texto"].apply(extraer_tipo)

# Verificamos los resultados
# print("\nEjemplos de tipos extraídos:")
# print(pokemon[["Pokemon", "Tipo"]].head(10))

stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

vec = TfidfVectorizer(stop_words=stopwords)

x = vec.fit_transform(pokemon["Tipo"])

print("\nEstos son los datos en forma de matriz:")
print(x.toarray())

print("\nTotal de palabras:")
print(len(sorted(vec.vocabulary_)))

print("\nEste es el vocabulario:")
print(vec.vocabulary_)

print("\nEste es el vocabulario ordenado:")
print(sorted(vec.vocabulary_))

# Ahora hacemos K-means
print("\nAhora hacemos K-means: ")

# IMPORTANTE: Ajustar el número de clusters según el tamaño de tu dataset
# Si tienes menos de 19 Pokémon, usa un número menor
num_pokemon = len(pokemon)
num_clusters = 19
print(f"Usando {num_clusters} clusters para {num_pokemon} Pokémon")

km = KMeans(n_clusters=num_clusters, n_init=20, random_state=42)
lista_de_clusters = km.fit_predict(x.toarray())
print(lista_de_clusters)

# Agregar columna de clusters al dataframe original
pokemon["Grupo"] = lista_de_clusters

# Mostrar resumen de tipos por grupo
print("\nTipos más frecuentes por cluster:")
for i in range(num_clusters):
    cluster_df = pokemon[pokemon["Grupo"] == i]
    print(f"\nCluster {i} - Total: {len(cluster_df)} Pokémon")
    print(cluster_df["Tipo"].value_counts().head(5))  # Top 5 tipos por cluster

print("")
# Vamos a crear un nuevo Data Frame:
print("Vamos a crear un nuevo Data Frame: ")

cabeceras = sorted(vec.vocabulary_)

tabla_de_frecuencias = pd.DataFrame(data=x.toarray(), columns=cabeceras)
print(tabla_de_frecuencias)

# Agregar la columna de clusters
tabla_de_frecuencias["Grupo"] = lista_de_clusters
print("\nLa tabla de frecuencias quedó así:")
print(tabla_de_frecuencias)

# Guardar el archivo CSV con información del tipo y grupo

resultado_final = tabla_de_frecuencias.copy()
resultado_final["Pokemon"] = pokemon["Pokemon"]
resultado_final["Tipo"] = pokemon["Tipo"]
resultado_final.to_csv("pokemon_clusters_p1_Tipo.csv", index=False)

print("\nArchivo 'pokemon_clusters_p1_Tipo.csv' guardado exitosamente!")

# Mostrar información de los clusters
print("\nInformación de los clusters:")
for i in range(num_clusters):
    pokemon_en_cluster = pokemon[lista_de_clusters == i]
    print(f"Cluster {i}: {len(pokemon_en_cluster)} Pokémon")
    if len(pokemon_en_cluster) > 0:
        print(f"  Ejemplos: {list(pokemon_en_cluster['Pokemon'].head(3))}")

print("\nAnálisis de clusters con tipo dominante:")
for i in range(num_clusters):
    cluster_df = pokemon[pokemon["Grupo"] == i]
    tipos = []
    for tipo in cluster_df["Tipo"]:
        if tipo != "Desconocido":
            tipos.extend(tipo.split("/"))

    if tipos:
        tipo_mas_comun, frecuencia = Counter(tipos).most_common(1)[0]
        porcentaje = (frecuencia / len(cluster_df)) * 100
        print(f"Cluster {i}: {tipo_mas_comun} ({frecuencia}/{len(cluster_df)} → {porcentaje:.1f}%)")
    else:
        print(f"Cluster {i}: Sin tipo identificado.")