import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# 1. Definir tipos oficiales
tipos_pokemon = [
    "Normal", "Fire", "Water", "Electric", "Grass", "Ice", "Fighting", "Poison",
    "Ground", "Flying", "Psychic", "Bug", "Rock", "Ghost", "Dragon", "Dark",
    "Steel", "Fairy"
]

# 2. Cargar datos
df = pd.read_csv("smogon.csv")


# 3. Función filtrar el texto a solo las palabras seleccionadas

def obtener_categorias_movimientos(movimientos):
    # Validar entrada
    if not isinstance(movimientos, str):
        movimientos = ""

    # Inicializar lista de categorías encontradas
    categorias_encontradas = []

    # Buscar cada tipo en la cadena de movimientos
    for categoria in tipos_pokemon:
        cantidad_ocurrencias = 0
        posicion = 0

        # Contar manualmente las ocurrencias
        while True:
            indice = movimientos.find(categoria, posicion)
            if indice == -1:
                break
            cantidad_ocurrencias += 1
            posicion = indice + 1

        # Agregar las categorías según las ocurrencias encontradas
        for _ in range(cantidad_ocurrencias):
            categorias_encontradas.append(categoria)

    # Convertir lista a string separado por espacios
    return ' '.join(categorias_encontradas)

# 4. Crear nueva columna con solo tipos
df["tipos_en_moves"] = df["moves"].apply(obtener_categorias_movimientos)


# 5. Vectorización TF-IDF con unigramas
vectorizer = TfidfVectorizer(ngram_range=(1, 1))
X_tfidf = vectorizer.fit_transform(df["tipos_en_moves"])

# Mostrar tokens
tokens = vectorizer.get_feature_names_out()
print("Tokens:", tokens)
print("Número total de tokens:", len(tokens))

# 6. Clustering
n_clusters = 18  # Se eligieron 18 clusters haciendo referencia a la cantidad de tipos existentes en pokemon
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df["cluster"] = kmeans.fit_predict(X_tfidf.toarray())


# 7. Interpretación de clusters

print("\nResumen de clusters:")

# Cantidad de Pokemones de ejemplo
cant_pok_ej = 5

# Iterar a través de cada cluster identificado
cluster_indices = list(range(n_clusters))
for i in cluster_indices:
    # Filtrar datos por cluster actual
    condicion_cluster = df["cluster"] == i
    grupo = df[condicion_cluster]

    # Mostrar información del cluster
    total_pokemon = len(grupo)
    print(f"Cluster {i}: {total_pokemon} Pokémon")

    # Obtener ejemplos representativos

    ejemplos_pokemon = grupo['Pokemon'].head(cant_pok_ej)
    lista_ejemplos = list(ejemplos_pokemon)
    print(f"  Ejemplos: {lista_ejemplos}")


# 8. Guardar resultado como CSV
df_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=tokens)
df_tfidf["cluster"] = df["cluster"]
df_tfidf["Pokemon"] = df["Pokemon"]
df_tfidf.to_csv("pregunta2_tfidf_clusters.csv", index=False)