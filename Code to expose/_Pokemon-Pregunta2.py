import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Cargar el archivo smogon.csv
pokemon = pd.read_csv("smogon.csv")

# Lista oficial de tipos Pokémon
tipos_pokemon = [
    "Normal", "Fire", "Water", "Electric", "Grass", "Ice", "Fighting", "Poison",
    "Ground", "Flying", "Psychic", "Bug", "Rock", "Ghost", "Dragon", "Dark",
    "Steel", "Fairy"
]

# Función para filtrar solo los tipos que aparecen en la cadena
def extraer_tipos_de_moves(moves):
    moves = moves if isinstance(moves, str) else ""
    resultado = []
    for tipo in tipos_pokemon:
        if tipo in moves:
            repeticiones = moves.count(tipo)
            resultado.extend([tipo] * repeticiones)
    return " ".join(resultado)

# Aplicar al DataFrame
pokemon["tipos_en_moves"] = pokemon["moves"].apply(extraer_tipos_de_moves)

# Crear la matriz TF-IDF (solo unigramas)
vec = TfidfVectorizer()
x = vec.fit_transform(pokemon["tipos_en_moves"])

print("\nEstos son los datos en forma de matriz:")
print(x.toarray())

# Mostrar tokens (vocabulario)
print("\nTokens (vocabulario controlado):")
print(sorted(vec.vocabulary_.keys()))

# Mostrar total de tokens
print("\nNúmero total de tokens:")
print(len(vec.vocabulary_))

# Aplicar KMeans
num_pokemon = len(pokemon)
num_clusters = 19
km = KMeans(n_clusters=num_clusters, n_init=20, random_state=42)
clusters = km.fit_predict(x)

print(f"\nUsando {num_clusters} clusters para {num_pokemon} Pokémon")
lista_de_clusters = km.fit_predict(x.toarray())
print(lista_de_clusters)

# Agregar los clusters al DataFrame
pokemon["Cluster_P2"] = clusters

# Mostrar resumen
print("\nResumen de clusters:")
for i in range(num_clusters):
    grupo = pokemon[pokemon["Cluster_P2"] == i]
    print(f"Cluster {i}: {len(grupo)} Pokémon")
    print(f"  Ejemplos: {list(grupo['Pokemon'].head(3))}")

# Guardar resultados
pokemon[["Pokemon", "tipos_en_moves", "Cluster_P2"]].to_csv("pokemon_clusters_p2.csv", index=False)
print("\nArchivo 'pokemon_clusters_p2.csv' guardado.")
