import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# 1. Cargar el archivo CSV de la Pregunta 1
pk = pd.read_csv("pokemon_clusters_p1_TfIDF.csv")

print(f"Dimensiones del DataFrame original: {pk.shape[0]} filas, {pk.shape[1]} columnas")

# 2. Eliminar la columna del cluster original
if "Grupo" in pk.columns:
    pk_sin_cluster = pk.drop(columns=["Grupo"])
else:
    print("No se encontró la columna 'Grupo'")

# 3. Verifica si la primera columna es un índice mal cargado
if pk_sin_cluster.columns[0] not in ["Pokemon", "Tipo"]:
    pk_sin_cluster = pk_sin_cluster.drop(columns=[pk_sin_cluster.columns[0]])

# 4. Separar columnas no numéricas para no meterlas al PCA
columnas_excluir = ["Pokemon", "Tipo"]
columnas_numericas = [col for col in pk_sin_cluster.columns if col not in columnas_excluir]
x = pk_sin_cluster[columnas_numericas]

"""
# PRIMERA ETAPA: Entrenar PCA sin limitar componentes para evaluar varianza acumulada
pca_test = PCA().fit(x)
varianza_acumulada = pca_test.explained_variance_ratio_.cumsum()

# Mostrar las primeras 200 componentes con su varianza acumulada
print("\nVarianza acumulada por número de componentes:")
for i, v in enumerate(varianza_acumulada[:200], start=1):
    print(f"Componentes: {i}, Varianza acumulada: {v:.4f}")
"""

# 5. Aplicar PCA
n_componentes = 107
pca = PCA(n_components=n_componentes)
x_pca = pca.fit_transform(x)

print(f"Dimensiones de la matriz PCA: {x_pca.shape[0]} filas, {x_pca.shape[1]} columnas")

print(f"\nMatriz PCA ({x_pca.shape[0]} filas, {x_pca.shape[1]} componentes):")
print(x_pca[:5])

# 6. Crear nuevo DataFrame con los componentes principales
pca_columns = [f"PC{i+1}" for i in range(n_componentes)]
pk_pca = pd.DataFrame(x_pca, columns=pca_columns)

# 7. Agrupar nuevamente con KMeans
num_clusters = 19
km = KMeans(n_clusters=num_clusters, n_init=20, random_state=42)
pk_pca["Cluster_P3"] = km.fit_predict(x_pca)

# 8. Guardar resultado final
pk_pca["Pokemon"] = pk["Pokemon"]
pk_pca["Tipo"] = pk["Tipo"]
pk_pca.to_csv("pokemon_clusters_p3_PCA.csv", index=False)
pk_pca[["Pokemon", "Tipo", "Cluster_P3"]].to_csv("pokemon_clusters_p3_PCA_Filtered.csv", index=False)

print("\nArchivo 'pokemon_clusters_p3_PCA.csv' guardado correctamente!")
print("\nArchivo 'pokemon_clusters_p3_PCA_Filtered.csv' guardado exitosamente!")

# 9. Mostrar interpretación de clusters
print("\nInterpretación de clusters con tipo dominante:")
for i in range(num_clusters):
    grupo = pk_pca[pk_pca["Cluster_P3"] == i]
    tipos = grupo["Tipo"].dropna().tolist()
    tipos_separados = []
    for t in tipos:
        if isinstance(t, str) and t != "Desconocido":
            tipos_separados.extend(t.lower().split("/"))
    if tipos_separados:
        tipo_mas_comun = pd.Series(tipos_separados).value_counts().idxmax()
        print(f"Cluster {i}: Dominante → {tipo_mas_comun} ({len(grupo)} Pokémon)")
    else:
        print(f"Cluster {i}: Sin tipo definido")
