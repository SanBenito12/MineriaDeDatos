#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLUSTERING NUMÉRICO - TÉCNICAS NO SUPERVISADAS
K-Means, Clustering Jerárquico y DBSCAN para agrupar poblaciones
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

def analizar_codo_kmeans(X, max_k=15):
    """Método del codo para encontrar K óptimo"""
    inercias = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        inercias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, cluster_labels))
    
    # Encontrar K óptimo usando método del codo
    diferencias = np.diff(inercias)
    diferencias2 = np.diff(diferencias)
    k_optimo_codo = k_range[np.argmax(diferencias2) + 2]
    
    # K óptimo por silhouette
    k_optimo_silhouette = k_range[np.argmax(silhouette_scores)]
    
    return k_range, inercias, silhouette_scores, k_optimo_codo, k_optimo_silhouette

def evaluar_clusters(X, labels, nombre_algoritmo):
    """Evaluar calidad de clusters usando múltiples métricas"""
    n_clusters = len(np.unique(labels[labels != -1]))  # Excluir ruido de DBSCAN
    
    if n_clusters < 2:
        return {
            'n_clusters': n_clusters,
            'silhouette': 0,
            'calinski_harabasz': 0,
            'davies_bouldin': float('inf'),
            'valido': False
        }
    
    try:
        silhouette = silhouette_score(X, labels)
        calinski = calinski_harabasz_score(X, labels)
        davies = davies_bouldin_score(X, labels)
        
        return {
            'n_clusters': n_clusters,
            'silhouette': silhouette,
            'calinski_harabasz': calinski,
            'davies_bouldin': davies,
            'valido': True
        }
    except:
        return {
            'n_clusters': n_clusters,
            'silhouette': 0,
            'calinski_harabasz': 0,
            'davies_bouldin': float('inf'),
            'valido': False
        }

def analizar_caracteristicas_clusters(datos_originales, labels, variables):
    """Analizar características demográficas de cada cluster"""
    datos_con_clusters = datos_originales.copy()
    datos_con_clusters['Cluster'] = labels
    
    analisis = {}
    
    for cluster_id in np.unique(labels):
        if cluster_id == -1:  # Ruido en DBSCAN
            continue
            
        datos_cluster = datos_con_clusters[datos_con_clusters['Cluster'] == cluster_id]
        
        estadisticas = {}
        for variable in variables:
            if variable in datos_cluster.columns:
                estadisticas[variable] = {
                    'promedio': datos_cluster[variable].mean(),
                    'mediana': datos_cluster[variable].median(),
                    'std': datos_cluster[variable].std(),
                    'min': datos_cluster[variable].min(),
                    'max': datos_cluster[variable].max()
                }
        
        analisis[cluster_id] = {
            'tamaño': len(datos_cluster),
            'porcentaje': len(datos_cluster) / len(datos_con_clusters) * 100,
            'estadisticas': estadisticas
        }
    
    return analisis

def ejecutar_clustering_numerico():
    print("📊 CLUSTERING NUMÉRICO - TÉCNICAS NO SUPERVISADAS")
    print("="*50)
    print("📝 Objetivo: Agrupar comunidades por similitud demográfica")
    print()
    
    # 1. CARGAR DATOS
    archivo = '/home/sedc/Proyectos/MineriaDeDatos/data/ceros_sin_columnasAB_limpio_weka.csv'
    try:
        datos = pd.read_csv(archivo)
        print(f"✅ Datos cargados: {datos.shape[0]:,} filas, {datos.shape[1]} columnas")
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        return
    
    # 2. SELECCIONAR VARIABLES PARA CLUSTERING
    variables_clustering = [
        'POBTOT', 'POBFEM', 'POBMAS', 'TOTHOG', 'VIVTOT', 
        'P_15YMAS', 'P_60YMAS', 'GRAPROES', 'PEA', 'POCUPADA'
    ]
    
    variables_disponibles = [v for v in variables_clustering if v in datos.columns]
    
    if len(variables_disponibles) < 4:
        print("❌ No hay suficientes variables para clustering")
        return
    
    print(f"📊 Variables para clustering: {', '.join(variables_disponibles)}")
    
    # 3. PREPARAR DATOS
    datos_limpios = datos[variables_disponibles].dropna()
    
    # Reducir muestra para eficiencia computacional
    if len(datos_limpios) > 5000:
        datos_limpios = datos_limpios.sample(n=5000, random_state=42)
        print(f"📝 Muestra reducida a {len(datos_limpios):,} registros para eficiencia")
    
    print(f"🧹 Datos finales: {len(datos_limpios):,} registros")
    
    # 4. ESCALADO DE DATOS (CRÍTICO PARA CLUSTERING)
    scaler = StandardScaler()
    datos_escalados = scaler.fit_transform(datos_limpios)
    
    print(f"🔢 Datos escalados: Media ≈ {np.mean(datos_escalados):.3f}, Std ≈ {np.std(datos_escalados):.3f}")
    print()
    
    # 5. ANÁLISIS DEL CODO PARA K-MEANS
    print("📈 ANALIZANDO NÚMERO ÓPTIMO DE CLUSTERS (K-MEANS)...")
    k_range, inercias, silhouette_scores, k_codo, k_silhouette = analizar_codo_kmeans(datos_escalados, max_k=12)
    
    print(f"   📊 K óptimo (método del codo): {k_codo}")
    print(f"   📊 K óptimo (silhouette): {k_silhouette}")
    
    # Usar el promedio de ambos métodos
    k_optimo = int((k_codo + k_silhouette) / 2)
    print(f"   🎯 K seleccionado: {k_optimo}")
    print()
    
    # 6. APLICAR DIFERENTES ALGORITMOS DE CLUSTERING
    algoritmos = {
        'K-Means': KMeans(n_clusters=k_optimo, random_state=42, n_init=10),
        'K-Means++': KMeans(n_clusters=k_optimo, init='k-means++', random_state=42, n_init=10),
        'Clustering Jerárquico': AgglomerativeClustering(n_clusters=k_optimo, linkage='ward'),
        'DBSCAN': DBSCAN(eps=0.5, min_samples=5)
    }
    
    print("🔄 EJECUTANDO ALGORITMOS DE CLUSTERING...")
    resultados = {}
    
    for nombre, algoritmo in algoritmos.items():
        print(f"   🔄 Aplicando {nombre}...")
        
        try:
            # Aplicar clustering
            labels = algoritmo.fit_predict(datos_escalados)
            
            # Evaluar calidad
            evaluacion = evaluar_clusters(datos_escalados, labels, nombre)
            
            # Analizar características de clusters
            analisis_clusters = analizar_caracteristicas_clusters(datos_limpios, labels, variables_disponibles)
            
            resultados[nombre] = {
                'algoritmo': algoritmo,
                'labels': labels,
                'evaluacion': evaluacion,
                'analisis': analisis_clusters
            }
            
            if evaluacion['valido']:
                print(f"   ✅ {nombre} → Clusters: {evaluacion['n_clusters']} | "
                      f"Silhouette: {evaluacion['silhouette']:.3f}")
            else:
                print(f"   ⚠️ {nombre} → Clusters: {evaluacion['n_clusters']} (no válido)")
                
        except Exception as e:
            print(f"   ❌ Error en {nombre}: {e}")
    
    # 7. ENCONTRAR EL MEJOR ALGORITMO
    algoritmos_validos = {k: v for k, v in resultados.items() if v['evaluacion']['valido']}
    
    if algoritmos_validos:
        mejor_nombre = max(algoritmos_validos.keys(), 
                          key=lambda x: algoritmos_validos[x]['evaluacion']['silhouette'])
        mejor_resultado = resultados[mejor_nombre]
        
        print()
        print(f"🏆 MEJOR ALGORITMO: {mejor_nombre}")
        print(f"   Clusters formados: {mejor_resultado['evaluacion']['n_clusters']}")
        print(f"   Silhouette Score: {mejor_resultado['evaluacion']['silhouette']:.3f}")
        print(f"   Calinski-Harabasz: {mejor_resultado['evaluacion']['calinski_harabasz']:.1f}")
        print(f"   Davies-Bouldin: {mejor_resultado['evaluacion']['davies_bouldin']:.3f}")
    else:
        print("❌ Ningún algoritmo produjo clusters válidos")
        return
    
    # 8. ANÁLISIS DETALLADO DE CLUSTERS
    print()
    print("🔍 ANÁLISIS DETALLADO DE CLUSTERS:")
    
    for cluster_id, info in mejor_resultado['analisis'].items():
        print(f"\n   📊 CLUSTER {cluster_id}:")
        print(f"      Tamaño: {info['tamaño']} comunidades ({info['porcentaje']:.1f}%)")
        print(f"      Características principales:")
        
        # Mostrar las 3 variables más distintivas
        promedios = {var: stats['promedio'] for var, stats in info['estadisticas'].items()}
        variables_ordenadas = sorted(promedios.items(), key=lambda x: x[1], reverse=True)
        
        for var, promedio in variables_ordenadas[:3]:
            print(f"         {var}: {promedio:.0f} (promedio)")
    
    # 9. APLICAR PCA PARA VISUALIZACIÓN
    print()
    print("🎨 PREPARANDO VISUALIZACIÓN CON PCA...")
    pca = PCA(n_components=2, random_state=42)
    datos_pca = pca.fit_transform(datos_escalados)
    
    varianza_explicada = pca.explained_variance_ratio_
    print(f"   📊 Varianza explicada: PC1={varianza_explicada[0]*100:.1f}%, PC2={varianza_explicada[1]*100:.1f}%")
    print(f"   📊 Varianza total: {sum(varianza_explicada)*100:.1f}%")
    
    # 10. VISUALIZACIONES COMPLETAS
    try:
        fig = plt.figure(figsize=(20, 16))
        
        # Gráfico 1: Método del codo
        plt.subplot(4, 4, 1)
        plt.plot(k_range, inercias, 'bo-', linewidth=2, markersize=8)
        plt.axvline(x=k_codo, color='red', linestyle='--', label=f'K óptimo = {k_codo}')
        plt.title('📈 Método del Codo (K-Means)', fontweight='bold')
        plt.xlabel('Número de Clusters (K)')
        plt.ylabel('Inercia')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Gráfico 2: Silhouette Score
        plt.subplot(4, 4, 2)
        plt.plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
        plt.axvline(x=k_silhouette, color='red', linestyle='--', label=f'K óptimo = {k_silhouette}')
        plt.title('📊 Silhouette Score', fontweight='bold')
        plt.xlabel('Número de Clusters (K)')
        plt.ylabel('Silhouette Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Gráfico 3: Comparación de algoritmos
        plt.subplot(4, 4, 3)
        nombres_alg = []
        silhouettes = []
        for nombre, resultado in resultados.items():
            if resultado['evaluacion']['valido']:
                nombres_alg.append(nombre.replace(' ', '\n'))
                silhouettes.append(resultado['evaluacion']['silhouette'])
        
        if silhouettes:
            colores = ['lightblue', 'lightgreen', 'orange', 'pink'][:len(nombres_alg)]
            barras = plt.bar(nombres_alg, silhouettes, color=colores)
            plt.title('🔄 Comparación Algoritmos', fontweight='bold')
            plt.ylabel('Silhouette Score')
            plt.xticks(rotation=45, ha='right')
            
            for i, (barra, score) in enumerate(zip(barras, silhouettes)):
                plt.text(i, score + 0.01, f'{score:.3f}', ha='center', fontweight='bold')
        
        # Gráfico 4: Distribución de tamaños de cluster
        plt.subplot(4, 4, 4)
        if mejor_resultado['analisis']:
            cluster_ids = list(mejor_resultado['analisis'].keys())
            tamaños = [mejor_resultado['analisis'][cid]['tamaño'] for cid in cluster_ids]
            
            plt.pie(tamaños, labels=[f'Cluster {cid}' for cid in cluster_ids], 
                   autopct='%1.1f%%', startangle=90)
            plt.title(f'📊 Distribución de Clusters\n{mejor_nombre}', fontweight='bold')
        
        # Gráfico 5: Visualización PCA - Mejor algoritmo
        plt.subplot(4, 4, 5)
        scatter = plt.scatter(datos_pca[:, 0], datos_pca[:, 1], 
                            c=mejor_resultado['labels'], cmap='viridis', alpha=0.6, s=50)
        plt.title(f'🎨 Clusters en Espacio PCA\n{mejor_nombre}', fontweight='bold')
        plt.xlabel(f'PC1 ({varianza_explicada[0]*100:.1f}%)')
        plt.ylabel(f'PC2 ({varianza_explicada[1]*100:.1f}%)')
        plt.colorbar(scatter)
        
        # Gráfico 6-8: Visualización PCA para otros algoritmos
        subplot_idx = 6
        for nombre, resultado in resultados.items():
            if nombre != mejor_nombre and resultado['evaluacion']['valido'] and subplot_idx <= 8:
                plt.subplot(4, 4, subplot_idx)
                scatter = plt.scatter(datos_pca[:, 0], datos_pca[:, 1], 
                                    c=resultado['labels'], cmap='viridis', alpha=0.6, s=50)
                plt.title(f'{nombre}\n({resultado["evaluacion"]["n_clusters"]} clusters)', fontweight='bold')
                plt.xlabel(f'PC1 ({varianza_explicada[0]*100:.1f}%)')
                plt.ylabel(f'PC2 ({varianza_explicada[1]*100:.1f}%)')
                subplot_idx += 1
        
        # Gráfico 9: Dendrograma (si hay clustering jerárquico)
        plt.subplot(4, 4, 9)
        try:
            # Calcular linkage para dendrograma
            linkage_matrix = linkage(datos_escalados[:500], method='ward')  # Muestra para visualización
            dendrogram(linkage_matrix, truncate_mode='level', p=5)
            plt.title('🌳 Dendrograma\n(Clustering Jerárquico)', fontweight='bold')
            plt.xlabel('Muestras')
            plt.ylabel('Distancia')
        except:
            plt.text(0.5, 0.5, 'Dendrograma\nno disponible', ha='center', va='center')
            plt.title('🌳 Dendrograma', fontweight='bold')
        
        # Gráfico 10: Heatmap de características por cluster
        plt.subplot(4, 4, 10)
        if mejor_resultado['analisis']:
            # Crear matriz de características promedio por cluster
            clusters = list(mejor_resultado['analisis'].keys())
            variables_principales = variables_disponibles[:5]  # Top 5 variables
            
            matriz_caracteristicas = np.zeros((len(clusters), len(variables_principales)))
            
            for i, cluster_id in enumerate(clusters):
                for j, variable in enumerate(variables_principales):
                    if variable in mejor_resultado['analisis'][cluster_id]['estadisticas']:
                        matriz_caracteristicas[i, j] = mejor_resultado['analisis'][cluster_id]['estadisticas'][variable]['promedio']
            
            # Normalizar para mejor visualización
            from sklearn.preprocessing import MinMaxScaler
            scaler_viz = MinMaxScaler()
            matriz_normalizada = scaler_viz.fit_transform(matriz_caracteristicas)
            
            sns.heatmap(matriz_normalizada, annot=True, fmt='.2f', cmap='YlOrRd',
                       xticklabels=[v[:8] for v in variables_principales],
                       yticklabels=[f'C{cid}' for cid in clusters])
            plt.title('🔥 Características por Cluster\n(Normalizadas)', fontweight='bold')
            plt.xlabel('Variables')
            plt.ylabel('Clusters')
        
        # Gráfico 11: Distribución de una variable clave por cluster
        plt.subplot(4, 4, 11)
        if 'POBTOT' in datos_limpios.columns:
            datos_con_clusters = datos_limpios.copy()
            datos_con_clusters['Cluster'] = mejor_resultado['labels']
            
            clusters_unicos = np.unique(mejor_resultado['labels'])
            clusters_unicos = clusters_unicos[clusters_unicos != -1]  # Excluir ruido
            
            for cluster_id in clusters_unicos:
                datos_cluster = datos_con_clusters[datos_con_clusters['Cluster'] == cluster_id]['POBTOT']
                plt.hist(datos_cluster, alpha=0.7, label=f'Cluster {cluster_id}', bins=20)
            
            plt.title('📊 Distribución POBTOT\npor Cluster', fontweight='bold')
            plt.xlabel('Población Total')
            plt.ylabel('Frecuencia')
            plt.legend()
        
        # Gráfico 12: Métricas de evaluación comparativas
        plt.subplot(4, 4, 12)
        algoritmos_validos_list = list(algoritmos_validos.keys())
        metricas = ['silhouette', 'calinski_harabasz', 'davies_bouldin']
        
        if len(algoritmos_validos_list) > 1:
            x = np.arange(len(algoritmos_validos_list))
            width = 0.25
            
            # Normalizar métricas para comparación
            silhouettes_norm = [algoritmos_validos[alg]['evaluacion']['silhouette'] for alg in algoritmos_validos_list]
            calinski_norm = [algoritmos_validos[alg]['evaluacion']['calinski_harabasz']/1000 for alg in algoritmos_validos_list]  # Escalar
            davies_norm = [1/algoritmos_validos[alg]['evaluacion']['davies_bouldin'] for alg in algoritmos_validos_list]  # Invertir (menor es mejor)
            
            plt.bar(x - width, silhouettes_norm, width, label='Silhouette', alpha=0.8)
            plt.bar(x, calinski_norm, width, label='Calinski/1000', alpha=0.8)
            plt.bar(x + width, davies_norm, width, label='1/Davies', alpha=0.8)
            
            plt.title('📊 Métricas Comparativas\n(Normalizadas)', fontweight='bold')
            plt.xlabel('Algoritmos')
            plt.ylabel('Score Normalizado')
            plt.xticks(x, [alg.replace(' ', '\n') for alg in algoritmos_validos_list])
            plt.legend()
        
        # Gráfico 13: Evolución de métricas con K (solo K-Means)
        plt.subplot(4, 4, 13)
        plt.plot(k_range, silhouette_scores, 'bo-', label='Silhouette', linewidth=2)
        plt.axvline(x=k_optimo, color='red', linestyle='--', label=f'K seleccionado = {k_optimo}')
        plt.title('📈 Evolución Silhouette\nvs Número de Clusters', fontweight='bold')
        plt.xlabel('K')
        plt.ylabel('Silhouette Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Gráfico 14: Componentes principales
        plt.subplot(4, 4, 14)
        componentes = pca.components_
        variables_cortas = [v[:8] for v in variables_disponibles]
        
        plt.imshow(componentes, cmap='coolwarm', aspect='auto')
        plt.colorbar()
        plt.title('🧮 Componentes Principales\n(Contribución Variables)', fontweight='bold')
        plt.xlabel('Variables')
        plt.ylabel('Componentes')
        plt.xticks(range(len(variables_cortas)), variables_cortas, rotation=45)
        plt.yticks([0, 1], ['PC1', 'PC2'])
        
        # Gráfico 15: Varianza explicada acumulada
        plt.subplot(4, 4, 15)
        # Calcular PCA completo para varianza
        pca_completo = PCA(random_state=42)
        pca_completo.fit(datos_escalados)
        varianza_acumulada = np.cumsum(pca_completo.explained_variance_ratio_)
        
        plt.plot(range(1, min(11, len(varianza_acumulada)+1)), 
                varianza_acumulada[:10], 'go-', linewidth=2, markersize=6)
        plt.axhline(y=0.8, color='red', linestyle='--', label='80% varianza')
        plt.title('📊 Varianza Explicada\nAcumulada (PCA)', fontweight='bold')
        plt.xlabel('Componentes')
        plt.ylabel('Varianza Acumulada')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Gráfico 16: Resumen final
        plt.subplot(4, 4, 16)
        plt.text(0.1, 0.8, f'🏆 MEJOR ALGORITMO:', fontsize=14, fontweight='bold')
        plt.text(0.1, 0.7, f'{mejor_nombre}', fontsize=12, color='blue')
        plt.text(0.1, 0.6, f'📊 Clusters: {mejor_resultado["evaluacion"]["n_clusters"]}', fontsize=11)
        plt.text(0.1, 0.5, f'📈 Silhouette: {mejor_resultado["evaluacion"]["silhouette"]:.3f}', fontsize=11)
        plt.text(0.1, 0.4, f'🔢 Muestras: {len(datos_limpios):,}', fontsize=11)
        plt.text(0.1, 0.3, f'📋 Variables: {len(variables_disponibles)}', fontsize=11)
        plt.text(0.1, 0.2, f'🎯 Varianza PCA: {sum(varianza_explicada)*100:.1f}%', fontsize=11)
        plt.text(0.1, 0.1, f'✅ Clustering Numérico Completado', fontsize=11, color='green')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        plt.title('📋 Resumen Final', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('/home/sedc/Proyectos/MineriaDeDatos/results/graficos/clustering_numerico.png', 
                   dpi=150, bbox_inches='tight')
        plt.show()
        
        print("💾 Gráficos guardados: results/graficos/clustering_numerico.png")
        
    except Exception as e:
        print(f"⚠️ Error en visualizaciones: {e}")
    
    # 11. GUARDAR RESULTADOS
    try:
        import joblib
        
        # Guardar mejor modelo y scaler
        joblib.dump(mejor_resultado['algoritmo'], '/home/sedc/Proyectos/MineriaDeDatos/results/modelos/mejor_clustering_numerico.pkl')
        joblib.dump(scaler, '/home/sedc/Proyectos/MineriaDeDatos/results/modelos/scaler_clustering.pkl')
        joblib.dump(pca, '/home/sedc/Proyectos/MineriaDeDatos/results/modelos/pca_clustering.pkl')
        
        # Crear reporte detallado
        reporte = f"""
REPORTE CLUSTERING NUMÉRICO - TÉCNICAS NO SUPERVISADAS
====================================================

MEJOR ALGORITMO: {mejor_nombre}
Número de clusters: {mejor_resultado['evaluacion']['n_clusters']}
Silhouette Score: {mejor_resultado['evaluacion']['silhouette']:.3f}
Calinski-Harabasz Index: {mejor_resultado['evaluacion']['calinski_harabasz']:.1f}
Davies-Bouldin Index: {mejor_resultado['evaluacion']['davies_bouldin']:.3f}

ANÁLISIS DE K ÓPTIMO:
- K por método del codo: {k_codo}
- K por silhouette: {k_silhouette}
- K seleccionado: {k_optimo}

COMPARACIÓN DE ALGORITMOS:
"""
        for nombre, resultado in resultados.items():
            if resultado['evaluacion']['valido']:
                reporte += f"""
{nombre}:
  - Clusters: {resultado['evaluacion']['n_clusters']}
  - Silhouette: {resultado['evaluacion']['silhouette']:.3f}
  - Calinski-Harabasz: {resultado['evaluacion']['calinski_harabasz']:.1f}
  - Davies-Bouldin: {resultado['evaluacion']['davies_bouldin']:.3f}"""

        reporte += f"""

ANÁLISIS PCA:
- Componentes utilizados: 2
- Varianza explicada PC1: {varianza_explicada[0]*100:.1f}%
- Varianza explicada PC2: {varianza_explicada[1]*100:.1f}%
- Varianza total explicada: {sum(varianza_explicada)*100:.1f}%

CARACTERÍSTICAS DE CLUSTERS:
"""
        for cluster_id, info in mejor_resultado['analisis'].items():
            reporte += f"""
Cluster {cluster_id}:
  - Tamaño: {info['tamaño']} comunidades ({info['porcentaje']:.1f}%)
  - Características principales:"""
            
            # Top 3 variables por promedio
            promedios = {var: stats['promedio'] for var, stats in info['estadisticas'].items()}
            variables_top = sorted(promedios.items(), key=lambda x: x[1], reverse=True)[:3]
            
            for var, promedio in variables_top:
                reporte += f"\n    * {var}: {promedio:.0f}"

        reporte += f"""

VARIABLES UTILIZADAS:
{', '.join(variables_disponibles)}

CONFIGURACIÓN:
- Registros procesados: {len(datos_limpios):,}
- Variables de clustering: {len(variables_disponibles)}
- Escalado aplicado: StandardScaler
- Reducción dimensional: PCA (visualización)

CALIDAD DEL CLUSTERING:
"""
        if mejor_resultado['evaluacion']['silhouette'] > 0.5:
            reporte += "- EXCELENTE: Clusters bien definidos y separados"
        elif mejor_resultado['evaluacion']['silhouette'] > 0.3:
            reporte += "- BUENO: Clusters moderadamente definidos"
        else:
            reporte += "- MODERADO: Clusters con cierta superposición"

        reporte += f"""

INTERPRETACIÓN:
- Silhouette Score: {mejor_resultado['evaluacion']['silhouette']:.3f} (rango: -1 a 1, mayor es mejor)
- Calinski-Harabasz: {mejor_resultado['evaluacion']['calinski_harabasz']:.1f} (mayor es mejor)
- Davies-Bouldin: {mejor_resultado['evaluacion']['davies_bouldin']:.3f} (menor es mejor)

RECOMENDACIONES:
- Los clusters identificados representan grupos de comunidades similares
- Usar estos grupos para políticas públicas diferenciadas
- Considerar características demográficas específicas de cada cluster
"""
        
        with open('/home/sedc/Proyectos/MineriaDeDatos/results/reportes/clustering_numerico_reporte.txt', 'w', encoding='utf-8') as f:
            f.write(reporte)
        
        print("💾 Modelo guardado: results/modelos/mejor_clustering_numerico.pkl")
        print("💾 Scaler guardado: results/modelos/scaler_clustering.pkl")
        print("💾 PCA guardado: results/modelos/pca_clustering.pkl")
        print("📄 Reporte guardado: results/reportes/clustering_numerico_reporte.txt")
        
    except Exception as e:
        print(f"⚠️ Error guardando archivos: {e}")
    
    # 12. RESUMEN FINAL
    print()
    print("📝 RESUMEN CLUSTERING NUMÉRICO:")
    print(f"   • Mejor algoritmo: {mejor_nombre}")
    print(f"   • Clusters formados: {mejor_resultado['evaluacion']['n_clusters']}")
    print(f"   • Calidad (Silhouette): {mejor_resultado['evaluacion']['silhouette']:.3f}")
    print(f"   • Variables analizadas: {len(variables_disponibles)}")
    print(f"   • Comunidades agrupadas: {len(datos_limpios):,}")
    
    if mejor_resultado['evaluacion']['silhouette'] > 0.5:
        print("   🎉 ¡Excelente agrupación! Clusters muy bien definidos")
    elif mejor_resultado['evaluacion']['silhouette'] > 0.3:
        print("   👍 Buena agrupación con clusters moderadamente definidos")
    else:
        print("   🔧 Agrupación moderada, considerar ajustar parámetros")
    
    print("   💡 Los clusters pueden usarse para segmentación demográfica")
    print("✅ CLUSTERING NUMÉRICO COMPLETADO")
    
    return resultados

if __name__ == "__main__":
    ejecutar_clustering_numerico()