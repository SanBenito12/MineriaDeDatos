#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLUSTERING CONCEPTUAL - TÉCNICAS NO SUPERVISADAS
Agrupación basada en conceptos demográficos y reglas interpretables
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

class ClusteringConceptual:
    """Algoritmo de clustering conceptual basado en reglas demográficas"""
    
    def __init__(self, max_clusters=8, min_cluster_size=50):
        self.max_clusters = max_clusters
        self.min_cluster_size = min_cluster_size
        self.conceptos = {}
        self.reglas_clusters = {}
        self.labels_ = None
        
    def _crear_conceptos_demograficos(self, datos, variables):
        """Crear conceptos demográficos interpretativos"""
        conceptos = {}
        
        for variable in variables:
            if variable in datos.columns:
                valores = datos[variable].dropna()
                
                # Crear rangos conceptuales
                if variable == 'POBTOT':
                    conceptos[variable] = {
                        'Muy Pequeña': (0, 500),
                        'Pequeña': (500, 2000),
                        'Mediana': (2000, 10000),
                        'Grande': (10000, 50000),
                        'Muy Grande': (50000, float('inf'))
                    }
                elif variable in ['POBFEM', 'POBMAS']:
                    conceptos[variable] = {
                        'Muy Bajo': (0, valores.quantile(0.2)),
                        'Bajo': (valores.quantile(0.2), valores.quantile(0.4)),
                        'Medio': (valores.quantile(0.4), valores.quantile(0.6)),
                        'Alto': (valores.quantile(0.6), valores.quantile(0.8)),
                        'Muy Alto': (valores.quantile(0.8), float('inf'))
                    }
                elif variable == 'TOTHOG':
                    conceptos[variable] = {
                        'Pocos Hogares': (0, valores.quantile(0.3)),
                        'Hogares Moderados': (valores.quantile(0.3), valores.quantile(0.7)),
                        'Muchos Hogares': (valores.quantile(0.7), float('inf'))
                    }
                elif variable in ['P_15YMAS', 'PEA', 'POCUPADA']:
                    conceptos[variable] = {
                        'Bajo': (0, valores.quantile(0.33)),
                        'Medio': (valores.quantile(0.33), valores.quantile(0.67)),
                        'Alto': (valores.quantile(0.67), float('inf'))
                    }
                elif variable == 'P_60YMAS':
                    conceptos[variable] = {
                        'Población Joven': (0, valores.quantile(0.3)),
                        'Población Mixta': (valores.quantile(0.3), valores.quantile(0.7)),
                        'Población Envejecida': (valores.quantile(0.7), float('inf'))
                    }
                else:
                    # Concepto genérico por cuartiles
                    conceptos[variable] = {
                        'Muy Bajo': (0, valores.quantile(0.25)),
                        'Bajo': (valores.quantile(0.25), valores.quantile(0.5)),
                        'Alto': (valores.quantile(0.5), valores.quantile(0.75)),
                        'Muy Alto': (valores.quantile(0.75), float('inf'))
                    }
        
        return conceptos
    
    def _asignar_concepto(self, valor, conceptos_variable):
        """Asignar un valor a su concepto correspondiente"""
        for concepto, (min_val, max_val) in conceptos_variable.items():
            if min_val <= valor < max_val:
                return concepto
        return list(conceptos_variable.keys())[-1]  # Último concepto por defecto
    
    def _convertir_a_conceptos(self, datos, variables):
        """Convertir datos numéricos a conceptos interpretativos"""
        datos_conceptuales = pd.DataFrame()
        
        for variable in variables:
            if variable in datos.columns and variable in self.conceptos:
                datos_conceptuales[variable] = datos[variable].apply(
                    lambda x: self._asignar_concepto(x, self.conceptos[variable])
                )
        
        return datos_conceptuales
    
    def _generar_reglas_cluster(self, datos_conceptuales, labels):
        """Generar reglas interpretables para cada cluster"""
        reglas = {}
        
        for cluster_id in np.unique(labels):
            if cluster_id == -1:  # Ruido
                continue
                
            indices_cluster = np.where(labels == cluster_id)[0]
            datos_cluster = datos_conceptuales.iloc[indices_cluster]
            
            # Encontrar patrones frecuentes en el cluster
            patron_cluster = {}
            for variable in datos_conceptuales.columns:
                conteos = datos_cluster[variable].value_counts()
                concepto_dominante = conteos.index[0]
                frecuencia = conteos.iloc[0] / len(datos_cluster)
                
                if frecuencia >= 0.4:  # Al menos 40% del cluster
                    patron_cluster[variable] = {
                        'concepto': concepto_dominante,
                        'frecuencia': frecuencia,
                        'casos': conteos.iloc[0]
                    }
            
            reglas[cluster_id] = {
                'patrones': patron_cluster,
                'tamaño': len(datos_cluster),
                'descripcion': self._generar_descripcion(patron_cluster)
            }
        
        return reglas
    
    def _generar_descripcion(self, patron_cluster):
        """Generar descripción legible del cluster"""
        if not patron_cluster:
            return "Cluster sin patrones dominantes claros"
        
        descripciones = []
        for variable, info in patron_cluster.items():
            descripciones.append(f"{variable}: {info['concepto']} ({info['frecuencia']*100:.0f}%)")
        
        return " | ".join(descripciones)
    
    def _clustering_jerarquico_conceptual(self, datos_conceptuales):
        """Aplicar clustering jerárquico basado en similitud conceptual"""
        # Calcular matriz de similitud conceptual
        n_muestras = len(datos_conceptuales)
        similitud = np.zeros((n_muestras, n_muestras))
        
        for i in range(n_muestras):
            for j in range(i, n_muestras):
                # Calcular similitud como proporción de conceptos compartidos
                coincidencias = 0
                total_variables = len(datos_conceptuales.columns)
                
                for variable in datos_conceptuales.columns:
                    if datos_conceptuales.iloc[i][variable] == datos_conceptuales.iloc[j][variable]:
                        coincidencias += 1
                
                sim = coincidencias / total_variables
                similitud[i, j] = similitud[j, i] = sim
        
        # Clustering aglomerativo simple basado en similitud
        clusters = list(range(n_muestras))  # Cada punto es su propio cluster inicialmente
        cluster_labels = np.arange(n_muestras)
        
        # Merger clusters similares iterativamente
        umbral_similitud = 0.7
        
        for _ in range(self.max_clusters):
            # Encontrar par de puntos más similar
            max_sim = 0
            best_pair = None
            
            for i in range(n_muestras):
                for j in range(i + 1, n_muestras):
                    if cluster_labels[i] != cluster_labels[j] and similitud[i, j] > max_sim:
                        max_sim = similitud[i, j]
                        best_pair = (i, j)
            
            if best_pair is None or max_sim < umbral_similitud:
                break
            
            # Merger clusters
            i, j = best_pair
            cluster_antiguo = cluster_labels[j]
            cluster_nuevo = cluster_labels[i]
            
            # Cambiar todas las etiquetas del cluster antiguo al nuevo
            cluster_labels[cluster_labels == cluster_antiguo] = cluster_nuevo
        
        # Renumerar clusters secuencialmente
        clusters_unicos = np.unique(cluster_labels)
        for idx, cluster_id in enumerate(clusters_unicos):
            cluster_labels[cluster_labels == cluster_id] = idx
        
        return cluster_labels
    
    def fit_predict(self, datos, variables):
        """Ajustar y predecir clusters conceptuales"""
        # Crear conceptos demográficos
        self.conceptos = self._crear_conceptos_demograficos(datos, variables)
        
        # Convertir datos a conceptos
        datos_conceptuales = self._convertir_a_conceptos(datos, variables)
        
        # Aplicar clustering conceptual
        labels = self._clustering_jerarquico_conceptual(datos_conceptuales)
        
        # Filtrar clusters muy pequeños
        conteos_clusters = Counter(labels)
        clusters_validos = {k: v for k, v in conteos_clusters.items() if v >= self.min_cluster_size}
        
        # Reasignar puntos de clusters pequeños al cluster más similar
        labels_filtrados = labels.copy()
        for i, label in enumerate(labels):
            if label not in clusters_validos:
                # Encontrar cluster más similar
                punto_conceptual = datos_conceptuales.iloc[i]
                mejor_cluster = self._encontrar_cluster_similar(punto_conceptual, datos_conceptuales, labels_filtrados, clusters_validos)
                labels_filtrados[i] = mejor_cluster
        
        self.labels_ = labels_filtrados
        
        # Generar reglas para clusters finales
        self.reglas_clusters = self._generar_reglas_cluster(datos_conceptuales, self.labels_)
        
        return self.labels_
    
    def _encontrar_cluster_similar(self, punto, datos_conceptuales, labels, clusters_validos):
        """Encontrar el cluster válido más similar para un punto"""
        if not clusters_validos:
            return 0
        
        mejor_similitud = -1
        mejor_cluster = list(clusters_validos.keys())[0]
        
        for cluster_id in clusters_validos.keys():
            indices_cluster = np.where(labels == cluster_id)[0]
            if len(indices_cluster) == 0:
                continue
            
            # Calcular similitud promedio con puntos del cluster
            similitudes = []
            for idx in indices_cluster[:min(20, len(indices_cluster))]:  # Muestra para eficiencia
                punto_cluster = datos_conceptuales.iloc[idx]
                coincidencias = sum(punto[var] == punto_cluster[var] for var in punto.index)
                similitud = coincidencias / len(punto.index)
                similitudes.append(similitud)
            
            similitud_promedio = np.mean(similitudes)
            if similitud_promedio > mejor_similitud:
                mejor_similitud = similitud_promedio
                mejor_cluster = cluster_id
        
        return mejor_cluster

def analizar_estabilidad_conceptual(datos, variables, n_repeticiones=5):
    """Analizar estabilidad del clustering conceptual"""
    resultados = []
    
    for i in range(n_repeticiones):
        # Muestra bootstrap
        muestra = datos.sample(n=min(2000, len(datos)), replace=True, random_state=i*42)
        
        clusterer = ClusteringConceptual(max_clusters=6, min_cluster_size=30)
        labels = clusterer.fit_predict(muestra, variables)
        
        resultados.append({
            'labels': labels,
            'n_clusters': len(np.unique(labels)),
            'reglas': clusterer.reglas_clusters
        })
    
    # Calcular estabilidad promedio
    aris = []
    for i in range(len(resultados)):
        for j in range(i + 1, len(resultados)):
            if len(resultados[i]['labels']) == len(resultados[j]['labels']):
                ari = adjusted_rand_score(resultados[i]['labels'], resultados[j]['labels'])
                aris.append(ari)
    
    estabilidad = np.mean(aris) if aris else 0
    return estabilidad, resultados

def ejecutar_clustering_conceptual():
    print("🧠 CLUSTERING CONCEPTUAL - TÉCNICAS NO SUPERVISADAS")
    print("="*52)
    print("📝 Objetivo: Agrupar comunidades por conceptos demográficos interpretativos")
    print()
    
    # 1. CARGAR DATOS
    archivo = '/home/sedc/Proyectos/MineriaDeDatos/data/ceros_sin_columnasAB_limpio_weka.csv'
    try:
        datos = pd.read_csv(archivo)
        print(f"✅ Datos cargados: {datos.shape[0]:,} filas, {datos.shape[1]} columnas")
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        return
    
    # 2. SELECCIONAR VARIABLES CONCEPTUALES
    variables_conceptuales = [
        'POBTOT', 'POBFEM', 'POBMAS', 'TOTHOG', 'VIVTOT',
        'P_15YMAS', 'P_60YMAS', 'GRAPROES', 'PEA', 'POCUPADA'
    ]
    
    variables_disponibles = [v for v in variables_conceptuales if v in datos.columns]
    
    if len(variables_disponibles) < 4:
        print("❌ No hay suficientes variables para clustering conceptual")
        return
    
    print(f"📊 Variables conceptuales: {', '.join(variables_disponibles)}")
    
    # 3. PREPARAR DATOS
    datos_limpios = datos[variables_disponibles].dropna()
    
    # Reducir muestra para clustering conceptual (computacionalmente intensivo)
    if len(datos_limpios) > 3000:
        datos_limpios = datos_limpios.sample(n=3000, random_state=42)
        print(f"📝 Muestra reducida a {len(datos_limpios):,} registros para clustering conceptual")
    
    print(f"🧹 Datos finales: {len(datos_limpios):,} registros")
    print()
    
    # 4. APLICAR CLUSTERING CONCEPTUAL
    print("🧠 EJECUTANDO CLUSTERING CONCEPTUAL...")
    
    clusterer_conceptual = ClusteringConceptual(max_clusters=8, min_cluster_size=50)
    labels_conceptual = clusterer_conceptual.fit_predict(datos_limpios, variables_disponibles)
    
    n_clusters_conceptual = len(np.unique(labels_conceptual))
    
    print(f"   ✅ Clustering conceptual completado")
    print(f"   📊 Clusters formados: {n_clusters_conceptual}")
    print(f"   📋 Conceptos creados: {len(clusterer_conceptual.conceptos)} variables")
    
    # 5. MOSTRAR CONCEPTOS DEFINIDOS
    print()
    print("📚 CONCEPTOS DEMOGRÁFICOS DEFINIDOS:")
    for variable, conceptos in clusterer_conceptual.conceptos.items():
        print(f"\n   📊 {variable}:")
        for concepto, (min_val, max_val) in conceptos.items():
            if max_val == float('inf'):
                print(f"      • {concepto}: ≥ {min_val:.0f}")
            else:
                print(f"      • {concepto}: {min_val:.0f} - {max_val:.0f}")
    
    # 6. MOSTRAR REGLAS DE CLUSTERS
    print()
    print("📏 REGLAS DE CLUSTERS CONCEPTUALES:")
    for cluster_id, info in clusterer_conceptual.reglas_clusters.items():
        print(f"\n   🔍 CLUSTER {cluster_id}:")
        print(f"      Tamaño: {info['tamaño']} comunidades")
        print(f"      Descripción: {info['descripcion']}")
        
        if info['patrones']:
            print(f"      Patrones dominantes:")
            for variable, patron in info['patrones'].items():
                print(f"         • {variable}: {patron['concepto']} "
                      f"({patron['casos']}/{info['tamaño']} casos, {patron['frecuencia']*100:.0f}%)")
    
    # 7. COMPARAR CON CLUSTERING NUMÉRICO (K-MEANS)
    print()
    print("⚖️ COMPARANDO CON CLUSTERING NUMÉRICO...")
    
    from sklearn.preprocessing import StandardScaler
    
    # K-means para comparación
    scaler = StandardScaler()
    datos_escalados = scaler.fit_transform(datos_limpios)
    
    kmeans = KMeans(n_clusters=n_clusters_conceptual, random_state=42, n_init=10)
    labels_kmeans = kmeans.fit_predict(datos_escalados)
    
    # Métricas de comparación
    ari_score = adjusted_rand_score(labels_conceptual, labels_kmeans)
    nmi_score = normalized_mutual_info_score(labels_conceptual, labels_kmeans)
    
    print(f"   📊 K-Means (numérico): {len(np.unique(labels_kmeans))} clusters")
    print(f"   📊 Conceptual: {n_clusters_conceptual} clusters")
    print(f"   📈 Adjusted Rand Index: {ari_score:.3f}")
    print(f"   📈 Normalized Mutual Info: {nmi_score:.3f}")
    
    # 8. ANÁLISIS DE ESTABILIDAD
    print()
    print("🔄 ANALIZANDO ESTABILIDAD DEL CLUSTERING...")
    
    estabilidad, resultados_estabilidad = analizar_estabilidad_conceptual(
        datos_limpios, variables_disponibles, n_repeticiones=3
    )
    
    print(f"   📊 Estabilidad promedio (ARI): {estabilidad:.3f}")
    print(f"   📊 Número de clusters (repeticiones): {[r['n_clusters'] for r in resultados_estabilidad]}")
    
    # 9. ANÁLISIS ESTADÍSTICO POR CLUSTER
    print()
    print("📊 ANÁLISIS ESTADÍSTICO POR CLUSTER:")
    
    datos_con_clusters = datos_limpios.copy()
    datos_con_clusters['Cluster_Conceptual'] = labels_conceptual
    datos_con_clusters['Cluster_KMeans'] = labels_kmeans
    
    for cluster_id in np.unique(labels_conceptual):
        datos_cluster = datos_con_clusters[datos_con_clusters['Cluster_Conceptual'] == cluster_id]
        
        print(f"\n   📈 CLUSTER {cluster_id} - Estadísticas:")
        print(f"      Tamaño: {len(datos_cluster)} ({len(datos_cluster)/len(datos_con_clusters)*100:.1f}%)")
        
        # Top 3 variables características
        promedios = {}
        for var in variables_disponibles:
            promedios[var] = datos_cluster[var].mean()
        
        # Comparar con promedio global
        promedios_globales = datos_limpios[variables_disponibles].mean()
        diferencias = {}
        for var in variables_disponibles:
            diferencias[var] = promedios[var] / promedios_globales[var] - 1
        
        # Variables más distintivas
        variables_distintivas = sorted(diferencias.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        
        print(f"      Variables más distintivas:")
        for var, diff in variables_distintivas:
            direction = "↑" if diff > 0 else "↓"
            print(f"         • {var}: {promedios[var]:.0f} ({direction} {abs(diff)*100:.0f}% vs promedio)")
    
    # 10. VISUALIZACIONES
    try:
        fig = plt.figure(figsize=(18, 14))
        
        # Gráfico 1: Distribución de clusters conceptuales
        plt.subplot(3, 4, 1)
        conteos_conceptual = Counter(labels_conceptual)
        clusters_ids = list(conteos_conceptual.keys())
        tamaños = list(conteos_conceptual.values())
        
        plt.pie(tamaños, labels=[f'Cluster {cid}' for cid in clusters_ids], 
               autopct='%1.1f%%', startangle=90)
        plt.title('🧠 Distribución Clustering\nConceptual', fontweight='bold')
        
        # Gráfico 2: Distribución K-means para comparación
        plt.subplot(3, 4, 2)
        conteos_kmeans = Counter(labels_kmeans)
        clusters_kmeans = list(conteos_kmeans.keys())
        tamaños_kmeans = list(conteos_kmeans.values())
        
        plt.pie(tamaños_kmeans, labels=[f'Cluster {cid}' for cid in clusters_kmeans], 
               autopct='%1.1f%%', startangle=90)
        plt.title('📊 Distribución K-Means\n(Comparación)', fontweight='bold')
        
        # Gráfico 3: Comparación de métricas
        plt.subplot(3, 4, 3)
        metricas = ['ARI', 'NMI', 'Estabilidad']
        valores = [ari_score, nmi_score, estabilidad]
        colores = ['lightblue', 'lightgreen', 'orange']
        
        barras = plt.bar(metricas, valores, color=colores)
        plt.title('⚖️ Métricas de Comparación', fontweight='bold')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        
        for i, (barra, valor) in enumerate(zip(barras, valores)):
            plt.text(i, valor + 0.02, f'{valor:.3f}', ha='center', fontweight='bold')
        
        # Gráfico 4: Heatmap de conceptos por cluster
        plt.subplot(3, 4, 4)
        
        # Crear matriz de conceptos dominantes
        variables_principales = variables_disponibles[:4]  # Top 4 para visualización
        matriz_conceptos = []
        etiquetas_clusters = []
        
        for cluster_id in sorted(clusterer_conceptual.reglas_clusters.keys()):
            fila_conceptos = []
            etiquetas_clusters.append(f'C{cluster_id}')
            
            for var in variables_principales:
                if (var in clusterer_conceptual.reglas_clusters[cluster_id]['patrones'] and 
                    var in clusterer_conceptual.conceptos):
                    # Obtener índice del concepto dominante
                    concepto_dom = clusterer_conceptual.reglas_clusters[cluster_id]['patrones'][var]['concepto']
                    conceptos_var = list(clusterer_conceptual.conceptos[var].keys())
                    if concepto_dom in conceptos_var:
                        indice_concepto = conceptos_var.index(concepto_dom)
                        fila_conceptos.append(indice_concepto)
                    else:
                        fila_conceptos.append(0)
                else:
                    fila_conceptos.append(0)
            
            matriz_conceptos.append(fila_conceptos)
        
        if matriz_conceptos:
            sns.heatmap(matriz_conceptos, annot=True, fmt='d', cmap='viridis',
                       xticklabels=[v[:8] for v in variables_principales],
                       yticklabels=etiquetas_clusters)
            plt.title('🔥 Conceptos Dominantes\npor Cluster', fontweight='bold')
            plt.xlabel('Variables')
            plt.ylabel('Clusters')
        
        # Gráfico 5: Análisis de una variable clave (POBTOT)
        plt.subplot(3, 4, 5)
        if 'POBTOT' in datos_con_clusters.columns:
            for cluster_id in np.unique(labels_conceptual):
                datos_cluster = datos_con_clusters[datos_con_clusters['Cluster_Conceptual'] == cluster_id]
                plt.hist(datos_cluster['POBTOT'], alpha=0.7, label=f'Cluster {cluster_id}', bins=15)
            
            plt.title('📊 Distribución POBTOT\npor Cluster Conceptual', fontweight='bold')
            plt.xlabel('Población Total')
            plt.ylabel('Frecuencia')
            plt.legend()
            plt.yscale('log')
        
        # Gráfico 6: Comparación directa conceptual vs K-means
        plt.subplot(3, 4, 6)
        # Crear matriz de confusión entre métodos
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(labels_conceptual, labels_kmeans)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('🔄 Confusión: Conceptual\nvs K-Means', fontweight='bold')
        plt.xlabel('K-Means')
        plt.ylabel('Conceptual')
        
        # Gráfico 7: Estabilidad por número de repeticiones
        plt.subplot(3, 4, 7)
        n_clusters_estabilidad = [r['n_clusters'] for r in resultados_estabilidad]
        plt.plot(range(1, len(n_clusters_estabilidad) + 1), n_clusters_estabilidad, 'bo-', linewidth=2)
        plt.title('🔄 Estabilidad: Número\nde Clusters', fontweight='bold')
        plt.xlabel('Repetición')
        plt.ylabel('Número de Clusters')
        plt.grid(True, alpha=0.3)
        
        # Gráfico 8: Distribución de tamaños de cluster
        plt.subplot(3, 4, 8)
        tamaños_clusters = [info['tamaño'] for info in clusterer_conceptual.reglas_clusters.values()]
        cluster_nombres = [f'C{cid}' for cid in clusterer_conceptual.reglas_clusters.keys()]
        
        plt.bar(range(len(tamaños_clusters)), tamaños_clusters, color='cyan')
        plt.title('📊 Tamaños de Clusters\nConceptuales', fontweight='bold')
        plt.xlabel('Cluster')
        plt.ylabel('Número de Comunidades')
        plt.xticks(range(len(cluster_nombres)), cluster_nombres)
        
        # Añadir valores en las barras
        for i, tamaño in enumerate(tamaños_clusters):
            plt.text(i, tamaño + max(tamaños_clusters)*0.01, str(tamaño), ha='center', fontweight='bold')
        
        # Gráfico 9: Análisis de pureza conceptual
        plt.subplot(3, 4, 9)
        purezas = []
        cluster_labels = []
        
        for cluster_id, info in clusterer_conceptual.reglas_clusters.items():
            if info['patrones']:
                pureza_promedio = np.mean([p['frecuencia'] for p in info['patrones'].values()])
                purezas.append(pureza_promedio)
                cluster_labels.append(f'C{cluster_id}')
        
        if purezas:
            plt.bar(cluster_labels, purezas, color='gold')
            plt.title('✨ Pureza Conceptual\npor Cluster', fontweight='bold')
            plt.ylabel('Pureza Promedio')
            plt.xlabel('Cluster')
            plt.ylim(0, 1)
            
            for i, pureza in enumerate(purezas):
                plt.text(i, pureza + 0.02, f'{pureza:.2f}', ha='center', fontweight='bold')
        
        # Gráfico 10: Comparación de variables clave
        plt.subplot(3, 4, 10)
        if len(variables_disponibles) >= 3:
            var1, var2, var3 = variables_disponibles[:3]
            
            for cluster_id in np.unique(labels_conceptual):
                datos_cluster = datos_con_clusters[datos_con_clusters['Cluster_Conceptual'] == cluster_id]
                plt.scatter(datos_cluster[var1], datos_cluster[var2], 
                           label=f'Cluster {cluster_id}', alpha=0.6, s=30)
            
            plt.title(f'📈 {var1[:8]} vs {var2[:8]}\npor Cluster', fontweight='bold')
            plt.xlabel(var1)
            plt.ylabel(var2)
            plt.legend()
        
        # Gráfico 11: Evolución de conceptos por variable
        plt.subplot(3, 4, 11)
        
        # Contar conceptos utilizados por variable
        conceptos_utilizados = defaultdict(set)
        for cluster_id, info in clusterer_conceptual.reglas_clusters.items():
            for variable, patron in info['patrones'].items():
                conceptos_utilizados[variable].add(patron['concepto'])
        
        variables_viz = list(conceptos_utilizados.keys())[:5]
        num_conceptos = [len(conceptos_utilizados[var]) for var in variables_viz]
        
        plt.bar(range(len(variables_viz)), num_conceptos, color='lightcoral')
        plt.title('🎨 Diversidad Conceptual\npor Variable', fontweight='bold')
        plt.xlabel('Variables')
        plt.ylabel('Conceptos Únicos Utilizados')
        plt.xticks(range(len(variables_viz)), [v[:8] for v in variables_viz], rotation=45)
        
        # Gráfico 12: Resumen final
        plt.subplot(3, 4, 12)
        plt.text(0.1, 0.9, f'🧠 CLUSTERING CONCEPTUAL', fontsize=14, fontweight='bold')
        plt.text(0.1, 0.8, f'📊 Clusters: {n_clusters_conceptual}', fontsize=12)
        plt.text(0.1, 0.7, f'📋 Variables: {len(variables_disponibles)}', fontsize=12)
        plt.text(0.1, 0.6, f'🔢 Muestras: {len(datos_limpios):,}', fontsize=12)
        plt.text(0.1, 0.5, f'📈 ARI vs K-means: {ari_score:.3f}', fontsize=12)
        plt.text(0.1, 0.4, f'🔄 Estabilidad: {estabilidad:.3f}', fontsize=12)
        plt.text(0.1, 0.3, f'🎯 Conceptos: {sum(len(c) for c in clusterer_conceptual.conceptos.values())}', fontsize=12)
        
        if estabilidad > 0.5:
            plt.text(0.1, 0.2, f'✅ Alta estabilidad', fontsize=11, color='green')
        else:
            plt.text(0.1, 0.2, f'⚠️ Estabilidad moderada', fontsize=11, color='orange')
        
        plt.text(0.1, 0.1, f'✅ Clustering Interpretable', fontsize=11, color='blue')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        plt.title('📋 Resumen Final', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('/home/sedc/Proyectos/MineriaDeDatos/results/graficos/clustering_conceptual.png', 
                   dpi=150, bbox_inches='tight')
        plt.show()
        
        print("💾 Gráficos guardados: results/graficos/clustering_conceptual.png")
        
    except Exception as e:
        print(f"⚠️ Error en visualizaciones: {e}")
    
    # 11. GUARDAR RESULTADOS
    try:
        import joblib
        
        # Guardar clusterer conceptual
        joblib.dump(clusterer_conceptual, '/home/sedc/Proyectos/MineriaDeDatos/results/modelos/clustering_conceptual.pkl')
        
        # Crear reporte detallado
        reporte = f"""
REPORTE CLUSTERING CONCEPTUAL - TÉCNICAS NO SUPERVISADAS
======================================================

RESUMEN GENERAL:
- Número de clusters: {n_clusters_conceptual}
- Registros procesados: {len(datos_limpios):,}
- Variables conceptuales: {len(variables_disponibles)}
- Conceptos totales definidos: {sum(len(c) for c in clusterer_conceptual.conceptos.values())}

COMPARACIÓN CON K-MEANS:
- Adjusted Rand Index: {ari_score:.3f}
- Normalized Mutual Information: {nmi_score:.3f}
- Estabilidad (ARI promedio): {estabilidad:.3f}

CONCEPTOS DEMOGRÁFICOS DEFINIDOS:
"""
        
        for variable, conceptos in clusterer_conceptual.conceptos.items():
            reporte += f"\n{variable}:"
            for concepto, (min_val, max_val) in conceptos.items():
                if max_val == float('inf'):
                    reporte += f"\n  - {concepto}: ≥ {min_val:.0f}"
                else:
                    reporte += f"\n  - {concepto}: {min_val:.0f} - {max_val:.0f}"
        
        reporte += f"""

CLUSTERS Y REGLAS IDENTIFICADAS:
"""
        
        for cluster_id, info in clusterer_conceptual.reglas_clusters.items():
            reporte += f"""
CLUSTER {cluster_id}:
  - Tamaño: {info['tamaño']} comunidades ({info['tamaño']/len(datos_limpios)*100:.1f}%)
  - Descripción: {info['descripción']}
  - Patrones dominantes:"""
            
            if info['patrones']:
                for variable, patron in info['patrones'].items():
                    reporte += f"""
    * {variable}: {patron['concepto']} ({patron['casos']}/{info['tamaño']} casos, {patron['frecuencia']*100:.0f}%)"""
            else:
                reporte += "\n    * Sin patrones dominantes claros"
        
        reporte += f"""

ANÁLISIS DE CALIDAD:
- Pureza conceptual promedio: {np.mean([np.mean([p['frecuencia'] for p in info['patrones'].values()]) for info in clusterer_conceptual.reglas_clusters.values() if info['patrones']]):.3f}
- Clusters con patrones claros: {len([info for info in clusterer_conceptual.reglas_clusters.values() if info['patrones']])}/{len(clusterer_conceptual.reglas_clusters)}

VARIABLES UTILIZADAS:
{', '.join(variables_disponibles)}

VENTAJAS DEL CLUSTERING CONCEPTUAL:
- Resultados interpretables y explicables
- Reglas demográficas claras
- Conceptos alineados con conocimiento del dominio
- Útil para políticas públicas segmentadas

LIMITACIONES:
- Dependiente de la definición de conceptos
- Puede ser menos preciso que métodos numéricos
- Requiere conocimiento del dominio para definir rangos

RECOMENDACIONES DE USO:
- Ideal para reportes ejecutivos y toma de decisiones
- Útil para comunicar hallazgos a audiencias no técnicas
- Combinar con clustering numérico para validación
- Ajustar conceptos según características del dominio específico
"""
        
        with open('/home/sedc/Proyectos/MineriaDeDatos/results/reportes/clustering_conceptual_reporte.txt', 'w', encoding='utf-8') as f:
            f.write(reporte)
        
        print("💾 Modelo guardado: results/modelos/clustering_conceptual.pkl")
        print("📄 Reporte guardado: results/reportes/clustering_conceptual_reporte.txt")
        
    except Exception as e:
        print(f"⚠️ Error guardando archivos: {e}")
    
    # 12. RESUMEN FINAL
    print()
    print("📝 RESUMEN CLUSTERING CONCEPTUAL:")
    print(f"   • Clusters conceptuales: {n_clusters_conceptual}")
    print(f"   • Conceptos definidos: {sum(len(c) for c in clusterer_conceptual.conceptos.values())}")
    print(f"   • Similitud con K-means (ARI): {ari_score:.3f}")
    print(f"   • Estabilidad: {estabilidad:.3f}")
    print(f"   • Comunidades agrupadas: {len(datos_limpios):,}")
    
    if estabilidad > 0.5:
        print("   🎉 ¡Clustering conceptual estable y consistente!")
    elif estabilidad > 0.3:
        print("   👍 Clustering conceptual moderadamente estable")
    else:
        print("   🔧 Clustering conceptual requiere ajuste de parámetros")
    
    if ari_score > 0.3:
        print("   🔗 Buena concordancia con clustering numérico")
    else:
        print("   🔄 Enfoque diferente al clustering numérico (normal)")
    
    print("   💡 Ideal para interpretación y toma de decisiones")
    print("   📋 Reglas claras para cada grupo demográfico")
    print("✅ CLUSTERING CONCEPTUAL COMPLETADO")
    
    return {
        'clusterer': clusterer_conceptual,
        'labels': labels_conceptual,
        'conceptos': clusterer_conceptual.conceptos,
        'reglas': clusterer_conceptual.reglas_clusters,
        'estabilidad': estabilidad,
        'ari_vs_kmeans': ari_score
    }

if __name__ == "__main__":
    ejecutar_clustering_conceptual()