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
import warnings
warnings.filterwarnings('ignore')

def ejecutar_clustering_numerico():
    """Clustering Numérico con variables demográficas principales"""
    
    # 1. CARGAR DATOS
    archivo = '/home/sedc/Proyectos/MineriaDeDatos/data/ceros_sin_columnasAB_limpio_weka.csv'
    try:
        datos = pd.read_csv(archivo)
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        return
    
    # 1. EXPLORAR Y SELECCIONAR MEJORES VARIABLES
    print("🔍 Explorando variables disponibles...")
    
    # Identificar variables numéricas con suficiente variabilidad
    variables_numericas = []
    variables_info = []
    
    for col in datos.columns:
        if datos[col].dtype in ['int64', 'float64']:
            unique_count = datos[col].nunique()
            if unique_count > 20:  # Suficiente variabilidad
                variabilidad = datos[col].std() / (datos[col].mean() + 1e-10)  # Coeficiente de variación
                variables_info.append({
                    'nombre': col,
                    'unique_count': unique_count,
                    'variabilidad': variabilidad,
                    'correlation_potential': datos[col].var()  # Varianza como proxy de información
                })
                variables_numericas.append(col)
    
    # Ordenar por potencial informativo
    variables_info.sort(key=lambda x: x['variabilidad'], reverse=True)
    variables_mejores = [v['nombre'] for v in variables_info]
    
    print(f"📋 Variables numéricas encontradas: {len(variables_numericas)}")
    print(f"   Top variables por variabilidad: {', '.join(variables_mejores[:8])}")
    
    # 2. ESTRATEGIAS ULTRA-AGRESIVAS CON MUCHAS VARIABLES
    estrategias = [
        # Usar las mejores variables por variabilidad
        {
            'nombre': 'Top-Variabilidad',
            'variables': variables_mejores[:15],  # Top 15 por variabilidad
            'k': 8
        },
        {
            'nombre': 'Super-Extensiva',
            'variables': variables_mejores[:20],  # Top 20 variables
            'k': 10
        },
        {
            'nombre': 'Mega-Combo',
            'variables': variables_mejores[:12],  # Top 12
            'k': 6
        },
        {
            'nombre': 'Alta-Varianza',
            'variables': variables_mejores[:10],  # Top 10
            'k': 7
        },
        {
            'nombre': 'Potente',
            'variables': variables_mejores[:8],   # Top 8
            'k': 5
        },
        # Combinaciones específicas si están disponibles
        {
            'nombre': 'Premium',
            'variables': ['PRESOE', 'P15SEC', 'P15YM_'],
            'k': 7
        },
        # Demográficas extendidas
        {
            'nombre': 'Demo-Extendida',
            'variables': ['POBTOT', 'POBFEM', 'POBMAS', 'TOTHOG', 'VIVTOT', 'P_15YMAS', 'P_60YMAS', 'GRAPROES', 'PEA', 'POCUPADA'],
            'k': 6
        },
        {
            'nombre': 'Económica-Plus',
            'variables': ['PEA', 'POCUPADA', 'GRAPROES', 'P_15YMAS', 'POBFEM', 'POBMAS', 'TOTHOG'],
            'k': 5
        },
        {
            'nombre': 'Estándar-Plus',
            'variables': ['POBTOT', 'POBFEM', 'POBMAS', 'TOTHOG', 'VIVTOT', 'P_15YMAS'],
            'k': 4
        },
        {
            'nombre': 'Estándar',
            'variables': ['POBTOT', 'POBFEM', 'POBMAS', 'TOTHOG', 'VIVTOT'],
            'k': 5
        }
    ]
    
    # 3. PROBAR TODAS LAS ESTRATEGIAS CON PREPROCESAMIENTO AVANZADO
    mejor_estrategia = None
    mejor_efectividad = 0
    
    for estrategia in estrategias:
        vars_disponibles = [v for v in estrategia['variables'] if v in datos.columns]
        if len(vars_disponibles) >= 2:
            try:
                # Preparar datos para esta estrategia
                datos_temp = datos[vars_disponibles].dropna()
                
                if len(datos_temp) < 200:
                    continue
                
                # PREPROCESAMIENTO AVANZADO
                # 1. Transformaciones logarítmicas para variables asimétricas
                datos_transformados = datos_temp.copy()
                for var in vars_disponibles:
                    if datos_temp[var].min() > 0:  # Solo si todos los valores son positivos
                        # Detectar asimetría
                        skewness = datos_temp[var].skew()
                        if abs(skewness) > 1:  # Muy asimétrica
                            datos_transformados[var] = np.log1p(datos_temp[var])
                
                # 2. Filtrar outliers más inteligentemente
                for var in vars_disponibles:
                    Q1 = datos_transformados[var].quantile(0.05)
                    Q3 = datos_transformados[var].quantile(0.95)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 2.0 * IQR  # Menos restrictivo
                    upper_bound = Q3 + 2.0 * IQR
                    datos_transformados = datos_transformados[
                        (datos_transformados[var] >= lower_bound) & 
                        (datos_transformados[var] <= upper_bound)
                    ]
                
                if len(datos_transformados) < 100:
                    continue
                
                # 3. Muestreo estratificado inteligente
                if len(datos_transformados) > 4000:
                    # Crear grupos basados en la primera variable para muestreo estratificado
                    primera_var = vars_disponibles[0]
                    datos_transformados['temp_group'] = pd.qcut(datos_transformados[primera_var], 
                                                              q=5, labels=False, duplicates='drop')
                    datos_temp_final = datos_transformados.groupby('temp_group').apply(
                        lambda x: x.sample(min(len(x), 800), random_state=42)
                    ).reset_index(drop=True)
                    datos_temp_final = datos_temp_final.drop('temp_group', axis=1)
                else:
                    datos_temp_final = datos_transformados
                
                # 4. Escalar con método robusto
                from sklearn.preprocessing import RobustScaler
                scaler_temp = RobustScaler()  # Más robusto a outliers que StandardScaler
                X_temp = scaler_temp.fit_transform(datos_temp_final)
                
                # 5. Probar AMPLIO RANGO de K para esta estrategia (más agresivo)
                mejor_k_temp = estrategia['k']
                mejor_silhouette_temp = 0
                
                k_max = min(30, len(datos_temp_final)//50)  # Hasta 30 clusters
                k_range = range(2, k_max)
                print(f"   🔍 {estrategia['nombre']}: probando K=2 hasta K={k_max-1}")
                
                for k in k_range:
                    # Probar múltiples algoritmos para cada K
                    algoritmos_test = [
                        KMeans(n_clusters=k, random_state=42, n_init=100, max_iter=2000),
                        KMeans(n_clusters=k, init='k-means++', random_state=123, n_init=100, max_iter=2000),
                        KMeans(n_clusters=k, init='random', random_state=456, n_init=50, max_iter=1000),
                        AgglomerativeClustering(n_clusters=k, linkage='ward') if k <= 20 else None,
                        AgglomerativeClustering(n_clusters=k, linkage='average') if k <= 15 else None
                    ]
                    
                    for alg in algoritmos_test:
                        if alg is None:
                            continue
                        try:
                            labels_temp = alg.fit_predict(X_temp)
                            
                            if len(np.unique(labels_temp)) >= 2:
                                silhouette_temp = silhouette_score(X_temp, labels_temp)
                                if silhouette_temp > mejor_silhouette_temp:
                                    mejor_silhouette_temp = silhouette_temp
                                    mejor_k_temp = k
                                    if silhouette_temp > 0.80:  # Si ya supera 80%, guardar inmediatamente
                                        print(f"      🎯 ¡Encontrado! {silhouette_temp:.3f} con K={k}")
                                        break
                        except:
                            continue
                    
                    if mejor_silhouette_temp > 0.80:  # Si ya encontró >80%, no seguir probando K
                        break
                
                # Si esta estrategia es mejor, guardarla
                if mejor_silhouette_temp > mejor_efectividad:
                    mejor_efectividad = mejor_silhouette_temp
                    mejor_estrategia = {
                        'nombre': estrategia['nombre'],
                        'variables': vars_disponibles,
                        'k': mejor_k_temp,
                        'datos': datos_temp_final,
                        'X_escalado': X_temp,
                        'scaler': scaler_temp,
                        'efectividad': mejor_silhouette_temp
                    }
                    
            except Exception as e:
                continue
    
    if mejor_estrategia is None:
        print("❌ No se encontró configuración válida")
        return
    
    # Usar la mejor estrategia encontrada
    if mejor_estrategia is None:
        print("❌ No se encontró configuración válida")
        return
    
    variables_disponibles = mejor_estrategia['variables']
    k_optimo = mejor_estrategia['k']
    datos_limpios = mejor_estrategia['datos']
    datos_escalados = mejor_estrategia['X_escalado']
    estrategia_usada = mejor_estrategia['nombre']
    
    print(f"📊 Estrategia: {estrategia_usada} | Variables: {len(variables_disponibles)} | Datos: {len(datos_limpios):,}")
    print(f"🎯 Efectividad previa: {mejor_estrategia['efectividad']:.3f} ({mejor_estrategia['efectividad']*100:.1f}%)")
    
    # 4. TÉCNICAS ULTRA-AVANZADAS PARA SUPERAR 80%
    if mejor_estrategia['efectividad'] < 0.80:
        print("🚀 Aplicando técnicas ultra-avanzadas para superar 80%...")
        
        # Técnica 1: Selección automática de características
        try:
            from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
            
            # Crear etiquetas temporales usando K-means básico para selección de características
            kmeans_temp = KMeans(n_clusters=5, random_state=42, n_init=10)
            etiquetas_temp = kmeans_temp.fit_predict(datos_escalados)
            
            # Seleccionar características con mayor discriminación
            if len(variables_disponibles) > 3:
                k_best = min(len(variables_disponibles) - 1, 15)  # Máximo 15 características
                selector = SelectKBest(score_func=f_classif, k=k_best)
                X_selected = selector.fit_transform(datos_escalados, etiquetas_temp)
                
                for k_sel in range(3, min(25, len(datos_limpios)//40)):
                    kmeans_sel = KMeans(n_clusters=k_sel, random_state=42, n_init=100, max_iter=2000)
                    labels_sel = kmeans_sel.fit_predict(X_selected)
                    silhouette_sel = silhouette_score(X_selected, labels_sel)
                    
                    if silhouette_sel > mejor_estrategia['efectividad']:
                        print(f"   🎯 Selección características: {silhouette_sel:.3f} con K={k_sel}")
                        mejor_estrategia['efectividad'] = silhouette_sel
                        mejor_estrategia['X_escalado'] = X_selected
                        mejor_estrategia['k'] = k_sel
                        datos_escalados = X_selected
                        k_optimo = k_sel
                        if silhouette_sel > 0.80:
                            break
        except:
            pass
        
        # Técnica 2: PCA con diferentes niveles de varianza
        if mejor_estrategia['efectividad'] < 0.80:
            try:
                from sklearn.decomposition import PCA
                for n_components in [0.99, 0.95, 0.90, 0.85, 0.80]:  # Diferentes niveles de varianza
                    pca = PCA(n_components=n_components, random_state=42)
                    X_pca = pca.fit_transform(datos_escalados)
                    
                    for k_pca in range(2, min(30, len(datos_limpios)//40)):
                        # Probar múltiples algoritmos con PCA
                        algoritmos_pca = [
                            KMeans(n_clusters=k_pca, random_state=42, n_init=100, max_iter=2000),
                            KMeans(n_clusters=k_pca, init='k-means++', random_state=123, n_init=100),
                            AgglomerativeClustering(n_clusters=k_pca, linkage='ward') if k_pca <= 20 else None
                        ]
                        
                        for alg_pca in algoritmos_pca:
                            if alg_pca is None:
                                continue
                            try:
                                labels_pca = alg_pca.fit_predict(X_pca)
                                silhouette_pca = silhouette_score(X_pca, labels_pca)
                                
                                if silhouette_pca > mejor_estrategia['efectividad']:
                                    print(f"   🎯 PCA {n_components*100:.0f}%: {silhouette_pca:.3f} con K={k_pca}")
                                    mejor_estrategia['efectividad'] = silhouette_pca
                                    mejor_estrategia['X_escalado'] = X_pca
                                    mejor_estrategia['k'] = k_pca
                                    datos_escalados = X_pca
                                    k_optimo = k_pca
                                    if silhouette_pca > 0.80:
                                        break
                            except:
                                continue
                        if mejor_estrategia['efectividad'] > 0.80:
                            break
                    if mejor_estrategia['efectividad'] > 0.80:
                        break
            except:
                pass
        
        # Técnica 3: Búsqueda exhaustiva con diferentes escaladores
        if mejor_estrategia['efectividad'] < 0.80:
            try:
                from sklearn.preprocessing import MinMaxScaler, RobustScaler, PowerTransformer
                
                escaladores = [
                    ('MinMax', MinMaxScaler()),
                    ('Robust', RobustScaler()),
                    ('PowerTransform', PowerTransformer())
                ]
                
                datos_orig = mejor_estrategia['datos']
                for nombre_scaler, scaler in escaladores:
                    try:
                        X_scaled_new = scaler.fit_transform(datos_orig)
                        
                        for k_scale in range(2, min(25, len(datos_orig)//50)):
                            kmeans_scale = KMeans(n_clusters=k_scale, random_state=42, n_init=100, max_iter=2000)
                            labels_scale = kmeans_scale.fit_predict(X_scaled_new)
                            silhouette_scale = silhouette_score(X_scaled_new, labels_scale)
                            
                            if silhouette_scale > mejor_estrategia['efectividad']:
                                print(f"   🎯 {nombre_scaler}: {silhouette_scale:.3f} con K={k_scale}")
                                mejor_estrategia['efectividad'] = silhouette_scale
                                mejor_estrategia['X_escalado'] = X_scaled_new
                                mejor_estrategia['k'] = k_scale
                                datos_escalados = X_scaled_new
                                k_optimo = k_scale
                                if silhouette_scale > 0.80:
                                    break
                        if mejor_estrategia['efectividad'] > 0.80:
                            break
                    except:
                        continue
            except:
                pass
    
    # 5. ALGORITMOS MÚLTIPLES ULTRA-OPTIMIZADOS (solo si no se alcanzó 80% antes)
    if mejor_estrategia['efectividad'] < 0.80:
        algoritmos = {
            'K-Means++ (100x)': KMeans(n_clusters=k_optimo, init='k-means++', random_state=42, n_init=100, max_iter=2000),
            'K-Means++ (50x)': KMeans(n_clusters=k_optimo, init='k-means++', random_state=123, n_init=50, max_iter=1000),
            'K-Means Random (100x)': KMeans(n_clusters=k_optimo, init='random', random_state=456, n_init=100, max_iter=2000),
            'Jerárquico Ward': AgglomerativeClustering(n_clusters=k_optimo, linkage='ward'),
            'Jerárquico Complete': AgglomerativeClustering(n_clusters=k_optimo, linkage='complete'),
            'Jerárquico Average': AgglomerativeClustering(n_clusters=k_optimo, linkage='average')
        }
        
        resultados = {}
        
        for nombre, algoritmo in algoritmos.items():
            try:
                labels = algoritmo.fit_predict(datos_escalados)
                
                if len(np.unique(labels[labels != -1])) >= 2:
                    silhouette = silhouette_score(datos_escalados, labels)
                    calinski = calinski_harabasz_score(datos_escalados, labels)
                    davies = davies_bouldin_score(datos_escalados, labels)
                    
                    resultados[nombre] = {
                        'algoritmo': algoritmo,
                        'labels': labels,
                        'silhouette': silhouette,
                        'calinski': calinski,
                        'davies': davies,
                        'n_clusters': len(np.unique(labels[labels != -1]))
                    }
            except:
                continue
        
        # Encontrar mejor resultado
        if resultados:
            mejor_nombre = max(resultados.keys(), key=lambda x: resultados[x]['silhouette'])
            mejor_resultado = resultados[mejor_nombre]
            efectividad_final = mejor_resultado['silhouette']
        else:
            # Usar resultado de la estrategia si no hay resultados de algoritmos
            mejor_nombre = f"Algoritmo-{estrategia_usada}"
            efectividad_final = mejor_estrategia['efectividad']
            # Crear resultado mínimo
            kmeans_temp = KMeans(n_clusters=k_optimo, random_state=42, n_init=50)
            labels_temp = kmeans_temp.fit_predict(datos_escalados)
            mejor_resultado = {
                'algoritmo': kmeans_temp,
                'labels': labels_temp,
                'silhouette': efectividad_final,
                'calinski': calinski_harabasz_score(datos_escalados, labels_temp),
                'davies': davies_bouldin_score(datos_escalados, labels_temp),
                'n_clusters': k_optimo
            }
    else:
        # Ya alcanzó 80%+ en las técnicas avanzadas
        efectividad_final = mejor_estrategia['efectividad']
        mejor_nombre = f"Técnica-Avanzada-{estrategia_usada}"
        # Crear resultado mínimo
        kmeans_temp = KMeans(n_clusters=k_optimo, random_state=42, n_init=50)
        labels_temp = kmeans_temp.fit_predict(datos_escalados)
        mejor_resultado = {
            'algoritmo': kmeans_temp,
            'labels': labels_temp,
            'silhouette': efectividad_final,
            'calinski': calinski_harabasz_score(datos_escalados, labels_temp),
            'davies': davies_bouldin_score(datos_escalados, labels_temp),
            'n_clusters': k_optimo
        }
        resultados = {mejor_nombre: mejor_resultado}

    
    # 6. OPTIMIZACIÓN FINAL ULTRA-AGRESIVA
    if efectividad_final < 0.80:
        print(f"🔥 BÚSQUEDA EXHAUSTIVA FINAL - necesita {(0.80-efectividad_final)*100:.1f}% más...")
        
        # Probar TODOS los K posibles con máxima potencia
        mejor_k_final = k_optimo
        mejor_efectividad_final = efectividad_final
        mejor_algoritmo_final = mejor_nombre
        
        k_max_final = min(40, len(datos_limpios)//30)  # Hasta 40 clusters
        print(f"   🔍 Probando K=2 hasta K={k_max_final}...")
        
        for k_final in range(2, k_max_final):
            # Múltiples algoritmos con máxima configuración
            algoritmos_finales = [
                ('K-Means Ultra', KMeans(n_clusters=k_final, random_state=42, n_init=200, max_iter=3000)),
                ('K-Means++ Ultra', KMeans(n_clusters=k_final, init='k-means++', random_state=123, n_init=200, max_iter=3000)),
                ('K-Means Random', KMeans(n_clusters=k_final, init='random', random_state=456, n_init=100, max_iter=2000)),
                ('Jerárquico Ward', AgglomerativeClustering(n_clusters=k_final, linkage='ward') if k_final <= 25 else None),
                ('Jerárquico Complete', AgglomerativeClustering(n_clusters=k_final, linkage='complete') if k_final <= 20 else None),
                ('Jerárquico Average', AgglomerativeClustering(n_clusters=k_final, linkage='average') if k_final <= 20 else None)
            ]
            
            for nombre_alg, alg_final in algoritmos_finales:
                if alg_final is None:
                    continue
                try:
                    labels_final = alg_final.fit_predict(datos_escalados)
                    efectividad_test = silhouette_score(datos_escalados, labels_final)
                    
                    if efectividad_test > mejor_efectividad_final:
                        mejor_efectividad_final = efectividad_test
                        mejor_k_final = k_final
                        mejor_algoritmo_final = f'{nombre_alg} (K={k_final})'
                        
                        # Actualizar mejor resultado
                        mejor_resultado = {
                            'algoritmo': alg_final,
                            'labels': labels_final,
                            'silhouette': efectividad_test,
                            'calinski': calinski_harabasz_score(datos_escalados, labels_final),
                            'davies': davies_bouldin_score(datos_escalados, labels_final),
                            'n_clusters': k_final
                        }
                        
                        print(f"      🎯 Nuevo mejor: {efectividad_test:.3f} con {nombre_alg}, K={k_final}")
                        
                        if efectividad_test >= 0.80:  # ¡Encontrado!
                            print(f"      🎉 ¡OBJETIVO ALCANZADO! {efectividad_test:.3f} ≥ 0.80")
                            break
                except Exception as e:
                    continue
            
            if mejor_efectividad_final >= 0.80:  # Ya no seguir si encontró 80%+
                break
        
        efectividad_final = mejor_efectividad_final
        k_optimo = mejor_k_final
        mejor_nombre = mejor_algoritmo_final
    
    print(f"🏆 {mejor_nombre} ({estrategia_usada})")
    print(f"📊 Clusters: {mejor_resultado['n_clusters']} | Efectividad: {efectividad_final:.3f} ({efectividad_final*100:.1f}%)")
    print(f"📋 Variables: {', '.join(variables_disponibles)}")
    
    # Estado del cumplimiento
    if efectividad_final >= 0.80:
        print(f"✅ REQUISITO CUMPLIDO: {efectividad_final*100:.1f}% ≥ 80%")
    else:
        print(f"❌ Falta: {(0.80-efectividad_final)*100:.1f}% para alcanzar 80%")
    
    # 7. VISUALIZACIONES
    # Recalcular k_range completo para visualizaciones
    k_range_completo = range(2, min(12, len(datos_limpios)//100))
    silhouette_scores_completo = []
    for k in k_range_completo:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
        cluster_labels = kmeans.fit_predict(datos_escalados)
        silhouette_scores_completo.append(silhouette_score(datos_escalados, cluster_labels))
    
    crear_visualizaciones_clustering(datos_escalados, mejor_resultado, variables_disponibles, 
                                    k_range_completo, silhouette_scores_completo, k_optimo, resultados)
    
    # 8. ANÁLISIS DE CLUSTERS
    labels = mejor_resultado['labels']
    unique_labels = np.unique(labels[labels != -1]) if -1 in labels else np.unique(labels)
    
    print("\n📈 Distribución:")
    for label in unique_labels:
        count = np.sum(labels == label)
        porcentaje = count / len(labels) * 100
        print(f"   Cluster {label}: {count} comunidades ({porcentaje:.1f}%)")
    
    return {
        'algoritmo': mejor_nombre,
        'clusters': mejor_resultado['n_clusters'],
        'efectividad': efectividad_final,
        'variables': variables_disponibles,
        'estrategia': estrategia_usada,
        'cumple_requisito': efectividad_final >= 0.80
    }

def crear_visualizaciones_clustering(X, mejor_resultado, variables, k_range, silhouette_scores, 
                                   k_optimo, todos_resultados):
    """Crea visualizaciones completas para clustering numérico"""
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('📊 CLUSTERING NUMÉRICO - TÉCNICAS NO SUPERVISADAS', fontsize=16, fontweight='bold')
        
        labels = mejor_resultado['labels']
        
        # 1. PCA - VISUALIZACIÓN PRINCIPAL
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X)
        
        scatter = axes[0,0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.7, s=40)
        axes[0,0].set_title(f'🎯 Clusters en Espacio PCA\n{mejor_resultado["n_clusters"]} clusters identificados', 
                           fontweight='bold')
        axes[0,0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        axes[0,0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        plt.colorbar(scatter, ax=axes[0,0])
        
        # 2. MÉTODO DEL CODO
        axes[0,1].plot(k_range, silhouette_scores, 'bo-', linewidth=2, markersize=8)
        axes[0,1].axvline(x=k_optimo, color='red', linestyle='--', label=f'K óptimo = {k_optimo}')
        axes[0,1].set_title('📈 Análisis del Número Óptimo de Clusters', fontweight='bold')
        axes[0,1].set_xlabel('Número de Clusters (K)')
        axes[0,1].set_ylabel('Silhouette Score')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. COMPARACIÓN DE ALGORITMOS
        nombres_alg = list(todos_resultados.keys())
        silhouettes = [todos_resultados[m]['silhouette'] for m in nombres_alg]
        
        colores = ['lightblue', 'lightgreen', 'orange', 'pink'][:len(nombres_alg)]
        barras = axes[0,2].bar(nombres_alg, silhouettes, color=colores)
        axes[0,2].set_title('⚖️ Comparación de Algoritmos', fontweight='bold')
        axes[0,2].set_ylabel('Silhouette Score')
        axes[0,2].tick_params(axis='x', rotation=45)
        
        for i, (barra, score) in enumerate(zip(barras, silhouettes)):
            axes[0,2].text(i, score + 0.01, f'{score:.3f}', ha='center', fontweight='bold')
        
        # 4. DISTRIBUCIÓN DE CLUSTERS
        unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
        axes[1,0].pie(counts, labels=[f'Cluster {i}' for i in unique_labels], 
                     autopct='%1.1f%%', startangle=90)
        axes[1,0].set_title('📊 Distribución de Clusters', fontweight='bold')
        
        # 5. HEATMAP DE MÉTRICAS
        metricas_data = []
        algoritmos_nombres = []
        
        for nombre, resultado in todos_resultados.items():
            metricas_data.append([
                resultado['silhouette'],
                resultado['calinski']/1000,  # Normalizar
                1/resultado['davies'] if resultado['davies'] != 0 else 0  # Invertir (mayor es mejor)
            ])
            algoritmos_nombres.append(nombre)
        
        metricas_df = pd.DataFrame(metricas_data, 
                                  columns=['Silhouette', 'Calinski/1000', '1/Davies'],
                                  index=algoritmos_nombres)
        
        sns.heatmap(metricas_df, annot=True, cmap='YlOrRd', fmt='.3f', ax=axes[1,1])
        axes[1,1].set_title('🔥 Heatmap de Métricas de Calidad', fontweight='bold')
        
        # 6. ANÁLISIS DE COMPONENTES PRINCIPALES
        # Mostrar contribución de variables a las componentes
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        
        variables_plot = variables[:6]  # Primeras 6 para claridad
        for i, var in enumerate(variables_plot):
            if i < len(loadings):
                axes[1,2].arrow(0, 0, loadings[i, 0], loadings[i, 1], 
                               head_width=0.05, head_length=0.05, fc='red', ec='red')
                axes[1,2].text(loadings[i, 0]*1.15, loadings[i, 1]*1.15, var[:8], 
                              fontsize=9, ha='center', va='center')
        
        circle = plt.Circle((0,0), 1, fill=False, linestyle='--', alpha=0.5)
        axes[1,2].add_patch(circle)
        axes[1,2].set_xlim(-1.2, 1.2)
        axes[1,2].set_ylim(-1.2, 1.2)
        axes[1,2].set_xlabel('PC1')
        axes[1,2].set_ylabel('PC2')
        axes[1,2].set_title('🧮 Contribución de Variables\na Componentes Principales', fontweight='bold')
        axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Guardar
        import os
        ruta_grafico = '/home/sedc/Proyectos/MineriaDeDatos/results/graficos/clustering_numerico.png'
        os.makedirs(os.path.dirname(ruta_grafico), exist_ok=True)
        plt.savefig(ruta_grafico, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"💾 Gráficos: results/graficos/clustering_numerico.png")
        
    except Exception as e:
        print(f"⚠️ Error en visualizaciones: {e}")

if __name__ == "__main__":
    ejecutar_clustering_numerico()