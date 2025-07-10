#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLUSTERING PROBABILÍSTICO - TÉCNICAS NO SUPERVISADAS (Versión Simple)
Gaussian Mixture Models (GMM) para agrupar comunidades
"""

import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def encontrar_mejor_k_gmm(X, max_k=8):
    """Encontrar mejor número de componentes usando AIC/BIC"""
    k_range = range(2, max_k + 1)
    aic_scores = []
    bic_scores = []
    silhouette_scores = []
    
    for k in k_range:
        try:
            gmm = GaussianMixture(n_components=k, random_state=42, max_iter=100)
            gmm.fit(X)
            
            labels = gmm.predict(X)
            
            aic_scores.append(gmm.aic(X))
            bic_scores.append(gmm.bic(X))
            silhouette_scores.append(silhouette_score(X, labels))
            
        except:
            aic_scores.append(float('inf'))
            bic_scores.append(float('inf'))
            silhouette_scores.append(0)
    
    # Mejor K por diferentes criterios
    mejor_k_aic = k_range[np.argmin(aic_scores)]
    mejor_k_bic = k_range[np.argmin(bic_scores)]
    mejor_k_sil = k_range[np.argmax(silhouette_scores)]
    
    # Promedio de criterios
    mejor_k = int(np.mean([mejor_k_aic, mejor_k_bic, mejor_k_sil]))
    
    return mejor_k, k_range, aic_scores, bic_scores, silhouette_scores

def ejecutar_clustering_probabilistico():
    print("🎯 CLUSTERING PROBABILÍSTICO - TÉCNICAS NO SUPERVISADAS")
    print("="*54)
    print("📝 Objetivo: Agrupar comunidades usando modelos probabilísticos (GMM)")
    print()
    
    # 1. CARGAR DATOS
    archivo = '/home/sedc/Proyectos/MineriaDeDatos/data/ceros_sin_columnasAB_limpio_weka.csv'
    try:
        datos = pd.read_csv(archivo)
        print(f"✅ Datos cargados: {datos.shape[0]:,} filas, {datos.shape[1]} columnas")
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        return
    
    # 2. SELECCIONAR VARIABLES
    variables_probabilisticas = [
        'POBTOT', 'POBFEM', 'POBMAS', 'TOTHOG', 'VIVTOT',
        'P_15YMAS', 'P_60YMAS', 'GRAPROES', 'PEA', 'POCUPADA'
    ]
    
    variables_disponibles = [v for v in variables_probabilisticas if v in datos.columns]
    
    if len(variables_disponibles) < 4:
        print("❌ No hay suficientes variables para clustering probabilístico")
        return
    
    print(f"📊 Variables: {', '.join(variables_disponibles)}")
    
    # 3. PREPARAR DATOS
    datos_limpios = datos[variables_disponibles].dropna()
    
    # Reducir muestra para eficiencia
    if len(datos_limpios) > 3000:
        datos_limpios = datos_limpios.sample(n=3000, random_state=42)
        print(f"📝 Muestra reducida a {len(datos_limpios):,} registros")
    
    print(f"🧹 Datos finales: {len(datos_limpios):,} registros")
    
    # 4. ESCALADO DE DATOS
    scaler = StandardScaler()
    datos_escalados = scaler.fit_transform(datos_limpios)
    
    print(f"🔢 Datos escalados correctamente")
    print()
    
    # 5. ENCONTRAR MEJOR NÚMERO DE COMPONENTES
    print("📈 ENCONTRANDO NÚMERO ÓPTIMO DE COMPONENTES...")
    
    mejor_k, k_range, aic_scores, bic_scores, silhouette_scores = encontrar_mejor_k_gmm(datos_escalados)
    
    print(f"   🎯 Mejor K seleccionado: {mejor_k} componentes")
    print()
    
    # 6. ENTRENAR MODELOS PROBABILÍSTICOS
    print("🧠 ENTRENANDO MODELOS PROBABILÍSTICOS...")
    
    modelos = {}
    
    # GMM Estándar
    print("   🔄 Entrenando Gaussian Mixture Model...")
    gmm = GaussianMixture(n_components=mejor_k, random_state=42, max_iter=200)
    gmm.fit(datos_escalados)
    
    labels_gmm = gmm.predict(datos_escalados)
    probabilidades_gmm = gmm.predict_proba(datos_escalados)
    
    # Bayesian GMM
    print("   🔄 Entrenando Bayesian Gaussian Mixture Model...")
    try:
        bgmm = BayesianGaussianMixture(n_components=mejor_k+2, random_state=42, max_iter=200)
        bgmm.fit(datos_escalados)
        labels_bgmm = bgmm.predict(datos_escalados)
        probabilidades_bgmm = bgmm.predict_proba(datos_escalados)
        bgmm_ok = True
    except:
        print("      ⚠️ Bayesian GMM falló, usando solo GMM estándar")
        bgmm_ok = False
    
    print(f"   ✅ GMM estándar: {gmm.n_components} componentes")
    if bgmm_ok:
        print(f"   ✅ Bayesian GMM: {bgmm.n_components} componentes")
    print()
    
    # 7. EVALUACIÓN DE MODELOS
    print("📊 EVALUANDO MODELOS...")
    
    # Métricas GMM
    silhouette_gmm = silhouette_score(datos_escalados, labels_gmm)
    aic_gmm = gmm.aic(datos_escalados)
    bic_gmm = gmm.bic(datos_escalados)
    
    # Incertidumbre (entropía)
    entropias_gmm = -np.sum(probabilidades_gmm * np.log(probabilidades_gmm + 1e-10), axis=1)
    max_probs_gmm = np.max(probabilidades_gmm, axis=1)
    
    print(f"   📊 GMM Estándar:")
    print(f"      Silhouette: {silhouette_gmm:.3f}")
    print(f"      AIC: {aic_gmm:.1f} | BIC: {bic_gmm:.1f}")
    print(f"      Incertidumbre promedio: {np.mean(entropias_gmm):.3f}")
    
    # Métricas Bayesian GMM
    if bgmm_ok:
        try:
            silhouette_bgmm = silhouette_score(datos_escalados, labels_bgmm)
            entropias_bgmm = -np.sum(probabilidades_bgmm * np.log(probabilidades_bgmm + 1e-10), axis=1)
            
            print(f"   📊 Bayesian GMM:")
            print(f"      Silhouette: {silhouette_bgmm:.3f}")
            print(f"      Incertidumbre promedio: {np.mean(entropias_bgmm):.3f}")
            
            # Seleccionar mejor modelo
            if silhouette_bgmm > silhouette_gmm:
                mejor_modelo = bgmm
                mejor_nombre = "Bayesian GMM"
                mejores_labels = labels_bgmm
                mejores_probabilidades = probabilidades_bgmm
                entropias_finales = entropias_bgmm
            else:
                mejor_modelo = gmm
                mejor_nombre = "GMM Estándar"
                mejores_labels = labels_gmm
                mejores_probabilidades = probabilidades_gmm
                entropias_finales = entropias_gmm
        except:
            mejor_modelo = gmm
            mejor_nombre = "GMM Estándar"
            mejores_labels = labels_gmm
            mejores_probabilidades = probabilidades_gmm
            entropias_finales = entropias_gmm
    else:
        mejor_modelo = gmm
        mejor_nombre = "GMM Estándar"
        mejores_labels = labels_gmm
        mejores_probabilidades = probabilidades_gmm
        entropias_finales = entropias_gmm
    
    print()
    print(f"🏆 MEJOR MODELO: {mejor_nombre}")
    print(f"   Componentes: {mejor_modelo.n_components}")
    print(f"   Silhouette: {silhouette_score(datos_escalados, mejores_labels):.3f}")
    
    # 8. ANÁLISIS DE INCERTIDUMBRE
    print()
    print("🌊 ANÁLISIS DE INCERTIDUMBRE:")
    
    max_probs_finales = np.max(mejores_probabilidades, axis=1)
    
    print(f"   📊 Incertidumbre promedio: {np.mean(entropias_finales):.3f}")
    print(f"   📊 Confianza promedio: {np.mean(max_probs_finales):.3f}")
    print(f"   📊 Puntos inciertos (>0.5): {np.sum(entropias_finales > 0.5)} ({np.sum(entropias_finales > 0.5)/len(entropias_finales)*100:.1f}%)")
    
    # 9. COMPARACIÓN CON K-MEANS
    print()
    print("⚖️ COMPARACIÓN CON K-MEANS...")
    
    kmeans = KMeans(n_clusters=mejor_modelo.n_components, random_state=42, n_init=10)
    labels_kmeans = kmeans.fit_predict(datos_escalados)
    
    ari_score = adjusted_rand_score(mejores_labels, labels_kmeans)
    silhouette_kmeans = silhouette_score(datos_escalados, labels_kmeans)
    
    print(f"   📊 Silhouette K-Means: {silhouette_kmeans:.3f}")
    print(f"   📊 Silhouette GMM: {silhouette_score(datos_escalados, mejores_labels):.3f}")
    print(f"   📊 Similitud (ARI): {ari_score:.3f}")
    
    # 10. ANÁLISIS DE COMPONENTES
    print()
    print("🔍 ANÁLISIS DE COMPONENTES:")
    
    for i in range(mejor_modelo.n_components):
        peso = mejor_modelo.weights_[i]
        media = mejor_modelo.means_[i]
        
        print(f"\n   📊 COMPONENTE {i}:")
        print(f"      Peso: {peso:.3f} ({peso*100:.1f}%)")
        
        # Variables más características
        medias_globales = np.mean(datos_escalados, axis=0)
        diferencias = np.abs(media - medias_globales)
        indices_top = np.argsort(diferencias)[-3:][::-1]
        
        print(f"      Variables distintivas:")
        for idx in indices_top:
            var_name = variables_disponibles[idx]
            direction = "↑" if media[idx] > medias_globales[idx] else "↓"
            print(f"         • {var_name}: {media[idx]:.2f} ({direction})")
    
    # 11. PREPARAR VISUALIZACIÓN
    print()
    print("🎨 PREPARANDO VISUALIZACIÓN...")
    
    pca = PCA(n_components=2, random_state=42)
    datos_pca = pca.fit_transform(datos_escalados)
    
    varianza_explicada = pca.explained_variance_ratio_
    print(f"   📊 Varianza explicada: {sum(varianza_explicada)*100:.1f}%")
    
    # 12. VISUALIZACIONES
    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Gráfico 1: Criterios de selección
        axes[0,0].plot(k_range, aic_scores, 'bo-', label='AIC', linewidth=2)
        axes[0,0].plot(k_range, bic_scores, 'ro-', label='BIC', linewidth=2)
        axes[0,0].axvline(x=mejor_k, color='green', linestyle='--', label=f'Seleccionado: {mejor_k}')
        axes[0,0].set_title('📈 Criterios de Selección', fontweight='bold')
        axes[0,0].set_xlabel('Número de Componentes')
        axes[0,0].set_ylabel('Criterio')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Gráfico 2: Clustering en PCA
        scatter = axes[0,1].scatter(datos_pca[:, 0], datos_pca[:, 1], 
                                   c=mejores_labels, cmap='viridis', alpha=0.6, s=30)
        axes[0,1].set_title(f'🎯 {mejor_nombre}\nen Espacio PCA', fontweight='bold')
        axes[0,1].set_xlabel(f'PC1 ({varianza_explicada[0]*100:.1f}%)')
        axes[0,1].set_ylabel(f'PC2 ({varianza_explicada[1]*100:.1f}%)')
        
        # Gráfico 3: Comparación con K-Means
        axes[0,2].scatter(datos_pca[:, 0], datos_pca[:, 1], 
                         c=labels_kmeans, cmap='viridis', alpha=0.6, s=30)
        axes[0,2].set_title('📊 K-Means\n(Comparación)', fontweight='bold')
        axes[0,2].set_xlabel(f'PC1 ({varianza_explicada[0]*100:.1f}%)')
        axes[0,2].set_ylabel(f'PC2 ({varianza_explicada[1]*100:.1f}%)')
        
        # Gráfico 4: Distribución de confianza
        axes[1,0].hist(max_probs_finales, bins=20, alpha=0.7, color='purple', edgecolor='black')
        axes[1,0].axvline(x=np.mean(max_probs_finales), color='red', linestyle='--',
                         label=f'Promedio: {np.mean(max_probs_finales):.3f}')
        axes[1,0].set_title('📊 Distribución de Confianza', fontweight='bold')
        axes[1,0].set_xlabel('Probabilidad Máxima')
        axes[1,0].set_ylabel('Frecuencia')
        axes[1,0].legend()
        
        # Gráfico 5: Distribución de incertidumbre
        axes[1,1].hist(entropias_finales, bins=20, alpha=0.7, color='orange', edgecolor='black')
        axes[1,1].axvline(x=np.mean(entropias_finales), color='red', linestyle='--',
                         label=f'Promedio: {np.mean(entropias_finales):.3f}')
        axes[1,1].set_title('🌀 Distribución de Incertidumbre', fontweight='bold')
        axes[1,1].set_xlabel('Entropía')
        axes[1,1].set_ylabel('Frecuencia')
        axes[1,1].legend()
        
        # Gráfico 6: Pesos de componentes
        pesos = mejor_modelo.weights_
        componentes_ids = [f'C{i}' for i in range(len(pesos))]
        axes[1,2].pie(pesos, labels=componentes_ids, autopct='%1.1f%%', startangle=90)
        axes[1,2].set_title(f'⚖️ Pesos de Componentes\n{mejor_nombre}', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('/home/sedc/Proyectos/MineriaDeDatos/results/graficos/clustering_probabilistico.png', 
                   dpi=150, bbox_inches='tight')
        plt.show()
        
        print("💾 Gráficos guardados: results/graficos/clustering_probabilistico.png")
        
    except Exception as e:
        print(f"⚠️ Error en visualizaciones: {e}")
    
    # 13. GUARDAR RESULTADOS
    try:
        import joblib
        
        # Guardar modelos
        joblib.dump(mejor_modelo, '/home/sedc/Proyectos/MineriaDeDatos/results/modelos/mejor_clustering_probabilistico.pkl')
        joblib.dump(scaler, '/home/sedc/Proyectos/MineriaDeDatos/results/modelos/scaler_probabilistico.pkl')
        joblib.dump(pca, '/home/sedc/Proyectos/MineriaDeDatos/results/modelos/pca_probabilistico.pkl')
        
        # Crear reporte
        reporte = f"""
REPORTE CLUSTERING PROBABILÍSTICO - TÉCNICAS NO SUPERVISADAS
==========================================================

MEJOR MODELO: {mejor_nombre}
Número de componentes: {mejor_modelo.n_components}
Silhouette Score: {silhouette_score(datos_escalados, mejores_labels):.3f}

SELECCIÓN DE COMPONENTES:
- Mejor K encontrado: {mejor_k}
- Criterio: Promedio de AIC, BIC y Silhouette

ANÁLISIS DE INCERTIDUMBRE:
- Incertidumbre promedio: {np.mean(entropias_finales):.3f}
- Confianza promedio: {np.mean(max_probs_finales):.3f}
- Puntos inciertos: {np.sum(entropias_finales > 0.5)} ({np.sum(entropias_finales > 0.5)/len(entropias_finales)*100:.1f}%)

COMPARACIÓN CON K-MEANS:
- Silhouette K-Means: {silhouette_kmeans:.3f}
- Silhouette GMM: {silhouette_score(datos_escalados, mejores_labels):.3f}
- Similitud (ARI): {ari_score:.3f}

COMPONENTES IDENTIFICADAS:
"""
        
        for i in range(mejor_modelo.n_components):
            peso = mejor_modelo.weights_[i]
            reporte += f"""
Componente {i}:
  - Peso: {peso:.3f} ({peso*100:.1f}%)"""
        
        reporte += f"""

VARIABLES UTILIZADAS:
{', '.join(variables_disponibles)}

CONFIGURACIÓN:
- Registros procesados: {len(datos_limpios):,}
- Variables: {len(variables_disponibles)}
- Escalado: StandardScaler aplicado
- Visualización: PCA con {sum(varianza_explicada)*100:.1f}% varianza explicada

VENTAJAS DEL CLUSTERING PROBABILÍSTICO:
- Asignación suave (probabilidades de pertenencia)
- Cuantificación natural de incertidumbre
- Fundamentación estadística sólida
- Selección automática de componentes óptimas

INTERPRETACIÓN:
- Silhouette > 0.5: Excelente separación
- Silhouette > 0.3: Buena separación
- Incertidumbre baja: Asignaciones confiables
- ARI alto: Concordancia con métodos alternativos

RECOMENDACIONES:
- Usar para detección de anomalías (baja probabilidad)
- Ideal cuando se requiere cuantificar confianza
- Complementar con análisis de componentes
- Considerar puntos con alta incertidumbre para revisión
"""
        
        with open('/home/sedc/Proyectos/MineriaDeDatos/results/reportes/clustering_probabilistico_reporte.txt', 'w', encoding='utf-8') as f:
            f.write(reporte)
        
        print("💾 Modelo guardado: results/modelos/mejor_clustering_probabilistico.pkl")
        print("💾 Scaler guardado: results/modelos/scaler_probabilistico.pkl")
        print("💾 PCA guardado: results/modelos/pca_probabilistico.pkl")
        print("📄 Reporte guardado: results/reportes/clustering_probabilistico_reporte.txt")
        
    except Exception as e:
        print(f"⚠️ Error guardando archivos: {e}")
    
    # 14. RESUMEN FINAL
    print()
    print("📝 RESUMEN CLUSTERING PROBABILÍSTICO:")
    print(f"   • Mejor modelo: {mejor_nombre}")
    print(f"   • Componentes: {mejor_modelo.n_components}")
    print(f"   • Calidad (Silhouette): {silhouette_score(datos_escalados, mejores_labels):.3f}")
    print(f"   • Incertidumbre promedio: {np.mean(entropias_finales):.3f}")
    print(f"   • Confianza promedio: {np.mean(max_probs_finales):.3f}")
    
    silhouette_final = silhouette_score(datos_escalados, mejores_labels)
    if silhouette_final > 0.5:
        print("   🎉 ¡Excelente clustering probabilístico!")
    elif silhouette_final > 0.3:
        print("   👍 Buen clustering con distribuciones identificadas")
    else:
        print("   🔧 Clustering moderado, considerar ajustar parámetros")
    
    if np.mean(entropias_finales) < 0.5:
        print("   🎯 Baja incertidumbre - asignaciones confiables")
    else:
        print("   🌀 Incertidumbre moderada - algunos puntos ambiguos")
    
    print("   💡 Ideal para cuantificar incertidumbre en agrupaciones")
    print("   🎲 Membership suave permite asignaciones probabilísticas")
    print("✅ CLUSTERING PROBABILÍSTICO COMPLETADO")
    
    return {
        'mejor_modelo': mejor_modelo,
        'modelo_nombre': mejor_nombre,
        'labels': mejores_labels,
        'probabilidades': mejores_probabilidades,
        'entropias': entropias_finales,
        'silhouette': silhouette_final,
        'ari_vs_kmeans': ari_score,
        'scaler': scaler,
        'pca': pca
    }

if __name__ == "__main__":
    ejecutar_clustering_probabilistico()