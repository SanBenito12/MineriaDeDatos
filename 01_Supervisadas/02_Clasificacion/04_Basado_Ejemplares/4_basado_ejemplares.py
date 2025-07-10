#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLASIFICACIÓN BASADA EN EJEMPLARES (K-NN)
Clasifica basándose en los vecinos más cercanos
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

def crear_categorias_poblacion(poblacion):
    """Crear categorías de población para clasificación"""
    if poblacion <= 1000:
        return 'Pequeña'
    elif poblacion <= 5000:
        return 'Mediana'
    elif poblacion <= 20000:
        return 'Grande'
    else:
        return 'Muy Grande'

def encontrar_k_optimo(X_train, y_train, k_max=20):
    """Encuentra el mejor valor de K usando validación cruzada"""
    k_values = range(1, min(k_max + 1, len(X_train) // 5))
    cv_scores = []
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
        scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
        cv_scores.append(scores.mean())
    
    mejor_k = k_values[np.argmax(cv_scores)]
    mejor_score = max(cv_scores)
    
    return mejor_k, mejor_score, list(k_values), cv_scores

def analizar_distancias(X_train, X_test, n_muestras=100):
    """Analiza la distribución de distancias entre puntos"""
    # Tomar muestra para análisis (computacionalmente costoso)
    if len(X_train) > n_muestras:
        indices = np.random.choice(len(X_train), n_muestras, replace=False)
        X_muestra = X_train[indices]
    else:
        X_muestra = X_train
    
    # Calcular distancias
    distancias = cdist(X_muestra, X_muestra, metric='euclidean')
    
    # Estadísticas de distancias
    distancias_no_cero = distancias[distancias > 0]
    
    stats = {
        'promedio': np.mean(distancias_no_cero),
        'mediana': np.median(distancias_no_cero),
        'std': np.std(distancias_no_cero),
        'min': np.min(distancias_no_cero),
        'max': np.max(distancias_no_cero)
    }
    
    return stats, distancias_no_cero

def ejecutar_clasificacion_ejemplares():
    print("👥 CLASIFICACIÓN BASADA EN EJEMPLARES (K-NN)")
    print("="*45)
    print("📝 Objetivo: Clasificar basándose en vecinos más cercanos")
    print()
    
    # 1. CARGAR DATOS
    archivo = '/home/sedc/Proyectos/MineriaDeDatos/data/ceros_sin_columnasAB_limpio_weka.csv'
    try:
        datos = pd.read_csv(archivo)
        print(f"✅ Datos cargados: {datos.shape[0]:,} filas, {datos.shape[1]} columnas")
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        return
    
    # 2. SELECCIONAR VARIABLES PREDICTORAS
    variables_predictoras = [
        'POBFEM', 'POBMAS', 'TOTHOG', 'VIVTOT', 'P_15YMAS', 
        'P_60YMAS', 'GRAPROES', 'PEA', 'POCUPADA', 'PDESOCUP'
    ]
    
    variables_disponibles = [v for v in variables_predictoras if v in datos.columns]
    
    if len(variables_disponibles) < 3:
        print("❌ No hay suficientes variables para clasificación")
        return
    
    print(f"📊 Variables usadas: {', '.join(variables_disponibles)}")
    
    # 3. CREAR VARIABLE OBJETIVO
    datos['CATEGORIA_POB'] = datos['POBTOT'].apply(crear_categorias_poblacion)
    
    # 4. PREPARAR DATOS
    datos_limpios = datos[variables_disponibles + ['CATEGORIA_POB']].dropna()
    
    # Reducir muestra si es muy grande (K-NN es costoso)
    if len(datos_limpios) > 5000:
        datos_limpios = datos_limpios.sample(n=5000, random_state=42)
        print(f"📝 Muestra reducida a {len(datos_limpios):,} registros (para eficiencia)")
    
    X = datos_limpios[variables_disponibles]
    y = datos_limpios['CATEGORIA_POB']
    
    print(f"🧹 Datos finales: {len(datos_limpios):,} registros")
    print(f"📈 Distribución de categorías:")
    for categoria, count in y.value_counts().items():
        print(f"   {categoria:12}: {count:,} ({count/len(y)*100:.1f}%)")
    
    # 5. DIVIDIR DATOS
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 6. ESCALAR DATOS (MUY IMPORTANTE PARA K-NN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"📊 Entrenamiento: {len(X_train):,} | Prueba: {len(X_test):,}")
    print()
    
    # 7. ENCONTRAR EL MEJOR VALOR DE K
    print("🔍 BUSCANDO EL MEJOR VALOR DE K...")
    mejor_k, mejor_cv_score, k_values, cv_scores = encontrar_k_optimo(X_train_scaled, y_train)
    
    print(f"   ✅ Mejor K: {mejor_k}")
    print(f"   📊 Score CV: {mejor_cv_score:.3f} ({mejor_cv_score*100:.1f}%)")
    print()
    
    # 8. ENTRENAR DIFERENTES CONFIGURACIONES DE K-NN
    configuraciones = {
        f'K-NN Óptimo (K={mejor_k})': {
            'n_neighbors': mejor_k,
            'weights': 'distance'
        },
        'K-NN Clásico (K=5)': {
            'n_neighbors': 5,
            'weights': 'uniform'
        },
        'K-NN Ponderado (K=7)': {
            'n_neighbors': 7,
            'weights': 'distance'
        },
        'K-NN Conservative (K=15)': {
            'n_neighbors': min(15, len(X_train) // 10),
            'weights': 'distance'
        }
    }
    
    print("👥 ENTRENANDO MODELOS K-NN...")
    resultados = {}
    
    for nombre, params in configuraciones.items():
        print(f"   🔄 Entrenando {nombre}...")
        
        try:
            # Crear y entrenar modelo
            modelo = KNeighborsClassifier(**params, metric='euclidean')
            modelo.fit(X_train_scaled, y_train)
            
            # Predicciones
            y_pred = modelo.predict(X_test_scaled)
            y_pred_proba = modelo.predict_proba(X_test_scaled)
            
            # Métricas
            precision = accuracy_score(y_test, y_pred)
            
            resultados[nombre] = {
                'modelo': modelo,
                'precision': precision,
                'predicciones': y_pred,
                'probabilidades': y_pred_proba,
                'k': params['n_neighbors'],
                'weights': params['weights']
            }
            
            print(f"   ✅ {nombre} → Precisión: {precision:.3f} ({precision*100:.1f}%)")
            
        except Exception as e:
            print(f"   ❌ Error en {nombre}: {e}")
    
    if not resultados:
        print("❌ No se pudo entrenar ningún modelo K-NN")
        return
    
    # 9. ENCONTRAR EL MEJOR MODELO
    mejor_nombre = max(resultados.keys(), key=lambda x: resultados[x]['precision'])
    mejor_modelo = resultados[mejor_nombre]['modelo']
    mejor_precision = resultados[mejor_nombre]['precision']
    
    print()
    print(f"🏆 MEJOR MODELO: {mejor_nombre}")
    print(f"   Precisión: {mejor_precision:.3f} ({mejor_precision*100:.1f}%)")
    print(f"   K utilizado: {resultados[mejor_nombre]['k']}")
    print(f"   Ponderación: {resultados[mejor_nombre]['weights']}")
    
    # 10. ANÁLISIS DETALLADO
    print()
    print("📊 ANÁLISIS DETALLADO:")
    y_pred_mejor = resultados[mejor_nombre]['predicciones']
    
    # Reporte por clase
    print("\n🎯 Métricas por Categoría:")
    reporte = classification_report(y_test, y_pred_mejor, output_dict=True)
    for categoria in ['Pequeña', 'Mediana', 'Grande', 'Muy Grande']:
        if categoria in reporte:
            precision = reporte[categoria]['precision']
            recall = reporte[categoria]['recall']
            f1 = reporte[categoria]['f1-score']
            support = reporte[categoria]['support']
            print(f"   {categoria:12}: Prec={precision:.3f} | Rec={recall:.3f} | F1={f1:.3f} | N={support}")
    
    # 11. ANÁLISIS DE DISTANCIAS
    print()
    print("📏 ANÁLISIS DE DISTANCIAS:")
    try:
        stats_dist, distancias = analizar_distancias(X_train_scaled, X_test_scaled)
        print(f"   📊 Distancia promedio: {stats_dist['promedio']:.3f}")
        print(f"   📊 Distancia mediana: {stats_dist['mediana']:.3f}")
        print(f"   📊 Desviación estándar: {stats_dist['std']:.3f}")
        print(f"   📊 Rango: [{stats_dist['min']:.3f}, {stats_dist['max']:.3f}]")
    except Exception as e:
        print(f"   ⚠️ Error calculando distancias: {e}")
        distancias = None
    
    # 12. ANÁLISIS DE VECINOS
    print()
    print("👥 ANÁLISIS DE VECINOS:")
    try:
        # Obtener distancias y índices de vecinos para algunas muestras
        if len(X_test_scaled) > 0:
            distancias_vecinos, indices_vecinos = mejor_modelo.kneighbors(X_test_scaled[:5])
            
            for i in range(min(3, len(distancias_vecinos))):
                print(f"\n   Muestra {i+1}:")
                print(f"   Clase real: {y_test.iloc[i]}")
                print(f"   Clase predicha: {y_pred_mejor[i]}")
                print(f"   Distancias a vecinos: {distancias_vecinos[i]}")
                
                # Clases de los vecinos
                clases_vecinos = y_train.iloc[indices_vecinos[i]].values
                print(f"   Clases de vecinos: {list(clases_vecinos)}")
    
    except Exception as e:
        print(f"   ⚠️ Error analizando vecinos: {e}")
    
    # 13. VISUALIZACIONES
    try:
        fig = plt.figure(figsize=(18, 12))
        
        # Gráfico 1: Evolución de precisión con K
        plt.subplot(3, 4, 1)
        plt.plot(k_values, cv_scores, 'b-o', linewidth=2, markersize=6)
        plt.axvline(x=mejor_k, color='red', linestyle='--', label=f'Mejor K={mejor_k}')
        plt.title('🔍 Precisión vs Valor de K', fontweight='bold')
        plt.xlabel('K (Número de Vecinos)')
        plt.ylabel('Precisión (CV)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Gráfico 2: Comparación de modelos
        plt.subplot(3, 4, 2)
        nombres = list(resultados.keys())
        precisiones = [resultados[m]['precision'] for m in nombres]
        colores = ['lightblue', 'lightgreen', 'orange', 'pink']
        
        barras = plt.bar(range(len(nombres)), precisiones, color=colores[:len(nombres)])
        plt.title('👥 Precisión por Configuración', fontweight='bold')
        plt.ylabel('Precisión')
        plt.xticks(range(len(nombres)), [n.split('(')[0] for n in nombres], rotation=45, ha='right')
        plt.ylim(0, 1)
        
        for i, (barra, precision) in enumerate(zip(barras, precisiones)):
            plt.text(i, precision + 0.02, f'{precision:.3f}', ha='center', fontweight='bold')
        
        # Gráfico 3: Matriz de confusión
        plt.subplot(3, 4, 3)
        cm = confusion_matrix(y_test, y_pred_mejor)
        clases = mejor_modelo.classes_
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=clases, yticklabels=clases)
        plt.title(f'🎯 Matriz de Confusión\n{mejor_nombre.split("(")[0]}', fontweight='bold')
        plt.xlabel('Predicción')
        plt.ylabel('Real')
        
        # Gráfico 4: Distribución de distancias (si disponible)
        plt.subplot(3, 4, 4)
        if distancias is not None and len(distancias) > 0:
            plt.hist(distancias, bins=30, alpha=0.7, color='purple', edgecolor='black')
            plt.title('📏 Distribución de Distancias', fontweight='bold')
            plt.xlabel('Distancia Euclidiana')
            plt.ylabel('Frecuencia')
        else:
            plt.text(0.5, 0.5, 'Análisis de distancias\nno disponible', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('📏 Distribución de Distancias', fontweight='bold')
        
        # Gráfico 5: F1-Score por categoría
        plt.subplot(3, 4, 5)
        f1_scores = []
        categorias_f1 = []
        for categoria in ['Pequeña', 'Mediana', 'Grande', 'Muy Grande']:
            if categoria in reporte:
                f1_scores.append(reporte[categoria]['f1-score'])
                categorias_f1.append(categoria)
        
        plt.bar(categorias_f1, f1_scores, color='gold')
        plt.title('🎯 F1-Score por Categoría', fontweight='bold')
        plt.ylabel('F1-Score')
        plt.xticks(rotation=45)
        
        # Gráfico 6: Comparación de K values utilizados
        plt.subplot(3, 4, 6)
        k_vals = [resultados[m]['k'] for m in nombres]
        weights = [resultados[m]['weights'] for m in nombres]
        
        colores_k = ['blue' if w == 'uniform' else 'red' for w in weights]
        plt.scatter(k_vals, precisiones, c=colores_k, s=100, alpha=0.7)
        for i, nombre in enumerate(nombres):
            plt.annotate(nombre.split('(')[0][:8], (k_vals[i], precisiones[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        plt.xlabel('Valor de K')
        plt.ylabel('Precisión')
        plt.title('📊 K vs Precisión\n🔵Uniform 🔴Distance', fontweight='bold')
        
        # Gráfico 7: Distribución de confianza
        plt.subplot(3, 4, 7)
        probabilidades = resultados[mejor_nombre]['probabilidades']
        max_probs = np.max(probabilidades, axis=1)
        plt.hist(max_probs, bins=20, alpha=0.7, color='green', edgecolor='black')
        plt.title('📈 Confianza de Predicciones', fontweight='bold')
        plt.xlabel('Confianza Máxima')
        plt.ylabel('Frecuencia')
        
        # Gráfico 8: Precisión por tamaño de K
        plt.subplot(3, 4, 8)
        k_sizes = ['Pequeño (1-5)', 'Mediano (6-10)', 'Grande (11-20)']
        precisions_by_size = [[], [], []]
        
        for i, k in enumerate(k_values):
            if k <= 5:
                precisions_by_size[0].append(cv_scores[i])
            elif k <= 10:
                precisions_by_size[1].append(cv_scores[i])
            else:
                precisions_by_size[2].append(cv_scores[i])
        
        avg_precisions = [np.mean(p) if p else 0 for p in precisions_by_size]
        plt.bar(k_sizes, avg_precisions, color=['lightcoral', 'lightblue', 'lightgreen'])
        plt.title('📊 Precisión por Tamaño K', fontweight='bold')
        plt.ylabel('Precisión Promedio')
        plt.xticks(rotation=45)
        
        # Gráfico 9: Errores por categoría
        plt.subplot(3, 4, 9)
        errores_por_categoria = {}
        for real, pred in zip(y_test, y_pred_mejor):
            if real != pred:
                errores_por_categoria[real] = errores_por_categoria.get(real, 0) + 1
        
        if errores_por_categoria:
            categorias_error = list(errores_por_categoria.keys())
            conteo_errores = list(errores_por_categoria.values())
            plt.bar(categorias_error, conteo_errores, color='red', alpha=0.7)
            plt.title('❌ Errores por Categoría', fontweight='bold')
            plt.ylabel('Número de Errores')
            plt.xticks(rotation=45)
        
        # Gráfico 10: Support por categoría
        plt.subplot(3, 4, 10)
        supports = []
        for categoria in categorias_f1:
            supports.append(reporte[categoria]['support'])
        
        plt.bar(categorias_f1, supports, color='cyan')
        plt.title('📊 Muestras por Categoría', fontweight='bold')
        plt.ylabel('Número de Muestras')
        plt.xticks(rotation=45)
        
        # Gráfico 11: Evolución de precisión (curva suavizada)
        plt.subplot(3, 4, 11)
        if len(cv_scores) > 5:
            from scipy.ndimage import gaussian_filter1d
            cv_scores_smooth = gaussian_filter1d(cv_scores, sigma=0.8)
            plt.plot(k_values, cv_scores, 'o-', alpha=0.5, label='Original')
            plt.plot(k_values, cv_scores_smooth, 'r-', linewidth=2, label='Suavizada')
            plt.axvline(x=mejor_k, color='green', linestyle='--', label=f'Óptimo K={mejor_k}')
            plt.title('📈 Curva de Aprendizaje K-NN', fontweight='bold')
            plt.xlabel('K')
            plt.ylabel('Precisión CV')
            plt.legend()
        
        # Gráfico 12: Heatmap de confusión normalizada
        plt.subplot(3, 4, 12)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Reds',
                   xticklabels=clases, yticklabels=clases)
        plt.title('🔥 Confusión Normalizada', fontweight='bold')
        plt.xlabel('Predicción')
        plt.ylabel('Real')
        
        plt.tight_layout()
        plt.savefig('/home/sedc/Proyectos/MineriaDeDatos/results/graficos/clasificacion_knn.png', 
                   dpi=150, bbox_inches='tight')
        plt.show()
        
        print("💾 Gráficos guardados en: results/graficos/clasificacion_knn.png")
        
    except Exception as e:
        print(f"⚠️ Error creando visualizaciones: {e}")
    
    # 14. GUARDAR RESULTADOS
    try:
        import joblib
        
        # Guardar el mejor modelo y el scaler
        joblib.dump(mejor_modelo, '/home/sedc/Proyectos/MineriaDeDatos/results/modelos/mejor_knn_clasificacion.pkl')
        joblib.dump(scaler, '/home/sedc/Proyectos/MineriaDeDatos/results/modelos/scaler_knn.pkl')
        
        # Crear reporte detallado
        reporte_completo = f"""
REPORTE CLASIFICACIÓN K-NN (BASADA EN EJEMPLARES)
===============================================

MEJOR MODELO: {mejor_nombre}
Precisión: {mejor_precision:.3f} ({mejor_precision*100:.1f}%)
K Óptimo: {mejor_k}
Valor K usado: {resultados[mejor_nombre]['k']}
Ponderación: {resultados[mejor_nombre]['weights']}

BÚSQUEDA DE K ÓPTIMO:
- Rango evaluado: {min(k_values)} - {max(k_values)}
- Mejor score CV: {mejor_cv_score:.3f}

COMPARACIÓN DE CONFIGURACIONES:
"""
        for nombre, resultado in resultados.items():
            reporte_completo += f"\n{nombre}:"
            reporte_completo += f"\n  - Precisión: {resultado['precision']:.3f}"
            reporte_completo += f"\n  - K: {resultado['k']}"
            reporte_completo += f"\n  - Weights: {resultado['weights']}"
        
        if 'stats_dist' in locals():
            reporte_completo += f"""

ANÁLISIS DE DISTANCIAS:
- Distancia promedio: {stats_dist['promedio']:.3f}
- Distancia mediana: {stats_dist['mediana']:.3f}
- Desviación estándar: {stats_dist['std']:.3f}
- Rango: [{stats_dist['min']:.3f}, {stats_dist['max']:.3f}]
"""
        
        reporte_completo += f"""

VARIABLES UTILIZADAS:
{', '.join(variables_disponibles)}

DATOS:
- Total registros: {len(datos_limpios):,}
- Variables predictoras: {len(variables_disponibles)}
- Categorías: {len(clases)}
- Entrenamiento: {len(X_train):,}
- Prueba: {len(X_test):,}

NOTAS:
- Los datos fueron escalados usando StandardScaler
- Se utilizó distancia Euclidiana
- Se aplicó validación cruzada para encontrar K óptimo
"""
        
        with open('/home/sedc/Proyectos/MineriaDeDatos/results/reportes/clasificacion_knn_reporte.txt', 'w', encoding='utf-8') as f:
            f.write(reporte_completo)
        
        print("💾 Modelo guardado en: results/modelos/mejor_knn_clasificacion.pkl")
        print("💾 Scaler guardado en: results/modelos/scaler_knn.pkl")
        print("📄 Reporte guardado en: results/reportes/clasificacion_knn_reporte.txt")
        
    except Exception as e:
        print(f"⚠️ Error guardando archivos: {e}")
    
    # 15. RESUMEN FINAL
    print()
    print("📝 RESUMEN K-NN (BASADO EN EJEMPLARES):")
    print(f"   • Mejor configuración: K={resultados[mejor_nombre]['k']}, weights={resultados[mejor_nombre]['weights']}")
    print(f"   • Precisión alcanzada: {mejor_precision*100:.1f}%")
    print(f"   • K óptimo encontrado: {mejor_k}")
    
    if mejor_precision > 0.8:
        print("   • ¡Excelente clasificación por similitud! 🎉")
    elif mejor_precision > 0.6:
        print("   • Buena clasificación basada en vecinos 👍")
    else:
        print("   • Clasificación moderada, considerar más datos 🔧")
    
    print("   • Ventaja: No requiere entrenamiento, se adapta a nuevos datos")
    print("   • Desventaja: Computacionalmente costoso para predicción")
    
    print("✅ CLASIFICACIÓN BASADA EN EJEMPLARES COMPLETADA")
    return resultados

if __name__ == "__main__":
    ejecutar_clasificacion_ejemplares()