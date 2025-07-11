#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLASIFICACIÓN BASADA EN EJEMPLARES - K-NN
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def cargar_datos():
    """Carga el dataset principal"""
    return pd.read_csv('data/ceros_sin_columnasAB_limpio_weka.csv')

def crear_categorias_poblacion(poblacion):
    """Crea categorías de población para clasificación"""
    if poblacion <= 500:
        return 'Pequeña'
    elif poblacion <= 2000:
        return 'Mediana'
    elif poblacion <= 8000:
        return 'Grande'
    else:
        return 'Muy_Grande'

def preparar_datos(datos, max_muestras=5000):
    """Prepara variables para K-NN (reduciendo muestra para eficiencia)"""
    variables = ['POBFEM', 'POBMAS', 'TOTHOG', 'VIVTOT', 'P_15YMAS', 'P_60YMAS', 'GRAPROES', 'PEA', 'POCUPADA', 'PDESOCUP']
    variables_disponibles = [v for v in variables if v in datos.columns]
    
    # Crear variable objetivo categórica
    datos['CATEGORIA_POB'] = datos['POBTOT'].apply(crear_categorias_poblacion)
    
    # Dataset limpio
    df = datos[variables_disponibles + ['CATEGORIA_POB']].dropna()
    
    # Reducir muestra para eficiencia (K-NN es costoso)
    if len(df) > max_muestras:
        df = df.sample(n=max_muestras, random_state=42)
        print(f"📝 Muestra reducida a {len(df):,} registros (para eficiencia K-NN)")
    
    X = df[variables_disponibles]
    y = df['CATEGORIA_POB']
    
    return X, y, variables_disponibles

def encontrar_k_optimo(X_train, y_train, k_max=20):
    """Encuentra el mejor valor de K usando validación cruzada"""
    k_values = range(1, min(k_max + 1, len(X_train) // 5))
    cv_scores = []
    
    print("    Probando diferentes valores de K...")
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
        scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
        cv_scores.append(scores.mean())
    
    mejor_k = k_values[np.argmax(cv_scores)]
    mejor_score = max(cv_scores)
    
    return mejor_k, mejor_score, list(k_values), cv_scores

def entrenar_modelos_knn(X_train, X_test, y_train, y_test, mejor_k):
    """Entrena diferentes configuraciones de K-NN"""
    
    # Escalar datos (CRÍTICO para K-NN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
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
        }
    }
    
    resultados = {}
    
    for nombre, params in configuraciones.items():
        try:
            # Crear y entrenar modelo
            modelo = KNeighborsClassifier(**params, metric='euclidean')
            modelo.fit(X_train_scaled, y_train)
            
            # Predicciones
            y_pred = modelo.predict(X_test_scaled)
            y_pred_proba = modelo.predict_proba(X_test_scaled)
            
            # Métricas
            resultados[nombre] = {
                'modelo': modelo,
                'accuracy': accuracy_score(y_test, y_pred),
                'predicciones': y_pred,
                'probabilidades': y_pred_proba,
                'k': params['n_neighbors'],
                'weights': params['weights']
            }
            
        except Exception as e:
            print(f"    ❌ Error en {nombre}: {str(e)[:50]}...")
            continue
    
    return resultados, scaler

def analizar_vecinos(modelo, X_test_scaled, y_test, y_pred, n_ejemplos=3):
    """Analiza los vecinos más cercanos para algunos ejemplos"""
    print(f"\n👥 Análisis de vecinos más cercanos:")
    
    try:
        # Obtener distancias y índices de vecinos para las primeras muestras
        distancias, indices = modelo.kneighbors(X_test_scaled[:n_ejemplos])
        
        for i in range(n_ejemplos):
            print(f"\n   Ejemplo {i+1}:")
            print(f"   Clase real: {y_test.iloc[i]}")
            print(f"   Clase predicha: {y_pred[i]}")
            print(f"   Distancias a vecinos: {distancias[i]}")
            
    except Exception as e:
        print(f"   ⚠️ Error analizando vecinos: {e}")

def visualizar_resultados_knn(resultados, y_test, k_values, cv_scores, mejor_k):
    """Crea visualizaciones de K-NN"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Evolución de precisión con K
    axes[0,0].plot(k_values, cv_scores, 'b-o', linewidth=2, markersize=6)
    axes[0,0].axvline(x=mejor_k, color='red', linestyle='--', label=f'Mejor K={mejor_k}')
    axes[0,0].set_title('Precisión vs Valor de K')
    axes[0,0].set_xlabel('K (Número de Vecinos)')
    axes[0,0].set_ylabel('Precisión (CV)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Comparación de modelos
    nombres = list(resultados.keys())
    accuracies = [resultados[m]['accuracy'] for m in nombres]
    colores = ['lightblue', 'lightgreen', 'orange']
    
    axes[0,1].bar(range(len(nombres)), accuracies, color=colores[:len(nombres)])
    axes[0,1].set_title('Precisión por Configuración K-NN')
    axes[0,1].set_ylabel('Accuracy')
    axes[0,1].set_xticks(range(len(nombres)))
    axes[0,1].set_xticklabels([n.split('(')[0] for n in nombres], rotation=45)
    for i, acc in enumerate(accuracies):
        axes[0,1].text(i, acc + 0.01, f'{acc:.3f}', ha='center')
    
    # 3. Matriz de confusión (mejor modelo)
    mejor = max(resultados.keys(), key=lambda x: resultados[x]['accuracy'])
    y_pred_mejor = resultados[mejor]['predicciones']
    
    cm = confusion_matrix(y_test, y_pred_mejor)
    clases = resultados[mejor]['modelo'].classes_
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=clases, yticklabels=clases, ax=axes[0,2])
    axes[0,2].set_title(f'Matriz de Confusión\n{mejor.split("(")[0]}')
    axes[0,2].set_xlabel('Predicción')
    axes[0,2].set_ylabel('Real')
    
    # 4. Distribución de confianza
    probabilidades = resultados[mejor]['probabilidades']
    max_probs = np.max(probabilidades, axis=1)
    axes[1,0].hist(max_probs, bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[1,0].set_title('Distribución de Confianza')
    axes[1,0].set_xlabel('Confianza Máxima')
    axes[1,0].set_ylabel('Frecuencia')
    
    # 5. Comparación K vs Precisión
    k_vals = [resultados[m]['k'] for m in nombres]
    accuracies_k = [resultados[m]['accuracy'] for m in nombres]
    weights = [resultados[m]['weights'] for m in nombres]
    
    colores_k = ['blue' if w == 'uniform' else 'red' for w in weights]
    axes[1,1].scatter(k_vals, accuracies_k, c=colores_k, s=100, alpha=0.7)
    for i, nombre in enumerate(nombres):
        axes[1,1].annotate(nombre.split('(')[0][:8], (k_vals[i], accuracies_k[i]), 
                          xytext=(5, 5), textcoords='offset points', fontsize=8)
    axes[1,1].set_xlabel('Valor de K')
    axes[1,1].set_ylabel('Precisión')
    axes[1,1].set_title('K vs Precisión\n🔵Uniform 🔴Distance')
    
    # 6. F1-Score por categoría
    reporte = classification_report(y_test, y_pred_mejor, output_dict=True)
    f1_scores = []
    categorias_f1 = []
    for categoria in clases:
        if categoria in reporte:
            f1_scores.append(reporte[categoria]['f1-score'])
            categorias_f1.append(categoria)
    
    axes[1,2].bar(categorias_f1, f1_scores, color='gold')
    axes[1,2].set_title('F1-Score por Categoría')
    axes[1,2].set_ylabel('F1-Score')
    axes[1,2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('results/graficos/clasificacion_knn.png', dpi=150, bbox_inches='tight')
    plt.show()

def guardar_resultados_knn(resultados, variables, total_registros, mejor_k, cv_scores, y_test):
    """Guarda reporte de K-NN"""
    mejor = max(resultados.keys(), key=lambda x: resultados[x]['accuracy'])
    mejor_acc = resultados[mejor]['accuracy']
    y_pred_mejor = resultados[mejor]['predicciones']
    
    reporte = f"""CLASIFICACIÓN BASADA EN EJEMPLARES (K-NN) - REPORTE
==================================================

MEJOR MODELO: {mejor}
Precisión (Accuracy): {mejor_acc:.3f} ({mejor_acc*100:.1f}%)
K Óptimo encontrado: {mejor_k}
K utilizado: {resultados[mejor]['k']}
Ponderación: {resultados[mejor]['weights']}

BÚSQUEDA DE K ÓPTIMO:
- Mejor K encontrado: {mejor_k}
- Mejor score CV: {max(cv_scores):.3f}

COMPARACIÓN CONFIGURACIONES:
"""
    for nombre, res in resultados.items():
        reporte += f"\n{nombre}:"
        reporte += f"\n  - Precisión: {res['accuracy']:.3f}"
        reporte += f"\n  - K: {res['k']}"
        reporte += f"\n  - Weights: {res['weights']}"
    
    # Métricas detalladas por clase
    reporte_sklearn = classification_report(y_test, y_pred_mejor, output_dict=True)
    reporte += f"\n\nMÉTRICAS POR CLASE ({mejor}):\n"
    clases = resultados[mejor]['modelo'].classes_
    for clase in clases:
        if clase in reporte_sklearn:
            prec = reporte_sklearn[clase]['precision']
            rec = reporte_sklearn[clase]['recall']
            f1 = reporte_sklearn[clase]['f1-score']
            support = reporte_sklearn[clase]['support']
            reporte += f"{clase}: Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}, N={support}\n"
    
    reporte += f"""
DATOS UTILIZADOS:
- Total registros: {total_registros:,}
- Variables: {', '.join(variables)}
- División: 70% entrenamiento, 30% prueba

CONFIGURACIÓN K-NN:
- Métrica de distancia: Euclidiana
- Escalado aplicado: StandardScaler
- Validación cruzada: 5-fold para encontrar K óptimo

PRINCIPIO K-NN:
- Clasifica según la mayoría de los K vecinos más cercanos
- No requiere entrenamiento (lazy learning)
- Se adapta automáticamente a nuevos datos
- Sensible a la escala de las variables (por eso se escala)

VENTAJAS:
- Simple de entender e implementar
- No hace suposiciones sobre la distribución de datos
- Funciona bien con datos no lineales
- Se adapta a cambios en los datos

DESVENTAJAS:
- Computacionalmente costoso para predicción
- Sensible al ruido y datos irrelevantes
- Sufre con alta dimensionalidad
- Requiere mucha memoria

APLICACIONES:
- Sistemas de recomendación
- Reconocimiento de patrones
- Clasificación de imágenes
- Análisis de similitud
"""
    
    with open('results/reportes/clasificacion_knn_reporte.txt', 'w', encoding='utf-8') as f:
        f.write(reporte)

def ejecutar_clasificacion_ejemplares():
    """Función principal"""
    print("👥 CLASIFICACIÓN BASADA EN EJEMPLARES (K-NN)")
    print("="*45)
    
    # Cargar y preparar datos
    datos = cargar_datos()
    X, y, variables = preparar_datos(datos)
    
    print(f"📊 Datos: {len(X):,} registros")
    print(f"📊 Variables: {', '.join(variables)}")
    
    # División train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Encontrar K óptimo
    print("\n🔍 Buscando K óptimo...")
    mejor_k, mejor_cv_score, k_values, cv_scores = encontrar_k_optimo(X_train, y_train)
    print(f"    ✅ Mejor K: {mejor_k} (CV Score: {mejor_cv_score:.3f})")
    
    # Entrenar modelos K-NN
    print("\n👥 Entrenando modelos K-NN...")
    resultados, scaler = entrenar_modelos_knn(X_train, X_test, y_train, y_test, mejor_k)
    
    if not resultados:
        print("❌ No se pudieron entrenar modelos K-NN")
        return
    
    # Mostrar resultados
    print("\nRESULTADOS:")
    for nombre, res in resultados.items():
        print(f"{nombre:25}: Accuracy = {res['accuracy']:.3f} ({res['accuracy']*100:.1f}%)")
    
    # Mejor modelo
    mejor = max(resultados.keys(), key=lambda x: resultados[x]['accuracy'])
    print(f"\n🏆 MEJOR: {mejor}")
    print(f"    Precisión: {resultados[mejor]['accuracy']:.3f}")
    print(f"    K utilizado: {resultados[mejor]['k']}")
    print(f"    Ponderación: {resultados[mejor]['weights']}")
    
    # Análisis de vecinos
    analizar_vecinos(resultados[mejor]['modelo'], scaler.transform(X_test), 
                    y_test, resultados[mejor]['predicciones'])
    
    # Visualizar resultados
    visualizar_resultados_knn(resultados, y_test, k_values, cv_scores, mejor_k)
    
    # Guardar resultados
    guardar_resultados_knn(resultados, variables, len(X), mejor_k, cv_scores, y_test)
    
    print("\n✅ COMPLETADO")
    
    return {
        'mejor_modelo': mejor,
        'precision': resultados[mejor]['accuracy'],
        'k_optimo': mejor_k,
        'resultados': resultados
    }

if __name__ == "__main__":
    ejecutar_clasificacion_ejemplares()