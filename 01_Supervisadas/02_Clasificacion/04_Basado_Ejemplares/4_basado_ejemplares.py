#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLASIFICACIÓN BASADA EN EJEMPLARES (K-NN) - Versión Optimizada
Clasificación por similitud con vecinos cercanos
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def crear_categorias_poblacion_dinamica(datos):
    """Crear categorías balanceadas basadas en cuartiles"""
    q1 = datos['POBTOT'].quantile(0.25)
    q2 = datos['POBTOT'].quantile(0.50)
    q3 = datos['POBTOT'].quantile(0.75)
    
    def categorizar(poblacion):
        if poblacion <= q1:
            return 'Pequeña'
        elif poblacion <= q2:
            return 'Mediana'
        elif poblacion <= q3:
            return 'Grande'
        else:
            return 'Muy_Grande'
    
    return categorizar

def preparar_datos_knn(datos):
    """Prepara datos específicamente para K-NN"""
    variables_predictoras = [
        'POBFEM', 'POBMAS', 'TOTHOG', 'VIVTOT', 'P_15YMAS', 
        'P_60YMAS', 'GRAPROES', 'PEA', 'POCUPADA'
    ]
    
    variables_disponibles = [v for v in variables_predictoras if v in datos.columns]
    
    if len(variables_disponibles) < 5:
        return None, None, None
    
    # Crear categorías dinámicas
    categorizador = crear_categorias_poblacion_dinamica(datos)
    datos['CATEGORIA_POB'] = datos['POBTOT'].apply(categorizador)
    
    # Limpiar datos
    datos_limpios = datos[variables_disponibles + ['CATEGORIA_POB']].dropna()
    
    # Reducir muestra para eficiencia (K-NN es costoso)
    if len(datos_limpios) > 20000:
        try:
            datos_limpios = datos_limpios.groupby('CATEGORIA_POB').apply(
                lambda x: x.sample(min(len(x), 10000), random_state=42)
            ).reset_index(drop=True)
        except:
            datos_limpios = datos_limpios.sample(n=20000, random_state=42)
    
    X = datos_limpios[variables_disponibles]
    y = datos_limpios['CATEGORIA_POB']
    
    return X, y, variables_disponibles

def encontrar_k_optimo(X_train, y_train, k_max=15):
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
            precision = accuracy_score(y_test, y_pred)
            
            resultados[nombre] = {
                'modelo': modelo,
                'precision': precision,
                'predicciones': y_pred,
                'probabilidades': y_pred_proba,
                'k': params['n_neighbors'],
                'weights': params['weights']
            }
            
        except Exception as e:
            continue
    
    return resultados, scaler

def crear_visualizaciones_knn(resultados, y_test, k_values, cv_scores, mejor_k):
    """Crear visualizaciones esenciales para K-NN"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('👥 CLASIFICACIÓN K-NN - ANÁLISIS', fontsize=14, fontweight='bold')
        
        # Gráfico 1: Evolución de precisión con K
        axes[0,0].plot(k_values, cv_scores, 'b-o', linewidth=2, markersize=6)
        axes[0,0].axvline(x=mejor_k, color='red', linestyle='--', label=f'Mejor K={mejor_k}')
        axes[0,0].set_title('🔍 Precisión vs Valor de K', fontweight='bold')
        axes[0,0].set_xlabel('K (Número de Vecinos)')
        axes[0,0].set_ylabel('Precisión (CV)')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Gráfico 2: Comparación de configuraciones
        nombres = list(resultados.keys())
        precisiones = [resultados[m]['precision'] for m in nombres]
        colores = ['lightblue', 'lightgreen', 'orange']
        
        axes[0,1].bar(range(len(nombres)), precisiones, color=colores[:len(nombres)])
        axes[0,1].set_title('👥 Precisión por Configuración', fontweight='bold')
        axes[0,1].set_ylabel('Precisión')
        axes[0,1].set_xticks(range(len(nombres)))
        axes[0,1].set_xticklabels([n.split('(')[0].strip() for n in nombres], rotation=45, ha='right')
        axes[0,1].set_ylim(0, 1)
        
        # Añadir valores en las barras
        for i, precision in enumerate(precisiones):
            axes[0,1].text(i, precision + 0.02, f'{precision:.3f}', ha='center', fontweight='bold')
        
        # Gráfico 3: Matriz de confusión del mejor modelo
        mejor_nombre = max(resultados.keys(), key=lambda x: resultados[x]['precision'])
        y_pred_mejor = resultados[mejor_nombre]['predicciones']
        clases = resultados[mejor_nombre]['modelo'].classes_
        
        cm = confusion_matrix(y_test, y_pred_mejor)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=clases, yticklabels=clases, ax=axes[1,0])
        axes[1,0].set_title(f'🎯 Matriz de Confusión\n{mejor_nombre.split("(")[0].strip()}', fontweight='bold')
        axes[1,0].set_xlabel('Predicción')
        axes[1,0].set_ylabel('Real')
        
        # Gráfico 4: Distribución de confianza
        probabilidades = resultados[mejor_nombre]['probabilidades']
        max_probs = np.max(probabilidades, axis=1)
        
        axes[1,1].hist(max_probs, bins=20, alpha=0.7, color='green', edgecolor='black')
        axes[1,1].set_title('📈 Distribución de Confianza', fontweight='bold')
        axes[1,1].set_xlabel('Confianza Máxima')
        axes[1,1].set_ylabel('Frecuencia')
        
        plt.tight_layout()
        plt.savefig('/home/sedc/Proyectos/MineriaDeDatos/results/graficos/clasificacion_knn.png', 
                   dpi=150, bbox_inches='tight')
        plt.show()
        
        return True
        
    except Exception as e:
        return False

def analizar_vecinos_cercanos(modelo, X_test_scaled, y_test, y_pred, n_ejemplos=3):
    """Analiza los vecinos más cercanos para algunos ejemplos"""
    print(f"\n👥 ANÁLISIS DE VECINOS:")
    
    try:
        # Obtener distancias y índices de vecinos
        distancias, indices = modelo.kneighbors(X_test_scaled[:n_ejemplos])
        
        for i in range(n_ejemplos):
            print(f"   Ejemplo {i+1}: Real={y_test.iloc[i]} | Predicho={y_pred[i]}")
            print(f"      Distancias: {distancias[i]}")
            
    except Exception as e:
        print(f"   ⚠️ Error: {e}")

def guardar_resultados_knn(resultados, variables_disponibles, mejor_k, datos_info):
    """Guardar modelo y reporte de manera optimizada"""
    try:
        import joblib
        
        # Mejor modelo
        mejor_nombre = max(resultados.keys(), key=lambda x: resultados[x]['precision'])
        mejor_modelo = resultados[mejor_nombre]['modelo']
        mejor_precision = resultados[mejor_nombre]['precision']
        
        # Guardar modelo
        joblib.dump(mejor_modelo, '/home/sedc/Proyectos/MineriaDeDatos/results/modelos/mejor_knn_clasificacion.pkl')
        
        # Crear reporte conciso
        reporte = f"""
REPORTE CLASIFICACIÓN K-NN (BASADA EN EJEMPLARES)
===============================================

MEJOR MODELO: {mejor_nombre}
Precisión: {mejor_precision:.3f} ({mejor_precision*100:.1f}%)
K Óptimo encontrado: {mejor_k}
K utilizado: {resultados[mejor_nombre]['k']}
Ponderación: {resultados[mejor_nombre]['weights']}

COMPARACIÓN DE CONFIGURACIONES:
"""
        for nombre, resultado in resultados.items():
            reporte += f"\n{nombre}:"
            reporte += f"\n  - Precisión: {resultado['precision']:.3f}"
            reporte += f"\n  - K: {resultado['k']}"
            reporte += f"\n  - Weights: {resultado['weights']}"
        
        reporte += f"""

DATOS PROCESADOS:
- Registros: {datos_info['n_registros']:,}
- Variables: {len(variables_disponibles)}
- Entrenamiento: {datos_info['n_train']:,}
- Prueba: {datos_info['n_test']:,}

VARIABLES UTILIZADAS:
{', '.join(variables_disponibles)}

PRINCIPIO K-NN:
- Clasifica según la mayoría de los K vecinos más cercanos
- No requiere entrenamiento (lazy learning)
- Sensible a la escala (por eso se aplica StandardScaler)
- Computacionalmente costoso para predicción

CONFIGURACIÓN:
- Métrica de distancia: Euclidiana
- Escalado aplicado: StandardScaler
- Validación cruzada: 5-fold para K óptimo
"""
        
        with open('/home/sedc/Proyectos/MineriaDeDatos/results/reportes/clasificacion_knn_reporte.txt', 'w', encoding='utf-8') as f:
            f.write(reporte)
        
        return True
        
    except Exception as e:
        return False

def ejecutar_clasificacion_ejemplares():
    """FUNCIÓN PRINCIPAL - Mantiene compatibilidad con menú"""
    print("👥 CLASIFICACIÓN BASADA EN EJEMPLARES (K-NN)")
    print("="*45)
    
    # 1. CARGAR DATOS
    archivo = '/home/sedc/Proyectos/MineriaDeDatos/data/ceros_sin_columnasAB_limpio_weka.csv'
    try:
        datos = pd.read_csv(archivo)
        print(f"✅ Datos cargados: {datos.shape[0]:,} registros")
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        return
    
    # 2. PREPARAR VARIABLES
    X, y, variables_disponibles = preparar_datos_knn(datos)
    if X is None:
        print("❌ No hay suficientes variables para K-NN")
        return
    
    print(f"📊 Variables: {len(variables_disponibles)} | Datos limpios: {len(X):,}")
    
    # Mostrar distribución de categorías
    distribucion = y.value_counts()
    print("📈 Categorías:")
    for categoria, count in distribucion.items():
        print(f"   {categoria}: {count:,} ({count/len(y)*100:.1f}%)")
    
    # 3. DIVIDIR DATOS
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        print(f"📊 División estratificada: {len(X_train):,} entrenamiento | {len(X_test):,} prueba")
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        print(f"📊 División simple: {len(X_train):,} entrenamiento | {len(X_test):,} prueba")
    
    print()
    
    # 4. ENCONTRAR K ÓPTIMO
    print("🔍 Buscando K óptimo...")
    mejor_k, mejor_cv_score, k_values, cv_scores = encontrar_k_optimo(X_train, y_train)
    print(f"   ✅ Mejor K: {mejor_k} (CV Score: {mejor_cv_score:.3f})")
    
    # 5. ENTRENAR MODELOS K-NN
    print("\n👥 Entrenando modelos K-NN...")
    resultados, scaler = entrenar_modelos_knn(X_train, X_test, y_train, y_test, mejor_k)
    
    if not resultados:
        print("❌ No se pudieron entrenar modelos K-NN")
        return
    
    # Mostrar resultados
    for nombre, resultado in resultados.items():
        print(f"   {nombre:20} → Precisión: {resultado['precision']:.3f} ({resultado['precision']*100:.1f}%)")
    
    # 6. ENCONTRAR MEJOR MODELO
    mejor_nombre = max(resultados.keys(), key=lambda x: resultados[x]['precision'])
    mejor_precision = resultados[mejor_nombre]['precision']
    
    print()
    print(f"🏆 MEJOR MODELO: {mejor_nombre}")
    print(f"   Precisión: {mejor_precision:.3f} ({mejor_precision*100:.1f}%)")
    print(f"   K utilizado: {resultados[mejor_nombre]['k']}")
    print(f"   Ponderación: {resultados[mejor_nombre]['weights']}")
    
    # 7. ANÁLISIS DETALLADO
    y_pred_mejor = resultados[mejor_nombre]['predicciones']
    reporte = classification_report(y_test, y_pred_mejor, output_dict=True)
    
    print("\n🎯 Métricas por categoría:")
    for categoria in y.unique():
        if categoria in reporte:
            precision = reporte[categoria]['precision']
            recall = reporte[categoria]['recall']
            f1 = reporte[categoria]['f1-score']
            print(f"   {categoria:12}: Prec={precision:.3f} | Rec={recall:.3f} | F1={f1:.3f}")
    
    # 8. ANÁLISIS DE VECINOS
    X_test_scaled = scaler.transform(X_test)
    analizar_vecinos_cercanos(resultados[mejor_nombre]['modelo'], X_test_scaled, 
                             y_test, y_pred_mejor)
    
    # 9. VISUALIZACIONES
    crear_visualizaciones_knn(resultados, y_test, k_values, cv_scores, mejor_k)
    
    # 10. GUARDAR RESULTADOS
    datos_info = {
        'n_registros': len(X),
        'n_train': len(X_train),
        'n_test': len(X_test)
    }
    
    guardar_resultados_knn(resultados, variables_disponibles, mejor_k, datos_info)
    
    # 11. RESUMEN FINAL
    print()
    print("📝 RESUMEN:")
    print(f"   • Configuración: {mejor_nombre}")
    print(f"   • Precisión: {mejor_precision*100:.1f}%")
    print(f"   • K óptimo: {mejor_k}")
    print(f"   • Variables: {len(variables_disponibles)}")
    
    if mejor_precision > 0.8:
        print("   🎉 ¡Excelente clasificación por similitud!")
    elif mejor_precision > 0.6:
        print("   👍 Buena clasificación basada en vecinos")
    else:
        print("   🔧 Clasificación moderada")
    
    print("✅ CLASIFICACIÓN BASADA EN EJEMPLARES COMPLETADA")
    return resultados

if __name__ == "__main__":
    ejecutar_clasificacion_ejemplares()