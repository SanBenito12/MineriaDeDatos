#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REDES NEURONALES - Clasificación con MLPClassifier
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def cargar_datos():
    """Carga el dataset principal"""
    return pd.read_csv('data/ceros_sin_columnasAB_limpio_weka.csv')

def crear_categorias_poblacion(poblacion):
    """Crea categorías balanceadas para redes neuronales"""
    if poblacion <= 200:
        return 'Pequeña'
    elif poblacion <= 800:
        return 'Mediana'
    elif poblacion <= 3000:
        return 'Grande'
    else:
        return 'Muy_Grande'

def preparar_datos(datos, max_muestras=3000):
    """Prepara variables para redes neuronales"""
    variables = ['POBFEM', 'POBMAS', 'TOTHOG', 'VIVTOT', 'P_15YMAS', 'P_60YMAS', 'GRAPROES', 'PEA', 'POCUPADA']
    variables_disponibles = [v for v in variables if v in datos.columns]
    
    # Crear variable objetivo categórica
    datos['CATEGORIA_POB'] = datos['POBTOT'].apply(crear_categorias_poblacion)
    
    # Dataset limpio
    df = datos[variables_disponibles + ['CATEGORIA_POB']].dropna()
    
    # Muestreo estratificado para balancear clases
    if len(df) > max_muestras:
        try:
            df, _ = train_test_split(df, test_size=1-max_muestras/len(df), 
                                   stratify=df['CATEGORIA_POB'], random_state=42)
        except:
            df = df.sample(n=max_muestras, random_state=42)
        
        print(f"📝 Muestra balanceada: {len(df):,} registros")
    
    # Verificar distribución de clases
    print(f"📊 Distribución de clases:")
    for categoria, count in df['CATEGORIA_POB'].value_counts().items():
        print(f"   {categoria}: {count:,} ({count/len(df)*100:.1f}%)")
    
    X = df[variables_disponibles]
    y = df['CATEGORIA_POB']
    
    return X, y, variables_disponibles

def crear_arquitecturas_redes():
    """Define diferentes arquitecturas de redes neuronales"""
    return {
        'Red Simple': {
            'hidden_layer_sizes': (20,),
            'activation': 'relu',
            'solver': 'lbfgs',
            'alpha': 0.01,
            'max_iter': 500,
            'descripcion': 'Una capa oculta con 20 neuronas'
        },
        'Red Profunda': {
            'hidden_layer_sizes': (30, 15),
            'activation': 'relu',
            'solver': 'lbfgs',
            'alpha': 0.01,
            'max_iter': 600,
            'descripcion': 'Dos capas: 30 y 15 neuronas'
        },
        'Red Tanh': {
            'hidden_layer_sizes': (25,),
            'activation': 'tanh',
            'solver': 'lbfgs',
            'alpha': 0.1,
            'max_iter': 400,
            'descripcion': 'Activación tangente hiperbólica'
        }
    }

def entrenar_redes_neuronales(X_train, X_test, y_train, y_test):
    """Entrena diferentes arquitecturas de redes neuronales"""
    
    # Escalar datos (CRÍTICO para redes neuronales)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    arquitecturas = crear_arquitecturas_redes()
    resultados = {}
    
    for nombre, config in arquitecturas.items():
        try:
            print(f"    Entrenando {nombre}...")
            
            # Crear configuración sin descripción
            config_modelo = {k: v for k, v in config.items() if k != 'descripcion'}
            config_modelo['random_state'] = 42
            
            # Crear y entrenar modelo
            modelo = MLPClassifier(**config_modelo)
            modelo.fit(X_train_scaled, y_train)
            
            # Predicciones
            y_pred = modelo.predict(X_test_scaled)
            y_pred_proba = modelo.predict_proba(X_test_scaled)
            
            # Calcular número de parámetros
            n_parametros = calcular_parametros_red(modelo, X_train_scaled.shape[1])
            
            # Métricas
            resultados[nombre] = {
                'modelo': modelo,
                'accuracy': accuracy_score(y_test, y_pred),
                'predicciones': y_pred,
                'probabilidades': y_pred_proba,
                'arquitectura': config['hidden_layer_sizes'],
                'activacion': config['activation'],
                'descripcion': config['descripcion'],
                'n_parametros': n_parametros,
                'convergencia': {
                    'convergio': modelo.n_iter_ < modelo.max_iter,
                    'iteraciones': modelo.n_iter_
                }
            }
            
        except Exception as e:
            print(f"    ❌ Error en {nombre}: {str(e)[:60]}...")
            continue
    
    return resultados, scaler

def calcular_parametros_red(modelo, n_entrada):
    """Calcula número aproximado de parámetros en la red"""
    try:
        capas = modelo.hidden_layer_sizes
        if isinstance(capas, int):
            capas = (capas,)
        
        n_salida = len(modelo.classes_)
        
        # Calcular parámetros
        n_parametros = n_entrada * capas[0]  # Primera capa
        for i in range(len(capas) - 1):
            n_parametros += capas[i] * capas[i + 1]  # Capas ocultas
        n_parametros += capas[-1] * n_salida  # Capa de salida
        n_parametros += sum(capas) + n_salida  # Bias
        
        return n_parametros
    except:
        return 0

def visualizar_resultados_redes(resultados, y_test, variables):
    """Crea visualizaciones de redes neuronales"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Comparación de accuracy
    nombres = list(resultados.keys())
    accuracies = [resultados[m]['accuracy'] for m in nombres]
    
    axes[0,0].bar(nombres, accuracies, color=['lightblue', 'lightgreen', 'orange'])
    axes[0,0].set_title('Precisión por Arquitectura')
    axes[0,0].set_ylabel('Accuracy')
    axes[0,0].tick_params(axis='x', rotation=45)
    for i, acc in enumerate(accuracies):
        axes[0,0].text(i, acc + 0.01, f'{acc:.3f}', ha='center')
    
    # 2. Matriz de confusión (mejor modelo)
    mejor = max(resultados.keys(), key=lambda x: resultados[x]['accuracy'])
    y_pred_mejor = resultados[mejor]['predicciones']
    
    cm = confusion_matrix(y_test, y_pred_mejor)
    clases = resultados[mejor]['modelo'].classes_
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=clases, yticklabels=clases, ax=axes[0,1])
    axes[0,1].set_title(f'Matriz de Confusión\n{mejor}')
    axes[0,1].set_xlabel('Predicción')
    axes[0,1].set_ylabel('Real')
    
    # 3. Complejidad vs Precisión
    n_params = [resultados[m]['n_parametros'] for m in nombres]
    axes[0,2].scatter(n_params, accuracies, s=100, alpha=0.7, 
                     c=['red', 'green', 'blue'][:len(n_params)])
    for i, nombre in enumerate(nombres):
        axes[0,2].annotate(nombre.split()[1], (n_params[i], accuracies[i]), 
                          xytext=(5, 5), textcoords='offset points', fontsize=8)
    axes[0,2].set_xlabel('Número de Parámetros')
    axes[0,2].set_ylabel('Precisión')
    axes[0,2].set_title('Complejidad vs Precisión')
    
    # 4. Distribución de confianza
    probabilidades = resultados[mejor]['probabilidades']
    max_probs = np.max(probabilidades, axis=1)
    axes[1,0].hist(max_probs, bins=15, alpha=0.7, color='purple', edgecolor='black')
    axes[1,0].set_title('Distribución de Confianza')
    axes[1,0].set_xlabel('Confianza Máxima')
    axes[1,0].set_ylabel('Frecuencia')
    
    # 5. Arquitectura del mejor modelo
    mejor_arq = resultados[mejor]['arquitectura']
    if isinstance(mejor_arq, int):
        mejor_arq = (mejor_arq,)
    
    capas = [len(variables)] + list(mejor_arq) + [len(clases)]
    
    x_pos = np.arange(len(capas))
    axes[1,1].bar(x_pos, capas, color=['blue', 'green', 'orange', 'red'][:len(capas)])
    axes[1,1].set_title(f'Arquitectura Mejor Red\n{mejor}')
    axes[1,1].set_xlabel('Capa')
    axes[1,1].set_ylabel('Neuronas')
    axes[1,1].set_xticks(x_pos)
    
    etiquetas = ['Entrada'] + [f'Oculta {i+1}' for i in range(len(mejor_arq))] + ['Salida']
    axes[1,1].set_xticklabels(etiquetas, rotation=45)
    
    for i, neurons in enumerate(capas):
        axes[1,1].text(i, neurons + max(capas)*0.02, str(neurons), ha='center', fontweight='bold')
    
    # 6. Convergencia por modelo
    convergencias = []
    for nombre in nombres:
        conv_info = resultados[nombre]['convergencia']
        convergencias.append(conv_info['iteraciones'])
    
    axes[1,2].bar(nombres, convergencias, color='cyan')
    axes[1,2].set_title('Iteraciones de Convergencia')
    axes[1,2].set_ylabel('Iteraciones')
    axes[1,2].tick_params(axis='x', rotation=45)
    
    for i, iter_val in enumerate(convergencias):
        axes[1,2].text(i, iter_val + max(convergencias)*0.02, str(iter_val), ha='center')
    
    plt.tight_layout()
    plt.savefig('results/graficos/redes_neuronas.png', dpi=150, bbox_inches='tight')
    plt.show()

def guardar_resultados_redes(resultados, variables, total_registros, y_test):
    """Guarda reporte de redes neuronales"""
    mejor = max(resultados.keys(), key=lambda x: resultados[x]['accuracy'])
    mejor_acc = resultados[mejor]['accuracy']
    y_pred_mejor = resultados[mejor]['predicciones']
    
    reporte = f"""REDES NEURONALES - CLASIFICACIÓN
===============================

MEJOR RED: {mejor}
Descripción: {resultados[mejor]['descripcion']}
Precisión (Accuracy): {mejor_acc:.3f} ({mejor_acc*100:.1f}%)
Arquitectura: {resultados[mejor]['arquitectura']}
Activación: {resultados[mejor]['activacion']}
Parámetros totales: {resultados[mejor]['n_parametros']:,}

COMPARACIÓN ARQUITECTURAS:
"""
    for nombre, res in resultados.items():
        conv_info = res['convergencia']
        status = "✅ Convergió" if conv_info['convergio'] else "⚠️ No convergió"
        reporte += f"\n{nombre}:"
        reporte += f"\n  - Precisión: {res['accuracy']:.3f}"
        reporte += f"\n  - Arquitectura: {res['arquitectura']}"
        reporte += f"\n  - Activación: {res['activacion']}"
        reporte += f"\n  - Parámetros: {res['n_parametros']:,}"
        reporte += f"\n  - Convergencia: {status} ({conv_info['iteraciones']} iter)"
    
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

CONFIGURACIÓN REDES:
- Escalado aplicado: StandardScaler
- Solver: lbfgs (optimización limitada)
- Regularización: L2 (alpha)
- Inicialización: aleatoria con semilla fija

PRINCIPIO REDES NEURONALES:
- Neuronas artificiales conectadas en capas
- Cada neurona aplica función de activación
- Aprende patrones complejos no lineales
- Backpropagation para ajustar pesos

FUNCIONES DE ACTIVACIÓN:
- ReLU: f(x) = max(0, x) - elimina negativos
- Tanh: f(x) = tanh(x) - salida entre -1 y 1
- Cada una captura diferentes tipos de patrones

VENTAJAS:
- Puede aprender patrones muy complejos
- Versátil para diferentes tipos de problemas
- Buena capacidad de generalización
- Funciona bien con grandes datasets

DESVENTAJAS:
- "Caja negra" - difícil de interpretar
- Requiere mucho ajuste de hiperparámetros
- Sensible al overfitting
- Computacionalmente intensivo

APLICACIONES:
- Reconocimiento de imágenes
- Procesamiento de lenguaje natural
- Diagnóstico médico
- Predicción financiera
- Sistemas de recomendación
"""
    
    with open('results/reportes/redes_neuronas_reporte.txt', 'w', encoding='utf-8') as f:
        f.write(reporte)

def ejecutar_redes_neuronas():
    """Función principal"""
    print("🧠 REDES NEURONALES - CLASIFICACIÓN")
    print("="*40)
    
    # Cargar y preparar datos
    datos = cargar_datos()
    X, y, variables = preparar_datos(datos)
    
    print(f"📊 Datos: {len(X):,} registros")
    print(f"📊 Variables: {', '.join(variables)}")
    print(f"🔢 Neuronas entrada: {len(variables)}")
    print(f"🎯 Clases salida: {len(y.unique())}")
    
    # División train/test
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        print(f"📊 División estratificada realizada")
    except:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        print(f"📊 División simple realizada")
    
    # Entrenar redes neuronales
    print(f"\n🧠 Entrenando redes neuronales...")
    resultados, scaler = entrenar_redes_neuronales(X_train, X_test, y_train, y_test)
    
    if not resultados:
        print("❌ No se pudieron entrenar redes neuronales")
        return
    
    # Mostrar resultados
    print("\nRESULTADOS:")
    for nombre, res in resultados.items():
        conv_status = "✅" if res['convergencia']['convergio'] else "⚠️"
        print(f"{nombre:15}: Accuracy = {res['accuracy']:.3f} ({res['accuracy']*100:.1f}%) {conv_status}")
    
    # Mejor modelo
    mejor = max(resultados.keys(), key=lambda x: resultados[x]['accuracy'])
    print(f"\n🏆 MEJOR: {mejor}")
    print(f"    Precisión: {resultados[mejor]['accuracy']:.3f}")
    print(f"    Arquitectura: {resultados[mejor]['arquitectura']}")
    print(f"    Parámetros: {resultados[mejor]['n_parametros']:,}")
    
    conv_mejor = resultados[mejor]['convergencia']
    if conv_mejor['convergio']:
        print(f"    ✅ Convergió en {conv_mejor['iteraciones']} iteraciones")
    else:
        print(f"    ⚠️ No convergió completamente")
    
    # Visualizar resultados
    visualizar_resultados_redes(resultados, y_test, variables)
    
    # Guardar resultados
    guardar_resultados_redes(resultados, variables, len(X), y_test)
    
    print("✅ COMPLETADO")
    
    return {
        'mejor_modelo': mejor,
        'precision': resultados[mejor]['accuracy'],
        'arquitectura': resultados[mejor]['arquitectura'],
        'resultados': resultados
    }

if __name__ == "__main__":
    ejecutar_redes_neuronas()