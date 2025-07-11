#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLASIFICACIÓN BAYESIANA - Naive Bayes
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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

def preparar_datos(datos):
    """Prepara variables para clasificación bayesiana"""
    variables = ['POBFEM', 'POBMAS', 'TOTHOG', 'VIVTOT', 'P_15YMAS', 'P_60YMAS', 'GRAPROES', 'PEA', 'POCUPADA', 'PDESOCUP']
    variables_disponibles = [v for v in variables if v in datos.columns]
    
    # Crear variable objetivo categórica
    datos['CATEGORIA_POB'] = datos['POBTOT'].apply(crear_categorias_poblacion)
    
    # Dataset limpio
    df = datos[variables_disponibles + ['CATEGORIA_POB']].dropna()
    
    X = df[variables_disponibles]
    y = df['CATEGORIA_POB']
    
    return X, y, variables_disponibles

def calcular_probabilidades_priori(y):
    """Calcula probabilidades a priori de cada clase"""
    conteos = y.value_counts()
    total = len(y)
    probabilidades = {}
    
    for clase, conteo in conteos.items():
        prob = conteo / total
        probabilidades[clase] = prob
    
    return probabilidades

def entrenar_modelos_bayesianos(X_train, X_test, y_train, y_test):
    """Entrena diferentes modelos Naive Bayes"""
    
    # Preparar datos para diferentes tipos de Naive Bayes
    
    # 1. Para Gaussian NB (datos continuos)
    scaler_gaussian = StandardScaler()
    X_train_gaussian = scaler_gaussian.fit_transform(X_train)
    X_test_gaussian = scaler_gaussian.transform(X_test)
    
    # 2. Para Multinomial NB (datos discretos positivos)
    scaler_multinomial = MinMaxScaler()
    X_train_multinomial = scaler_multinomial.fit_transform(X_train)
    X_test_multinomial = scaler_multinomial.transform(X_test)
    
    # 3. Para Bernoulli NB (datos binarios)
    X_train_bernoulli = (X_train > X_train.median()).astype(int)
    X_test_bernoulli = (X_test > X_train.median()).astype(int)
    
    modelos = {
        'Gaussiano': {
            'modelo': GaussianNB(),
            'X_train': X_train_gaussian,
            'X_test': X_test_gaussian,
            'descripcion': 'Asume distribución normal'
        },
        'Multinomial': {
            'modelo': MultinomialNB(alpha=1.0),
            'X_train': X_train_multinomial,
            'X_test': X_test_multinomial,
            'descripcion': 'Para conteos y frecuencias'
        },
        'Bernoulli': {
            'modelo': BernoulliNB(alpha=1.0),
            'X_train': X_train_bernoulli,
            'X_test': X_test_bernoulli,
            'descripcion': 'Para características binarias'
        }
    }
    
    resultados = {}
    
    for nombre, config in modelos.items():
        try:
            # Entrenar modelo
            modelo = config['modelo']
            modelo.fit(config['X_train'], y_train)
            
            # Predicciones
            y_pred = modelo.predict(config['X_test'])
            y_pred_proba = modelo.predict_proba(config['X_test'])
            
            # Métricas
            resultados[nombre] = {
                'modelo': modelo,
                'accuracy': accuracy_score(y_test, y_pred),
                'predicciones': y_pred,
                'probabilidades': y_pred_proba,
                'descripcion': config['descripcion']
            }
            
        except Exception as e:
            print(f"    ❌ Error en {nombre}: {str(e)[:50]}...")
            continue
    
    return resultados

def analizar_probabilidades(resultados, y_test):
    """Analiza las probabilidades de predicción"""
    mejor = max(resultados.keys(), key=lambda x: resultados[x]['accuracy'])
    probabilidades = resultados[mejor]['probabilidades']
    clases = resultados[mejor]['modelo'].classes_
    
    # Confianza promedio por clase
    analisis = {}
    y_pred_mejor = resultados[mejor]['predicciones']
    
    for i, clase in enumerate(clases):
        indices_clase = np.where(y_pred_mejor == clase)[0]
        if len(indices_clase) > 0:
            confianza_promedio = np.mean(probabilidades[indices_clase, i])
            analisis[clase] = {
                'confianza_promedio': confianza_promedio,
                'predicciones_clase': len(indices_clase)
            }
    
    return analisis

def visualizar_resultados_bayesianos(resultados, y_test, prob_priori):
    """Crea visualizaciones de clasificación bayesiana"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Comparación de accuracy
    nombres = list(resultados.keys())
    accuracies = [resultados[m]['accuracy'] for m in nombres]
    
    axes[0,0].bar(nombres, accuracies, color=['lightblue', 'lightgreen', 'orange'])
    axes[0,0].set_title('Precisión por Modelo Bayesiano')
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
    axes[0,1].set_title(f'Matriz de Confusión - {mejor}')
    axes[0,1].set_xlabel('Predicción')
    axes[0,1].set_ylabel('Real')
    
    # 3. Probabilidades a priori
    clases_priori = list(prob_priori.keys())
    probs_priori = list(prob_priori.values())
    axes[0,2].pie(probs_priori, labels=clases_priori, autopct='%1.1f%%', startangle=90)
    axes[0,2].set_title('Probabilidades A Priori')
    
    # 4. Distribución de confianza
    probabilidades = resultados[mejor]['probabilidades']
    max_probs = np.max(probabilidades, axis=1)
    axes[1,0].hist(max_probs, bins=20, alpha=0.7, color='purple', edgecolor='black')
    axes[1,0].set_title('Distribución de Confianza')
    axes[1,0].set_xlabel('Confianza Máxima')
    axes[1,0].set_ylabel('Frecuencia')
    
    # 5. F1-Score por clase
    reporte = classification_report(y_test, y_pred_mejor, output_dict=True)
    f1_scores = []
    categorias_f1 = []
    for categoria in clases:
        if categoria in reporte:
            f1_scores.append(reporte[categoria]['f1-score'])
            categorias_f1.append(categoria)
    
    axes[1,1].bar(categorias_f1, f1_scores, color='gold')
    axes[1,1].set_title('F1-Score por Categoría')
    axes[1,1].set_ylabel('F1-Score')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    # 6. Comparación de características de modelos
    descripciones = [resultados[m]['descripcion'] for m in nombres]
    axes[1,2].text(0.1, 0.9, 'CARACTERÍSTICAS DE MODELOS:', fontsize=12, fontweight='bold')
    
    for i, (nombre, desc) in enumerate(zip(nombres, descripciones)):
        axes[1,2].text(0.1, 0.7 - i*0.2, f'{nombre}:', fontsize=10, fontweight='bold')
        axes[1,2].text(0.1, 0.65 - i*0.2, desc, fontsize=9)
    
    axes[1,2].set_xlim(0, 1)
    axes[1,2].set_ylim(0, 1)
    axes[1,2].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/graficos/clasificacion_bayesiana.png', dpi=150, bbox_inches='tight')
    plt.show()

def guardar_resultados_bayesianos(resultados, variables, total_registros, prob_priori, y_test):
    """Guarda reporte de clasificación bayesiana"""
    mejor = max(resultados.keys(), key=lambda x: resultados[x]['accuracy'])
    mejor_acc = resultados[mejor]['accuracy']
    y_pred_mejor = resultados[mejor]['predicciones']
    
    # Análisis de probabilidades
    analisis_prob = analizar_probabilidades(resultados, y_test)
    
    reporte = f"""CLASIFICACIÓN BAYESIANA - REPORTE
================================

MEJOR MODELO: {mejor} Naive Bayes
Descripción: {resultados[mejor]['descripcion']}
Precisión (Accuracy): {mejor_acc:.3f} ({mejor_acc*100:.1f}%)

COMPARACIÓN MODELOS:
"""
    for nombre, res in resultados.items():
        reporte += f"\n{nombre} NB:"
        reporte += f"\n  - Precisión: {res['accuracy']:.3f}"
        reporte += f"\n  - Descripción: {res['descripcion']}"
    
    reporte += f"\n\nPROBABILIDADES A PRIORI:\n"
    for clase, prob in prob_priori.items():
        reporte += f"P({clase}) = {prob:.3f} ({prob*100:.1f}%)\n"
    
    # Métricas detalladas por clase
    reporte_sklearn = classification_report(y_test, y_pred_mejor, output_dict=True)
    reporte += f"\nMÉTRICAS POR CLASE ({mejor}):\n"
    for clase in ['Pequeña', 'Mediana', 'Grande', 'Muy_Grande']:
        if clase in reporte_sklearn:
            prec = reporte_sklearn[clase]['precision']
            rec = reporte_sklearn[clase]['recall']
            f1 = reporte_sklearn[clase]['f1-score']
            support = reporte_sklearn[clase]['support']
            reporte += f"{clase}: Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}, N={support}\n"
    
    # Análisis de confianza
    reporte += f"\nANÁLISIS DE CONFIANZA:\n"
    for clase, info in analisis_prob.items():
        reporte += f"{clase}: Confianza promedio = {info['confianza_promedio']:.3f}, Predicciones = {info['predicciones_clase']}\n"
    
    reporte += f"""
DATOS UTILIZADOS:
- Total registros: {total_registros:,}
- Variables: {', '.join(variables)}
- División: 70% entrenamiento, 30% prueba

PRINCIPIO BAYESIANO:
- Usa el Teorema de Bayes: P(Clase|Datos) = P(Datos|Clase) × P(Clase) / P(Datos)
- Asume independencia entre variables (naive)
- Calcula probabilidades en lugar de dar respuestas definitivas

TIPOS DE NAIVE BAYES:
- Gaussiano: Para variables continuas con distribución normal
- Multinomial: Para conteos y frecuencias (ej: palabras en texto)
- Bernoulli: Para características binarias (presencia/ausencia)

VENTAJAS:
- Rápido y eficiente
- Funciona bien con pocos datos
- Proporciona probabilidades de clasificación
- Robusto al ruido

APLICACIONES:
- Filtros de spam
- Análisis de sentimientos
- Diagnóstico médico
- Clasificación de documentos
"""
    
    with open('results/reportes/clasificacion_bayesiana_reporte.txt', 'w', encoding='utf-8') as f:
        f.write(reporte)

def ejecutar_clasificacion_bayesiana():
    """Función principal"""
    print("🎲 CLASIFICACIÓN BAYESIANA")
    print("="*35)
    
    # Cargar y preparar datos
    datos = cargar_datos()
    X, y, variables = preparar_datos(datos)
    
    print(f"📊 Datos: {len(X):,} registros")
    print(f"📊 Variables: {', '.join(variables)}")
    
    # Calcular probabilidades a priori
    prob_priori = calcular_probabilidades_priori(y)
    print(f"\n🎯 Probabilidades A Priori:")
    for clase, prob in prob_priori.items():
        print(f"   P({clase}) = {prob:.3f} ({prob*100:.1f}%)")
    
    # División train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Entrenar modelos bayesianos
    print("\n🧠 Entrenando modelos bayesianos...")
    resultados = entrenar_modelos_bayesianos(X_train, X_test, y_train, y_test)
    
    if not resultados:
        print("❌ No se pudieron entrenar modelos bayesianos")
        return
    
    # Mostrar resultados
    print("\nRESULTADOS:")
    for nombre, res in resultados.items():
        print(f"{nombre:12}: Accuracy = {res['accuracy']:.3f} ({res['accuracy']*100:.1f}%)")
    
    # Mejor modelo
    mejor = max(resultados.keys(), key=lambda x: resultados[x]['accuracy'])
    print(f"\n🏆 MEJOR: {mejor} (Accuracy = {resultados[mejor]['accuracy']:.3f})")
    print(f"    Descripción: {resultados[mejor]['descripcion']}")
    
    # Análisis de probabilidades
    print(f"\n📊 Análisis de confianza:")
    analisis_prob = analizar_probabilidades(resultados, y_test)
    for clase, info in analisis_prob.items():
        print(f"   {clase}: Confianza = {info['confianza_promedio']:.3f}")
    
    # Visualizar resultados
    visualizar_resultados_bayesianos(resultados, y_test, prob_priori)
    
    # Guardar resultados
    guardar_resultados_bayesianos(resultados, variables, len(X), prob_priori, y_test)
    
    print("✅ COMPLETADO")
    
    return {
        'mejor_modelo': mejor,
        'precision': resultados[mejor]['accuracy'],
        'resultados': resultados
    }

if __name__ == "__main__":
    ejecutar_clasificacion_bayesiana()