#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ÁRBOLES DE DECISIÓN - Clasificación de Comunidades
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
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
    """Prepara variables para clasificación"""
    variables = ['POBFEM', 'POBMAS', 'TOTHOG', 'VIVTOT', 'P_15YMAS', 'P_60YMAS', 'GRAPROES', 'PEA', 'POCUPADA']
    variables_disponibles = [v for v in variables if v in datos.columns]
    
    # Crear variable objetivo categórica
    datos['CATEGORIA_POB'] = datos['POBTOT'].apply(crear_categorias_poblacion)
    
    # Dataset limpio
    df = datos[variables_disponibles + ['CATEGORIA_POB']].dropna()
    
    X = df[variables_disponibles]
    y = df['CATEGORIA_POB']
    
    return X, y, variables_disponibles

def entrenar_arboles_clasificacion(X_train, X_test, y_train, y_test):
    """Entrena diferentes árboles de clasificación"""
    modelos = {
        'Decision Tree': DecisionTreeClassifier(
            max_depth=8,
            min_samples_split=50,
            min_samples_leaf=20,
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            random_state=42
        )
    }
    
    resultados = {}
    
    for nombre, modelo in modelos.items():
        # Entrenar
        modelo.fit(X_train, y_train)
        
        # Predecir
        y_pred = modelo.predict(X_test)
        
        # Métricas
        resultados[nombre] = {
            'modelo': modelo,
            'accuracy': accuracy_score(y_test, y_pred),
            'predicciones': y_pred,
            'importancias': modelo.feature_importances_
        }
    
    return resultados

def crear_arbol_simple_visual(X_train, X_test, y_train, y_test, variables):
    """Crea un árbol simple para visualización"""
    arbol_visual = DecisionTreeClassifier(
        max_depth=3,
        min_samples_split=100,
        min_samples_leaf=50,
        random_state=42
    )
    
    arbol_visual.fit(X_train, y_train)
    y_pred_visual = arbol_visual.predict(X_test)
    accuracy_visual = accuracy_score(y_test, y_pred_visual)
    
    # Visualizar árbol
    plt.figure(figsize=(20, 12))
    plot_tree(arbol_visual,
              feature_names=variables,
              class_names=arbol_visual.classes_,
              filled=True,
              rounded=True,
              fontsize=10)
    plt.title(f'Árbol de Decisión - Clasificación (Precisión: {accuracy_visual:.3f})', 
              fontsize=16, fontweight='bold')
    plt.savefig('results/graficos/arbol_clasificacion_visual.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return accuracy_visual

def visualizar_resultados_clasificacion(resultados, y_test, variables):
    """Crea visualizaciones de clasificación"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Comparación de accuracy
    nombres = list(resultados.keys())
    accuracies = [resultados[m]['accuracy'] for m in nombres]
    
    axes[0,0].bar(nombres, accuracies, color=['lightcoral', 'lightgreen'])
    axes[0,0].set_title('Precisión por Modelo')
    axes[0,0].set_ylabel('Accuracy')
    for i, acc in enumerate(accuracies):
        axes[0,0].text(i, acc + 0.01, f'{acc:.3f}', ha='center')
    
    # 2. Matriz de confusión (mejor modelo)
    mejor = max(resultados.keys(), key=lambda x: resultados[x]['accuracy'])
    y_pred_mejor = resultados[mejor]['predicciones']
    
    cm = confusion_matrix(y_test, y_pred_mejor)
    clases = np.unique(y_test)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=clases, yticklabels=clases, ax=axes[0,1])
    axes[0,1].set_title(f'Matriz de Confusión - {mejor}')
    axes[0,1].set_xlabel('Predicción')
    axes[0,1].set_ylabel('Real')
    
    # 3. Importancia de variables
    importancias = resultados[mejor]['importancias']
    axes[1,0].barh(variables, importancias, color='orange')
    axes[1,0].set_title(f'Importancia Variables - {mejor}')
    axes[1,0].set_xlabel('Importancia')
    
    # 4. Distribución de clases
    distribucion = pd.Series(y_test).value_counts()
    axes[1,1].pie(distribucion.values, labels=distribucion.index, autopct='%1.1f%%')
    axes[1,1].set_title('Distribución de Clases')
    
    plt.tight_layout()
    plt.savefig('results/graficos/arboles_clasificacion.png', dpi=150, bbox_inches='tight')
    plt.show()

def guardar_resultados_clasificacion(resultados, variables, total_registros, y_test):
    """Guarda reporte de clasificación"""
    mejor = max(resultados.keys(), key=lambda x: resultados[x]['accuracy'])
    mejor_acc = resultados[mejor]['accuracy']
    y_pred_mejor = resultados[mejor]['predicciones']
    
    # Reporte detallado por clase
    reporte_sklearn = classification_report(y_test, y_pred_mejor, output_dict=True)
    
    reporte = f"""ÁRBOLES DE DECISIÓN - CLASIFICACIÓN
==================================

MEJOR MODELO: {mejor}
Precisión (Accuracy): {mejor_acc:.3f} ({mejor_acc*100:.1f}%)

COMPARACIÓN MODELOS:
"""
    for nombre, res in resultados.items():
        reporte += f"\n{nombre}: Accuracy = {res['accuracy']:.3f}"
    
    reporte += f"\n\nMÉTRICAS POR CLASE ({mejor}):\n"
    for clase in ['Pequeña', 'Mediana', 'Grande', 'Muy_Grande']:
        if clase in reporte_sklearn:
            prec = reporte_sklearn[clase]['precision']
            rec = reporte_sklearn[clase]['recall']
            f1 = reporte_sklearn[clase]['f1-score']
            support = reporte_sklearn[clase]['support']
            reporte += f"{clase}: Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}, N={support}\n"
    
    # Importancia de variables
    importancias = resultados[mejor]['importancias']
    reporte += f"\nIMPORTANCIA VARIABLES ({mejor}):\n"
    for var, imp in zip(variables, importancias):
        reporte += f"- {var}: {imp:.3f}\n"
    
    reporte += f"""
DATOS UTILIZADOS:
- Total registros: {total_registros:,}
- Variables: {', '.join(variables)}
- División: 70% entrenamiento, 30% prueba

CATEGORÍAS DE POBLACIÓN:
- Pequeña: ≤ 500 habitantes
- Mediana: 501 - 2,000 habitantes  
- Grande: 2,001 - 8,000 habitantes
- Muy Grande: > 8,000 habitantes

INTERPRETACIÓN:
- Los árboles pueden identificar patrones para clasificar comunidades
- Son interpretables: se pueden ver las reglas de decisión
- Útil para políticas públicas diferenciadas por tipo de comunidad
"""
    
    with open('results/reportes/arboles_clasificacion_reporte.txt', 'w', encoding='utf-8') as f:
        f.write(reporte)

def ejecutar_arboles_decision():
    """Función principal"""
    print("🌳 ÁRBOLES DE DECISIÓN - CLASIFICACIÓN")
    print("="*40)
    
    # Cargar y preparar datos
    datos = cargar_datos()
    X, y, variables = preparar_datos(datos)
    
    print(f"📊 Datos: {len(X):,} registros")
    print(f"📊 Variables: {', '.join(variables)}")
    
    # Mostrar distribución de clases
    print(f"📊 Distribución de categorías:")
    for categoria, count in y.value_counts().items():
        print(f"   {categoria}: {count:,} ({count/len(y)*100:.1f}%)")
    
    # División train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Entrenar modelos
    print("\n🌱 Entrenando árboles de clasificación...")
    resultados = entrenar_arboles_clasificacion(X_train, X_test, y_train, y_test)
    
    # Mostrar resultados
    print("\nRESULTADOS:")
    for nombre, res in resultados.items():
        print(f"{nombre:15}: Accuracy = {res['accuracy']:.3f} ({res['accuracy']*100:.1f}%)")
    
    # Mejor modelo
    mejor = max(resultados.keys(), key=lambda x: resultados[x]['accuracy'])
    print(f"\n🏆 MEJOR: {mejor} (Accuracy = {resultados[mejor]['accuracy']:.3f})")
    
    # Crear árbol visual
    print("\n🌱 Creando árbol visual...")
    accuracy_visual = crear_arbol_simple_visual(X_train, X_test, y_train, y_test, variables)
    print(f"Árbol visual: Accuracy = {accuracy_visual:.3f}")
    
    # Visualizar resultados
    visualizar_resultados_clasificacion(resultados, y_test, variables)
    
    # Guardar resultados
    guardar_resultados_clasificacion(resultados, variables, len(X), y_test)
    
    print("✅ COMPLETADO")
    
    return {
        'mejor_modelo': mejor,
        'precision': resultados[mejor]['accuracy'],
        'resultados': resultados
    }

if __name__ == "__main__":
    ejecutar_arboles_decision()