#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
INDUCCIÓN DE REGLAS - Generación de Reglas IF-THEN
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

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
    """Prepara variables para inducción de reglas"""
    variables = ['POBFEM', 'POBMAS', 'TOTHOG', 'VIVTOT', 'P_15YMAS', 'GRAPROES', 'PEA', 'POCUPADA']
    variables_disponibles = [v for v in variables if v in datos.columns]
    
    # Crear variable objetivo categórica
    datos['CATEGORIA_POB'] = datos['POBTOT'].apply(crear_categorias_poblacion)
    
    # Dataset limpio
    df = datos[variables_disponibles + ['CATEGORIA_POB']].dropna()
    
    X = df[variables_disponibles]
    y = df['CATEGORIA_POB']
    
    return X, y, variables_disponibles

def extraer_reglas_del_arbol(arbol, nombres_variables, nombres_clases):
    """Extrae reglas legibles del árbol de decisión"""
    tree = arbol.tree_
    reglas = []
    
    def obtener_reglas(nodo, condiciones=[]):
        if tree.children_left[nodo] != tree.children_right[nodo]:  # No es hoja
            variable = nombres_variables[tree.feature[nodo]]
            umbral = tree.threshold[nodo]
            
            # Rama izquierda (<=)
            condiciones_izq = condiciones + [f"{variable} <= {umbral:.0f}"]
            obtener_reglas(tree.children_left[nodo], condiciones_izq)
            
            # Rama derecha (>)
            condiciones_der = condiciones + [f"{variable} > {umbral:.0f}"]
            obtener_reglas(tree.children_right[nodo], condiciones_der)
        else:  # Es una hoja
            clase_idx = np.argmax(tree.value[nodo])
            clase = nombres_clases[clase_idx]
            muestras = tree.n_node_samples[nodo]
            pureza = tree.value[nodo][0][clase_idx] / muestras
            
            if condiciones:
                regla = {
                    'condiciones': condiciones,
                    'clase': clase,
                    'confianza': pureza,
                    'muestras': muestras,
                    'texto': f"SI {' Y '.join(condiciones)} ENTONCES {clase}"
                }
                reglas.append(regla)
    
    obtener_reglas(0)
    return reglas

def generar_diferentes_conjuntos_reglas(X_train, X_test, y_train, y_test, variables):
    """Genera diferentes conjuntos de reglas con distintas configuraciones"""
    configuraciones = {
        'Reglas Simples': {
            'max_depth': 4,
            'min_samples_split': 200,
            'min_samples_leaf': 100
        },
        'Reglas Detalladas': {
            'max_depth': 6,
            'min_samples_split': 100,
            'min_samples_leaf': 50
        },
        'Reglas Precisas': {
            'max_depth': 8,
            'min_samples_split': 50,
            'min_samples_leaf': 25
        }
    }
    
    resultados = {}
    
    for nombre, params in configuraciones.items():
        # Crear árbol para extraer reglas
        arbol = DecisionTreeClassifier(**params, random_state=42)
        arbol.fit(X_train, y_train)
        
        # Predicciones y métricas
        y_pred = arbol.predict(X_test)
        precision = accuracy_score(y_test, y_pred)
        
        # Extraer reglas
        reglas = extraer_reglas_del_arbol(arbol, variables, arbol.classes_)
        
        # Filtrar reglas por confianza mínima
        reglas_confiables = [r for r in reglas if r['confianza'] >= 0.7]
        
        resultados[nombre] = {
            'modelo': arbol,
            'precision': precision,
            'reglas': reglas,
            'reglas_confiables': reglas_confiables,
            'n_reglas': len(reglas),
            'n_reglas_confiables': len(reglas_confiables)
        }
    
    return resultados

def mostrar_mejores_reglas(reglas, titulo="TOP REGLAS", limite=10):
    """Muestra las mejores reglas ordenadas por confianza"""
    print(f"\n📏 {titulo}:")
    print("-" * 60)
    
    # Ordenar por confianza
    reglas_ordenadas = sorted(reglas, key=lambda x: x['confianza'], reverse=True)
    
    for i, regla in enumerate(reglas_ordenadas[:limite], 1):
        print(f"\n{i:2d}. {regla['texto']}")
        print(f"    Confianza: {regla['confianza']*100:.1f}% | Muestras: {regla['muestras']}")

def visualizar_reglas(resultados):
    """Crea visualizaciones de las reglas generadas"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Comparación de precisión
    nombres = list(resultados.keys())
    precisiones = [resultados[m]['precision'] for m in nombres]
    
    axes[0,0].bar(nombres, precisiones, color=['lightblue', 'lightgreen', 'orange'])
    axes[0,0].set_title('Precisión por Conjunto de Reglas')
    axes[0,0].set_ylabel('Accuracy')
    axes[0,0].tick_params(axis='x', rotation=45)
    for i, prec in enumerate(precisiones):
        axes[0,0].text(i, prec + 0.01, f'{prec:.3f}', ha='center')
    
    # 2. Número de reglas generadas
    n_reglas = [resultados[m]['n_reglas'] for m in nombres]
    n_confiables = [resultados[m]['n_reglas_confiables'] for m in nombres]
    
    x = np.arange(len(nombres))
    width = 0.35
    axes[0,1].bar(x - width/2, n_reglas, width, label='Total Reglas', color='lightcoral')
    axes[0,1].bar(x + width/2, n_confiables, width, label='Reglas Confiables', color='darkgreen')
    axes[0,1].set_title('Número de Reglas Generadas')
    axes[0,1].set_ylabel('Cantidad')
    axes[0,1].set_xticks(x)
    axes[0,1].set_xticklabels(nombres, rotation=45)
    axes[0,1].legend()
    
    # 3. Distribución de confianza
    mejor = max(resultados.keys(), key=lambda x: resultados[x]['precision'])
    confianzas = [r['confianza'] for r in resultados[mejor]['reglas']]
    
    axes[1,0].hist(confianzas, bins=15, alpha=0.7, color='purple', edgecolor='black')
    axes[1,0].set_title(f'Distribución de Confianza - {mejor}')
    axes[1,0].set_xlabel('Confianza')
    axes[1,0].set_ylabel('Número de Reglas')
    
    # 4. Cobertura por clase
    cobertura_por_clase = {}
    for regla in resultados[mejor]['reglas_confiables']:
        clase = regla['clase']
        cobertura_por_clase[clase] = cobertura_por_clase.get(clase, 0) + regla['muestras']
    
    if cobertura_por_clase:
        clases = list(cobertura_por_clase.keys())
        muestras = list(cobertura_por_clase.values())
        axes[1,1].pie(muestras, labels=clases, autopct='%1.1f%%', startangle=90)
        axes[1,1].set_title('Cobertura por Clase')
    
    plt.tight_layout()
    plt.savefig('results/graficos/induccion_reglas.png', dpi=150, bbox_inches='tight')
    plt.show()

def guardar_reglas_texto(resultados, variables, total_registros):
    """Guarda las reglas en formato texto"""
    mejor = max(resultados.keys(), key=lambda x: resultados[x]['precision'])
    mejores_reglas = resultados[mejor]['reglas_confiables']
    
    # Ordenar reglas por confianza
    mejores_reglas.sort(key=lambda x: x['confianza'], reverse=True)
    
    reporte = f"""INDUCCIÓN DE REGLAS - REPORTE
===========================

MEJOR CONJUNTO: {mejor}
Precisión: {resultados[mejor]['precision']:.3f} ({resultados[mejor]['precision']*100:.1f}%)
Total reglas: {len(resultados[mejor]['reglas'])}
Reglas confiables: {len(mejores_reglas)}

COMPARACIÓN CONJUNTOS:
"""
    for nombre, resultado in resultados.items():
        reporte += f"\n{nombre}:"
        reporte += f"\n  - Precisión: {resultado['precision']:.3f}"
        reporte += f"\n  - Total reglas: {resultado['n_reglas']}"
        reporte += f"\n  - Reglas confiables: {resultado['n_reglas_confiables']}"
    
    reporte += f"\n\nTOP 10 REGLAS MÁS CONFIABLES:\n"
    reporte += "=" * 50 + "\n"
    
    for i, regla in enumerate(mejores_reglas[:10], 1):
        reporte += f"\n{i:2d}. {regla['texto']}"
        reporte += f"\n    Confianza: {regla['confianza']*100:.1f}%"
        reporte += f"\n    Casos que cubre: {regla['muestras']}"
        reporte += f"\n    Condiciones:"
        for condicion in regla['condiciones']:
            reporte += f"\n      • {condicion}"
        reporte += f"\n{'-'*50}"
    
    reporte += f"""

DATOS UTILIZADOS:
- Total registros: {total_registros:,}
- Variables: {', '.join(variables)}
- División: 70% entrenamiento, 30% prueba

INTERPRETACIÓN DE REGLAS:
- Cada regla es una secuencia IF-THEN
- La confianza indica qué tan segura es la regla
- Se pueden usar para tomar decisiones automáticas
- Son completamente interpretables y explicables

APLICACIONES PRÁCTICAS:
- Clasificar automáticamente nuevas comunidades
- Guías para políticas públicas específicas
- Sistemas de apoyo a la decisión
- Auditoría de criterios de clasificación
"""
    
    with open('results/reportes/induccion_reglas_reporte.txt', 'w', encoding='utf-8') as f:
        f.write(reporte)

def ejecutar_induccion_reglas():
    """Función principal"""
    print("📏 INDUCCIÓN DE REGLAS")
    print("="*30)
    
    # Cargar y preparar datos
    datos = cargar_datos()
    X, y, variables = preparar_datos(datos)
    
    print(f"📊 Datos: {len(X):,} registros")
    print(f"📊 Variables: {', '.join(variables)}")
    
    # División train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Generar diferentes conjuntos de reglas
    print("\n📏 Generando conjuntos de reglas...")
    resultados = generar_diferentes_conjuntos_reglas(X_train, X_test, y_train, y_test, variables)
    
    # Mostrar resultados por conjunto
    print("\nRESULTADOS POR CONJUNTO:")
    for nombre, res in resultados.items():
        print(f"{nombre:18}: Accuracy = {res['precision']:.3f} | Reglas = {res['n_reglas']} | Confiables = {res['n_reglas_confiables']}")
    
    # Mejor conjunto
    mejor = max(resultados.keys(), key=lambda x: resultados[x]['precision'])
    print(f"\n🏆 MEJOR CONJUNTO: {mejor}")
    print(f"    Precisión: {resultados[mejor]['precision']:.3f}")
    print(f"    Reglas confiables: {resultados[mejor]['n_reglas_confiables']}")
    
    # Mostrar mejores reglas
    mejores_reglas = resultados[mejor]['reglas_confiables']
    mostrar_mejores_reglas(mejores_reglas, "REGLAS MÁS CONFIABLES", 8)
    
    # Visualizar
    visualizar_reglas(resultados)
    
    # Guardar reglas
    guardar_reglas_texto(resultados, variables, len(X))
    
    print("\n✅ COMPLETADO")
    
    return {
        'mejor_conjunto': mejor,
        'precision': resultados[mejor]['precision'],
        'num_reglas': resultados[mejor]['n_reglas_confiables'],
        'resultados': resultados
    }

if __name__ == "__main__":
    ejecutar_induccion_reglas()