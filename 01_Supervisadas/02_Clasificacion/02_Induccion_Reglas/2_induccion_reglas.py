#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
INDUCCIÓN DE REGLAS - CLASIFICACIÓN
Genera reglas IF-THEN para clasificar poblaciones
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.pyplot as plt
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

def extraer_reglas_del_arbol(arbol, nombres_variables, nombres_clases):
    """Extrae reglas legibles del árbol de decisión"""
    tree = arbol.tree_
    reglas = []
    
    def obtener_reglas(nodo, condiciones=[]):
        if tree.children_left[nodo] != tree.children_right[nodo]:  # No es hoja
            variable = nombres_variables[tree.feature[nodo]]
            umbral = tree.threshold[nodo]
            
            # Rama izquierda (<=)
            condiciones_izq = condiciones + [f"{variable} <= {umbral:.2f}"]
            obtener_reglas(tree.children_left[nodo], condiciones_izq)
            
            # Rama derecha (>)
            condiciones_der = condiciones + [f"{variable} > {umbral:.2f}"]
            obtener_reglas(tree.children_right[nodo], condiciones_der)
        else:  # Es una hoja
            clase_idx = np.argmax(tree.value[nodo])
            clase = nombres_clases[clase_idx]
            muestras = tree.n_node_samples[nodo]
            pureza = tree.value[nodo][0][clase_idx] / muestras
            
            if condiciones:
                regla = f"SI {' Y '.join(condiciones)} ENTONCES {clase} (Confianza: {pureza:.2f}, Muestras: {muestras})"
                reglas.append({
                    'condiciones': condiciones,
                    'clase': clase,
                    'confianza': pureza,
                    'muestras': muestras,
                    'texto': regla
                })
    
    obtener_reglas(0)
    return reglas

def discretizar_variables(X, nombres_variables, n_bins=5):
    """Discretiza variables continuas para generar reglas más legibles"""
    discretizador = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    X_discreto = discretizador.fit_transform(X)
    
    # Crear nombres descriptivos para los bins
    nombres_discretos = []
    for i, variable in enumerate(nombres_variables):
        nombres_discretos.append(f"{variable}_bin")
    
    return X_discreto, discretizador, nombres_discretos

def ejecutar_induccion_reglas():
    print("📏 INDUCCIÓN DE REGLAS - CLASIFICACIÓN")
    print("="*40)
    print("📝 Objetivo: Generar reglas IF-THEN para clasificar poblaciones")
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
        'GRAPROES', 'PEA', 'POCUPADA', 'PDESOCUP'
    ]
    
    variables_disponibles = [v for v in variables_predictoras if v in datos.columns]
    
    if len(variables_disponibles) < 3:
        print("❌ No hay suficientes variables para generar reglas")
        return
    
    print(f"📊 Variables usadas: {', '.join(variables_disponibles)}")
    
    # 3. CREAR VARIABLE OBJETIVO
    datos['CATEGORIA_POB'] = datos['POBTOT'].apply(crear_categorias_poblacion)
    
    # 4. PREPARAR DATOS
    datos_limpios = datos[variables_disponibles + ['CATEGORIA_POB']].dropna()
    X = datos_limpios[variables_disponibles]
    y = datos_limpios['CATEGORIA_POB']
    
    print(f"🧹 Datos limpios: {len(datos_limpios):,} registros")
    print(f"📈 Distribución de categorías:")
    for categoria, count in y.value_counts().items():
        print(f"   {categoria:12}: {count:,} ({count/len(y)*100:.1f}%)")
    
    # 5. DIVIDIR DATOS
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"📊 Entrenamiento: {len(X_train):,} | Prueba: {len(X_test):,}")
    print()
    
    # 6. GENERAR DIFERENTES CONJUNTOS DE REGLAS
    configuraciones = {
        'Reglas Simples': {
            'max_depth': 4,
            'min_samples_split': 200,
            'min_samples_leaf': 100,
            'max_leaf_nodes': 15
        },
        'Reglas Precisas': {
            'max_depth': 6,
            'min_samples_split': 100,
            'min_samples_leaf': 50,
            'max_leaf_nodes': 25
        },
        'Reglas Complejas': {
            'max_depth': 8,
            'min_samples_split': 50,
            'min_samples_leaf': 25,
            'max_leaf_nodes': 40
        }
    }
    
    print("📏 GENERANDO CONJUNTOS DE REGLAS...")
    resultados = {}
    
    for nombre, params in configuraciones.items():
        print(f"   🔄 Generando {nombre}...")
        
        # Crear árbol para extraer reglas
        arbol = DecisionTreeClassifier(
            **params,
            random_state=42,
            criterion='gini'
        )
        
        arbol.fit(X_train, y_train)
        
        # Predicciones y métricas
        y_pred = arbol.predict(X_test)
        precision = accuracy_score(y_test, y_pred)
        
        # Extraer reglas
        reglas = extraer_reglas_del_arbol(arbol, variables_disponibles, arbol.classes_)
        
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
        
        print(f"   ✅ {nombre} → Precisión: {precision:.3f} | Reglas: {len(reglas)} | Confiables: {len(reglas_confiables)}")
    
    # 7. MOSTRAR LAS MEJORES REGLAS
    mejor_nombre = max(resultados.keys(), key=lambda x: resultados[x]['precision'])
    mejores_reglas = resultados[mejor_nombre]['reglas_confiables']
    
    print()
    print(f"🏆 MEJOR CONJUNTO: {mejor_nombre}")
    print(f"📏 TOP 10 REGLAS MÁS CONFIABLES:")
    print()
    
    # Ordenar reglas por confianza
    mejores_reglas.sort(key=lambda x: x['confianza'], reverse=True)
    
    for i, regla in enumerate(mejores_reglas[:10], 1):
        print(f"{i:2d}. {regla['texto']}")
        print()
    
    # 8. ANÁLISIS DE COBERTURA DE REGLAS
    print("📊 ANÁLISIS DE COBERTURA:")
    
    # Contar muestras por clase en las reglas
    cobertura_por_clase = {}
    for regla in mejores_reglas:
        clase = regla['clase']
        if clase not in cobertura_por_clase:
            cobertura_por_clase[clase] = 0
        cobertura_por_clase[clase] += regla['muestras']
    
    total_muestras = sum(cobertura_por_clase.values())
    for clase, muestras in cobertura_por_clase.items():
        porcentaje = (muestras / total_muestras) * 100
        print(f"   {clase:12}: {muestras:,} muestras ({porcentaje:.1f}%)")
    
    # 9. GENERAR REGLAS EN FORMATO TEXTO LEGIBLE
    print()
    print("📝 REGLAS EN FORMATO LEGIBLE:")
    
    reglas_texto = []
    for i, regla in enumerate(mejores_reglas[:5], 1):
        texto_legible = f"""
REGLA #{i}:
---------
{regla['texto']}

Interpretación:
Si se cumplen todas estas condiciones:
"""
        for condicion in regla['condiciones']:
            texto_legible += f"  • {condicion}\n"
        
        texto_legible += f"Entonces la población es: {regla['clase']}\n"
        texto_legible += f"Confianza: {regla['confianza']*100:.1f}%\n"
        texto_legible += f"Basado en: {regla['muestras']} casos\n"
        
        reglas_texto.append(texto_legible)
        print(texto_legible)
    
    # 10. VISUALIZACIONES
    try:
        fig = plt.figure(figsize=(16, 10))
        
        # Gráfico 1: Comparación de precisión y número de reglas
        plt.subplot(2, 3, 1)
        nombres = list(resultados.keys())
        precisiones = [resultados[m]['precision'] for m in nombres]
        
        x = np.arange(len(nombres))
        barras = plt.bar(x, precisiones, color=['lightblue', 'lightgreen', 'orange'])
        plt.title('📊 Precisión por Conjunto de Reglas', fontweight='bold')
        plt.ylabel('Precisión')
        plt.xticks(x, nombres, rotation=45, ha='right')
        plt.ylim(0, 1)
        
        for i, (barra, precision) in enumerate(zip(barras, precisiones)):
            plt.text(i, precision + 0.02, f'{precision:.3f}', ha='center', fontweight='bold')
        
        # Gráfico 2: Número de reglas generadas
        plt.subplot(2, 3, 2)
        n_reglas = [resultados[m]['n_reglas'] for m in nombres]
        n_confiables = [resultados[m]['n_reglas_confiables'] for m in nombres]
        
        x = np.arange(len(nombres))
        plt.bar(x - 0.2, n_reglas, 0.4, label='Total Reglas', color='lightcoral')
        plt.bar(x + 0.2, n_confiables, 0.4, label='Reglas Confiables', color='darkgreen')
        plt.title('📏 Número de Reglas Generadas', fontweight='bold')
        plt.ylabel('Cantidad')
        plt.xticks(x, nombres, rotation=45, ha='right')
        plt.legend()
        
        # Gráfico 3: Distribución de confianza de reglas
        plt.subplot(2, 3, 3)
        todas_confianzas = [r['confianza'] for r in mejores_reglas]
        plt.hist(todas_confianzas, bins=10, alpha=0.7, color='purple', edgecolor='black')
        plt.title('📈 Distribución de Confianza', fontweight='bold')
        plt.xlabel('Confianza')
        plt.ylabel('Número de Reglas')
        
        # Gráfico 4: Cobertura por clase
        plt.subplot(2, 3, 4)
        clases = list(cobertura_por_clase.keys())
        muestras = list(cobertura_por_clase.values())
        colores = ['gold', 'lightblue', 'lightgreen', 'lightcoral']
        
        plt.pie(muestras, labels=clases, autopct='%1.1f%%', 
               colors=colores[:len(clases)], startangle=90)
        plt.title('🎯 Cobertura por Clase', fontweight='bold')
        
        # Gráfico 5: Tamaño promedio de reglas
        plt.subplot(2, 3, 5)
        tamaños = [len(r['condiciones']) for r in mejores_reglas]
        plt.hist(tamaños, bins=range(1, max(tamaños)+2), alpha=0.7, 
                color='cyan', edgecolor='black')
        plt.title('📐 Complejidad de Reglas', fontweight='bold')
        plt.xlabel('Número de Condiciones')
        plt.ylabel('Frecuencia')
        
        # Gráfico 6: Top variables en reglas
        plt.subplot(2, 3, 6)
        contador_variables = {}
        for regla in mejores_reglas:
            for condicion in regla['condiciones']:
                variable = condicion.split()[0]
                contador_variables[variable] = contador_variables.get(variable, 0) + 1
        
        if contador_variables:
            variables = list(contador_variables.keys())[:6]
            frecuencias = [contador_variables[v] for v in variables]
            plt.barh(range(len(variables)), frecuencias, color='orange')
            plt.yticks(range(len(variables)), variables)
            plt.xlabel('Frecuencia en Reglas')
            plt.title('🔝 Variables Más Usadas', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('/home/sedc/Proyectos/MineriaDeDatos/results/graficos/induccion_reglas.png', 
                   dpi=150, bbox_inches='tight')
        plt.show()
        
        print("💾 Gráficos guardados en: results/graficos/induccion_reglas.png")
        
    except Exception as e:
        print(f"⚠️ Error creando visualizaciones: {e}")
    
    # 11. GUARDAR REGLAS EN ARCHIVO
    try:
        # Guardar todas las reglas en formato texto
        with open('/home/sedc/Proyectos/MineriaDeDatos/results/reportes/reglas_clasificacion.txt', 'w', encoding='utf-8') as f:
            f.write("REGLAS DE CLASIFICACIÓN GENERADAS\n")
            f.write("="*40 + "\n\n")
            f.write(f"Mejor conjunto: {mejor_nombre}\n")
            f.write(f"Precisión: {resultados[mejor_nombre]['precision']:.3f}\n")
            f.write(f"Total de reglas: {len(mejores_reglas)}\n\n")
            
            for texto in reglas_texto:
                f.write(texto)
                f.write("\n" + "-"*50 + "\n\n")
        
        print("📄 Reglas guardadas en: results/reportes/reglas_clasificacion.txt")
        
    except Exception as e:
        print(f"⚠️ Error guardando reglas: {e}")
    
    # 12. RESUMEN FINAL
    print()
    print("📝 RESUMEN DE INDUCCIÓN DE REGLAS:")
    print(f"   • Mejor conjunto: {mejor_nombre}")
    print(f"   • Precisión: {resultados[mejor_nombre]['precision']*100:.1f}%")
    print(f"   • Reglas generadas: {len(mejores_reglas)}")
    print(f"   • Confianza promedio: {np.mean([r['confianza'] for r in mejores_reglas]):.3f}")
    
    if resultados[mejor_nombre]['precision'] > 0.8:
        print("   • ¡Excelentes reglas generadas! 🎉")
    elif resultados[mejor_nombre]['precision'] > 0.6:
        print("   • Buenas reglas de clasificación 👍")
    else:
        print("   • Reglas moderadas, revisar parámetros 🔧")
    
    print("✅ INDUCCIÓN DE REGLAS COMPLETADA")
    return resultados

if __name__ == "__main__":
    ejecutar_induccion_reglas()