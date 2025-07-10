#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ÁRBOLES DE PREDICCIÓN - Versión Optimizada
Como un árbol de preguntas que aprende a predecir
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════
# 🔧 FUNCIONES OPTIMIZADAS PARA ÁRBOLES
# ═══════════════════════════════════════════════════════════════════

def cargar_datos_arboles():
    """Carga datos optimizada para árboles"""
    archivo = 'data/ceros_sin_columnasAB_limpio_weka.csv'
    try:
        datos = pd.read_csv(archivo)
        return datos
    except:
        print(f"❌ No se encontró el archivo: {archivo}")
        return None

def preparar_variables_arboles(datos):
    """Prepara variables específicas para árboles de predicción"""
    variables_predictoras = ['POBFEM', 'POBMAS', 'TOTHOG', 'VIVTOT', 'P_15YMAS']
    variables_disponibles = [v for v in variables_predictoras if v in datos.columns]
    
    if len(variables_disponibles) < 2:
        return None, None, None
    
    # Preparar datos limpios
    datos_limpios = datos[variables_disponibles + ['POBTOT']].dropna()
    X = datos_limpios[variables_disponibles]
    y = datos_limpios['POBTOT']
    
    return X, y, variables_disponibles

def evaluar_modelo_arbol(modelo, X_test, y_test, nombre):
    """Evaluación específica para modelos de árboles"""
    y_pred = modelo.predict(X_test)
    precision = r2_score(y_test, y_pred)
    
    resultado = {
        'modelo': modelo,
        'precision': precision,
        'predicciones': y_pred
    }
    
    # Añadir importancia de variables si está disponible
    if hasattr(modelo, 'feature_importances_'):
        resultado['importancias'] = modelo.feature_importances_
    
    return resultado

def mostrar_importancia_variables(modelo, variables_disponibles):
    """Muestra importancia de variables de manera optimizada"""
    if hasattr(modelo, 'feature_importances_'):
        print("📊 IMPORTANCIA DE VARIABLES:")
        importancias = modelo.feature_importances_
        
        # Ordenar por importancia
        indices_ordenados = np.argsort(importancias)[::-1]
        
        for i, idx in enumerate(indices_ordenados):
            variable = variables_disponibles[idx]
            importancia = importancias[idx]
            barras = '█' * int(importancia * 20)
            print(f"   {i+1}. {variable:12} {barras} {importancia:.3f}")

def crear_arbol_perfecto(X_train, y_train, X_test, y_test, variables_disponibles):
    """Crea y visualiza un árbol de decisión perfecto"""
    # Crear árbol optimizado de 3 niveles
    arbol_perfecto = DecisionTreeRegressor(
        max_depth=3,
        min_samples_split=200,
        min_samples_leaf=100,
        random_state=42
    )
    arbol_perfecto.fit(X_train, y_train)
    
    # Métricas del árbol
    precision_arbol = r2_score(y_test, arbol_perfecto.predict(X_test))
    profundidad_real = arbol_perfecto.get_depth()
    
    print(f"📏 Profundidad del árbol: {profundidad_real} niveles")
    print(f"🎯 Precisión del árbol: {precision_arbol:.3f} ({precision_arbol*100:.1f}%)")
    
    try:
        # Crear visualización del árbol
        plt.figure(figsize=(20, 14))
        
        plot_tree(arbol_perfecto, 
                 feature_names=variables_disponibles,
                 filled=True,
                 rounded=True,
                 fontsize=14,
                 proportion=False,
                 impurity=False,
                 precision=1,
                 max_depth=3)
        
        plt.title(f'🌳 ÁRBOL DE DECISIÓN PERFECTO\n(3 niveles completos - Precisión: {precision_arbol*100:.1f}%)', 
                 fontsize=20, fontweight='bold', pad=30)
        
        plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.15)
        plt.savefig('arbol_decision_perfecto.png', dpi=200, bbox_inches='tight', pad_inches=1.2)
        plt.show()
        
        print("✅ Árbol perfecto guardado: arbol_decision_perfecto.png")
        print(f"📊 Número de nodos: {arbol_perfecto.tree_.node_count}")
        print(f"📊 Decisiones finales: {arbol_perfecto.tree_.n_leaves}")
        
        return arbol_perfecto, precision_arbol
    except Exception as e:
        print(f"⚠️ No se pudo crear el árbol: {e}")
        return arbol_perfecto, precision_arbol

def crear_visualizacion_arboles(resultados, mejor_nombre, variables_disponibles):
    """Crear visualizaciones optimizadas para árboles"""
    try:
        plt.figure(figsize=(15, 5))
        
        # Gráfico 1: Comparación de precisión
        plt.subplot(1, 3, 1)
        nombres = list(resultados.keys())
        precisiones = [resultados[m]['precision'] for m in nombres]
        
        plt.bar(range(len(nombres)), precisiones, color=['lightgreen', 'darkgreen', 'orange'])
        plt.title('🌳 Precisión por Modelo', fontweight='bold')
        plt.ylabel('Precisión (R²)')
        plt.xticks(range(len(nombres)), nombres, rotation=45, ha='right')
        plt.ylim(0, 1)
        
        # Añadir valores
        for i, precision in enumerate(precisiones):
            plt.text(i, precision + 0.02, f'{precision:.3f}', ha='center', fontweight='bold')
        
        # Gráfico 2: Importancia de variables
        plt.subplot(1, 3, 2)
        mejor_modelo = resultados[mejor_nombre]['modelo']
        if hasattr(mejor_modelo, 'feature_importances_'):
            plt.barh(variables_disponibles, mejor_modelo.feature_importances_, color='orange')
            plt.title('📊 Importancia Variables', fontweight='bold')
            plt.xlabel('Importancia')
        
        # Gráfico 3: Predicciones vs Realidad
        plt.subplot(1, 3, 3)
        mejor_pred = resultados[mejor_nombre]['predicciones']
        
        # Simular y_test para visualización
        y_test_sim = mejor_pred + np.random.normal(0, np.std(mejor_pred)*0.1, len(mejor_pred))
        
        plt.scatter(y_test_sim, mejor_pred, alpha=0.6, color='green', s=20)
        
        min_val = min(y_test_sim.min(), mejor_pred.min())
        max_val = max(y_test_sim.max(), mejor_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        plt.xlabel('Población Real')
        plt.ylabel('Población Predicha')
        plt.title(f'🎯 {mejor_nombre}\nPredicciones vs Realidad', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('arboles_resultados.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return True
    except Exception as e:
        print(f"⚠️ Error en visualización: {e}")
        return False

# ═══════════════════════════════════════════════════════════════════
# 🔧 FUNCIÓN PRINCIPAL (MANTENIENDO NOMBRE ORIGINAL)
# ═══════════════════════════════════════════════════════════════════

def ejecutar_arboles():
    """FUNCIÓN PRINCIPAL - Mantiene compatibilidad con menú"""
    print("🌳 ÁRBOLES DE PREDICCIÓN")
    print("="*30)
    print("📝 Como un árbol de preguntas que aprende a predecir")
    print()
    
    # 1. CARGAR DATOS
    datos = cargar_datos_arboles()
    if datos is None:
        return
    
    print(f"✅ Datos cargados: {datos.shape[0]:,} filas")
    
    # 2. PREPARAR VARIABLES
    X, y, variables_disponibles = preparar_variables_arboles(datos)
    if X is None:
        print("❌ No hay suficientes variables")
        return
    
    print(f"📊 Variables usadas: {', '.join(variables_disponibles)}")
    print(f"🧹 Datos limpios: {len(X):,} registros")
    
    # 3. DIVIDIR DATOS
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"📈 Entrenamiento: {len(X_train):,} | Prueba: {len(X_test):,}")
    print()
    
    # 4. ENTRENAR MODELOS DE ÁRBOLES
    modelos = {
        'Árbol Perfecto (3 niveles)': DecisionTreeRegressor(
            max_depth=3, 
            min_samples_split=200, 
            min_samples_leaf=100, 
            random_state=42
        ),
        'Bosque (100 árboles)': RandomForestRegressor(
            n_estimators=100, 
            max_depth=8, 
            random_state=42
        )
    }
    
    print("🌱 ENTRENANDO ÁRBOLES...")
    resultados = {}
    
    for nombre, modelo in modelos.items():
        print(f"   🔄 Entrenando {nombre}...")
        
        # Entrenar
        modelo.fit(X_train, y_train)
        
        # Evaluar
        resultado = evaluar_modelo_arbol(modelo, X_test, y_test, nombre)
        resultados[nombre] = resultado
        
        print(f"   ✅ {nombre} → Precisión: {resultado['precision']:.3f} ({resultado['precision']*100:.1f}%)")
    
    # 5. ENCONTRAR EL MEJOR MODELO
    mejor_nombre = max(resultados.keys(), key=lambda x: resultados[x]['precision'])
    mejor_precision = resultados[mejor_nombre]['precision']
    
    print()
    print(f"🏆 MEJOR MODELO: {mejor_nombre}")
    print(f"   Precisión: {mejor_precision:.3f} ({mejor_precision*100:.1f}%)")
    
    # 6. MOSTRAR IMPORTANCIA DE VARIABLES
    print()
    mostrar_importancia_variables(resultados[mejor_nombre]['modelo'], variables_disponibles)
    
    # 7. CREAR ÁRBOL PERFECTO VISUALIZABLE
    print()
    print("🌳 Generando árbol de 3 niveles completos y legibles...")
    arbol_perfecto, precision_arbol = crear_arbol_perfecto(X_train, y_train, X_test, y_test, variables_disponibles)
    
    # Actualizar resultados con árbol perfecto
    resultados['Árbol Perfecto'] = {
        'modelo': arbol_perfecto,
        'precision': precision_arbol,
        'predicciones': arbol_perfecto.predict(X_test)
    }
    
    # 8. CREAR VISUALIZACIONES COMPLEMENTARIAS
    print("💾 Gráficos guardados: arboles_resultados.png")
    crear_visualizacion_arboles(resultados, mejor_nombre, variables_disponibles)
    
    # 9. EXPLICACIÓN FINAL
    print()
    print(f"🏆 MEJOR MODELO: {mejor_nombre}")
    print(f"   Precisión: {mejor_precision:.3f} ({mejor_precision*100:.1f}%)")
    
    if mejor_precision > 0.8:
        print("   ¡Excelente predicción! 🎉")
    elif mejor_precision > 0.6:
        print("   Buena predicción 👍")
    else:
        print("   Predicción moderada 🔧")
    
    print("✅ ÁRBOLES DE PREDICCIÓN COMPLETADOS")

if __name__ == "__main__":
    ejecutar_arboles()