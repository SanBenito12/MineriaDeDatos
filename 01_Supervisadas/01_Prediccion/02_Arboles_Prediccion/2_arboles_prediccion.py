#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ÁRBOLES DE PREDICCIÓN - Versión Ultra Simple
Como un árbol de preguntas que predice la población
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def ejecutar_arboles():
    print("🌳 ÁRBOLES DE PREDICCIÓN")
    print("="*30)
    print("📝 Como un árbol de preguntas que aprende a predecir")
    print()
    
    # 1. CARGAR DATOS
    archivo = 'ceros_sin_columnasAB_limpio_weka.csv'
    try:
        datos = pd.read_csv(archivo)
        print(f"✅ Datos cargados: {datos.shape[0]:,} filas")
    except:
        print(f"❌ No se encontró el archivo: {archivo}")
        return
    
    # 2. SELECCIONAR VARIABLES IMPORTANTES
    variables_predictoras = ['POBFEM', 'POBMAS', 'TOTHOG', 'VIVTOT', 'P_15YMAS']
    variables_disponibles = [v for v in variables_predictoras if v in datos.columns]
    
    if len(variables_disponibles) < 2:
        print("❌ No hay suficientes variables")
        return
    
    print(f"📊 Variables usadas: {', '.join(variables_disponibles)}")
    
    # 3. PREPARAR DATOS
    datos_limpios = datos[variables_disponibles + ['POBTOT']].dropna()
    X = datos_limpios[variables_disponibles]
    y = datos_limpios['POBTOT']
    
    print(f"🧹 Datos limpios: {len(datos_limpios):,} registros")
    
    # 4. DIVIDIR DATOS
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"📈 Entrenamiento: {len(X_train):,} | Prueba: {len(X_test):,}")
    print()
    
    # 5. ENTRENAR 2 MODELOS DIFERENTES
    modelos = {
        'Árbol Perfecto (3 niveles)': DecisionTreeRegressor(max_depth=3, min_samples_split=200, min_samples_leaf=100, random_state=42),
        'Bosque (100 árboles)': RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    }
    
    print("🌱 ENTRENANDO ÁRBOLES...")
    resultados = {}
    
    for nombre, modelo in modelos.items():
        print(f"   🔄 Entrenando {nombre}...")
        
        # Entrenar
        modelo.fit(X_train, y_train)
        
        # Predecir
        y_pred = modelo.predict(X_test)
        
        # Calcular precisión
        precision = r2_score(y_test, y_pred)
        
        resultados[nombre] = {
            'modelo': modelo,
            'precision': precision,
            'predicciones': y_pred
        }
        
        print(f"   ✅ {nombre} → Precisión: {precision:.3f} ({precision*100:.1f}%)")
    
    # 6. ENCONTRAR EL MEJOR
    mejor_nombre = max(resultados.keys(), key=lambda x: resultados[x]['precision'])
    mejor_precision = resultados[mejor_nombre]['precision']
    
    print()
    print(f"🏆 MEJOR MODELO: {mejor_nombre}")
    print(f"   Precisión: {mejor_precision:.3f} ({mejor_precision*100:.1f}%)")
    
    # 7. MOSTRAR IMPORTANCIA DE VARIABLES (qué es más importante para predecir)
    print()
    print("📊 IMPORTANCIA DE VARIABLES:")
    mejor_modelo = resultados[mejor_nombre]['modelo']
    
    if hasattr(mejor_modelo, 'feature_importances_'):
        importancias = mejor_modelo.feature_importances_
        
        # Ordenar por importancia
        indices_ordenados = np.argsort(importancias)[::-1]
        
        for i, idx in enumerate(indices_ordenados):
            variable = variables_disponibles[idx]
            importancia = importancias[idx]
            barras = '█' * int(importancia * 20)
            print(f"   {i+1}. {variable:12} {barras} {importancia:.3f}")
    
    # 8. VISUALIZAR EL ÁRBOL DE DECISIÓN PERFECTO (3 NIVELES COMPLETOS)
    print()
    print("🌳 Generando árbol de 3 niveles completos y legibles...")
    
    try:
        from sklearn.tree import plot_tree
        
        # Crear un árbol de 3 niveles con más ramas (más útil)
        arbol_perfecto = DecisionTreeRegressor(
            max_depth=3,              # 3 niveles (se ve perfecto)
            min_samples_split=200,    # Menos restrictivo = más ramas
            min_samples_leaf=100,     # Hojas medianas
            random_state=42
        )
        arbol_perfecto.fit(X_train, y_train)
        
        # Obtener métricas del árbol
        profundidad_real = arbol_perfecto.get_depth()
        precision_arbol = r2_score(y_test, arbol_perfecto.predict(X_test))
        
        print(f"📏 Profundidad del árbol: {profundidad_real} niveles")
        print(f"🎯 Precisión del árbol: {precision_arbol:.3f} ({precision_arbol*100:.1f}%)")
        
        # Crear figura perfecta para 3 niveles con más espacio abajo
        plt.figure(figsize=(20, 14))
        
        # Gráfico del árbol perfecto
        plot_tree(arbol_perfecto, 
                 feature_names=variables_disponibles,
                 filled=True,
                 rounded=True,
                 fontsize=14,           # Texto grande y legible
                 proportion=False,
                 impurity=False,
                 precision=1,
                 max_depth=3)          # Forzar 3 niveles máximo
        
        plt.title(f'🌳 ÁRBOL DE DECISIÓN PERFECTO\n(3 niveles completos - Precisión: {precision_arbol*100:.1f}%)', 
                 fontsize=20, fontweight='bold', pad=30)
        
        # Ajustar espaciado con MÁS espacio abajo
        plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.15)
        
        plt.savefig('arbol_decision_perfecto.png', dpi=200, bbox_inches='tight', pad_inches=1.2)
        plt.show()
        
        print("✅ Árbol perfecto guardado: arbol_decision_perfecto.png")
        print(f"📊 Número de nodos: {arbol_perfecto.tree_.node_count}")
        print(f"📊 Decisiones finales: {arbol_perfecto.tree_.n_leaves}")
        
        # Actualizar el modelo en resultados
        resultados['Árbol Perfecto'] = {
            'modelo': arbol_perfecto,
            'precision': precision_arbol,
            'predicciones': arbol_perfecto.predict(X_test)
        }
        
    except Exception as e:
        print(f"⚠️ No se pudo crear el árbol: {e}")
    
    # 9. GRÁFICOS COMPLEMENTARIOS
    try:
        plt.figure(figsize=(12, 4))
        
        # Gráfico 1: Comparación de precisión
        plt.subplot(1, 3, 1)
        nombres = list(resultados.keys())
        precisiones = [resultados[m]['precision'] for m in nombres]
        
        plt.bar(range(len(nombres)), precisiones, color=['lightgreen', 'darkgreen'])
        plt.title('🌳 Precisión por Modelo', fontweight='bold')
        plt.ylabel('Precisión (R²)')
        plt.xticks(range(len(nombres)), nombres, rotation=45, ha='right')
        plt.ylim(0, 1)
        
        # Añadir valores
        for i, precision in enumerate(precisiones):
            plt.text(i, precision + 0.02, f'{precision:.3f}', ha='center', fontweight='bold')
        
        # Gráfico 2: Importancia de variables
        plt.subplot(1, 3, 2)
        if hasattr(mejor_modelo, 'feature_importances_'):
            plt.barh(variables_disponibles, mejor_modelo.feature_importances_, color='orange')
            plt.title('📊 Importancia Variables', fontweight='bold')
            plt.xlabel('Importancia')
        
        # Gráfico 3: Predicciones vs Realidad
        plt.subplot(1, 3, 3)
        mejor_pred = resultados[mejor_nombre]['predicciones']
        
        plt.scatter(y_test, mejor_pred, alpha=0.6, color='green', s=20)
        
        min_val = min(y_test.min(), mejor_pred.min())
        max_val = max(y_test.max(), mejor_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        plt.xlabel('Población Real')
        plt.ylabel('Población Predicha')
        plt.title(f'🎯 {mejor_nombre}\nPredicciones vs Realidad', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('arboles_resultados.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("💾 Gráficos guardados: arboles_resultados.png")
        
    except Exception as e:
        print(f"⚠️ No se pudo crear los gráficos: {e}")
    
    # 10. EXPLICACIÓN SIMPLE DEL ÁRBOL
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