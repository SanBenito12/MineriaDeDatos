#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REGRESIÓN LINEAL - Versión Ultra Simple
Predice la población total usando otras variables demográficas
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def ejecutar_regresion():
    print("🔵 REGRESIÓN LINEAL")
    print("="*30)
    print("📝 Objetivo: Predecir POBLACIÓN TOTAL usando otras variables")
    print()
    
    # 1. CARGAR DATOS
    archivo = 'ceros_sin_columnasAB_limpio_weka.csv'
    try:
        datos = pd.read_csv(archivo)
        print(f"✅ Datos cargados: {datos.shape[0]:,} filas")
    except:
        print(f"❌ No se encontró el archivo: {archivo}")
        return
    
    # 2. SELECCIONAR VARIABLES MÁS IMPORTANTES
    variables_predictoras = ['POBFEM', 'POBMAS', 'TOTHOG', 'VIVTOT']
    variables_disponibles = [v for v in variables_predictoras if v in datos.columns]
    
    if len(variables_disponibles) < 2:
        print("❌ No hay suficientes variables para el análisis")
        return
    
    print(f"📊 Variables usadas: {', '.join(variables_disponibles)}")
    
    # 3. PREPARAR DATOS (eliminar filas vacías)
    datos_limpios = datos[variables_disponibles + ['POBTOT']].dropna()
    X = datos_limpios[variables_disponibles]  # Variables predictoras
    y = datos_limpios['POBTOT']              # Variable a predecir
    
    print(f"🧹 Datos limpios: {len(datos_limpios):,} registros")
    
    # 4. DIVIDIR EN ENTRENAMIENTO Y PRUEBA
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 5. ESCALAR DATOS (importante para Ridge y Lasso)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"📈 Entrenamiento: {len(X_train):,} | Prueba: {len(X_test):,}")
    print()
    
    # 6. ENTRENAR 3 MODELOS DIFERENTES
    modelos = {
        'Lineal Simple': LinearRegression(),
        'Ridge (L2)': Ridge(alpha=1.0),
        'Lasso (L1)': Lasso(alpha=1.0, max_iter=1000)
    }
    
    print("🤖 ENTRENANDO MODELOS...")
    resultados = {}
    
    for nombre, modelo in modelos.items():
        # Entrenar modelo
        modelo.fit(X_train_scaled, y_train)
        
        # Hacer predicciones
        y_pred = modelo.predict(X_test_scaled)
        
        # Calcular precisión (R²)
        precision = r2_score(y_test, y_pred)
        
        resultados[nombre] = {
            'modelo': modelo,
            'precision': precision,
            'predicciones': y_pred
        }
        
        print(f"   {nombre:15} → Precisión: {precision:.3f} ({precision*100:.1f}%)")
    
    # 7. ENCONTRAR EL MEJOR MODELO
    mejor_nombre = max(resultados.keys(), key=lambda x: resultados[x]['precision'])
    mejor_precision = resultados[mejor_nombre]['precision']
    
    print()
    print(f"🏆 MEJOR MODELO: {mejor_nombre}")
    print(f"   Precisión: {mejor_precision:.3f} ({mejor_precision*100:.1f}%)")
    
    # 8. GRÁFICO SIMPLE Y CLARO
    try:
        plt.figure(figsize=(10, 5))
        
        # Gráfico 1: Comparación de modelos
        plt.subplot(1, 2, 1)
        nombres = list(resultados.keys())
        precisiones = [resultados[m]['precision'] for m in nombres]
        colores = ['lightblue', 'lightgreen', 'orange']
        
        barras = plt.bar(range(len(nombres)), precisiones, color=colores)
        plt.title('📊 Precisión por Modelo', fontsize=12, fontweight='bold')
        plt.ylabel('Precisión (R²)')
        plt.xticks(range(len(nombres)), nombres, rotation=45, ha='right')
        plt.ylim(0, 1)
        
        # Añadir valores en las barras
        for i, (barra, precision) in enumerate(zip(barras, precisiones)):
            plt.text(i, precision + 0.02, f'{precision:.3f}', 
                    ha='center', fontweight='bold')
        
        # Gráfico 2: Predicciones vs Realidad (mejor modelo)
        plt.subplot(1, 2, 2)
        mejor_pred = resultados[mejor_nombre]['predicciones']
        
        plt.scatter(y_test, mejor_pred, alpha=0.6, color='blue', s=20)
        
        # Línea de predicción perfecta
        min_val = min(y_test.min(), mejor_pred.min())
        max_val = max(y_test.max(), mejor_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
                label='Predicción Perfecta')
        
        plt.xlabel('Población Real')
        plt.ylabel('Población Predicha')
        plt.title(f'🎯 {mejor_nombre}\nPredicciones vs Realidad', fontsize=12, fontweight='bold')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('regresion_resultados.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("💾 Gráfico guardado: regresion_resultados.png")
        
    except Exception as e:
        print(f"⚠️ No se pudo crear el gráfico: {e}")
    
    # 9. EXPLICACIÓN SIMPLE
    print()
    print("📝 EXPLICACIÓN:")
    print(f"   • El modelo {mejor_nombre} es el más preciso")
    print(f"   • Puede explicar el {mejor_precision*100:.1f}% de la variación en población")
    if mejor_precision > 0.8:
        print("   • ¡Excelente precisión! 🎉")
    elif mejor_precision > 0.6:
        print("   • Buena precisión 👍")
    else:
        print("   • Precisión moderada, se puede mejorar 🔧")
    
    print("✅ REGRESIÓN LINEAL COMPLETADA")

if __name__ == "__main__":
    ejecutar_regresion()