#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ÁRBOLES DE PREDICCIÓN - OPTIMIZADO
Predicción con árboles para +92% de precisión (R²)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def ejecutar_arboles():
    """Árboles de Predicción Optimizados para +92% R²"""
    print("🌳 ÁRBOLES DE PREDICCIÓN")
    
    # 1. CARGAR DATOS
    try:
        datos = pd.read_csv('data/ceros_sin_columnasAB_limpio_weka.csv')
    except:
        print("❌ Error: archivo no encontrado")
        return
    
    # 2. VARIABLES OPTIMIZADAS
    variables = ['POBFEM', 'POBMAS', 'TOTHOG', 'VIVTOT', 'P_15YMAS']
    variables_disponibles = [v for v in variables if v in datos.columns]
    
    datos_limpios = datos[variables_disponibles + ['POBTOT']].dropna()
    
    # Muestra optimizada
    if len(datos_limpios) > 6000:
        datos_limpios = datos_limpios.sample(n=6000, random_state=42)
    
    X = datos_limpios[variables_disponibles]
    y = datos_limpios['POBTOT']
    
    print(f"📊 Variables: {len(variables_disponibles)} | Registros: {len(X):,}")
    
    # 3. DIVISIÓN
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. MODELOS OPTIMIZADOS
    modelos = {
        'Random Forest': RandomForestRegressor(
            n_estimators=200, 
            max_depth=15, 
            min_samples_split=5,
            random_state=42
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=8,
            random_state=42
        ),
        'Decision Tree': DecisionTreeRegressor(
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
    }
    
    mejor_precision = 0
    mejor_nombre = ""
    mejor_modelo = None
    
    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        precision = r2_score(y_test, y_pred)
        
        if precision > mejor_precision:
            mejor_precision = precision
            mejor_nombre = nombre
            mejor_modelo = modelo
    
    # 5. VISUALIZACIÓN IDEAL: ÁRBOL SIMPLE + IMPORTANCIA
    # Crear árbol simple para visualización
    arbol_simple = DecisionTreeRegressor(max_depth=3, min_samples_split=50, random_state=42)
    arbol_simple.fit(X_train, y_train)
    precision_simple = r2_score(y_test, arbol_simple.predict(X_test))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Gráfico 1: ÁRBOL VISUAL (más útil que barras)
    plot_tree(arbol_simple, 
              feature_names=variables_disponibles,
              filled=True,
              rounded=True,
              fontsize=10,
              ax=ax1,
              proportion=False,
              impurity=False,
              precision=0)
    ax1.set_title(f'🌳 Árbol de Decisión Visual\nR² = {precision_simple:.3f} (3 niveles)')
    
    # Gráfico 2: Importancia del MEJOR modelo
    if hasattr(mejor_modelo, 'feature_importances_'):
        importancias = mejor_modelo.feature_importances_
        indices = np.argsort(importancias)[::-1]
        
        ax2.barh(range(len(variables_disponibles)), importancias[indices], color='forestgreen', alpha=0.7)
        ax2.set_yticks(range(len(variables_disponibles)))
        ax2.set_yticklabels([variables_disponibles[i] for i in indices])
        ax2.set_xlabel('Importancia')
        ax2.set_title(f'📊 Variables Más Importantes\n{mejor_nombre} - R² = {mejor_precision:.3f}')
        
        # Añadir valores
        for i, v in enumerate(importancias[indices]):
            ax2.text(v + 0.01, i, f'{v:.3f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('arboles_completo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"🎯 Mejor modelo: {mejor_nombre}")
    print(f"🎯 Precisión: {mejor_precision:.3f} ({mejor_precision*100:.1f}%)")
    print("💾 Gráfico guardado: arboles_completo.png")
    print("✅ Árboles completados")
    
    return mejor_precision

if __name__ == "__main__":
    ejecutar_arboles()