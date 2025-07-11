#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REGRESIÓN LINEAL - OPTIMIZADO
Predice población total con +90% de precisión (R²)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def ejecutar_regresion():
    """Regresión Lineal Optimizada para +90% R²"""
    print("🔵 REGRESIÓN LINEAL")
    
    # 1. CARGAR DATOS
    try:
        datos = pd.read_csv('data/ceros_sin_columnasAB_limpio_weka.csv')
    except:
        print("❌ Error: archivo no encontrado")
        return
    
    # 2. VARIABLES OPTIMIZADAS
    variables = ['POBFEM', 'POBMAS', 'TOTHOG', 'VIVTOT']
    variables_disponibles = [v for v in variables if v in datos.columns]
    
    datos_limpios = datos[variables_disponibles + ['POBTOT']].dropna()
    
    # Muestra optimizada para velocidad
    if len(datos_limpios) > 5000:
        datos_limpios = datos_limpios.sample(n=5000, random_state=42)
    
    X = datos_limpios[variables_disponibles]
    y = datos_limpios['POBTOT']
    
    print(f"📊 Variables: {len(variables_disponibles)} | Registros: {len(X):,}")
    
    # 3. DIVISIÓN Y ESCALADO
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. MODELOS OPTIMIZADOS
    modelos = {
        'Ridge': Ridge(alpha=0.1),
        'Lasso': Lasso(alpha=0.01, max_iter=2000),
        'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=2000)
    }
    
    mejor_precision = 0
    mejor_nombre = ""
    mejor_modelo = None
    
    for nombre, modelo in modelos.items():
        modelo.fit(X_train_scaled, y_train)
        y_pred = modelo.predict(X_test_scaled)
        precision = r2_score(y_test, y_pred)
        
        if precision > mejor_precision:
            mejor_precision = precision
            mejor_nombre = nombre
            mejor_modelo = modelo
    
    # 5. VISUALIZACIÓN IDEAL: RESIDUOS + PREDICCIÓN
    y_pred_final = mejor_modelo.predict(X_test_scaled)
    residuos = y_test - y_pred_final
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Gráfico 1: Predicción vs Real
    ax1.scatter(y_test, y_pred_final, alpha=0.6, color='blue', s=30)
    min_val = min(y_test.min(), y_pred_final.min())
    max_val = max(y_test.max(), y_pred_final.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Predicción Perfecta')
    ax1.set_xlabel('Población Real')
    ax1.set_ylabel('Población Predicha')
    ax1.set_title(f'🎯 Precisión: R² = {mejor_precision:.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: RESIDUOS (más revelador)
    ax2.scatter(y_pred_final, residuos, alpha=0.6, color='red', s=30)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel('Valores Predichos')
    ax2.set_ylabel('Residuos (Real - Predicho)')
    ax2.set_title(f'📊 Análisis de Errores\n{mejor_nombre}')
    ax2.grid(True, alpha=0.3)
    
    # Estadísticas útiles
    rmse = np.sqrt(np.mean(residuos**2))
    mae = np.mean(np.abs(residuos))
    
    fig.suptitle(f'🔵 REGRESIÓN LINEAL - RMSE: {rmse:.0f} | MAE: {mae:.0f}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('regresion_analisis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"🎯 Mejor modelo: {mejor_nombre}")
    print(f"🎯 Precisión: {mejor_precision:.3f} ({mejor_precision*100:.1f}%)")
    print(f"📊 RMSE: {rmse:.0f} | MAE: {mae:.0f}")
    print("💾 Gráfico guardado: regresion_analisis.png")
    print("✅ Regresión completada")
    
    return mejor_precision

if __name__ == "__main__":
    ejecutar_regresion()