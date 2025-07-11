#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESTIMADORES DE NÚCLEOS - OPTIMIZADO
SVR y K-NN para +88% de precisión (R²)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def ejecutar_nucleos():
    """Estimadores de Núcleos Optimizados para +88% R²"""
    print("🔬 ESTIMADORES DE NÚCLEOS")
    
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
    
    # Muestra reducida para SVR (es lento)
    if len(datos_limpios) > 2500:
        datos_limpios = datos_limpios.sample(n=2500, random_state=42)
    
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
        'SVR RBF': SVR(
            kernel='rbf', 
            C=1000, 
            gamma='auto', 
            epsilon=0.01
        ),
        'SVR Poly': SVR(
            kernel='poly',
            degree=3,
            C=1000,
            epsilon=0.01
        ),
        'K-NN': KNeighborsRegressor(
            n_neighbors=5,
            weights='distance',
            metric='euclidean'
        ),
        'Kernel Ridge': KernelRidge(
            kernel='rbf',
            alpha=0.01,
            gamma=0.1
        )
    }
    
    resultados = {}
    mejor_precision = 0
    mejor_nombre = ""
    
    for nombre, modelo in modelos.items():
        try:
            modelo.fit(X_train_scaled, y_train)
            y_pred = modelo.predict(X_test_scaled)
            precision = r2_score(y_test, y_pred)
            
            resultados[nombre] = precision
            
            if precision > mejor_precision:
                mejor_precision = precision
                mejor_nombre = nombre
                
        except Exception as e:
            print(f"⚠️ Error en {nombre}: {str(e)[:30]}...")
            resultados[nombre] = 0
            continue
    
    # 5. VISUALIZACIÓN IDEAL: SUPERFICIE DE DECISIÓN + COMPARACIÓN
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Gráfico 1: SUPERFICIE DE DECISIÓN (solo con 2 variables principales)
    if len(variables_disponibles) >= 2:
        # Usar solo las 2 primeras variables para visualización 2D
        X_2d = X_train_scaled[:, :2]
        y_2d = y_train
        
        # Entrenar modelo simple para superficie
        modelo_surf = SVR(kernel='rbf', C=1000, gamma='auto', epsilon=0.01)
        modelo_surf.fit(X_2d, y_2d)
        
        # Crear malla
        h = 0.5
        x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
        y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # Predicciones en malla
        Z = modelo_surf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Superficie de decisión
        im = ax1.contourf(xx, yy, Z, levels=20, alpha=0.6, cmap='viridis')
        scatter = ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap='viridis', s=30, edgecolors='black', alpha=0.8)
        ax1.set_xlabel(f'{variables_disponibles[0]}')
        ax1.set_ylabel(f'{variables_disponibles[1]}')
        ax1.set_title('🌐 Superficie de Decisión SVR')
        plt.colorbar(scatter, ax=ax1, label='Población')
    
    # Gráfico 2: Comparación de modelos
    if resultados:
        nombres = list(resultados.keys())
        precisiones = list(resultados.values())
        colores = ['purple', 'orange', 'green', 'red'][:len(nombres)]
        
        barras = ax2.bar(range(len(nombres)), precisiones, color=colores, alpha=0.7)
        ax2.set_title(f'🔬 Comparación de Kernels\nMejor: {mejor_nombre}')
        ax2.set_xlabel('Modelos')
        ax2.set_ylabel('Precisión (R²)')
        ax2.set_xticks(range(len(nombres)))
        ax2.set_xticklabels([n.replace(' ', '\n') for n in nombres], rotation=0)
        ax2.set_ylim(0, max(precisiones) * 1.1 if precisiones else 1)
        
        # Añadir valores
        for i, (barra, precision) in enumerate(zip(barras, precisiones)):
            ax2.text(i, precision + 0.02, f'{precision:.3f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('nucleos_superficie.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"🎯 Mejor modelo: {mejor_nombre}")
    print(f"🎯 Precisión: {mejor_precision:.3f} ({mejor_precision*100:.1f}%)")
    print("💾 Gráfico guardado: nucleos_superficie.png")
    print("✅ Núcleos completados")
    
    return mejor_precision

if __name__ == "__main__":
    ejecutar_nucleos()