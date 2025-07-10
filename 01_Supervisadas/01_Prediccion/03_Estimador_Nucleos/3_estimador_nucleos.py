#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESTIMADORES DE NÚCLEOS - Versión Ultra Simple
Métodos inteligentes que encuentran patrones complejos
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def ejecutar_nucleos():
    print("🔬 ESTIMADORES DE NÚCLEOS")
    print("="*30)
    print("📝 Métodos inteligentes que encuentran patrones complejos")
    print()
    
    # 1. CARGAR DATOS
    archivo = 'ceros_sin_columnasAB_limpio_weka.csv'
    try:
        datos = pd.read_csv(archivo)
        print(f"✅ Datos cargados: {datos.shape[0]:,} filas")
    except:
        print(f"❌ No se encontró el archivo: {archivo}")
        return
    
    # 2. SELECCIONAR VARIABLES
    variables_predictoras = ['POBFEM', 'POBMAS', 'TOTHOG', 'VIVTOT']
    variables_disponibles = [v for v in variables_predictoras if v in datos.columns]
    
    if len(variables_disponibles) < 2:
        print("❌ No hay suficientes variables")
        return
    
    print(f"📊 Variables usadas: {', '.join(variables_disponibles)}")
    
    # 3. PREPARAR DATOS (reducir muestra para rapidez)
    datos_limpios = datos[variables_disponibles + ['POBTOT']].dropna()
    
    # Tomar muestra más pequeña para SVR (es lento con muchos datos)
    if len(datos_limpios) > 2000:
        datos_limpios = datos_limpios.sample(n=2000, random_state=42)
        print(f"📝 Muestra reducida a {len(datos_limpios):,} registros (para rapidez)")
    
    X = datos_limpios[variables_disponibles]
    y = datos_limpios['POBTOT']
    
    # 4. DIVIDIR DATOS
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 5. ESCALAR DATOS (muy importante para estos métodos)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"📈 Entrenamiento: {len(X_train):,} | Prueba: {len(X_test):,}")
    print()
    
    # 6. ENTRENAR 2 MODELOS DIFERENTES
    modelos = {
        'SVR (Patrones Complejos)': SVR(kernel='rbf', C=100, epsilon=0.1),
        'KNN (Vecinos Cercanos)': KNeighborsRegressor(n_neighbors=5, weights='distance')
    }
    
    print("🧠 ENTRENANDO MODELOS INTELIGENTES...")
    resultados = {}
    
    for nombre, modelo in modelos.items():
        print(f"   🔄 Entrenando {nombre}...")
        
        try:
            # Entrenar
            modelo.fit(X_train_scaled, y_train)
            
            # Predecir
            y_pred = modelo.predict(X_test_scaled)
            
            # Calcular precisión
            precision = r2_score(y_test, y_pred)
            
            resultados[nombre] = {
                'modelo': modelo,
                'precision': precision,
                'predicciones': y_pred
            }
            
            print(f"   ✅ {nombre} → Precisión: {precision:.3f} ({precision*100:.1f}%)")
            
        except Exception as e:
            print(f"   ❌ Error en {nombre}: {str(e)[:50]}...")
    
    if not resultados:
        print("❌ No se pudo entrenar ningún modelo")
        return
    
    # 7. ENCONTRAR EL MEJOR
    mejor_nombre = max(resultados.keys(), key=lambda x: resultados[x]['precision'])
    mejor_precision = resultados[mejor_nombre]['precision']
    
    print()
    print(f"🏆 MEJOR MODELO: {mejor_nombre}")
    print(f"   Precisión: {mejor_precision:.3f} ({mejor_precision*100:.1f}%)")
    
    # 8. GRÁFICO SIMPLE Y CLARO
    try:
        plt.figure(figsize=(10, 4))
        
        # Gráfico 1: Comparación de modelos
        plt.subplot(1, 2, 1)
        nombres = list(resultados.keys())
        precisiones = [resultados[m]['precision'] for m in nombres]
        colores = ['purple', 'orange']
        
        barras = plt.bar(range(len(nombres)), precisiones, color=colores[:len(nombres)])
        plt.title('🔬 Precisión por Modelo', fontsize=12, fontweight='bold')
        plt.ylabel('Precisión (R²)')
        plt.xticks(range(len(nombres)), [n.split('(')[0].strip() for n in nombres], rotation=45, ha='right')
        plt.ylim(0, 1)
        
        # Añadir valores en las barras
        for i, (barra, precision) in enumerate(zip(barras, precisiones)):
            plt.text(i, precision + 0.02, f'{precision:.3f}', 
                    ha='center', fontweight='bold')
        
        # Gráfico 2: Predicciones vs Realidad (mejor modelo)
        plt.subplot(1, 2, 2)
        mejor_pred = resultados[mejor_nombre]['predicciones']
        
        plt.scatter(y_test, mejor_pred, alpha=0.6, color='purple', s=30)
        
        # Línea de predicción perfecta
        min_val = min(y_test.min(), mejor_pred.min())
        max_val = max(y_test.max(), mejor_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
                label='Predicción Perfecta')
        
        plt.xlabel('Población Real')
        plt.ylabel('Población Predicha')
        plt.title(f'🎯 {mejor_nombre.split("(")[0].strip()}\nPredicciones vs Realidad', 
                 fontsize=12, fontweight='bold')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('nucleos_resultados.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("💾 Gráfico guardado: nucleos_resultados.png")
        
    except Exception as e:
        print(f"⚠️ No se pudo crear el gráfico: {e}")
    
    # 9. EXPLICACIÓN SIMPLE
    print()
    print("📝 EXPLICACIÓN:")
    print("   • SVR: Encuentra patrones complejos y no lineales")
    print("   • KNN: Busca casos similares para hacer predicciones")
    print(f"   • El {mejor_nombre.split('(')[0].strip()} funciona mejor aquí")
    print(f"   • Puede explicar el {mejor_precision*100:.1f}% de la variación")
    
    if mejor_precision > 0.8:
        print("   • ¡Excelente detección de patrones! 🎉")
    elif mejor_precision > 0.6:
        print("   • Buena detección de patrones 👍")
    else:
        print("   • Patrones moderados detectados 🔧")
    
    print("✅ ESTIMADORES DE NÚCLEOS COMPLETADOS")

if __name__ == "__main__":
    ejecutar_nucleos()#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESTIMADORES DE NÚCLEOS - Versión Simplificada
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def ejecutar_nucleos():
    print("🔬 INICIANDO ESTIMADORES DE NÚCLEOS")
    print("="*40)
    
    # Cargar datos
    archivo = 'ceros_sin_columnasAB_limpio_weka.csv'
    try:
        datos = pd.read_csv(archivo)
        print(f"✅ Datos cargados: {datos.shape}")
    except:
        print(f"❌ Error: No se encontró {archivo}")
        return
    
    # Preparar datos
    variables = ['POBFEM', 'POBMAS', 'P_0A2', 'P_5YMAS', 'P_15YMAS', 'TOTHOG']
    variables = [v for v in variables if v in datos.columns]
    
    if not variables:
        print("❌ No se encontraron variables válidas")
        return
    
    # Limpiar y muestrear datos (importante para SVR)
    datos_limpios = datos[variables + ['POBTOT']].dropna()
    
    # Tomar muestra si es muy grande
    if len(datos_limpios) > 3000:
        datos_limpios = datos_limpios.sample(n=3000, random_state=42)
        print(f"📝 Muestra reducida a {len(datos_limpios)} registros")
    
    X = datos_limpios[variables]
    y = datos_limpios['POBTOT']
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Escalar (MUY importante para métodos kernel)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"📊 Entrenamiento: {len(X_train)} | Prueba: {len(X_test)}")
    
    # Modelos (parámetros simplificados)
    modelos = {
        'SVR': SVR(kernel='rbf', C=100, epsilon=0.1),
        'KernelRidge': KernelRidge(kernel='rbf', alpha=1.0),
        'KNN': KNeighborsRegressor(n_neighbors=5, weights='distance')
    }
    
    # Entrenar y evaluar
    resultados = {}
    for nombre, modelo in modelos.items():
        print(f"🔄 Entrenando {nombre}...")
        
        try:
            # Entrenar
            modelo.fit(X_train_scaled, y_train)
            
            # Predecir
            y_pred = modelo.predict(X_test_scaled)
            
            # Métricas
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            resultados[nombre] = {'R2': r2, 'MSE': mse, 'Predicciones': y_pred}
            
            print(f"🔍 {nombre:12} | R² = {r2:.4f} | MSE = {mse:.0f}")
            
        except Exception as e:
            print(f"❌ Error en {nombre}: {e}")
    
    if not resultados:
        print("❌ No se pudieron entrenar modelos")
        return
    
    # Mejor modelo
    mejor = max(resultados.items(), key=lambda x: x[1]['R2'])
    print(f"\n🏆 MEJOR: {mejor[0]} (R² = {mejor[1]['R2']:.4f})")
    
    # Prueba de diferentes kernels en SVR
    print(f"\n🧪 PROBANDO DIFERENTES KERNELS:")
    kernels = ['linear', 'rbf', 'poly']
    for kernel in kernels:
        try:
            svr_test = SVR(kernel=kernel, C=10)
            svr_test.fit(X_train_scaled, y_train)
            y_pred_kernel = svr_test.predict(X_test_scaled)
            r2_kernel = r2_score(y_test, y_pred_kernel)
            print(f"   {kernel:8} | R² = {r2_kernel:.4f}")
        except:
            print(f"   {kernel:8} | Error")
    
    # Gráficos
    try:
        plt.figure(figsize=(12, 4))
        
        # Gráfico 1: Comparación R²
        plt.subplot(1, 3, 1)
        nombres = list(resultados.keys())
        r2_vals = [resultados[m]['R2'] for m in nombres]
        colores = ['skyblue', 'lightgreen', 'orange']
        plt.bar(nombres, r2_vals, color=colores[:len(nombres)])
        plt.title('R² por Modelo')
        plt.ylabel('R² Score')
        plt.xticks(rotation=45)
        
        # Gráfico 2: Distribución de errores
        plt.subplot(1, 3, 2)
        mejor_pred = resultados[mejor[0]]['Predicciones']
        errores = y_test - mejor_pred
        plt.hist(errores, bins=20, alpha=0.7, edgecolor='black')
        plt.title(f'Distribución Errores - {mejor[0]}')
        plt.xlabel('Error')
        plt.ylabel('Frecuencia')
        
        # Gráfico 3: Predicciones vs Reales
        plt.subplot(1, 3, 3)
        plt.scatter(y_test, mejor_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Valores Reales')
        plt.ylabel('Predicciones')
        plt.title(f'{mejor[0]} - Pred vs Real')
        
        plt.tight_layout()
        plt.savefig('nucleos_resultados.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("📊 Gráfico guardado: nucleos_resultados.png")
    except Exception as e:
        print(f"⚠️ No se pudo generar el gráfico: {e}")
    
    print("✅ ESTIMADORES DE NÚCLEOS COMPLETADOS")

if __name__ == "__main__":
    ejecutar_nucleos()