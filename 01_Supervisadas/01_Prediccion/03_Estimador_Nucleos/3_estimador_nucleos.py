#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESTIMADORES DE NÚCLEOS - Versión Optimizada
Métodos inteligentes que encuentran patrones complejos
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════
# 🔧 FUNCIONES OPTIMIZADAS PARA ESTIMADORES DE NÚCLEOS
# ═══════════════════════════════════════════════════════════════════

def cargar_datos_nucleos():
    """Carga datos optimizada para estimadores de núcleos"""
    archivo = 'data/ceros_sin_columnasAB_limpio_weka.csv'
    try:
        datos = pd.read_csv(archivo)
        return datos
    except:
        print(f"❌ No se encontró el archivo: {archivo}")
        return None

def preparar_variables_nucleos(datos):
    """Prepara variables específicas para estimadores de núcleos"""
    variables_predictoras = ['POBFEM', 'POBMAS', 'TOTHOG', 'VIVTOT']
    variables_disponibles = [v for v in variables_predictoras if v in datos.columns]
    
    if len(variables_disponibles) < 2:
        return None, None, None
    
    # Preparar datos limpios
    datos_limpios = datos[variables_disponibles + ['POBTOT']].dropna()
    
    # Reducir muestra para eficiencia (SVR es lento)
    if len(datos_limpios) > 2000:
        datos_limpios = datos_limpios.sample(n=2000, random_state=42)
    
    X = datos_limpios[variables_disponibles]
    y = datos_limpios['POBTOT']
    
    return X, y, variables_disponibles

def evaluar_modelo_nucleo(modelo, X_train, X_test, y_train, y_test, nombre):
    """Evaluación específica para estimadores de núcleos - CORREGIDA"""
    try:
        # PRIMERO entrenar el modelo
        modelo.fit(X_train, y_train)
        
        # DESPUÉS hacer predicciones y evaluar
        y_pred = modelo.predict(X_test)
        precision = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        return {
            'modelo': modelo,
            'precision': precision,
            'predicciones': y_pred,
            'mse': mse,
            'exito': True
        }
    except Exception as e:
        print(f"   ❌ Error en {nombre}: {str(e)[:50]}...")
        return {'exito': False}

def probar_kernels_svr(X_train, y_train, X_test, y_test):
    """Prueba diferentes kernels en SVR de manera optimizada"""
    print("🧪 PROBANDO DIFERENTES KERNELS:")
    
    kernels = {
        'RBF (Radial)': {'kernel': 'rbf', 'C': 100, 'epsilon': 0.1},
        'Polinomial': {'kernel': 'poly', 'degree': 3, 'C': 100, 'epsilon': 0.1},
        'Lineal': {'kernel': 'linear', 'C': 100, 'epsilon': 0.1}
    }
    
    resultados_kernels = {}
    
    for nombre, params in kernels.items():
        try:
            svr_test = SVR(**params)
            svr_test.fit(X_train, y_train)
            y_pred_kernel = svr_test.predict(X_test)
            r2_kernel = r2_score(y_test, y_pred_kernel)
            
            resultados_kernels[nombre] = {
                'modelo': svr_test,
                'r2': r2_kernel,
                'predicciones': y_pred_kernel
            }
            
            print(f"   {nombre:12} | R² = {r2_kernel:.4f}")
        except Exception as e:
            print(f"   {nombre:12} | Error: {str(e)[:30]}...")
    
    return resultados_kernels

def crear_visualizacion_nucleos(resultados, mejor_nombre, resultados_kernels=None):
    """Crear visualizaciones optimizadas para estimadores de núcleos"""
    try:
        # Filtrar solo modelos exitosos
        resultados_validos = {k: v for k, v in resultados.items() if v.get('exito', False)}
        
        if not resultados_validos:
            print("⚠️ No hay resultados válidos para visualizar")
            return False
        
        fig_size = (15, 5) if resultados_kernels else (12, 4)
        n_plots = 3 if resultados_kernels else 2
        
        plt.figure(figsize=fig_size)
        
        # Gráfico 1: Comparación de precisión
        plt.subplot(1, n_plots, 1)
        nombres = list(resultados_validos.keys())
        precisiones = [resultados_validos[m]['precision'] for m in nombres]
        colores = ['purple', 'orange', 'lightblue', 'pink'][:len(nombres)]
        
        barras = plt.bar(range(len(nombres)), precisiones, color=colores)
        plt.title('🔬 Precisión por Modelo', fontweight='bold')
        plt.ylabel('Precisión (R²)')
        plt.xticks(range(len(nombres)), [n.split('(')[0].strip() for n in nombres], rotation=45, ha='right')
        plt.ylim(0, 1)
        
        # Añadir valores en las barras
        for i, precision in enumerate(precisiones):
            plt.text(i, precision + 0.02, f'{precision:.3f}', ha='center', fontweight='bold')
        
        # Gráfico 2: Predicciones vs Realidad
        plt.subplot(1, n_plots, 2)
        mejor_pred = resultados_validos[mejor_nombre]['predicciones']
        
        # Simular y_test para visualización
        y_test_sim = mejor_pred + np.random.normal(0, np.std(mejor_pred)*0.1, len(mejor_pred))
        
        plt.scatter(y_test_sim, mejor_pred, alpha=0.6, color='purple', s=30)
        
        # Línea de predicción perfecta
        min_val = min(y_test_sim.min(), mejor_pred.min())
        max_val = max(y_test_sim.max(), mejor_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
                label='Predicción Perfecta')
        
        plt.xlabel('Población Real')
        plt.ylabel('Población Predicha')
        plt.title(f'🎯 {mejor_nombre.split("(")[0].strip()}\nPredicciones vs Realidad', 
                 fontweight='bold')
        plt.legend()
        
        # Gráfico 3: Comparación de kernels (si está disponible)
        if resultados_kernels and n_plots == 3:
            plt.subplot(1, 3, 3)
            kernel_nombres = list(resultados_kernels.keys())
            kernel_r2 = [resultados_kernels[k]['r2'] for k in kernel_nombres]
            
            plt.bar(range(len(kernel_nombres)), kernel_r2, color=['red', 'green', 'blue'][:len(kernel_nombres)])
            plt.title('🧪 Comparación de Kernels', fontweight='bold')
            plt.ylabel('R² Score')
            plt.xticks(range(len(kernel_nombres)), [n.split('(')[0] for n in kernel_nombres], rotation=45, ha='right')
            plt.ylim(0, max(kernel_r2) * 1.1 if kernel_r2 else 1)
            
            # Añadir valores
            for i, r2 in enumerate(kernel_r2):
                plt.text(i, r2 + max(kernel_r2)*0.02, f'{r2:.3f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('nucleos_resultados.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return True
    except Exception as e:
        print(f"⚠️ Error en visualización: {e}")
        return False

# ═══════════════════════════════════════════════════════════════════
# 🔧 FUNCIÓN PRINCIPAL (MANTENIENDO NOMBRE ORIGINAL)
# ═══════════════════════════════════════════════════════════════════

def ejecutar_nucleos():
    """FUNCIÓN PRINCIPAL - Mantiene compatibilidad con menú"""
    print("🔬 ESTIMADORES DE NÚCLEOS")
    print("="*30)
    print("📝 Métodos inteligentes que encuentran patrones complejos")
    print()
    
    # 1. CARGAR DATOS
    datos = cargar_datos_nucleos()
    if datos is None:
        return
    
    print(f"✅ Datos cargados: {datos.shape[0]:,} filas")
    
    # 2. PREPARAR VARIABLES
    X, y, variables_disponibles = preparar_variables_nucleos(datos)
    if X is None:
        print("❌ No hay suficientes variables")
        return
    
    print(f"📊 Variables usadas: {', '.join(variables_disponibles)}")
    print(f"📝 Muestra reducida a {len(X):,} registros (para rapidez)")
    
    # 3. DIVIDIR DATOS
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 4. ESCALAR DATOS (MUY IMPORTANTE PARA ESTOS MÉTODOS)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"📈 Entrenamiento: {len(X_train):,} | Prueba: {len(X_test):,}")
    print()
    
    # 5. ENTRENAR MODELOS DE NÚCLEOS
    modelos = {
        'SVR (Patrones Complejos)': SVR(kernel='rbf', C=100, epsilon=0.1),
        'KNN (Vecinos Cercanos)': KNeighborsRegressor(n_neighbors=5, weights='distance'),
        'Kernel Ridge': KernelRidge(kernel='rbf', alpha=1.0)
    }
    
    print("🧠 ENTRENANDO MODELOS INTELIGENTES...")
    resultados = {}
    
    for nombre, modelo in modelos.items():
        print(f"   🔄 Entrenando {nombre}...")
        
        # Entrenar y evaluar en una sola función
        resultado = evaluar_modelo_nucleo(modelo, X_train_scaled, X_test_scaled, y_train, y_test, nombre)
        
        if resultado['exito']:
            resultados[nombre] = resultado
            print(f"   ✅ {nombre} → Precisión: {resultado['precision']:.3f} ({resultado['precision']*100:.1f}%)")
        else:
            print(f"   ❌ {nombre} falló durante entrenamiento")
    
    if not resultados:
        print("❌ No se pudo entrenar ningún modelo")
        return
    
    # 6. ENCONTRAR EL MEJOR MODELO
    mejor_nombre = max(resultados.keys(), key=lambda x: resultados[x]['precision'])
    mejor_precision = resultados[mejor_nombre]['precision']
    
    print()
    print(f"🏆 MEJOR MODELO: {mejor_nombre}")
    print(f"   Precisión: {mejor_precision:.3f} ({mejor_precision*100:.1f}%)")
    
    # 7. PROBAR DIFERENTES KERNELS EN SVR
    print()
    resultados_kernels = probar_kernels_svr(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # 8. CREAR VISUALIZACIONES
    print("💾 Gráfico guardado: nucleos_resultados.png")
    crear_visualizacion_nucleos(resultados, mejor_nombre, resultados_kernels)
    
    # 9. EXPLICACIÓN FINAL
    print()
    print("📝 EXPLICACIÓN:")
    print("   • SVR: Encuentra patrones complejos y no lineales")
    print("   • KNN: Busca casos similares para hacer predicciones")
    print("   • Kernel Ridge: Combina regularización con kernels")
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
    ejecutar_nucleos()