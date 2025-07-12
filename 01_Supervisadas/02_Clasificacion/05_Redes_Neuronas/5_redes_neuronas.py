#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REDES NEURONALES - CLASIFICACIÓN (Versión Optimizada)
Redes neuronales artificiales para clasificar poblaciones
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def crear_categorias_poblacion_dinamica(datos):
    """Crear categorías balanceadas basadas en cuartiles"""
    q1 = datos['POBTOT'].quantile(0.25)
    q2 = datos['POBTOT'].quantile(0.50)
    q3 = datos['POBTOT'].quantile(0.75)
    
    def categorizar(poblacion):
        if poblacion <= q1:
            return 'Pequeña'
        elif poblacion <= q2:
            return 'Mediana'
        elif poblacion <= q3:
            return 'Grande'
        else:
            return 'Muy_Grande'
    
    return categorizar

def preparar_datos_redes(datos):
    """Prepara datos específicamente para redes neuronales"""
    variables_predictoras = [
        'POBFEM', 'POBMAS', 'TOTHOG', 'VIVTOT', 'P_15YMAS', 
        'P_60YMAS', 'GRAPROES', 'PEA', 'POCUPADA'
    ]
    
    variables_disponibles = [v for v in variables_predictoras if v in datos.columns]
    
    if len(variables_disponibles) < 5:
        return None, None, None
    
    # Crear categorías dinámicas
    categorizador = crear_categorias_poblacion_dinamica(datos)
    datos['CATEGORIA_POB'] = datos['POBTOT'].apply(categorizador)
    
    # Limpiar datos
    datos_limpios = datos[variables_disponibles + ['CATEGORIA_POB']].dropna()
    
    # Muestreo estratificado para balancear clases
    if len(datos_limpios) > 5000:
        try:
            datos_limpios = datos_limpios.groupby('CATEGORIA_POB').apply(
                lambda x: x.sample(min(len(x), 1250), random_state=42)
            ).reset_index(drop=True)
        except:
            datos_limpios = datos_limpios.sample(n=5000, random_state=42)
    
    # Convertir variables a tipo numérico
    for var in variables_disponibles:
        datos_limpios[var] = pd.to_numeric(datos_limpios[var], errors='coerce')
    
    # Eliminar filas con NaN después de conversión
    datos_limpios = datos_limpios.dropna()
    
    X = datos_limpios[variables_disponibles]
    y = datos_limpios['CATEGORIA_POB']
    
    return X, y, variables_disponibles

def crear_arquitecturas_redes():
    """Define diferentes arquitecturas de redes neuronales optimizadas"""
    return {
        'Red Simple': {
            'hidden_layer_sizes': (20,),
            'activation': 'relu',
            'solver': 'lbfgs',
            'alpha': 0.01,
            'max_iter': 500,
            'descripcion': 'Una capa oculta con 20 neuronas'
        },
        'Red Profunda': {
            'hidden_layer_sizes': (30, 15),
            'activation': 'relu',
            'solver': 'lbfgs',
            'alpha': 0.01,
            'max_iter': 600,
            'descripcion': 'Dos capas: 30 y 15 neuronas'
        },
        'Red Tanh': {
            'hidden_layer_sizes': (25,),
            'activation': 'tanh',
            'solver': 'lbfgs',
            'alpha': 0.1,
            'max_iter': 400,
            'descripcion': 'Activación tangente hiperbólica'
        }
    }

def calcular_parametros_red(modelo, n_entrada):
    """Calcula número aproximado de parámetros en la red"""
    try:
        capas = modelo.hidden_layer_sizes
        if isinstance(capas, int):
            capas = (capas,)
        
        n_salida = len(modelo.classes_)
        
        # Calcular parámetros
        n_parametros = n_entrada * capas[0]  # Primera capa
        for i in range(len(capas) - 1):
            n_parametros += capas[i] * capas[i + 1]  # Capas ocultas
        n_parametros += capas[-1] * n_salida  # Capa de salida
        n_parametros += sum(capas) + n_salida  # Bias
        
        return n_parametros
    except:
        return 0

def entrenar_redes_neuronales(X_train, X_test, y_train, y_test):
    """Entrena diferentes arquitecturas de redes neuronales"""
    
    # Escalar datos (CRÍTICO para redes neuronales)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    arquitecturas = crear_arquitecturas_redes()
    resultados = {}
    
    for nombre, config in arquitecturas.items():
        try:
            # Crear configuración sin descripción
            config_modelo = {k: v for k, v in config.items() if k != 'descripcion'}
            config_modelo['random_state'] = 42
            
            # Crear y entrenar modelo
            modelo = MLPClassifier(**config_modelo)
            modelo.fit(X_train_scaled, y_train)
            
            # Predicciones
            y_pred = modelo.predict(X_test_scaled)
            y_pred_proba = modelo.predict_proba(X_test_scaled)
            
            # Calcular número de parámetros
            n_parametros = calcular_parametros_red(modelo, X_train_scaled.shape[1])
            
            # Análisis de convergencia
            convergencia = {
                'convergio': modelo.n_iter_ < modelo.max_iter,
                'iteraciones': modelo.n_iter_,
                'loss_curve': getattr(modelo, 'loss_curve_', None)
            }
            
            # Métricas
            precision = accuracy_score(y_test, y_pred)
            
            resultados[nombre] = {
                'modelo': modelo,
                'precision': precision,
                'predicciones': y_pred,
                'probabilidades': y_pred_proba,
                'arquitectura': config['hidden_layer_sizes'],
                'activacion': config['activation'],
                'descripcion': config['descripcion'],
                'n_parametros': n_parametros,
                'convergencia': convergencia
            }
            
        except Exception as e:
            continue
    
    return resultados, scaler

def crear_visualizaciones_redes(resultados, y_test, variables_disponibles):
    """Crear visualizaciones esenciales para redes neuronales"""
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('🧠 REDES NEURONALES - ANÁLISIS', fontsize=14, fontweight='bold')
        
        # Gráfico 1: Comparación de precisión por arquitectura
        nombres = list(resultados.keys())
        precisiones = [resultados[m]['precision'] for m in nombres]
        
        axes[0].bar(nombres, precisiones, color=['lightblue', 'lightgreen', 'orange'])
        axes[0].set_title('🧠 Precisión por Arquitectura', fontweight='bold')
        axes[0].set_ylabel('Precisión')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].set_ylim(0, 1)
        
        # Añadir valores en las barras
        for i, precision in enumerate(precisiones):
            axes[0].text(i, precision + 0.02, f'{precision:.3f}', ha='center', fontweight='bold')
        
        # Gráfico 2: Matriz de confusión del mejor modelo
        mejor_nombre = max(resultados.keys(), key=lambda x: resultados[x]['precision'])
        y_pred_mejor = resultados[mejor_nombre]['predicciones']
        clases = resultados[mejor_nombre]['modelo'].classes_
        
        cm = confusion_matrix(y_test, y_pred_mejor)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=clases, yticklabels=clases, ax=axes[1])
        axes[1].set_title(f'🎯 Matriz de Confusión\n{mejor_nombre}', fontweight='bold')
        axes[1].set_xlabel('Predicción')
        axes[1].set_ylabel('Real')
        
        # Gráfico 3: Arquitectura del mejor modelo
        mejor_arq = resultados[mejor_nombre]['arquitectura']
        if isinstance(mejor_arq, int):
            mejor_arq = (mejor_arq,)
        
        capas = [len(variables_disponibles)] + list(mejor_arq) + [len(clases)]
        
        x_pos = np.arange(len(capas))
        axes[2].bar(x_pos, capas, color=['blue', 'green', 'orange', 'red'][:len(capas)])
        axes[2].set_title(f'🏗️ Arquitectura\n{mejor_nombre}', fontweight='bold')
        axes[2].set_xlabel('Capa')
        axes[2].set_ylabel('Neuronas')
        axes[2].set_xticks(x_pos)
        
        etiquetas = ['Entrada'] + [f'Oculta {i+1}' for i in range(len(mejor_arq))] + ['Salida']
        axes[2].set_xticklabels(etiquetas, rotation=45, ha='right')
        
        # Añadir valores en las barras
        for i, neurons in enumerate(capas):
            axes[2].text(i, neurons + max(capas)*0.02, str(neurons), ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('/home/sedc/Proyectos/MineriaDeDatos/results/graficos/redes_neuronas.png', 
                   dpi=150, bbox_inches='tight')
        plt.show()
        
        return True
        
    except Exception as e:
        return False

def guardar_resultados_redes(resultados, variables_disponibles, datos_info):
    """Guardar modelo y reporte de manera optimizada"""
    try:
        import joblib
        
        # Mejor modelo
        mejor_nombre = max(resultados.keys(), key=lambda x: resultados[x]['precision'])
        mejor_modelo = resultados[mejor_nombre]['modelo']
        mejor_precision = resultados[mejor_nombre]['precision']
        
        # Guardar modelo
        joblib.dump(mejor_modelo, '/home/sedc/Proyectos/MineriaDeDatos/results/modelos/mejor_red_neuronal.pkl')
        
        # Crear reporte conciso
        reporte = f"""
REPORTE REDES NEURONALES - CLASIFICACIÓN
=======================================

MEJOR RED: {mejor_nombre}
Descripción: {resultados[mejor_nombre]['descripcion']}
Precisión: {mejor_precision:.3f} ({mejor_precision*100:.1f}%)
Arquitectura: {resultados[mejor_nombre]['arquitectura']}
Activación: {resultados[mejor_nombre]['activacion']}
Parámetros totales: {resultados[mejor_nombre]['n_parametros']:,}

COMPARACIÓN DE ARQUITECTURAS:
"""
        for nombre, resultado in resultados.items():
            convergencia = resultado['convergencia']
            status = "✅ Convergió" if convergencia['convergio'] else "⚠️ No convergió"
            reporte += f"\n{nombre}:"
            reporte += f"\n  - Precisión: {resultado['precision']:.3f}"
            reporte += f"\n  - Arquitectura: {resultado['arquitectura']}"
            reporte += f"\n  - Activación: {resultado['activacion']}"
            reporte += f"\n  - Parámetros: {resultado['n_parametros']:,}"
            reporte += f"\n  - Convergencia: {status} ({convergencia['iteraciones']} iter)"
        
        reporte += f"""

DATOS PROCESADOS:
- Registros: {datos_info['n_registros']:,}
- Variables: {len(variables_disponibles)}
- Entrenamiento: {datos_info['n_train']:,}
- Prueba: {datos_info['n_test']:,}

VARIABLES UTILIZADAS:
{', '.join(variables_disponibles)}

CONFIGURACIÓN:
- Escalado aplicado: StandardScaler
- Solver: lbfgs (optimización limitada)
- Regularización: L2 (alpha)
- Inicialización: aleatoria con semilla fija

PRINCIPIO REDES NEURONALES:
- Neuronas artificiales conectadas en capas
- Cada neurona aplica función de activación
- Aprende patrones complejos no lineales
- Backpropagation para ajustar pesos

VENTAJAS:
- Puede aprender patrones muy complejos
- Versátil para diferentes tipos de problemas
- Buena capacidad de generalización

DESVENTAJAS:
- "Caja negra" - difícil de interpretar
- Requiere ajuste de hiperparámetros
- Sensible al overfitting
"""
        
        with open('/home/sedc/Proyectos/MineriaDeDatos/results/reportes/redes_neuronas_reporte.txt', 'w', encoding='utf-8') as f:
            f.write(reporte)
        
        return True
        
    except Exception as e:
        return False

def ejecutar_redes_neuronas():
    """FUNCIÓN PRINCIPAL - Mantiene compatibilidad con menú"""
    print("🧠 REDES NEURONALES - CLASIFICACIÓN")
    print("="*40)
    
    # 1. CARGAR DATOS
    archivo = '/home/sedc/Proyectos/MineriaDeDatos/data/ceros_sin_columnasAB_limpio_weka.csv'
    try:
        datos = pd.read_csv(archivo)
        print(f"✅ Datos cargados: {datos.shape[0]:,} registros")
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        return
    
    # 2. PREPARAR VARIABLES
    X, y, variables_disponibles = preparar_datos_redes(datos)
    if X is None:
        print("❌ No hay suficientes variables para redes neuronales")
        return
    
    print(f"📊 Variables: {len(variables_disponibles)} | Datos limpios: {len(X):,}")
    print(f"🔢 Neuronas entrada: {len(variables_disponibles)}")
    print(f"🎯 Clases salida: {len(y.unique())}")
    
    # Mostrar distribución de categorías
    distribucion = y.value_counts()
    print("📈 Categorías:")
    for categoria, count in distribucion.items():
        print(f"   {categoria}: {count:,} ({count/len(y)*100:.1f}%)")
    
    # 3. DIVIDIR DATOS
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        print(f"📊 División estratificada: {len(X_train):,} entrenamiento | {len(X_test):,} prueba")
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        print(f"📊 División simple: {len(X_train):,} entrenamiento | {len(X_test):,} prueba")
    
    print()
    
    # 4. ENTRENAR REDES NEURONALES
    print("🧠 Entrenando redes neuronales...")
    resultados, scaler = entrenar_redes_neuronales(X_train, X_test, y_train, y_test)
    
    if not resultados:
        print("❌ No se pudieron entrenar redes neuronales")
        return
    
    # Mostrar resultados
    for nombre, resultado in resultados.items():
        convergencia = resultado['convergencia']
        conv_status = "✅" if convergencia['convergio'] else "⚠️"
        print(f"   {nombre:15} → Precisión: {resultado['precision']:.3f} ({resultado['precision']*100:.1f}%) {conv_status}")
    
    # 5. ENCONTRAR MEJOR MODELO
    mejor_nombre = max(resultados.keys(), key=lambda x: resultados[x]['precision'])
    mejor_precision = resultados[mejor_nombre]['precision']
    
    print()
    print(f"🏆 MEJOR RED: {mejor_nombre}")
    print(f"   Descripción: {resultados[mejor_nombre]['descripcion']}")
    print(f"   Precisión: {mejor_precision:.3f} ({mejor_precision*100:.1f}%)")
    print(f"   Arquitectura: {resultados[mejor_nombre]['arquitectura']}")
    print(f"   Parámetros: {resultados[mejor_nombre]['n_parametros']:,}")
    
    # Información de convergencia
    convergencia_mejor = resultados[mejor_nombre]['convergencia']
    if convergencia_mejor['convergio']:
        print(f"   ✅ Convergió en {convergencia_mejor['iteraciones']} iteraciones")
    else:
        print(f"   ⚠️ No convergió completamente ({convergencia_mejor['iteraciones']} iter)")
    
    # 6. ANÁLISIS DETALLADO
    y_pred_mejor = resultados[mejor_nombre]['predicciones']
    reporte = classification_report(y_test, y_pred_mejor, output_dict=True)
    
    print("\n🎯 Métricas por categoría:")
    for categoria in y.unique():
        if categoria in reporte:
            precision = reporte[categoria]['precision']
            recall = reporte[categoria]['recall']
            f1 = reporte[categoria]['f1-score']
            print(f"   {categoria:12}: Prec={precision:.3f} | Rec={recall:.3f} | F1={f1:.3f}")
    
    # 7. VISUALIZACIONES
    crear_visualizaciones_redes(resultados, y_test, variables_disponibles)
    
    # 8. GUARDAR RESULTADOS
    datos_info = {
        'n_registros': len(X),
        'n_train': len(X_train),
        'n_test': len(X_test)
    }
    
    guardar_resultados_redes(resultados, variables_disponibles, datos_info)
    
    # 9. RESUMEN FINAL
    print()
    print("📝 RESUMEN:")
    print(f"   • Mejor arquitectura: {resultados[mejor_nombre]['arquitectura']}")
    print(f"   • Función activación: {resultados[mejor_nombre]['activacion']}")
    print(f"   • Precisión: {mejor_precision*100:.1f}%")
    print(f"   • Parámetros: {resultados[mejor_nombre]['n_parametros']:,}")
    
    if mejor_precision > 0.8:
        print("   🎉 ¡Excelente aprendizaje neuronal!")
    elif mejor_precision > 0.65:
        print("   👍 Buen aprendizaje de la red neuronal")
    else:
        print("   🔧 Aprendizaje moderado")
    
    print("✅ REDES NEURONALES COMPLETADAS")
    return resultados

if __name__ == "__main__":
    ejecutar_redes_neuronas()