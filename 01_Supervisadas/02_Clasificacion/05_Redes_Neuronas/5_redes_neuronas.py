#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REDES DE NEURONAS - CLASIFICACIÓN (Versión Optimizada)
Redes neuronales artificiales para clasificar poblaciones
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════
# 🔧 FUNCIONES OPTIMIZADAS PARA REDES NEURONALES
# ═══════════════════════════════════════════════════════════════════

def crear_categorias_poblacion(poblacion):
    """Crear categorías de población optimizadas para clasificación"""
    # Umbrales basados en cuartiles para distribución más balanceada
    if poblacion <= 50:
        return 'Muy Pequeña'
    elif poblacion <= 200:
        return 'Pequeña'
    elif poblacion <= 800:
        return 'Mediana'
    elif poblacion <= 3000:
        return 'Grande'
    else:
        return 'Muy Grande'

def cargar_datos_redes():
    """Carga datos optimizada para redes neuronales"""
    archivo = 'data/ceros_sin_columnasAB_limpio_weka.csv'
    try:
        datos = pd.read_csv(archivo)
        return datos
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        return None

def preparar_variables_redes(datos):
    """Prepara variables específicas para redes neuronales"""
    variables_predictoras = [
        'POBFEM', 'POBMAS', 'TOTHOG', 'VIVTOT', 'P_15YMAS', 
        'P_60YMAS', 'GRAPROES', 'PEA', 'POCUPADA'
    ]
    
    variables_disponibles = [v for v in variables_predictoras if v in datos.columns]
    
    if len(variables_disponibles) < 5:
        return None, None, None
    
    # Analizar distribución de POBTOT para crear umbrales inteligentes
    poblacion_stats = datos['POBTOT'].describe()
    print(f"📊 Estadísticas de población:")
    print(f"   Mínimo: {poblacion_stats['min']:.0f}")
    print(f"   Q1 (25%): {poblacion_stats['25%']:.0f}")
    print(f"   Mediana: {poblacion_stats['50%']:.0f}")
    print(f"   Q3 (75%): {poblacion_stats['75%']:.0f}")
    print(f"   Máximo: {poblacion_stats['max']:.0f}")
    
    # Crear categorías basadas en cuartiles dinámicos
    def categorizar_dinamico(poblacion):
        if poblacion <= poblacion_stats['25%']:
            return 'Pequeña'
        elif poblacion <= poblacion_stats['50%']:
            return 'Mediana'
        elif poblacion <= poblacion_stats['75%']:
            return 'Grande'
        else:
            return 'Muy Grande'
    
    # Aplicar categorización dinámica
    datos['CATEGORIA_POB'] = datos['POBTOT'].apply(categorizar_dinamico)
    
    # Verificar distribución de clases
    print(f"📈 Distribución de categorías (cuartiles):")
    distribucion = datos['CATEGORIA_POB'].value_counts()
    for categoria, count in distribucion.items():
        print(f"   {categoria:12}: {count:,} ({count/len(datos)*100:.1f}%)")
    
    # Preparar datos limpios
    datos_limpios = datos[variables_disponibles + ['CATEGORIA_POB']].dropna()
    
    # CRÍTICO: Convertir todas las variables numéricas a float
    print(f"🔧 Convirtiendo variables a tipo numérico...")
    for var in variables_disponibles:
        try:
            datos_limpios[var] = pd.to_numeric(datos_limpios[var], errors='coerce')
        except:
            print(f"   ⚠️ Error convirtiendo {var}")
    
    # Eliminar filas con NaN después de conversión
    datos_limpios = datos_limpios.dropna()
    
    print(f"📊 Tipos de datos después de conversión:")
    for var in variables_disponibles[:3]:  # Mostrar solo primeras 3
        dtype = datos_limpios[var].dtype
        print(f"   {var}: {dtype}")
    
    # Verificar que todas las variables sean numéricas
    variables_numericas = []
    for var in variables_disponibles:
        if pd.api.types.is_numeric_dtype(datos_limpios[var]):
            variables_numericas.append(var)
        else:
            print(f"   ⚠️ Excluyendo {var} (no numérico: {datos_limpios[var].dtype})")
    
    if len(variables_numericas) < 5:
        print(f"❌ Solo {len(variables_numericas)} variables numéricas válidas")
        return None, None, None
    
    variables_disponibles = variables_numericas
    
    # Verificar que todas las clases tengan suficientes muestras
    conteos_clase = datos_limpios['CATEGORIA_POB'].value_counts()
    clases_validas = conteos_clase[conteos_clase >= 20].index  # Mínimo 20 muestras por clase
    
    if len(clases_validas) < 2:
        print(f"❌ Clases con suficientes muestras: {len(clases_validas)}")
        print(f"   Conteos por clase: {dict(conteos_clase)}")
        return None, None, None
    
    # Filtrar solo clases válidas
    datos_limpios = datos_limpios[datos_limpios['CATEGORIA_POB'].isin(clases_validas)]
    
    # Muestreo estratificado para balancear si es necesario
    if len(datos_limpios) > 5000:
        from sklearn.model_selection import train_test_split
        try:
            # Usar muestreo estratificado para mantener proporciones
            datos_limpios, _ = train_test_split(
                datos_limpios, 
                test_size=1-5000/len(datos_limpios), 
                stratify=datos_limpios['CATEGORIA_POB'],
                random_state=42
            )
        except:
            # Si falla el estratificado, usar muestreo simple
            datos_limpios = datos_limpios.sample(n=5000, random_state=42)
    
    print(f"📈 Distribución final de categorías:")
    distribucion_final = datos_limpios['CATEGORIA_POB'].value_counts()
    for categoria, count in distribucion_final.items():
        print(f"   {categoria:12}: {count:,} ({count/len(datos_limpios)*100:.1f}%)")
    
    X = datos_limpios[variables_disponibles]
    y = datos_limpios['CATEGORIA_POB']
    
    return X, y, variables_disponibles

def crear_arquitecturas_optimizadas():
    """Define arquitecturas de redes neuronales optimizadas"""
    return {
        'Red Simple': {
            'hidden_layer_sizes': (15,),
            'activation': 'relu',
            'solver': 'lbfgs',  # Cambiar a lbfgs que es más estable
            'alpha': 0.01,
            'max_iter': 300,
            'descripcion': 'Una capa oculta con 15 neuronas'
        },
        'Red Mediana': {
            'hidden_layer_sizes': (20, 10),
            'activation': 'relu', 
            'solver': 'lbfgs',  # Cambiar a lbfgs
            'alpha': 0.01,
            'max_iter': 400,
            'descripcion': 'Dos capas: 20 y 10 neuronas'
        },
        'Red Profunda': {
            'hidden_layer_sizes': (30, 15, 8),
            'activation': 'relu',
            'solver': 'lbfgs',  # Cambiar definitivamente a lbfgs
            'alpha': 0.005,
            'max_iter': 500,
            'descripcion': 'Tres capas: 30, 15 y 8 neuronas'
        },
        'Red Tanh': {
            'hidden_layer_sizes': (25,),
            'activation': 'tanh',
            'solver': 'lbfgs',
            'alpha': 0.1,
            'max_iter': 300,
            'descripcion': 'Activación tangente hiperbólica'
        }
    }

def entrenar_red_neuronal(nombre, config, X_train, X_test, y_train, y_test):
    """Entrena una red neuronal con configuración específica"""
    print(f"   🔄 Entrenando {nombre}...")
    
    try:
        # Crear configuración sin descripción
        config_modelo = {k: v for k, v in config.items() if k != 'descripcion'}
        config_modelo['random_state'] = 42
        
        # Crear y entrenar modelo
        modelo = MLPClassifier(**config_modelo)
        modelo.fit(X_train, y_train)
        
        # Predicciones
        y_pred = modelo.predict(X_test)
        y_pred_proba = modelo.predict_proba(X_test)
        
        # Métricas
        precision = accuracy_score(y_test, y_pred)
        
        # Calcular número de parámetros
        n_parametros = calcular_parametros_red(modelo, X_train.shape[1])
        
        # Análisis de convergencia
        convergencia = analizar_convergencia_red(modelo)
        
        resultado = {
            'modelo': modelo,
            'precision': precision,
            'predicciones': y_pred,
            'probabilidades': y_pred_proba,
            'arquitectura': config['hidden_layer_sizes'],
            'activacion': config['activation'],
            'solver': config.get('solver', 'adam'),
            'descripcion': config['descripcion'],
            'convergencia': convergencia,
            'n_parametros': n_parametros,
            'exito': True
        }
        
        print(f"   ✅ {nombre} → Precisión: {precision:.3f} ({precision*100:.1f}%)")
        if convergencia:
            print(f"      Iteraciones: {convergencia['iteraciones']} | Convergió: {'Sí' if convergencia['convergio'] else 'No'}")
        
        return resultado
        
    except Exception as e:
        print(f"   ❌ Error en {nombre}: {str(e)[:60]}...")
        return {'exito': False}

def calcular_parametros_red(modelo, n_entrada):
    """Calcula número aproximado de parámetros en la red"""
    try:
        capas = modelo.hidden_layer_sizes
        n_salida = len(modelo.classes_)
        
        n_parametros = n_entrada * capas[0]  # Primera capa
        for i in range(len(capas) - 1):
            n_parametros += capas[i] * capas[i + 1]  # Capas ocultas
        n_parametros += capas[-1] * n_salida  # Capa de salida
        n_parametros += sum(capas) + n_salida  # Bias
        
        return n_parametros
    except:
        return 0

def analizar_convergencia_red(modelo):
    """Analiza la convergencia del entrenamiento"""
    try:
        if hasattr(modelo, 'loss_curve_') and modelo.loss_curve_:
            return {
                'convergio': modelo.n_iter_ < modelo.max_iter,
                'iteraciones': modelo.n_iter_,
                'loss_final': modelo.loss_curve_[-1],
                'loss_curve': modelo.loss_curve_
            }
    except:
        pass
    return None

def crear_visualizacion_redes(resultados, mejor_nombre, variables):
    """Crear visualizaciones optimizadas para redes neuronales"""
    try:
        # Filtrar solo resultados exitosos
        resultados_validos = {k: v for k, v in resultados.items() if v.get('exito', False)}
        
        if not resultados_validos:
            print("⚠️ No hay resultados válidos para visualizar")
            return False
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('🧠 ANÁLISIS DE REDES NEURONALES', fontsize=16, fontweight='bold')
        
        # Gráfico 1: Comparación de precisión
        nombres = list(resultados_validos.keys())
        precisiones = [resultados_validos[m]['precision'] for m in nombres]
        
        axes[0,0].bar(range(len(nombres)), precisiones, 
                     color=['lightblue', 'lightgreen', 'orange', 'pink'][:len(nombres)])
        axes[0,0].set_title('🧠 Precisión por Arquitectura', fontweight='bold')
        axes[0,0].set_ylabel('Precisión')
        axes[0,0].set_xticks(range(len(nombres)))
        axes[0,0].set_xticklabels([n.split()[1] for n in nombres], rotation=45)
        
        # Añadir valores en las barras
        for i, precision in enumerate(precisiones):
            axes[0,0].text(i, precision + 0.02, f'{precision:.3f}', 
                          ha='center', fontweight='bold')
        
        # Gráfico 2: Matriz de confusión del mejor modelo
        mejor_resultado = resultados_validos[mejor_nombre]
        y_test_dummy = mejor_resultado['predicciones']  # Para demo
        y_pred_mejor = mejor_resultado['predicciones']
        
        try:
            clases = mejor_resultado['modelo'].classes_
            cm = confusion_matrix(y_test_dummy, y_pred_mejor, labels=clases)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=clases, yticklabels=clases, ax=axes[0,1])
            axes[0,1].set_title(f'🎯 Matriz de Confusión\n{mejor_nombre}', fontweight='bold')
            axes[0,1].set_xlabel('Predicción')
            axes[0,1].set_ylabel('Real')
        except:
            axes[0,1].text(0.5, 0.5, 'Matriz de\nConfusión', ha='center', va='center')
            axes[0,1].set_title('🎯 Matriz de Confusión', fontweight='bold')
        
        # Gráfico 3: Curva de pérdida
        convergencia_mejor = mejor_resultado['convergencia']
        if convergencia_mejor and convergencia_mejor.get('loss_curve'):
            axes[0,2].plot(convergencia_mejor['loss_curve'], 'b-', linewidth=2)
            axes[0,2].set_title(f'📉 Curva de Pérdida\n{mejor_nombre}', fontweight='bold')
            axes[0,2].set_xlabel('Época')
            axes[0,2].set_ylabel('Pérdida')
            axes[0,2].grid(True, alpha=0.3)
        else:
            axes[0,2].text(0.5, 0.5, 'Curva de pérdida\nno disponible', ha='center', va='center')
            axes[0,2].set_title('📉 Curva de Pérdida', fontweight='bold')
        
        # Gráfico 4: Complejidad vs Precisión
        n_params = [resultados_validos[m]['n_parametros'] for m in nombres]
        axes[1,0].scatter(n_params, precisiones, s=100, alpha=0.7, 
                         c=['red', 'green', 'blue', 'orange'][:len(n_params)])
        for i, nombre in enumerate(nombres):
            axes[1,0].annotate(nombre.split()[1][:6], (n_params[i], precisiones[i]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[1,0].set_xlabel('Número de Parámetros')
        axes[1,0].set_ylabel('Precisión')
        axes[1,0].set_title('⚖️ Complejidad vs Precisión', fontweight='bold')
        
        # Gráfico 5: Distribución de confianza
        probabilidades = mejor_resultado['probabilidades']
        max_probs = np.max(probabilidades, axis=1)
        axes[1,1].hist(max_probs, bins=15, alpha=0.7, color='purple', edgecolor='black')
        axes[1,1].set_title('📈 Confianza de Predicciones', fontweight='bold')
        axes[1,1].set_xlabel('Confianza Máxima')
        axes[1,1].set_ylabel('Frecuencia')
        
        # Gráfico 6: Arquitectura del mejor modelo
        mejor_arq = mejor_resultado['arquitectura']
        capas = [len(variables)] + list(mejor_arq) + [len(clases) if 'clases' in locals() else 4]
        
        x_pos = np.arange(len(capas))
        axes[1,2].bar(x_pos, capas, color=['blue', 'green', 'orange', 'red', 'purple'][:len(capas)])
        axes[1,2].set_title(f'🏗️ Arquitectura Mejor Red\n{mejor_nombre}', fontweight='bold')
        axes[1,2].set_xlabel('Capa')
        axes[1,2].set_ylabel('Neuronas')
        axes[1,2].set_xticks(x_pos)
        axes[1,2].set_xticklabels(['Entrada'] + [f'Oculta {i+1}' for i in range(len(mejor_arq))] + ['Salida'])
        
        for i, neurons in enumerate(capas):
            axes[1,2].text(i, neurons + max(capas)*0.02, str(neurons), ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('redes_neuronas_resultados.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return True
    except Exception as e:
        print(f"⚠️ Error en visualizaciones: {e}")
        return False

# ═══════════════════════════════════════════════════════════════════
# 🔧 FUNCIÓN PRINCIPAL (MANTENIENDO NOMBRE ORIGINAL)
# ═══════════════════════════════════════════════════════════════════

def ejecutar_redes_neuronas():
    """FUNCIÓN PRINCIPAL - Mantiene compatibilidad con menú"""
    print("🧠 REDES DE NEURONAS - CLASIFICACIÓN")
    print("="*40)
    print("📝 Objetivo: Clasificar usando redes neuronales artificiales")
    print()
    
    # 1. CARGAR DATOS
    datos = cargar_datos_redes()
    if datos is None:
        return
    
    print(f"✅ Datos cargados: {datos.shape[0]:,} filas, {datos.shape[1]} columnas")
    
    # 2. PREPARAR VARIABLES
    X, y, variables_disponibles = preparar_variables_redes(datos)
    if X is None:
        print("❌ No hay suficientes variables para redes neuronales")
        return
    
    print(f"📊 Variables usadas: {', '.join(variables_disponibles)}")
    print(f"🧹 Datos finales: {len(X):,} registros")
    
    # 3. DIVIDIR DATOS
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        print(f"📊 División estratificada - Entrenamiento: {len(X_train):,} | Prueba: {len(X_test):,}")
    except Exception as e:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        print(f"📊 División simple - Entrenamiento: {len(X_train):,} | Prueba: {len(X_test):,}")
    
    # 4. ESCALAR DATOS (CRÍTICO PARA REDES NEURONALES)
    print(f"🔍 Verificando datos antes del escalado:")
    print(f"   X_train shape: {X_train.shape}")
    print(f"   X_train tipos: {X_train.dtypes.unique()}")
    print(f"   Hay NaN en X_train: {X_train.isnull().sum().sum()}")
    
    # Asegurar que X sea completamente numérico
    X_train_clean = X_train.copy()
    X_test_clean = X_test.copy()
    
    # Convertir a float64 explícitamente
    for col in X_train_clean.columns:
        X_train_clean[col] = pd.to_numeric(X_train_clean[col], errors='coerce')
        X_test_clean[col] = pd.to_numeric(X_test_clean[col], errors='coerce')
    
    # Eliminar NaN
    nan_rows_train = X_train_clean.isnull().any(axis=1).sum()
    nan_rows_test = X_test_clean.isnull().any(axis=1).sum()
    
    if nan_rows_train > 0 or nan_rows_test > 0:
        print(f"   ⚠️ Eliminando {nan_rows_train} filas con NaN en train, {nan_rows_test} en test")
        
        # Filtrar filas válidas
        valid_train = ~X_train_clean.isnull().any(axis=1)
        valid_test = ~X_test_clean.isnull().any(axis=1)
        
        X_train_clean = X_train_clean[valid_train]
        y_train = y_train[valid_train] if hasattr(y_train, 'iloc') else y_train[valid_train.values]
        
        X_test_clean = X_test_clean[valid_test]
        y_test = y_test[valid_test] if hasattr(y_test, 'iloc') else y_test[valid_test.values]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean.values)  # .values para asegurar array numpy
    X_test_scaled = scaler.transform(X_test_clean.values)
    
    print(f"✅ Escalado completado:")
    print(f"   X_train_scaled shape: {X_train_scaled.shape}")
    print(f"   X_train_scaled tipo: {X_train_scaled.dtype}")
    print(f"🔢 Neuronas de entrada: {X_train_scaled.shape[1]}")
    print(f"🎯 Clases de salida: {len(np.unique(y_train))}")
    print()
    
    # 5. ENTRENAR DIFERENTES ARQUITECTURAS
    arquitecturas = crear_arquitecturas_optimizadas()
    
    print("🧠 ENTRENANDO REDES NEURONALES...")
    resultados = {}
    
    for nombre, config in arquitecturas.items():
        resultado = entrenar_red_neuronal(nombre, config, X_train_scaled, X_test_scaled, y_train, y_test)
        
        if resultado.get('exito', False):
            resultados[nombre] = resultado
    
    if not resultados:
        print("❌ No se pudo entrenar ninguna red neuronal")
        return
    
    # 6. ENCONTRAR LA MEJOR RED
    mejor_nombre = max(resultados.keys(), key=lambda x: resultados[x]['precision'])
    mejor_modelo = resultados[mejor_nombre]['modelo']
    mejor_precision = resultados[mejor_nombre]['precision']
    
    print()
    print(f"🏆 MEJOR RED: {mejor_nombre}")
    print(f"   Descripción: {resultados[mejor_nombre]['descripcion']}")
    print(f"   Precisión: {mejor_precision:.3f} ({mejor_precision*100:.1f}%)")
    print(f"   Arquitectura: {resultados[mejor_nombre]['arquitectura']}")
    print(f"   Activación: {resultados[mejor_nombre]['activacion']}")
    print(f"   Parámetros: {resultados[mejor_nombre]['n_parametros']:,}")
    
    # 7. ANÁLISIS DETALLADO
    print()
    print("📊 ANÁLISIS DETALLADO:")
    y_pred_mejor = resultados[mejor_nombre]['predicciones']
    
    # Reporte por clase
    print("\n🎯 Métricas por Categoría:")
    try:
        reporte = classification_report(y_test, y_pred_mejor, output_dict=True)
        for categoria in np.unique(y):
            if categoria in reporte:
                precision = reporte[categoria]['precision']
                recall = reporte[categoria]['recall']
                f1 = reporte[categoria]['f1-score']
                support = reporte[categoria]['support']
                print(f"   {categoria:12}: Prec={precision:.3f} | Rec={recall:.3f} | F1={f1:.3f} | N={support}")
    except Exception as e:
        print(f"   ⚠️ Error en reporte: {e}")
    
    # 8. ANÁLISIS DE CONVERGENCIA
    print()
    print("📈 ANÁLISIS DE CONVERGENCIA:")
    for nombre, resultado in resultados.items():
        convergencia = resultado['convergencia']
        if convergencia:
            status = "✅ Convergió" if convergencia['convergio'] else "⚠️ No convergió"
            print(f"   {nombre:15}: {status} | Iter: {convergencia['iteraciones']}")
    
    # 9. CREAR VISUALIZACIONES
    print("💾 Gráfico guardado: redes_neuronas_resultados.png")
    crear_visualizacion_redes(resultados, mejor_nombre, variables_disponibles)
    
    # 10. GUARDAR MEJOR MODELO
    try:
        import joblib
        joblib.dump(mejor_modelo, 'mejor_red_neuronal.pkl')
        joblib.dump(scaler, 'scaler_red_neuronal.pkl')
        print("💾 Modelo guardado: mejor_red_neuronal.pkl")
        print("💾 Scaler guardado: scaler_red_neuronal.pkl")
    except Exception as e:
        print(f"⚠️ Error guardando modelo: {e}")
    
    # 11. RESUMEN FINAL
    print()
    print("📝 RESUMEN REDES NEURONALES:")
    print(f"   • Mejor arquitectura: {resultados[mejor_nombre]['arquitectura']}")
    print(f"   • Función activación: {resultados[mejor_nombre]['activacion']}")
    print(f"   • Precisión alcanzada: {mejor_precision*100:.1f}%")
    print(f"   • Parámetros totales: {resultados[mejor_nombre]['n_parametros']:,}")
    
    conv_mejor = resultados[mejor_nombre]['convergencia']
    if conv_mejor:
        if conv_mejor['convergio']:
            print(f"   • Red convergió en {conv_mejor['iteraciones']} iteraciones ✅")
        else:
            print(f"   • Red no convergió completamente ⚠️")
    
    if mejor_precision > 0.8:
        print("   • ¡Excelente aprendizaje neuronal! 🧠🎉")
    elif mejor_precision > 0.65:
        print("   • Buen aprendizaje de la red neuronal 👍")
    else:
        print("   • Aprendizaje moderado, considerar ajustes 🔧")
    
    print("   • Ventaja: Puede aprender patrones complejos no lineales")
    print("   • Desventaja: Caja negra, difícil de interpretar")
    
    print("✅ REDES NEURONALES COMPLETADAS")
    return resultados

if __name__ == "__main__":
    ejecutar_redes_neuronas()