#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REDES DE NEURONAS - CLASIFICACIÓN (Versión Arreglada)
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

def crear_categorias_poblacion(poblacion):
    """Crear categorías de población para clasificación (umbrales ajustados)"""
    if poblacion <= 500:
        return 'Pequeña'
    elif poblacion <= 2000:
        return 'Mediana'
    elif poblacion <= 8000:
        return 'Grande'
    else:
        return 'Muy Grande'

def muestreo_estratificado_balanceado(datos, variable_objetivo, n_muestra=3000, min_por_clase=200):
    """Muestreo balanceado para redes neuronales"""
    clases_disponibles = datos[variable_objetivo].value_counts()
    clases_validas = clases_disponibles[clases_disponibles >= min_por_clase]
    
    if len(clases_validas) < 2:
        print(f"⚠️ Solo {len(clases_validas)} clases tienen suficientes muestras")
        return datos.sample(n=min(n_muestra, len(datos)), random_state=42)
    
    # Calcular muestras por clase
    n_clases = len(clases_validas)
    muestras_por_clase = max(min_por_clase, n_muestra // n_clases)
    
    datos_balanceados = []
    for clase in clases_validas.index:
        datos_clase = datos[datos[variable_objetivo] == clase]
        n_tomar = min(muestras_por_clase, len(datos_clase))
        muestra_clase = datos_clase.sample(n=n_tomar, random_state=42)
        datos_balanceados.append(muestra_clase)
    
    return pd.concat(datos_balanceados, ignore_index=True)

def crear_arquitecturas_red():
    """Define diferentes arquitecturas de redes neuronales optimizadas"""
    arquitecturas = {
        'Red Simple': {
            'hidden_layer_sizes': (20,),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.001,
            'max_iter': 300,
            'descripcion': 'Una capa oculta con 20 neuronas'
        },
        'Red Mediana': {
            'hidden_layer_sizes': (30, 15),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.001,
            'max_iter': 400,
            'descripcion': 'Dos capas: 30 y 15 neuronas'
        },
        'Red Profunda': {
            'hidden_layer_sizes': (40, 20, 10),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.001,
            'max_iter': 500,
            'descripcion': 'Tres capas: 40, 20 y 10 neuronas'
        },
        'Red Tanh': {
            'hidden_layer_sizes': (25, 15),
            'activation': 'tanh',
            'solver': 'lbfgs',
            'alpha': 0.01,
            'max_iter': 300,
            'descripcion': 'Activación tangente hiperbólica'
        }
    }
    return arquitecturas

def analizar_convergencia(modelo):
    """Analiza la convergencia del entrenamiento"""
    if hasattr(modelo, 'loss_curve_'):
        return {
            'convergió': modelo.n_iter_ < modelo.max_iter,
            'iteraciones': modelo.n_iter_,
            'loss_final': modelo.loss_curve_[-1] if modelo.loss_curve_ else None,
            'loss_curve': modelo.loss_curve_
        }
    return None

def ejecutar_redes_neuronas():
    print("🧠 REDES DE NEURONAS - CLASIFICACIÓN")
    print("="*40)
    print("📝 Objetivo: Clasificar usando redes neuronales artificiales")
    print()
    
    # 1. CARGAR DATOS
    archivo = '/home/sedc/Proyectos/MineriaDeDatos/data/ceros_sin_columnasAB_limpio_weka.csv'
    try:
        datos = pd.read_csv(archivo)
        print(f"✅ Datos cargados: {datos.shape[0]:,} filas, {datos.shape[1]} columnas")
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        return
    
    # 2. SELECCIONAR VARIABLES PREDICTORAS
    variables_predictoras = [
        'POBFEM', 'POBMAS', 'TOTHOG', 'VIVTOT', 'P_15YMAS', 
        'P_60YMAS', 'GRAPROES', 'PEA', 'POCUPADA'
    ]
    
    variables_disponibles = [v for v in variables_predictoras if v in datos.columns]
    
    if len(variables_disponibles) < 5:
        print("❌ No hay suficientes variables para redes neuronales")
        return
    
    print(f"📊 Variables usadas: {', '.join(variables_disponibles)}")
    
    # 3. CREAR CATEGORÍAS CON UMBRALES AJUSTADOS
    datos['CATEGORIA_POB'] = datos['POBTOT'].apply(crear_categorias_poblacion)
    
    print(f"\n📈 Distribución original de categorías:")
    distribucion_original = datos['CATEGORIA_POB'].value_counts()
    for categoria, count in distribucion_original.items():
        print(f"   {categoria:12}: {count:,} ({count/len(datos)*100:.1f}%)")
    
    # 4. MUESTREO ESTRATIFICADO BALANCEADO
    print(f"\n🎯 Aplicando muestreo estratificado balanceado...")
    datos_balanceados = muestreo_estratificado_balanceado(datos, 'CATEGORIA_POB', n_muestra=3000, min_por_clase=200)
    
    print(f"📝 Muestra balanceada: {len(datos_balanceados):,} registros")
    print(f"📈 Nueva distribución:")
    for categoria, count in datos_balanceados['CATEGORIA_POB'].value_counts().items():
        print(f"   {categoria:12}: {count:,} ({count/len(datos_balanceados)*100:.1f}%)")
    
    # 5. PREPARAR DATOS FINALES
    datos_limpios = datos_balanceados[variables_disponibles + ['CATEGORIA_POB']].dropna()
    X = datos_limpios[variables_disponibles]
    y = datos_limpios['CATEGORIA_POB']
    
    print(f"\n🧹 Datos finales: {len(datos_limpios):,} registros")
    
    # 6. DIVIDIR DATOS
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        print(f"📊 División estratificada - Entrenamiento: {len(X_train):,} | Prueba: {len(X_test):,}")
    except Exception as e:
        print(f"⚠️ Error en división estratificada: {e}")
        # División simple sin estratificación
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        print(f"📊 División simple - Entrenamiento: {len(X_train):,} | Prueba: {len(X_test):,}")
    
    # 7. ESCALAR DATOS (CRÍTICO PARA REDES NEURONALES)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"🔢 Neuronas de entrada: {X_train_scaled.shape[1]}")
    print(f"🎯 Clases de salida: {len(np.unique(y))}")
    print()
    
    # 8. ENTRENAR DIFERENTES ARQUITECTURAS
    arquitecturas = crear_arquitecturas_red()
    
    print("🧠 ENTRENANDO REDES NEURONALES...")
    resultados = {}
    
    for nombre, config in arquitecturas.items():
        print(f"   🔄 Entrenando {nombre}...")
        print(f"      Arquitectura: {config['hidden_layer_sizes']}")
        
        try:
            # Crear configuración sin descripción para el modelo
            config_modelo = {k: v for k, v in config.items() if k != 'descripcion'}
            config_modelo['random_state'] = 42
            config_modelo['early_stopping'] = True
            config_modelo['validation_fraction'] = 0.2
            
            # Crear y entrenar modelo
            modelo = MLPClassifier(**config_modelo)
            modelo.fit(X_train_scaled, y_train)
            
            # Predicciones
            y_pred = modelo.predict(X_test_scaled)
            y_pred_proba = modelo.predict_proba(X_test_scaled)
            
            # Métricas
            precision = accuracy_score(y_test, y_pred)
            
            # Análisis de convergencia
            convergencia = analizar_convergencia(modelo)
            
            # Calcular número de parámetros aproximado
            n_entrada = X_train_scaled.shape[1]
            capas = config['hidden_layer_sizes']
            n_salida = len(np.unique(y))
            
            n_parametros = n_entrada * capas[0]  # Primera capa
            for i in range(len(capas) - 1):
                n_parametros += capas[i] * capas[i + 1]  # Capas ocultas
            n_parametros += capas[-1] * n_salida  # Capa de salida
            n_parametros += sum(capas) + n_salida  # Bias
            
            resultados[nombre] = {
                'modelo': modelo,
                'precision': precision,
                'predicciones': y_pred,
                'probabilidades': y_pred_proba,
                'arquitectura': config['hidden_layer_sizes'],
                'activacion': config['activation'],
                'solver': config['solver'],
                'descripcion': config['descripcion'],
                'convergencia': convergencia,
                'n_parametros': n_parametros
            }
            
            print(f"   ✅ {nombre} → Precisión: {precision:.3f} ({precision*100:.1f}%)")
            if convergencia:
                print(f"      Iteraciones: {convergencia['iteraciones']}")
                print(f"      Convergió: {'Sí' if convergencia['convergió'] else 'No'}")
            
        except Exception as e:
            print(f"   ❌ Error en {nombre}: {e}")
    
    if not resultados:
        print("❌ No se pudo entrenar ninguna red neuronal")
        return
    
    # 9. ENCONTRAR LA MEJOR RED
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
    
    # 10. ANÁLISIS DETALLADO
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
    
    # 11. ANÁLISIS DE CONVERGENCIA
    print()
    print("📈 ANÁLISIS DE CONVERGENCIA:")
    for nombre, resultado in resultados.items():
        convergencia = resultado['convergencia']
        if convergencia:
            status = "✅ Convergió" if convergencia['convergió'] else "⚠️ No convergió"
            print(f"   {nombre:15}: {status} | Iter: {convergencia['iteraciones']}")
    
    # 12. VISUALIZACIONES
    try:
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        
        # Gráfico 1: Comparación de precisión
        axes[0,0].bar(range(len(resultados)), 
                     [resultados[m]['precision'] for m in resultados.keys()],
                     color=['lightblue', 'lightgreen', 'orange', 'pink'][:len(resultados)])
        axes[0,0].set_title('🧠 Precisión por Arquitectura', fontweight='bold')
        axes[0,0].set_ylabel('Precisión')
        axes[0,0].set_xticks(range(len(resultados)))
        axes[0,0].set_xticklabels([n.split()[1] for n in resultados.keys()], rotation=45)
        
        # Añadir valores en las barras
        for i, (nombre, resultado) in enumerate(resultados.items()):
            axes[0,0].text(i, resultado['precision'] + 0.01, f"{resultado['precision']:.3f}", 
                          ha='center', fontweight='bold')
        
        # Gráfico 2: Matriz de confusión
        try:
            cm = confusion_matrix(y_test, y_pred_mejor)
            clases = np.unique(y)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=clases, yticklabels=clases, ax=axes[0,1])
            axes[0,1].set_title(f'🎯 Matriz de Confusión\n{mejor_nombre}', fontweight='bold')
            axes[0,1].set_xlabel('Predicción')
            axes[0,1].set_ylabel('Real')
        except:
            axes[0,1].text(0.5, 0.5, 'Matriz no\ndisponible', ha='center', va='center')
            axes[0,1].set_title('🎯 Matriz de Confusión', fontweight='bold')
        
        # Gráfico 3: Curvas de pérdida
        mejor_convergencia = resultados[mejor_nombre]['convergencia']
        if mejor_convergencia and mejor_convergencia['loss_curve']:
            axes[0,2].plot(mejor_convergencia['loss_curve'], 'b-', linewidth=2)
            axes[0,2].set_title(f'📉 Curva de Pérdida\n{mejor_nombre}', fontweight='bold')
            axes[0,2].set_xlabel('Época')
            axes[0,2].set_ylabel('Pérdida')
            axes[0,2].grid(True, alpha=0.3)
        else:
            axes[0,2].text(0.5, 0.5, 'Curva de pérdida\nno disponible', ha='center', va='center')
            axes[0,2].set_title('📉 Curva de Pérdida', fontweight='bold')
        
        # Gráfico 4: Número de parámetros vs precisión
        n_params = [resultados[m]['n_parametros'] for m in resultados.keys()]
        precisiones = [resultados[m]['precision'] for m in resultados.keys()]
        axes[1,0].scatter(n_params, precisiones, s=100, alpha=0.7, c=['red', 'green', 'blue', 'orange'][:len(n_params)])
        for i, nombre in enumerate(resultados.keys()):
            axes[1,0].annotate(nombre.split()[1][:6], (n_params[i], precisiones[i]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[1,0].set_xlabel('Número de Parámetros')
        axes[1,0].set_ylabel('Precisión')
        axes[1,0].set_title('⚖️ Complejidad vs Precisión', fontweight='bold')
        
        # Gráfico 5: Distribución de confianza
        probabilidades = resultados[mejor_nombre]['probabilidades']
        max_probs = np.max(probabilidades, axis=1)
        axes[1,1].hist(max_probs, bins=15, alpha=0.7, color='purple', edgecolor='black')
        axes[1,1].set_title('📈 Confianza de Predicciones', fontweight='bold')
        axes[1,1].set_xlabel('Confianza Máxima')
        axes[1,1].set_ylabel('Frecuencia')
        
        # Gráfico 6: Iteraciones hasta convergencia
        iteraciones = []
        nombres_conv = []
        for nombre, resultado in resultados.items():
            if resultado['convergencia']:
                iteraciones.append(resultado['convergencia']['iteraciones'])
                nombres_conv.append(nombre.split()[1])
        
        if iteraciones:
            axes[1,2].bar(nombres_conv, iteraciones, color='cyan')
            axes[1,2].set_title('⏱️ Iteraciones para Convergencia', fontweight='bold')
            axes[1,2].set_ylabel('Iteraciones')
            axes[1,2].tick_params(axis='x', rotation=45)
        
        # Gráfico 7: Arquitectura de la mejor red
        mejor_arq = resultados[mejor_nombre]['arquitectura']
        capas = [X_train_scaled.shape[1]] + list(mejor_arq) + [len(np.unique(y))]
        
        x_pos = np.arange(len(capas))
        axes[2,0].bar(x_pos, capas, color=['blue', 'green', 'orange', 'red', 'purple'][:len(capas)])
        axes[2,0].set_title(f'🏗️ Arquitectura Mejor Red\n{mejor_nombre}', fontweight='bold')
        axes[2,0].set_xlabel('Capa')
        axes[2,0].set_ylabel('Neuronas')
        axes[2,0].set_xticks(x_pos)
        axes[2,0].set_xticklabels(['Entrada'] + [f'Oculta {i+1}' for i in range(len(mejor_arq))] + ['Salida'])
        
        for i, neurons in enumerate(capas):
            axes[2,0].text(i, neurons + max(capas)*0.02, str(neurons), ha='center', fontweight='bold')
        
        # Gráfico 8: Distribución de categorías
        axes[2,1].pie(datos_balanceados['CATEGORIA_POB'].value_counts().values,
                     labels=datos_balanceados['CATEGORIA_POB'].value_counts().index,
                     autopct='%1.1f%%', startangle=90)
        axes[2,1].set_title('📊 Distribución Final', fontweight='bold')
        
        # Gráfico 9: Curvas de pérdida comparativas
        for nombre, resultado in resultados.items():
            if resultado['convergencia'] and resultado['convergencia']['loss_curve']:
                loss_curve = resultado['convergencia']['loss_curve']
                axes[2,2].plot(loss_curve, label=nombre.split()[1], alpha=0.7, linewidth=2)
        
        axes[2,2].set_title('📉 Curvas de Pérdida Comparativas', fontweight='bold')
        axes[2,2].set_xlabel('Época')
        axes[2,2].set_ylabel('Pérdida')
        axes[2,2].legend()
        axes[2,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/sedc/Proyectos/MineriaDeDatos/results/graficos/redes_neuronas_clasificacion.png', 
                   dpi=150, bbox_inches='tight')
        plt.show()
        
        print("💾 Gráficos guardados: results/graficos/redes_neuronas_clasificacion.png")
        
    except Exception as e:
        print(f"⚠️ Error en visualizaciones: {e}")
    
    # 13. GUARDAR RESULTADOS
    try:
        import joblib
        
        # Guardar el mejor modelo y el scaler
        joblib.dump(mejor_modelo, '/home/sedc/Proyectos/MineriaDeDatos/results/modelos/mejor_red_neuronal.pkl')
        joblib.dump(scaler, '/home/sedc/Proyectos/MineriaDeDatos/results/modelos/scaler_red_neuronal.pkl')
        
        # Crear reporte detallado
        reporte_completo = f"""
REPORTE REDES NEURONALES - CLASIFICACIÓN
=======================================

MEJOR RED: {mejor_nombre}
Descripción: {resultados[mejor_nombre]['descripcion']}
Precisión: {mejor_precision:.3f} ({mejor_precision*100:.1f}%)
Arquitectura: {resultados[mejor_nombre]['arquitectura']}
Activación: {resultados[mejor_nombre]['activacion']}
Solver: {resultados[mejor_nombre]['solver']}
Parámetros totales: {resultados[mejor_nombre]['n_parametros']:,}

CONVERGENCIA:
"""
        conv_mejor = resultados[mejor_nombre]['convergencia']
        if conv_mejor:
            reporte_completo += f"- Convergió: {'Sí' if conv_mejor['convergió'] else 'No'}\n"
            reporte_completo += f"- Iteraciones: {conv_mejor['iteraciones']}\n"
            reporte_completo += f"- Pérdida final: {conv_mejor['loss_final']:.6f}\n"
        
        reporte_completo += f"""

COMPARACIÓN DE ARQUITECTURAS:
"""
        for nombre, resultado in resultados.items():
            reporte_completo += f"\n{nombre}:"
            reporte_completo += f"\n  - Precisión: {resultado['precision']:.3f}"
            reporte_completo += f"\n  - Arquitectura: {resultado['arquitectura']}"
            reporte_completo += f"\n  - Activación: {resultado['activacion']}"
            reporte_completo += f"\n  - Parámetros: {resultado['n_parametros']:,}"
            if resultado['convergencia']:
                reporte_completo += f"\n  - Iteraciones: {resultado['convergencia']['iteraciones']}"
        
        reporte_completo += f"""

VARIABLES UTILIZADAS:
{', '.join(variables_disponibles)}

CONFIGURACIÓN:
- Neuronas de entrada: {X_train_scaled.shape[1]}
- Clases de salida: {len(np.unique(y))}
- Datos de entrenamiento: {len(X_train):,}
- Datos de prueba: {len(X_test):,}

DATOS PROCESADOS:
- Registros originales: {len(datos):,}
- Muestra balanceada: {len(datos_balanceados):,}
- Muestreo estratificado aplicado

NOTAS:
- Se aplicó escalado estándar a todas las variables
- Se utilizó early stopping para evitar overfitting
- Se reservó 20% de datos de entrenamiento para validación
- Umbrales ajustados para mejor distribución de clases
"""
        
        with open('/home/sedc/Proyectos/MineriaDeDatos/results/reportes/redes_neuronas_reporte.txt', 'w', encoding='utf-8') as f:
            f.write(reporte_completo)
        
        print("💾 Modelo guardado: results/modelos/mejor_red_neuronal.pkl")
        print("💾 Scaler guardado: results/modelos/scaler_red_neuronal.pkl")
        print("📄 Reporte guardado: results/reportes/redes_neuronas_reporte.txt")
        
    except Exception as e:
        print(f"⚠️ Error guardando archivos: {e}")
    
    # 14. RESUMEN FINAL
    print()
    print("📝 RESUMEN REDES NEURONALES:")
    print(f"   • Mejor arquitectura: {resultados[mejor_nombre]['arquitectura']}")
    print(f"   • Función activación: {resultados[mejor_nombre]['activacion']}")
    print(f"   • Precisión alcanzada: {mejor_precision*100:.1f}%")
    print(f"   • Parámetros totales: {resultados[mejor_nombre]['n_parametros']:,}")
    
    if conv_mejor:
        if conv_mejor['convergió']:
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