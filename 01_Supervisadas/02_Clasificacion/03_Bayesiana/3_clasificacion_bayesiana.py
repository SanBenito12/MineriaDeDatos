#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLASIFICACIÓN BAYESIANA
Usa probabilidades para clasificar poblaciones
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def crear_categorias_poblacion(poblacion):
    """Crear categorías de población para clasificación"""
    if poblacion <= 1000:
        return 'Pequeña'
    elif poblacion <= 5000:
        return 'Mediana'
    elif poblacion <= 20000:
        return 'Grande'
    else:
        return 'Muy Grande'

def calcular_probabilidades_clase(y):
    """Calcular probabilidades a priori de cada clase"""
    conteos = pd.Series(y).value_counts()
    total = len(y)
    probabilidades = {}
    
    for clase, conteo in conteos.items():
        prob = conteo / total
        probabilidades[clase] = prob
    
    return probabilidades

def ejecutar_clasificacion_bayesiana():
    print("🎲 CLASIFICACIÓN BAYESIANA")
    print("="*40)
    print("📝 Objetivo: Clasificar usando probabilidades y teorema de Bayes")
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
        'P_60YMAS', 'GRAPROES', 'PEA', 'POCUPADA', 'PDESOCUP'
    ]
    
    variables_disponibles = [v for v in variables_predictoras if v in datos.columns]
    
    if len(variables_disponibles) < 3:
        print("❌ No hay suficientes variables para clasificación bayesiana")
        return
    
    print(f"📊 Variables usadas: {', '.join(variables_disponibles)}")
    
    # 3. CREAR VARIABLE OBJETIVO
    datos['CATEGORIA_POB'] = datos['POBTOT'].apply(crear_categorias_poblacion)
    
    # 4. PREPARAR DATOS
    datos_limpios = datos[variables_disponibles + ['CATEGORIA_POB']].dropna()
    X = datos_limpios[variables_disponibles]
    y = datos_limpios['CATEGORIA_POB']
    
    print(f"🧹 Datos limpios: {len(datos_limpios):,} registros")
    
    # 5. CALCULAR PROBABILIDADES A PRIORI
    prob_priori = calcular_probabilidades_clase(y)
    print()
    print("🎯 PROBABILIDADES A PRIORI:")
    for clase, prob in prob_priori.items():
        print(f"   P({clase:12}) = {prob:.3f} ({prob*100:.1f}%)")
    
    # 6. DIVIDIR DATOS
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\n📊 Entrenamiento: {len(X_train):,} | Prueba: {len(X_test):,}")
    print()
    
    # 7. PREPARAR DATOS PARA DIFERENTES MODELOS BAYESIANOS
    
    # Para Gaussian Naive Bayes (datos continuos)
    scaler_gaussian = StandardScaler()
    X_train_gaussian = scaler_gaussian.fit_transform(X_train)
    X_test_gaussian = scaler_gaussian.transform(X_test)
    
    # Para Multinomial Naive Bayes (datos discretos positivos)
    scaler_multinomial = MinMaxScaler()
    X_train_multinomial = scaler_multinomial.fit_transform(X_train)
    X_test_multinomial = scaler_multinomial.transform(X_test)
    
    # Para Bernoulli Naive Bayes (datos binarios)
    X_train_bernoulli = (X_train > X_train.median()).astype(int)
    X_test_bernoulli = (X_test > X_train.median()).astype(int)
    
    # 8. ENTRENAR DIFERENTES MODELOS BAYESIANOS
    modelos = {
        'Gaussiano': {
            'modelo': GaussianNB(),
            'X_train': X_train_gaussian,
            'X_test': X_test_gaussian,
            'descripcion': 'Asume distribución normal'
        },
        'Multinomial': {
            'modelo': MultinomialNB(alpha=1.0),
            'X_train': X_train_multinomial,
            'X_test': X_test_multinomial,
            'descripcion': 'Para conteos y frecuencias'
        },
        'Bernoulli': {
            'modelo': BernoulliNB(alpha=1.0),
            'X_train': X_train_bernoulli,
            'X_test': X_test_bernoulli,
            'descripcion': 'Para características binarias'
        }
    }
    
    print("🧠 ENTRENANDO MODELOS BAYESIANOS...")
    resultados = {}
    
    for nombre, config in modelos.items():
        print(f"   🔄 Entrenando {nombre} Naive Bayes...")
        
        try:
            # Entrenar modelo
            modelo = config['modelo']
            modelo.fit(config['X_train'], y_train)
            
            # Predicciones
            y_pred = modelo.predict(config['X_test'])
            y_pred_proba = modelo.predict_proba(config['X_test'])
            
            # Métricas
            precision = accuracy_score(y_test, y_pred)
            
            resultados[nombre] = {
                'modelo': modelo,
                'precision': precision,
                'predicciones': y_pred,
                'probabilidades': y_pred_proba,
                'descripcion': config['descripcion']
            }
            
            print(f"   ✅ {nombre:12} → Precisión: {precision:.3f} ({precision*100:.1f}%)")
            
        except Exception as e:
            print(f"   ❌ Error en {nombre}: {e}")
    
    if not resultados:
        print("❌ No se pudo entrenar ningún modelo bayesiano")
        return
    
    # 9. ENCONTRAR EL MEJOR MODELO
    mejor_nombre = max(resultados.keys(), key=lambda x: resultados[x]['precision'])
    mejor_modelo = resultados[mejor_nombre]['modelo']
    mejor_precision = resultados[mejor_nombre]['precision']
    
    print()
    print(f"🏆 MEJOR MODELO: {mejor_nombre} Naive Bayes")
    print(f"   Descripción: {resultados[mejor_nombre]['descripcion']}")
    print(f"   Precisión: {mejor_precision:.3f} ({mejor_precision*100:.1f}%)")
    
    # 10. ANÁLISIS DETALLADO DEL MEJOR MODELO
    print()
    print("📊 ANÁLISIS DETALLADO:")
    y_pred_mejor = resultados[mejor_nombre]['predicciones']
    
    # Reporte por clase
    print("\n🎯 Métricas por Categoría:")
    reporte = classification_report(y_test, y_pred_mejor, output_dict=True)
    for categoria in ['Pequeña', 'Mediana', 'Grande', 'Muy Grande']:
        if categoria in reporte:
            precision = reporte[categoria]['precision']
            recall = reporte[categoria]['recall']
            f1 = reporte[categoria]['f1-score']
            print(f"   {categoria:12}: Prec={precision:.3f} | Rec={recall:.3f} | F1={f1:.3f}")
    
    # 11. ANÁLISIS DE PROBABILIDADES
    print()
    print("🎲 ANÁLISIS DE PROBABILIDADES:")
    
    # Obtener probabilidades del mejor modelo
    probabilidades = resultados[mejor_nombre]['probabilidades']
    clases = mejor_modelo.classes_
    
    # Confianza promedio por clase
    print("\n📈 Confianza Promedio por Predicción:")
    for i, clase in enumerate(clases):
        indices_clase = np.where(y_pred_mejor == clase)[0]
        if len(indices_clase) > 0:
            confianza_promedio = np.mean(probabilidades[indices_clase, i])
            print(f"   {clase:12}: {confianza_promedio:.3f} ({confianza_promedio*100:.1f}%)")
    
    # 12. VISUALIZACIONES
    try:
        fig = plt.figure(figsize=(16, 12))
        
        # Gráfico 1: Comparación de precisión entre modelos
        plt.subplot(3, 3, 1)
        nombres = list(resultados.keys())
        precisiones = [resultados[m]['precision'] for m in nombres]
        colores = ['lightblue', 'lightgreen', 'orange']
        
        barras = plt.bar(nombres, precisiones, color=colores[:len(nombres)])
        plt.title('🎲 Precisión por Modelo Bayesiano', fontweight='bold')
        plt.ylabel('Precisión')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        
        for i, (barra, precision) in enumerate(zip(barras, precisiones)):
            plt.text(i, precision + 0.02, f'{precision:.3f}', ha='center', fontweight='bold')
        
        # Gráfico 2: Matriz de confusión del mejor modelo
        plt.subplot(3, 3, 2)
        cm = confusion_matrix(y_test, y_pred_mejor)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=clases, yticklabels=clases)
        plt.title(f'🎯 Matriz de Confusión\n{mejor_nombre} NB', fontweight='bold')
        plt.xlabel('Predicción')
        plt.ylabel('Real')
        
        # Gráfico 3: Distribución de probabilidades a priori
        plt.subplot(3, 3, 3)
        clases_priori = list(prob_priori.keys())
        probs_priori = list(prob_priori.values())
        plt.pie(probs_priori, labels=clases_priori, autopct='%1.1f%%', startangle=90)
        plt.title('📊 Probabilidades A Priori', fontweight='bold')
        
        # Gráfico 4: Distribución de confianza de predicciones
        plt.subplot(3, 3, 4)
        max_probs = np.max(probabilidades, axis=1)
        plt.hist(max_probs, bins=20, alpha=0.7, color='purple', edgecolor='black')
        plt.title('📈 Distribución de Confianza', fontweight='bold')
        plt.xlabel('Confianza Máxima')
        plt.ylabel('Frecuencia')
        
        # Gráfico 5: Comparación de F1-Score por clase
        plt.subplot(3, 3, 5)
        f1_scores = []
        categorias_f1 = []
        for categoria in ['Pequeña', 'Mediana', 'Grande', 'Muy Grande']:
            if categoria in reporte:
                f1_scores.append(reporte[categoria]['f1-score'])
                categorias_f1.append(categoria)
        
        plt.bar(categorias_f1, f1_scores, color='gold')
        plt.title('🎯 F1-Score por Categoría', fontweight='bold')
        plt.ylabel('F1-Score')
        plt.xticks(rotation=45)
        
        # Gráfico 6: Heatmap de probabilidades por clase
        plt.subplot(3, 3, 6)
        prob_media_por_clase = np.zeros((len(clases), len(clases)))
        for i, clase_real in enumerate(clases):
            indices = np.where(y_test == clase_real)[0]
            if len(indices) > 0:
                prob_media_por_clase[i] = np.mean(probabilidades[indices], axis=0)
        
        sns.heatmap(prob_media_por_clase, annot=True, fmt='.3f', cmap='Reds',
                   xticklabels=clases, yticklabels=clases)
        plt.title('🔥 Probabilidades Promedio\nReal vs Predicho', fontweight='bold')
        plt.xlabel('Clase Predicha')
        plt.ylabel('Clase Real')
        
        # Gráfico 7: Precisión vs Recall por modelo
        plt.subplot(3, 3, 7)
        precision_macro = []
        recall_macro = []
        for nombre in nombres:
            y_pred_temp = resultados[nombre]['predicciones']
            reporte_temp = classification_report(y_test, y_pred_temp, output_dict=True)
            precision_macro.append(reporte_temp['macro avg']['precision'])
            recall_macro.append(reporte_temp['macro avg']['recall'])
        
        plt.scatter(recall_macro, precision_macro, s=100, c=colores[:len(nombres)], alpha=0.7)
        for i, nombre in enumerate(nombres):
            plt.annotate(nombre, (recall_macro[i], precision_macro[i]), 
                        xytext=(5, 5), textcoords='offset points')
        plt.xlabel('Recall Macro')
        plt.ylabel('Precisión Macro')
        plt.title('📈 Precisión vs Recall', fontweight='bold')
        
        # Gráfico 8: Distribución de errores por categoría
        plt.subplot(3, 3, 8)
        errores_por_categoria = {}
        for i, (real, pred) in enumerate(zip(y_test, y_pred_mejor)):
            if real != pred:
                if real not in errores_por_categoria:
                    errores_por_categoria[real] = 0
                errores_por_categoria[real] += 1
        
        if errores_por_categoria:
            categorias_error = list(errores_por_categoria.keys())
            conteo_errores = list(errores_por_categoria.values())
            plt.bar(categorias_error, conteo_errores, color='lightcoral')
            plt.title('❌ Errores por Categoría Real', fontweight='bold')
            plt.ylabel('Número de Errores')
            plt.xticks(rotation=45)
        
        # Gráfico 9: Evolución de confianza por tamaño de muestra
        plt.subplot(3, 3, 9)
        if len(max_probs) > 100:
            ventana = len(max_probs) // 20
            confianza_promedio = []
            for i in range(0, len(max_probs), ventana):
                fin = min(i + ventana, len(max_probs))
                confianza_promedio.append(np.mean(max_probs[i:fin]))
            
            plt.plot(range(len(confianza_promedio)), confianza_promedio, 'b-', linewidth=2)
            plt.title('📊 Confianza vs Muestras', fontweight='bold')
            plt.xlabel('Segmento de Muestra')
            plt.ylabel('Confianza Promedio')
        
        plt.tight_layout()
        plt.savefig('/home/sedc/Proyectos/MineriaDeDatos/results/graficos/clasificacion_bayesiana.png', 
                   dpi=150, bbox_inches='tight')
        plt.show()
        
        print("💾 Gráficos guardados en: results/graficos/clasificacion_bayesiana.png")
        
    except Exception as e:
        print(f"⚠️ Error creando visualizaciones: {e}")
    
    # 13. GUARDAR RESULTADOS
    try:
        import joblib
        
        # Guardar el mejor modelo
        joblib.dump(mejor_modelo, '/home/sedc/Proyectos/MineriaDeDatos/results/modelos/mejor_modelo_bayesiano.pkl')
        
        # Crear reporte detallado
        reporte_completo = f"""
REPORTE CLASIFICACIÓN BAYESIANA
==============================

MEJOR MODELO: {mejor_nombre} Naive Bayes
Descripción: {resultados[mejor_nombre]['descripcion']}
Precisión: {mejor_precision:.3f} ({mejor_precision*100:.1f}%)

COMPARACIÓN DE MODELOS:
"""
        for nombre, resultado in resultados.items():
            reporte_completo += f"\n{nombre} NB:"
            reporte_completo += f"\n  - Precisión: {resultado['precision']:.3f}"
            reporte_completo += f"\n  - Descripción: {resultado['descripcion']}"
        
        reporte_completo += f"""

PROBABILIDADES A PRIORI:
"""
        for clase, prob in prob_priori.items():
            reporte_completo += f"\nP({clase}) = {prob:.3f}"
        
        reporte_completo += f"""

VARIABLES UTILIZADAS:
{', '.join(variables_disponibles)}

DATOS:
- Total registros: {len(datos_limpios):,}
- Variables predictoras: {len(variables_disponibles)}
- Categorías: {len(clases)}
"""
        
        with open('/home/sedc/Proyectos/MineriaDeDatos/results/reportes/clasificacion_bayesiana_reporte.txt', 'w', encoding='utf-8') as f:
            f.write(reporte_completo)
        
        print("💾 Modelo guardado en: results/modelos/mejor_modelo_bayesiano.pkl")
        print("📄 Reporte guardado en: results/reportes/clasificacion_bayesiana_reporte.txt")
        
    except Exception as e:
        print(f"⚠️ Error guardando archivos: {e}")
    
    # 14. RESUMEN FINAL
    print()
    print("📝 RESUMEN CLASIFICACIÓN BAYESIANA:")
    print(f"   • Mejor modelo: {mejor_nombre} Naive Bayes")
    print(f"   • Precisión: {mejor_precision*100:.1f}%")
    print(f"   • Principio: {resultados[mejor_nombre]['descripcion']}")
    
    if mejor_precision > 0.8:
        print("   • ¡Excelente clasificación probabilística! 🎉")
    elif mejor_precision > 0.6:
        print("   • Buena clasificación bayesiana 👍")
    else:
        print("   • Clasificación moderada, revisar distribuciones 🔧")
    
    print("✅ CLASIFICACIÓN BAYESIANA COMPLETADA")
    return resultados

if __name__ == "__main__":
    ejecutar_clasificacion_bayesiana()