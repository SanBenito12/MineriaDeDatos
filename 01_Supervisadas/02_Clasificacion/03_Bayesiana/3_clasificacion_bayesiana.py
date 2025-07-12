#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLASIFICACIÓN BAYESIANA - Versión Optimizada
Clasificación probabilística usando teorema de Bayes
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
            return 'Muy Grande'
    
    return categorizar

def preparar_datos_bayesianos(datos):
    """Prepara datos específicamente para clasificación bayesiana"""
    variables_predictoras = [
        'POBFEM', 'POBMAS', 'TOTHOG', 'VIVTOT', 'P_15YMAS', 
        'P_60YMAS', 'GRAPROES', 'PEA', 'POCUPADA'
    ]
    
    variables_disponibles = [v for v in variables_predictoras if v in datos.columns]
    
    if len(variables_disponibles) < 5:
        return None, None, None
    
    # Crear categorías dinámicas basadas en cuartiles
    categorizador = crear_categorias_poblacion_dinamica(datos)
    datos['CATEGORIA_POB'] = datos['POBTOT'].apply(categorizador)
    
    # Verificar distribución antes de limpiar
    distribucion_inicial = datos['CATEGORIA_POB'].value_counts()
    
    # Limpiar datos
    datos_limpios = datos[variables_disponibles + ['CATEGORIA_POB']].dropna()
    
    # Verificar que todas las categorías tengan suficientes muestras
    distribucion = datos_limpios['CATEGORIA_POB'].value_counts()
    categorias_validas = distribucion[distribucion >= 50].index  # Mínimo 50 por categoría
    
    if len(categorias_validas) < 2:
        # Si no hay suficientes categorías, usar categorización más simple
        def categorizar_simple(poblacion):
            mediana = datos['POBTOT'].median()
            return 'Pequeña' if poblacion <= mediana else 'Grande'
        
        datos['CATEGORIA_POB'] = datos['POBTOT'].apply(categorizar_simple)
        datos_limpios = datos[variables_disponibles + ['CATEGORIA_POB']].dropna()
    else:
        # Filtrar solo categorías válidas
        datos_limpios = datos_limpios[datos_limpios['CATEGORIA_POB'].isin(categorias_validas)]
    
    # Reducir muestra si es muy grande, manteniendo estratificación
    if len(datos_limpios) > 5000:
        try:
            datos_limpios = datos_limpios.groupby('CATEGORIA_POB').apply(
                lambda x: x.sample(min(len(x), 1250), random_state=42)
            ).reset_index(drop=True)
        except:
            datos_limpios = datos_limpios.sample(n=5000, random_state=42)
    
    X = datos_limpios[variables_disponibles]
    y = datos_limpios['CATEGORIA_POB']
    
    return X, y, variables_disponibles

def entrenar_modelos_bayesianos(X_train, X_test, y_train, y_test):
    """Entrena diferentes modelos bayesianos de manera optimizada"""
    
    # Preparar datos para diferentes modelos
    scaler_gaussian = StandardScaler()
    X_train_gaussian = scaler_gaussian.fit_transform(X_train)
    X_test_gaussian = scaler_gaussian.transform(X_test)
    
    scaler_multinomial = MinMaxScaler()
    X_train_multinomial = scaler_multinomial.fit_transform(X_train)
    X_test_multinomial = scaler_multinomial.transform(X_test)
    
    X_train_bernoulli = (X_train > X_train.median()).astype(int)
    X_test_bernoulli = (X_test > X_train.median()).astype(int)
    
    # Configuración de modelos
    modelos = {
        'Gaussiano': {
            'modelo': GaussianNB(),
            'X_train': X_train_gaussian,
            'X_test': X_test_gaussian,
            'scaler': scaler_gaussian
        },
        'Multinomial': {
            'modelo': MultinomialNB(alpha=1.0),
            'X_train': X_train_multinomial,
            'X_test': X_test_multinomial,
            'scaler': scaler_multinomial
        },
        'Bernoulli': {
            'modelo': BernoulliNB(alpha=1.0),
            'X_train': X_train_bernoulli,
            'X_test': X_test_bernoulli,
            'scaler': None
        }
    }
    
    resultados = {}
    
    for nombre, config in modelos.items():
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
                'scaler': config['scaler']
            }
            
        except Exception as e:
            continue
    
    return resultados

def crear_visualizaciones_bayesianas(resultados, mejor_nombre, y_test):
    """Crear visualizaciones esenciales para clasificación bayesiana"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('🎲 CLASIFICACIÓN BAYESIANA - ANÁLISIS', fontsize=14, fontweight='bold')
        
        # Gráfico 1: Comparación de precisión
        nombres = list(resultados.keys())
        precisiones = [resultados[m]['precision'] for m in nombres]
        colores = ['lightblue', 'lightgreen', 'orange']
        
        axes[0,0].bar(nombres, precisiones, color=colores[:len(nombres)])
        axes[0,0].set_title('📊 Precisión por Modelo Bayesiano', fontweight='bold')
        axes[0,0].set_ylabel('Precisión')
        axes[0,0].set_ylim(0, 1)
        
        # Añadir valores en las barras
        for i, precision in enumerate(precisiones):
            axes[0,0].text(i, precision + 0.02, f'{precision:.3f}', 
                          ha='center', fontweight='bold')
        
        # Gráfico 2: Matriz de confusión del mejor modelo
        mejor_pred = resultados[mejor_nombre]['predicciones']
        clases = resultados[mejor_nombre]['modelo'].classes_
        
        cm = confusion_matrix(y_test, mejor_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=clases, yticklabels=clases, ax=axes[0,1])
        axes[0,1].set_title(f'🎯 Matriz de Confusión\n{mejor_nombre}', fontweight='bold')
        axes[0,1].set_xlabel('Predicción')
        axes[0,1].set_ylabel('Real')
        
        # Gráfico 3: Distribución de confianza
        probabilidades = resultados[mejor_nombre]['probabilidades']
        max_probs = np.max(probabilidades, axis=1)
        
        axes[1,0].hist(max_probs, bins=20, alpha=0.7, color='purple', edgecolor='black')
        axes[1,0].set_title('📈 Distribución de Confianza', fontweight='bold')
        axes[1,0].set_xlabel('Confianza Máxima')
        axes[1,0].set_ylabel('Frecuencia')
        
        # Gráfico 4: F1-Score por categoría
        reporte = classification_report(y_test, mejor_pred, output_dict=True)
        categorias_f1 = []
        f1_scores = []
        
        for categoria in ['Pequeña', 'Mediana', 'Grande', 'Muy Grande']:
            if categoria in reporte:
                f1_scores.append(reporte[categoria]['f1-score'])
                categorias_f1.append(categoria)
        
        if categorias_f1:
            axes[1,1].bar(categorias_f1, f1_scores, color='gold')
            axes[1,1].set_title('🎯 F1-Score por Categoría', fontweight='bold')
            axes[1,1].set_ylabel('F1-Score')
            axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('/home/sedc/Proyectos/MineriaDeDatos/results/graficos/clasificacion_bayesiana.png', 
                   dpi=150, bbox_inches='tight')
        plt.show()
        
        return True
        
    except Exception as e:
        return False

def guardar_resultados_bayesianos(mejor_modelo, mejor_nombre, mejor_precision, variables_disponibles, datos_info):
    """Guardar modelo y reporte de manera optimizada"""
    try:
        import joblib
        
        # Guardar modelo
        joblib.dump(mejor_modelo, '/home/sedc/Proyectos/MineriaDeDatos/results/modelos/mejor_modelo_bayesiano.pkl')
        
        # Crear reporte conciso
        reporte = f"""
REPORTE CLASIFICACIÓN BAYESIANA
==============================

MEJOR MODELO: {mejor_nombre} Naive Bayes
Precisión: {mejor_precision:.3f} ({mejor_precision*100:.1f}%)

DATOS PROCESADOS:
- Registros: {datos_info['n_registros']:,}
- Variables: {len(variables_disponibles)}
- Entrenamiento: {datos_info['n_train']:,}
- Prueba: {datos_info['n_test']:,}

VARIABLES UTILIZADAS:
{', '.join(variables_disponibles)}

INTERPRETACIÓN:
- Usa probabilidades para clasificar poblaciones
- Asume independencia entre variables (naive)
- Eficiente y rápido para datos grandes
"""
        
        with open('/home/sedc/Proyectos/MineriaDeDatos/results/reportes/clasificacion_bayesiana_reporte.txt', 'w', encoding='utf-8') as f:
            f.write(reporte)
        
        return True
        
    except Exception as e:
        return False

def ejecutar_clasificacion_bayesiana():
    """FUNCIÓN PRINCIPAL - Mantiene compatibilidad con menú"""
    print("🎲 CLASIFICACIÓN BAYESIANA")
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
    X, y, variables_disponibles = preparar_datos_bayesianos(datos)
    if X is None:
        print("❌ No hay suficientes variables para clasificación bayesiana")
        return
    
    print(f"📊 Variables: {len(variables_disponibles)} | Datos limpios: {len(X):,}")
    
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
        # Si stratify falla, usar división simple
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        print(f"📊 División simple: {len(X_train):,} entrenamiento | {len(X_test):,} prueba")
    print()
    
    # 4. ENTRENAR MODELOS BAYESIANOS
    print("🧠 Entrenando modelos bayesianos...")
    resultados = entrenar_modelos_bayesianos(X_train, X_test, y_train, y_test)
    
    if not resultados:
        print("❌ No se pudo entrenar ningún modelo bayesiano")
        return
    
    # Mostrar resultados
    for nombre, resultado in resultados.items():
        print(f"   {nombre:12} → Precisión: {resultado['precision']:.3f} ({resultado['precision']*100:.1f}%)")
    
    # 5. ENCONTRAR MEJOR MODELO
    mejor_nombre = max(resultados.keys(), key=lambda x: resultados[x]['precision'])
    mejor_modelo = resultados[mejor_nombre]['modelo']
    mejor_precision = resultados[mejor_nombre]['precision']
    
    print()
    print(f"🏆 MEJOR MODELO: {mejor_nombre} Naive Bayes")
    print(f"   Precisión: {mejor_precision:.3f} ({mejor_precision*100:.1f}%)")
    
    # 6. ANÁLISIS DETALLADO
    y_pred_mejor = resultados[mejor_nombre]['predicciones']
    reporte = classification_report(y_test, y_pred_mejor, output_dict=True)
    
    print("\n🎯 Métricas por categoría:")
    for categoria in ['Pequeña', 'Mediana', 'Grande', 'Muy Grande']:
        if categoria in reporte:
            precision = reporte[categoria]['precision']
            recall = reporte[categoria]['recall']
            f1 = reporte[categoria]['f1-score']
            print(f"   {categoria:12}: Prec={precision:.3f} | Rec={recall:.3f} | F1={f1:.3f}")
    
    # 7. VISUALIZACIONES
    crear_visualizaciones_bayesianas(resultados, mejor_nombre, y_test)
    
    # 8. GUARDAR RESULTADOS
    datos_info = {
        'n_registros': len(X),
        'n_train': len(X_train),
        'n_test': len(X_test)
    }
    
    guardar_resultados_bayesianos(mejor_modelo, mejor_nombre, mejor_precision, 
                                 variables_disponibles, datos_info)
    
    # 9. RESUMEN FINAL
    print()
    print("📝 RESUMEN:")
    print(f"   • Modelo: {mejor_nombre} Naive Bayes")
    print(f"   • Precisión: {mejor_precision*100:.1f}%")
    print(f"   • Variables: {len(variables_disponibles)}")
    
    if mejor_precision > 0.8:
        print("   🎉 ¡Excelente clasificación probabilística!")
    elif mejor_precision > 0.6:
        print("   👍 Buena clasificación bayesiana")
    else:
        print("   🔧 Clasificación moderada")
    
    print("✅ CLASIFICACIÓN BAYESIANA COMPLETADA")
    return resultados

if __name__ == "__main__":
    ejecutar_clasificacion_bayesiana()