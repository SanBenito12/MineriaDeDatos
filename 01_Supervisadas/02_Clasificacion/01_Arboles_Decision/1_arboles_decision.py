

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
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

def ejecutar_arboles_decision():
    print("🌳 ÁRBOLES DE DECISIÓN - CLASIFICACIÓN")
    print("="*40)
    print("📝 Objetivo: Clasificar comunidades por tamaño de población")
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
    
    # Verificar qué variables están disponibles
    variables_disponibles = [v for v in variables_predictoras if v in datos.columns]
    
    if len(variables_disponibles) < 3:
        print("❌ No hay suficientes variables para clasificación")
        return
    
    print(f"📊 Variables usadas: {', '.join(variables_disponibles)}")
    
    # 3. CREAR VARIABLE OBJETIVO (CLASIFICACIÓN)
    datos['CATEGORIA_POB'] = datos['POBTOT'].apply(crear_categorias_poblacion)
    
    # 4. PREPARAR DATOS
    datos_limpios = datos[variables_disponibles + ['CATEGORIA_POB']].dropna()
    X = datos_limpios[variables_disponibles]
    y = datos_limpios['CATEGORIA_POB']
    
    print(f"🧹 Datos limpios: {len(datos_limpios):,} registros")
    print(f"📈 Distribución de categorías:")
    for categoria, count in y.value_counts().items():
        print(f"   {categoria:12}: {count:,} ({count/len(y)*100:.1f}%)")
    
    # 5. DIVIDIR DATOS
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"📊 Entrenamiento: {len(X_train):,} | Prueba: {len(X_test):,}")
    print()
    
    # 6. ENTRENAR DIFERENTES CONFIGURACIONES DE ÁRBOLES
    configuraciones = {
        'Árbol Simple': {
            'max_depth': 3,
            'min_samples_split': 100,
            'min_samples_leaf': 50
        },
        'Árbol Balanceado': {
            'max_depth': 5,
            'min_samples_split': 50,
            'min_samples_leaf': 20
        },
        'Árbol Profundo': {
            'max_depth': 10,
            'min_samples_split': 20,
            'min_samples_leaf': 10
        }
    }
    
    print("🌱 ENTRENANDO ÁRBOLES DE DECISIÓN...")
    resultados = {}
    
    for nombre, params in configuraciones.items():
        print(f"   🔄 Entrenando {nombre}...")
        
        # Crear y entrenar modelo
        modelo = DecisionTreeClassifier(
            **params,
            random_state=42,
            class_weight='balanced'  # Para balancear clases
        )
        
        modelo.fit(X_train, y_train)
        
        # Predicciones
        y_pred = modelo.predict(X_test)
        
        # Métricas
        precision = accuracy_score(y_test, y_pred)
        
        resultados[nombre] = {
            'modelo': modelo,
            'precision': precision,
            'predicciones': y_pred,
            'profundidad': modelo.get_depth(),
            'n_hojas': modelo.get_n_leaves()
        }
        
        print(f"   ✅ {nombre} → Precisión: {precision:.3f} ({precision*100:.1f}%)")
        print(f"      Profundidad: {modelo.get_depth()} | Hojas: {modelo.get_n_leaves()}")
    
    # 7. ENCONTRAR EL MEJOR MODELO
    mejor_nombre = max(resultados.keys(), key=lambda x: resultados[x]['precision'])
    mejor_modelo = resultados[mejor_nombre]['modelo']
    mejor_precision = resultados[mejor_nombre]['precision']
    
    print()
    print(f"🏆 MEJOR MODELO: {mejor_nombre}")
    print(f"   Precisión: {mejor_precision:.3f} ({mejor_precision*100:.1f}%)")
    
    # 8. REPORTE DETALLADO DEL MEJOR MODELO
    print()
    print("📊 REPORTE DETALLADO:")
    y_pred_mejor = resultados[mejor_nombre]['predicciones']
    
    print("\n🎯 Precisión por Categoría:")
    reporte = classification_report(y_test, y_pred_mejor, output_dict=True)
    for categoria in ['Pequeña', 'Mediana', 'Grande', 'Muy Grande']:
        if categoria in reporte:
            precision = reporte[categoria]['precision']
            recall = reporte[categoria]['recall']
            f1 = reporte[categoria]['f1-score']
            print(f"   {categoria:12}: Prec={precision:.3f} | Rec={recall:.3f} | F1={f1:.3f}")
    
    # 9. IMPORTANCIA DE VARIABLES
    print()
    print("📈 IMPORTANCIA DE VARIABLES:")
    importancias = mejor_modelo.feature_importances_
    variables_importancia = list(zip(variables_disponibles, importancias))
    variables_importancia.sort(key=lambda x: x[1], reverse=True)
    
    for i, (variable, importancia) in enumerate(variables_importancia):
        barras = '█' * int(importancia * 30)
        print(f"   {i+1}. {variable:12} {barras} {importancia:.3f}")
    
    # 10. VISUALIZACIONES
    try:
        # Crear figura con múltiples gráficos
        fig = plt.figure(figsize=(20, 12))
        
        # Gráfico 1: Árbol de decisión
        plt.subplot(2, 3, 1)
        plot_tree(mejor_modelo, 
                 feature_names=variables_disponibles,
                 class_names=mejor_modelo.classes_,
                 filled=True,
                 rounded=True,
                 fontsize=8,
                 max_depth=3)  # Limitar profundidad para visualización
        plt.title(f'🌳 {mejor_nombre}\n(Primeros 3 niveles)', fontweight='bold')
        
        # Gráfico 2: Matriz de confusión
        plt.subplot(2, 3, 2)
        cm = confusion_matrix(y_test, y_pred_mejor)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=mejor_modelo.classes_,
                   yticklabels=mejor_modelo.classes_)
        plt.title('🎯 Matriz de Confusión', fontweight='bold')
        plt.xlabel('Predicción')
        plt.ylabel('Real')
        
        # Gráfico 3: Comparación de precisión
        plt.subplot(2, 3, 3)
        nombres = list(resultados.keys())
        precisiones = [resultados[m]['precision'] for m in nombres]
        colores = ['lightgreen', 'orange', 'lightcoral']
        
        barras = plt.bar(range(len(nombres)), precisiones, color=colores)
        plt.title('📊 Precisión por Configuración', fontweight='bold')
        plt.ylabel('Precisión')
        plt.xticks(range(len(nombres)), nombres, rotation=45, ha='right')
        plt.ylim(0, 1)
        
        for i, (barra, precision) in enumerate(zip(barras, precisiones)):
            plt.text(i, precision + 0.02, f'{precision:.3f}', 
                    ha='center', fontweight='bold')
        
        # Gráfico 4: Importancia de variables
        plt.subplot(2, 3, 4)
        variables_top = [v[0] for v in variables_importancia[:6]]
        importancias_top = [v[1] for v in variables_importancia[:6]]
        plt.barh(range(len(variables_top)), importancias_top, color='purple')
        plt.yticks(range(len(variables_top)), variables_top)
        plt.xlabel('Importancia')
        plt.title('📈 Top 6 Variables Importantes', fontweight='bold')
        
        # Gráfico 5: Distribución de profundidades
        plt.subplot(2, 3, 5)
        nombres_cortos = [n.split()[1] for n in nombres]
        profundidades = [resultados[m]['profundidad'] for m in nombres]
        plt.bar(range(len(nombres_cortos)), profundidades, color='cyan')
        plt.title('🌳 Profundidad por Modelo', fontweight='bold')
        plt.ylabel('Niveles')
        plt.xticks(range(len(nombres_cortos)), nombres_cortos)
        
        # Gráfico 6: Número de hojas
        plt.subplot(2, 3, 6)
        n_hojas = [resultados[m]['n_hojas'] for m in nombres]
        plt.bar(range(len(nombres_cortos)), n_hojas, color='gold')
        plt.title('🍃 Número de Decisiones Finales', fontweight='bold')
        plt.ylabel('Hojas')
        plt.xticks(range(len(nombres_cortos)), nombres_cortos)
        
        plt.tight_layout()
        plt.savefig('/home/sedc/Proyectos/MineriaDeDatos/results/graficos/arboles_decision_clasificacion.png', 
                   dpi=150, bbox_inches='tight')
        plt.show()
        
        print("💾 Gráficos guardados en: results/graficos/arboles_decision_clasificacion.png")
        
    except Exception as e:
        print(f"⚠️ Error creando visualizaciones: {e}")
    
    # 11. GUARDAR MODELO Y RESULTADOS
    try:
        import joblib
        
        # Guardar el mejor modelo
        joblib.dump(mejor_modelo, '/home/sedc/Proyectos/MineriaDeDatos/results/modelos/mejor_arbol_decision.pkl')
        
        # Guardar reporte
        reporte_texto = f"""
REPORTE ÁRBOLES DE DECISIÓN - CLASIFICACIÓN
==========================================

MEJOR MODELO: {mejor_nombre}
Precisión General: {mejor_precision:.3f} ({mejor_precision*100:.1f}%)
Profundidad: {resultados[mejor_nombre]['profundidad']} niveles
Decisiones Finales: {resultados[mejor_nombre]['n_hojas']} hojas

VARIABLES MÁS IMPORTANTES:
{chr(10).join([f"{i+1}. {var}: {imp:.3f}" for i, (var, imp) in enumerate(variables_importancia[:5])])}

DATOS UTILIZADOS:
- Total registros: {len(datos_limpios):,}
- Variables predictoras: {len(variables_disponibles)}
- Categorías: {len(mejor_modelo.classes_)}
"""
        
        with open('/home/sedc/Proyectos/MineriaDeDatos/results/reportes/arboles_decision_reporte.txt', 'w', encoding='utf-8') as f:
            f.write(reporte_texto)
        
        print("💾 Modelo guardado en: results/modelos/mejor_arbol_decision.pkl")
        print("📄 Reporte guardado en: results/reportes/arboles_decision_reporte.txt")
        
    except Exception as e:
        print(f"⚠️ Error guardando archivos: {e}")
    
    # 12. RESUMEN FINAL
    print()
    print("📝 RESUMEN:")
    print(f"   • Mejor configuración: {mejor_nombre}")
    print(f"   • Precisión alcanzada: {mejor_precision*100:.1f}%")
    
    if mejor_precision > 0.8:
        print("   • ¡Excelente clasificación! 🎉")
    elif mejor_precision > 0.6:
        print("   • Buena clasificación 👍")
    else:
        print("   • Clasificación moderada, revisar datos 🔧")
    
    print("✅ ÁRBOLES DE DECISIÓN COMPLETADOS")
    return resultados

if __name__ == "__main__":
    ejecutar_arboles_decision()