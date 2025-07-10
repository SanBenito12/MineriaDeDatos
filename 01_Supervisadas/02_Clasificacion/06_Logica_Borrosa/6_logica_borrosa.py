#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LÓGICA BORROSA - CLASIFICACIÓN (Versión Arreglada)
Clasificación usando conjuntos borrosos y reglas difusas
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def crear_categorias_poblacion(poblacion):
    """Crear categorías de población para clasificación"""
    if poblacion <= 500:
        return 'Pequeña'
    elif poblacion <= 2000:
        return 'Mediana'
    elif poblacion <= 10000:
        return 'Grande'
    else:
        return 'Muy Grande'

def funcion_membresia_triangular(x, a, b, c):
    """Función de membresía triangular"""
    return np.maximum(0, np.minimum((x - a) / (b - a + 1e-6), (c - x) / (c - b + 1e-6)))

def funcion_membresia_gaussiana(x, centro, sigma):
    """Función de membresía gaussiana"""
    return np.exp(-0.5 * ((x - centro) / (sigma + 1e-6)) ** 2)

class ClasificadorBorroso:
    """Clasificador basado en lógica borrosa simplificado"""
    
    def __init__(self, n_conjuntos=3):
        self.n_conjuntos = n_conjuntos
        self.conjuntos_borrosos = {}
        self.reglas = []
        self.clases = None
        
    def _crear_conjuntos_borrosos(self, X, nombres_variables):
        """Crear conjuntos borrosos triangulares para cada variable"""
        self.conjuntos_borrosos = {}
        
        for i, variable in enumerate(nombres_variables):
            valores = X[:, i]
            min_val, max_val = np.min(valores), np.max(valores)
            
            # Evitar división por cero
            if max_val == min_val:
                max_val = min_val + 1
            
            conjuntos = {}
            nombres = ['Bajo', 'Medio', 'Alto']
            
            # Crear 3 conjuntos triangulares
            paso = (max_val - min_val) / 4
            for j in range(self.n_conjuntos):
                nombre = f"{variable}_{nombres[j]}"
                a = min_val + j * paso
                b = min_val + (j + 1) * paso
                c = min_val + (j + 2) * paso
                conjuntos[nombre] = ('triangular', a, b, c)
            
            self.conjuntos_borrosos[variable] = conjuntos
    
    def _calcular_membresia(self, x, conjunto_params):
        """Calcular grado de membresía"""
        tipo, a, b, c = conjunto_params
        return funcion_membresia_triangular(x, a, b, c)
    
    def _generar_reglas_simples(self, X, y, nombres_variables):
        """Generar reglas simples basadas en estadísticas"""
        self.reglas = []
        
        for clase in self.clases:
            indices_clase = np.where(y == clase)[0]
            if len(indices_clase) == 0:
                continue
                
            X_clase = X[indices_clase]
            
            # Para cada variable, encontrar el conjunto con mayor activación
            condiciones = {}
            for i, variable in enumerate(nombres_variables):
                valores_var = X_clase[:, i]
                mejor_membresia = 0
                mejor_conjunto = None
                
                for nombre_conjunto, params in self.conjuntos_borrosos[variable].items():
                    membresia_promedio = np.mean(self._calcular_membresia(valores_var, params))
                    
                    if membresia_promedio > mejor_membresia:
                        mejor_membresia = membresia_promedio
                        mejor_conjunto = nombre_conjunto
                
                if mejor_conjunto and mejor_membresia > 0.2:
                    condiciones[variable] = (mejor_conjunto, mejor_membresia)
            
            if condiciones:
                self.reglas.append({
                    'condiciones': condiciones,
                    'conclusion': clase,
                    'confianza': np.mean([conf for _, conf in condiciones.values()]),
                    'muestras': len(indices_clase)
                })
    
    def fit(self, X, y, nombres_variables):
        """Entrenar el clasificador borroso"""
        self.clases = np.unique(y)
        self._crear_conjuntos_borrosos(X, nombres_variables)
        self._generar_reglas_simples(X, y, nombres_variables)
        return self
    
    def predict(self, X, nombres_variables):
        """Predecir clases usando lógica borrosa"""
        predicciones = []
        
        for i in range(X.shape[0]):
            x = X[i]
            puntuaciones_clase = {}
            
            # Evaluar cada regla
            for regla in self.reglas:
                activaciones = []
                
                for j, variable in enumerate(nombres_variables):
                    if variable in regla['condiciones']:
                        conjunto_nombre, _ = regla['condiciones'][variable]
                        if variable in self.conjuntos_borrosos:
                            conjunto_params = self.conjuntos_borrosos[variable][conjunto_nombre]
                            membresia = self._calcular_membresia(x[j], conjunto_params)
                            activaciones.append(membresia)
                
                if activaciones:
                    # Usar mínimo (operador AND fuzzy)
                    activacion_regla = np.min(activaciones) * regla['confianza']
                    clase = regla['conclusion']
                    
                    if clase not in puntuaciones_clase:
                        puntuaciones_clase[clase] = 0
                    puntuaciones_clase[clase] = max(puntuaciones_clase[clase], activacion_regla)
            
            # Predecir clase con mayor puntuación
            if puntuaciones_clase:
                clase_predicha = max(puntuaciones_clase.keys(), key=lambda k: puntuaciones_clase[k])
                predicciones.append(clase_predicha)
            else:
                # Predicción por defecto
                predicciones.append(self.clases[0])
        
        return np.array(predicciones)

def muestreo_estratificado_balanceado(datos, variable_objetivo, n_muestra=2000, min_por_clase=50):
    """Muestreo que garantiza representación de todas las clases"""
    clases_disponibles = datos[variable_objetivo].value_counts()
    
    # Filtrar clases con suficientes muestras
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

def ejecutar_logica_borrosa():
    print("🌫️ LÓGICA BORROSA - CLASIFICACIÓN")
    print("="*40)
    print("📝 Objetivo: Clasificar usando conjuntos borrosos y reglas difusas")
    print()
    
    # 1. CARGAR DATOS
    archivo = '/home/sedc/Proyectos/MineriaDeDatos/data/ceros_sin_columnasAB_limpio_weka.csv'
    try:
        datos = pd.read_csv(archivo)
        print(f"✅ Datos cargados: {datos.shape[0]:,} filas, {datos.shape[1]} columnas")
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        return
    
    # 2. SELECCIONAR VARIABLES
    variables_predictoras = [
        'POBFEM', 'POBMAS', 'TOTHOG', 'VIVTOT', 'P_15YMAS', 
        'GRAPROES', 'PEA', 'POCUPADA'
    ]
    
    variables_disponibles = [v for v in variables_predictoras if v in datos.columns]
    print(f"📊 Variables usadas: {', '.join(variables_disponibles)}")
    
    # 3. CREAR CATEGORÍAS CON UMBRALES AJUSTADOS
    datos['CATEGORIA_POB'] = datos['POBTOT'].apply(crear_categorias_poblacion)
    
    print(f"\n📈 Distribución original de categorías:")
    distribucion_original = datos['CATEGORIA_POB'].value_counts()
    for categoria, count in distribucion_original.items():
        print(f"   {categoria:12}: {count:,} ({count/len(datos)*100:.1f}%)")
    
    # 4. MUESTREO ESTRATIFICADO BALANCEADO
    print(f"\n🎯 Aplicando muestreo estratificado balanceado...")
    datos_balanceados = muestreo_estratificado_balanceado(datos, 'CATEGORIA_POB', n_muestra=2000, min_por_clase=100)
    
    print(f"📝 Muestra balanceada: {len(datos_balanceados):,} registros")
    print(f"📈 Nueva distribución:")
    for categoria, count in datos_balanceados['CATEGORIA_POB'].value_counts().items():
        print(f"   {categoria:12}: {count:,} ({count/len(datos_balanceados)*100:.1f}%)")
    
    # 5. PREPARAR DATOS FINALES
    datos_limpios = datos_balanceados[variables_disponibles + ['CATEGORIA_POB']].dropna()
    X = datos_limpios[variables_disponibles].values
    y = datos_limpios['CATEGORIA_POB'].values
    
    print(f"\n🧹 Datos finales: {len(datos_limpios):,} registros")
    
    # 6. DIVIDIR DATOS
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        print(f"📊 Entrenamiento: {len(X_train):,} | Prueba: {len(X_test):,}")
    except Exception as e:
        print(f"⚠️ Error en división estratificada: {e}")
        # División simple sin estratificación
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        print(f"📊 División simple - Entrenamiento: {len(X_train):,} | Prueba: {len(X_test):,}")
    
    print()
    
    # 7. ENTRENAR CLASIFICADOR BORROSO
    print("🌫️ ENTRENANDO CLASIFICADOR BORROSO...")
    
    try:
        clasificador = ClasificadorBorroso(n_conjuntos=3)
        clasificador.fit(X_train, y_train, variables_disponibles)
        
        print(f"   ✅ Conjuntos borrosos creados")
        print(f"   📏 Reglas generadas: {len(clasificador.reglas)}")
        
        # 8. REALIZAR PREDICCIONES
        y_pred = clasificador.predict(X_test, variables_disponibles)
        precision = accuracy_score(y_test, y_pred)
        
        print(f"   🎯 Precisión: {precision:.3f} ({precision*100:.1f}%)")
        
    except Exception as e:
        print(f"❌ Error en entrenamiento: {e}")
        return
    
    # 9. MOSTRAR REGLAS GENERADAS
    print()
    print("📋 REGLAS BORROSAS GENERADAS:")
    print("-" * 40)
    
    for i, regla in enumerate(clasificador.reglas[:5], 1):  # Mostrar solo las primeras 5
        print(f"\nRegla #{i}: {regla['conclusion']}")
        print(f"   Confianza: {regla['confianza']:.3f}")
        print(f"   Muestras: {regla['muestras']}")
        print("   Condiciones:")
        for variable, (conjunto, membresia) in regla['condiciones'].items():
            print(f"      {variable} es {conjunto.split('_')[1]} (grado: {membresia:.3f})")
    
    # 10. ANÁLISIS DETALLADO
    print()
    print("📊 ANÁLISIS DETALLADO:")
    
    try:
        reporte = classification_report(y_test, y_pred, output_dict=True)
        print("\n🎯 Métricas por Categoría:")
        for categoria in np.unique(y):
            if categoria in reporte:
                prec = reporte[categoria]['precision']
                rec = reporte[categoria]['recall']
                f1 = reporte[categoria]['f1-score']
                support = reporte[categoria]['support']
                print(f"   {categoria:12}: Prec={prec:.3f} | Rec={rec:.3f} | F1={f1:.3f} | N={support}")
    except Exception as e:
        print(f"   ⚠️ Error en reporte: {e}")
    
    # 11. VISUALIZACIONES
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Gráfico 1: Distribución de datos
        axes[0,0].pie(datos_balanceados['CATEGORIA_POB'].value_counts().values,
                     labels=datos_balanceados['CATEGORIA_POB'].value_counts().index,
                     autopct='%1.1f%%', startangle=90)
        axes[0,0].set_title('📊 Distribución de Categorías', fontweight='bold')
        
        # Gráfico 2: Matriz de confusión
        try:
            cm = confusion_matrix(y_test, y_pred)
            clases = np.unique(y)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=clases, yticklabels=clases, ax=axes[0,1])
            axes[0,1].set_title('🎯 Matriz de Confusión', fontweight='bold')
            axes[0,1].set_xlabel('Predicción')
            axes[0,1].set_ylabel('Real')
        except:
            axes[0,1].text(0.5, 0.5, 'Matriz no\ndisponible', ha='center', va='center')
            axes[0,1].set_title('🎯 Matriz de Confusión', fontweight='bold')
        
        # Gráfico 3: Funciones de membresía para primera variable
        if variables_disponibles:
            var_ejemplo = variables_disponibles[0]
            x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
            
            if var_ejemplo in clasificador.conjuntos_borrosos:
                for nombre_conjunto, params in clasificador.conjuntos_borrosos[var_ejemplo].items():
                    y_vals = clasificador._calcular_membresia(x_vals, params)
                    axes[1,0].plot(x_vals, y_vals, label=nombre_conjunto.split('_')[1], linewidth=2)
                
                axes[1,0].set_title(f'📈 Funciones Membresía\n{var_ejemplo}', fontweight='bold')
                axes[1,0].set_xlabel('Valor')
                axes[1,0].set_ylabel('Grado de Membresía')
                axes[1,0].legend()
                axes[1,0].grid(True, alpha=0.3)
        
        # Gráfico 4: Número de reglas por clase
        reglas_por_clase = {}
        for regla in clasificador.reglas:
            clase = regla['conclusion']
            reglas_por_clase[clase] = reglas_por_clase.get(clase, 0) + 1
        
        if reglas_por_clase:
            clases_reglas = list(reglas_por_clase.keys())
            conteo_reglas = list(reglas_por_clase.values())
            axes[1,1].bar(clases_reglas, conteo_reglas, color='lightcoral')
            axes[1,1].set_title('📏 Reglas por Clase', fontweight='bold')
            axes[1,1].set_ylabel('Número de Reglas')
            axes[1,1].tick_params(axis='x', rotation=45)
            
            # Añadir valores en las barras
            for i, valor in enumerate(conteo_reglas):
                axes[1,1].text(i, valor + 0.1, str(valor), ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('/home/sedc/Proyectos/MineriaDeDatos/results/graficos/logica_borrosa_clasificacion.png', 
                   dpi=150, bbox_inches='tight')
        plt.show()
        
        print("💾 Gráficos guardados: results/graficos/logica_borrosa_clasificacion.png")
        
    except Exception as e:
        print(f"⚠️ Error en visualizaciones: {e}")
    
    # 12. GUARDAR RESULTADOS
    try:
        import pickle
        
        # Guardar modelo
        with open('/home/sedc/Proyectos/MineriaDeDatos/results/modelos/mejor_modelo_borroso.pkl', 'wb') as f:
            pickle.dump(clasificador, f)
        
        # Crear reporte
        reporte_texto = f"""
REPORTE LÓGICA BORROSA - CLASIFICACIÓN
====================================

RESULTADOS:
- Precisión: {precision:.3f} ({precision*100:.1f}%)
- Reglas generadas: {len(clasificador.reglas)}
- Variables utilizadas: {len(variables_disponibles)}

DATOS PROCESADOS:
- Registros totales: {len(datos):,}
- Muestra balanceada: {len(datos_balanceados):,}
- Entrenamiento: {len(X_train):,}
- Prueba: {len(X_test):,}

CONFIGURACIÓN BORROSA:
- Conjuntos por variable: {clasificador.n_conjuntos}
- Función de membresía: Triangular
- Operador lógico: Mínimo (AND)

REGLAS PRINCIPALES:
"""
        
        for i, regla in enumerate(clasificador.reglas[:3], 1):
            reporte_texto += f"\nRegla {i}: {regla['conclusion']} (Confianza: {regla['confianza']:.3f})\n"
            for variable, (conjunto, membresia) in regla['condiciones'].items():
                reporte_texto += f"  - {variable} es {conjunto.split('_')[1]}\n"
        
        with open('/home/sedc/Proyectos/MineriaDeDatos/results/reportes/logica_borrosa_reporte.txt', 'w', encoding='utf-8') as f:
            f.write(reporte_texto)
        
        print("💾 Modelo guardado: results/modelos/mejor_modelo_borroso.pkl")
        print("📄 Reporte guardado: results/reportes/logica_borrosa_reporte.txt")
        
    except Exception as e:
        print(f"⚠️ Error guardando: {e}")
    
    # 13. RESUMEN FINAL
    print()
    print("📝 RESUMEN LÓGICA BORROSA:")
    print(f"   • Precisión: {precision*100:.1f}%")
    print(f"   • Reglas interpretables: {len(clasificador.reglas)}")
    print(f"   • Muestreo balanceado aplicado correctamente")
    
    if precision > 0.7:
        print("   🎉 ¡Buena clasificación borrosa!")
    else:
        print("   🔧 Clasificación moderada, ajustar parámetros")
    
    print("✅ LÓGICA BORROSA COMPLETADA")
    
    return {
        'precision': precision,
        'reglas': len(clasificador.reglas),
        'modelo': clasificador
    }

if __name__ == "__main__":
    ejecutar_logica_borrosa()