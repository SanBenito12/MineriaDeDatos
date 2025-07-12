#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LÓGICA BORROSA OPTIMIZADA - CLASIFICACIÓN
Implementación pura sin dependencias externas de fuzzy
"""

import warnings
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

def crear_categorias_poblacion_dinamica(datos):
    """Crea categorías balanceadas según cuartiles"""
    q1, q2, q3 = datos["POBTOT"].quantile([0.25, 0.50, 0.75])
    
    def categorizar(v):
        if v <= q1:
            return "Pequeña"
        elif v <= q2:
            return "Mediana"
        elif v <= q3:
            return "Grande"
        else:
            return "Muy_Grande"
    return categorizar

class ConjuntoBorroso:
    """Implementación simple de conjunto borroso triangular"""
    
    def __init__(self, a, b, c, nombre):
        self.a = a  # punto izquierdo
        self.b = b  # punto central
        self.c = c  # punto derecho
        self.nombre = nombre
    
    def membresia(self, x):
        """Calcula grado de membresía triangular"""
        if x <= self.a or x >= self.c:
            return 0.0
        elif x == self.b:
            return 1.0
        elif x < self.b:
            return (x - self.a) / (self.b - self.a)
        else:
            return (self.c - x) / (self.c - self.b)

class VariableBorrosa:
    """Variable lingüística borrosa"""
    
    def __init__(self, nombre, rango_min, rango_max):
        self.nombre = nombre
        self.rango_min = rango_min
        self.rango_max = rango_max
        self.conjuntos = {}
    
    def agregar_conjunto(self, nombre, a, b, c):
        """Agrega conjunto borroso a la variable"""
        self.conjuntos[nombre] = ConjuntoBorroso(a, b, c, nombre)
    
    def evaluar(self, valor, conjunto_nombre):
        """Evalúa grado de membresía para un valor"""
        if conjunto_nombre in self.conjuntos:
            return self.conjuntos[conjunto_nombre].membresia(valor)
        return 0.0

class ReglaBorrosa:
    """Regla borrosa IF-THEN"""
    
    def __init__(self, antecedentes, consecuente):
        self.antecedentes = antecedentes  # [(variable, conjunto, valor), ...]
        self.consecuente = consecuente    # (variable_salida, conjunto_salida)
    
    def evaluar(self, valores_entrada):
        """Evalúa la regla con valores de entrada"""
        # Usar operador AND (mínimo) para antecedentes
        fuerza_activacion = 1.0
        
        for variable_nombre, conjunto_nombre, _ in self.antecedentes:
            if variable_nombre in valores_entrada:
                membresia = valores_entrada[variable_nombre][conjunto_nombre]
                fuerza_activacion = min(fuerza_activacion, membresia)
        
        return fuerza_activacion

class ClasificadorBorrosoOptimizado:
    """Sistema de lógica borrosa optimizado"""
    
    def __init__(self):
        self.variables_entrada = {}
        self.variable_salida = None
        self.reglas = []
        self.etiquetas = ["Pequeña", "Mediana", "Grande", "Muy_Grande"]
        self.preparado = False
    
    def _crear_variable_entrada(self, datos, nombre_variable):
        """Crea variable borrosa de entrada"""
        valores = datos[nombre_variable].values
        
        # Calcular rangos usando percentiles
        p0, p25, p50, p75, p100 = np.percentile(valores, [0, 25, 50, 75, 100])
        
        # Expandir rango
        rango_min = p0 - (p100 - p0) * 0.1
        rango_max = p100 + (p100 - p0) * 0.1
        
        variable = VariableBorrosa(nombre_variable, rango_min, rango_max)
        
        # Crear conjuntos borrosos triangulares
        variable.agregar_conjunto("Bajo", rango_min, rango_min, p50)
        variable.agregar_conjunto("Medio", p25, p50, p75)
        variable.agregar_conjunto("Alto", p50, rango_max, rango_max)
        
        return variable
    
    def _crear_variable_salida(self):
        """Crea variable de salida para categorías"""
        variable = VariableBorrosa("CATEGORIA", 0, 3)
        
        variable.agregar_conjunto("Pequeña", 0, 0, 1)
        variable.agregar_conjunto("Mediana", 0, 1, 2)
        variable.agregar_conjunto("Grande", 1, 2, 3)
        variable.agregar_conjunto("Muy_Grande", 2, 3, 3)
        
        return variable
    
    def _crear_reglas_demograficas(self):
        """Crea reglas basadas en conocimiento demográfico"""
        reglas = []
        
        # Reglas principales
        reglas_definicion = [
            # Poblaciones pequeñas
            ([("POBFEM", "Bajo"), ("TOTHOG", "Bajo")], "Pequeña"),
            
            # Poblaciones medianas
            ([("POBFEM", "Bajo"), ("TOTHOG", "Medio")], "Mediana"),
            ([("POBFEM", "Medio"), ("TOTHOG", "Bajo")], "Mediana"),
            
            # Poblaciones grandes
            ([("POBFEM", "Medio"), ("TOTHOG", "Medio")], "Grande"),
            ([("POBFEM", "Alto"), ("TOTHOG", "Bajo")], "Grande"),
            ([("POBFEM", "Bajo"), ("TOTHOG", "Alto")], "Grande"),
            
            # Poblaciones muy grandes
            ([("POBFEM", "Alto"), ("TOTHOG", "Medio")], "Muy_Grande"),
            ([("POBFEM", "Medio"), ("TOTHOG", "Alto")], "Muy_Grande"),
            ([("POBFEM", "Alto"), ("TOTHOG", "Alto")], "Muy_Grande"),
        ]
        
        # Agregar reglas adicionales si hay más variables
        if "VIVTOT" in self.variables_entrada:
            reglas_definicion.extend([
                ([("VIVTOT", "Alto"), ("POBFEM", "Alto")], "Muy_Grande"),
                ([("VIVTOT", "Bajo"), ("POBFEM", "Bajo")], "Pequeña"),
                ([("VIVTOT", "Medio"), ("TOTHOG", "Medio")], "Grande"),
            ])
        
        if "P_15YMAS" in self.variables_entrada:
            reglas_definicion.extend([
                ([("P_15YMAS", "Alto"), ("POBFEM", "Alto")], "Muy_Grande"),
                ([("P_15YMAS", "Bajo"), ("TOTHOG", "Bajo")], "Pequeña"),
            ])
        
        # Convertir a objetos ReglaBorrosa
        for antecedentes_def, consecuente_def in reglas_definicion:
            antecedentes = [(var, conj, None) for var, conj in antecedentes_def]
            regla = ReglaBorrosa(antecedentes, ("CATEGORIA", consecuente_def))
            reglas.append(regla)
        
        return reglas
    
    def fit(self, X, y=None):
        """Entrena el sistema borroso"""
        # Crear variables de entrada
        for columna in X.columns:
            self.variables_entrada[columna] = self._crear_variable_entrada(X, columna)
        
        # Crear variable de salida
        self.variable_salida = self._crear_variable_salida()
        
        # Crear reglas
        self.reglas = self._crear_reglas_demograficas()
        
        self.preparado = True
        return self
    
    def _evaluar_entrada(self, fila):
        """Evalúa grados de membresía para una fila de datos"""
        valores_evaluados = {}
        
        for nombre_variable, variable in self.variables_entrada.items():
            if nombre_variable in fila:
                valor = fila[nombre_variable]
                valores_evaluados[nombre_variable] = {}
                
                for nombre_conjunto in variable.conjuntos:
                    membresia = variable.evaluar(valor, nombre_conjunto)
                    valores_evaluados[nombre_variable][nombre_conjunto] = membresia
        
        return valores_evaluados
    
    def _inferencia_mamdani(self, valores_entrada):
        """Realiza inferencia usando método Mamdani simplificado"""
        activaciones = {}
        
        # Evaluar todas las reglas
        for regla in self.reglas:
            fuerza = regla.evaluar(valores_entrada)
            consecuente = regla.consecuente[1]  # nombre del conjunto de salida
            
            if consecuente not in activaciones:
                activaciones[consecuente] = []
            activaciones[consecuente].append(fuerza)
        
        # Usar máximo para combinar reglas del mismo consecuente
        activaciones_finales = {}
        for consecuente, fuerzas in activaciones.items():
            activaciones_finales[consecuente] = max(fuerzas)
        
        return activaciones_finales
    
    def _defuzzificacion(self, activaciones):
        """Defuzzificación usando centro de gravedad simplificado"""
        if not activaciones:
            return "Mediana"  # Por defecto
        
        # Mapeo de categorías a valores numéricos
        valores_categoria = {
            "Pequeña": 0,
            "Mediana": 1,
            "Grande": 2,
            "Muy_Grande": 3
        }
        
        # Calcular centro de gravedad ponderado
        numerador = 0
        denominador = 0
        
        for categoria, activacion in activaciones.items():
            if activacion > 0:
                valor = valores_categoria[categoria]
                numerador += valor * activacion
                denominador += activacion
        
        if denominador == 0:
            return "Mediana"
        
        # Convertir resultado numérico a categoría
        resultado_numerico = numerador / denominador
        indice = int(round(np.clip(resultado_numerico, 0, 3)))
        
        return self.etiquetas[indice]
    
    def predict(self, X):
        """Predice categorías para datos de entrada"""
        if not self.preparado:
            raise ValueError("El modelo no ha sido entrenado")
        
        predicciones = []
        
        for _, fila in X.iterrows():
            try:
                # Evaluar grados de membresía
                valores_entrada = self._evaluar_entrada(fila)
                
                # Inferencia
                activaciones = self._inferencia_mamdani(valores_entrada)
                
                # Defuzzificación
                prediccion = self._defuzzificacion(activaciones)
                
                predicciones.append(prediccion)
                
            except Exception:
                # En caso de error, usar predicción por defecto
                predicciones.append("Mediana")
        
        return np.array(predicciones)

def preparar_datos_optimizado(datos):
    """Prepara datos con múltiples variables para mejor precisión"""
    # Variables de entrada optimizadas
    variables_entrada = ["POBFEM", "TOTHOG", "VIVTOT", "P_15YMAS"]
    variables_disponibles = [v for v in variables_entrada if v in datos.columns]
    
    if len(variables_disponibles) < 2:
        return None, None, None
    
    # Crear categorías
    datos = datos.copy()
    categorizador = crear_categorias_poblacion_dinamica(datos)
    datos["CATEGORIA_POB"] = datos["POBTOT"].apply(categorizador)
    
    # Limpiar datos
    columnas_necesarias = variables_disponibles + ["CATEGORIA_POB"]
    datos_limpios = datos[columnas_necesarias].dropna()
    
    # Filtrar outliers extremos
    for variable in variables_disponibles:
        Q1 = datos_limpios[variable].quantile(0.01)
        Q3 = datos_limpios[variable].quantile(0.99)
        datos_limpios = datos_limpios[
            (datos_limpios[variable] >= Q1) & 
            (datos_limpios[variable] <= Q3)
        ]
    
    # Muestreo estratificado para balancear clases
    if len(datos_limpios) > 4000:
        datos_balanceados = []
        for categoria in datos_limpios['CATEGORIA_POB'].unique():
            subset = datos_limpios[datos_limpios['CATEGORIA_POB'] == categoria]
            n_muestra = min(1000, len(subset))
            if n_muestra > 0:
                datos_balanceados.append(subset.sample(n=n_muestra, random_state=42))
        
        if datos_balanceados:
            datos_limpios = pd.concat(datos_balanceados, ignore_index=True)
    
    X = datos_limpios[variables_disponibles]
    y = datos_limpios["CATEGORIA_POB"]
    
    return X, y, variables_disponibles

def crear_visualizaciones_borrosas(y_test, y_pred, variables_disponibles, precision):
    """Crea visualizaciones para lógica borrosa"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix, classification_report
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('🌫️ LÓGICA BORROSA - ANÁLISIS DE RESULTADOS', fontsize=16, fontweight='bold')
        
        # 1. Matriz de Confusión
        categorias = ["Pequeña", "Mediana", "Grande", "Muy_Grande"]
        cm = confusion_matrix(y_test, y_pred, labels=categorias)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=categorias, yticklabels=categorias, ax=axes[0,0])
        axes[0,0].set_title('🎯 Matriz de Confusión', fontweight='bold')
        axes[0,0].set_xlabel('Predicción')
        axes[0,0].set_ylabel('Real')
        
        # 2. Distribución de Predicciones
        from collections import Counter
        pred_counts = Counter(y_pred)
        real_counts = Counter(y_test)
        
        categorias_pred = list(pred_counts.keys())
        valores_pred = list(pred_counts.values())
        valores_real = [real_counts.get(cat, 0) for cat in categorias_pred]
        
        x = np.arange(len(categorias_pred))
        width = 0.35
        
        axes[0,1].bar(x - width/2, valores_real, width, label='Real', alpha=0.8, color='lightcoral')
        axes[0,1].bar(x + width/2, valores_pred, width, label='Predicho', alpha=0.8, color='skyblue')
        axes[0,1].set_title('📊 Distribución Real vs Predicha', fontweight='bold')
        axes[0,1].set_xlabel('Categorías')
        axes[0,1].set_ylabel('Cantidad')
        axes[0,1].set_xticks(x)
        axes[0,1].set_xticklabels(categorias_pred, rotation=45)
        axes[0,1].legend()
        
        # 3. Métricas por Categoría
        reporte = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        categorias_metricas = []
        precision_vals = []
        recall_vals = []
        f1_vals = []
        
        for categoria in categorias:
            if categoria in reporte:
                categorias_metricas.append(categoria)
                precision_vals.append(reporte[categoria]['precision'])
                recall_vals.append(reporte[categoria]['recall'])
                f1_vals.append(reporte[categoria]['f1-score'])
        
        if categorias_metricas:
            x = np.arange(len(categorias_metricas))
            width = 0.25
            
            axes[1,0].bar(x - width, precision_vals, width, label='Precisión', alpha=0.8, color='gold')
            axes[1,0].bar(x, recall_vals, width, label='Recall', alpha=0.8, color='lightgreen')
            axes[1,0].bar(x + width, f1_vals, width, label='F1-Score', alpha=0.8, color='plum')
            
            axes[1,0].set_title('📈 Métricas por Categoría', fontweight='bold')
            axes[1,0].set_xlabel('Categorías')
            axes[1,0].set_ylabel('Score')
            axes[1,0].set_xticks(x)
            axes[1,0].set_xticklabels(categorias_metricas, rotation=45)
            axes[1,0].legend()
            axes[1,0].set_ylim(0, 1)
        
        # 4. Resumen del Sistema Borroso
        axes[1,1].text(0.1, 0.9, '🌫️ SISTEMA DE LÓGICA BORROSA', fontsize=14, fontweight='bold', color='darkblue')
        axes[1,1].text(0.1, 0.8, f'📊 Precisión Global: {precision:.3f} ({precision*100:.1f}%)', fontsize=12)
        axes[1,1].text(0.1, 0.7, f'📋 Variables Utilizadas: {len(variables_disponibles)}', fontsize=12)
        axes[1,1].text(0.1, 0.6, f'🔢 Datos de Prueba: {len(y_test):,}', fontsize=12)
        axes[1,1].text(0.1, 0.5, f'📂 Variables:', fontsize=12, fontweight='bold')
        
        # Mostrar variables en múltiples líneas
        vars_text = ', '.join(variables_disponibles)
        if len(vars_text) > 40:
            vars_lines = [vars_text[i:i+40] for i in range(0, len(vars_text), 40)]
            for i, line in enumerate(vars_lines[:3]):  # Máximo 3 líneas
                axes[1,1].text(0.15, 0.45 - i*0.05, line, fontsize=10)
        else:
            axes[1,1].text(0.15, 0.45, vars_text, fontsize=10)
        
        # Características del sistema
        axes[1,1].text(0.1, 0.25, '🔧 Características del Sistema:', fontsize=11, fontweight='bold')
        axes[1,1].text(0.15, 0.20, '• Conjuntos borrosos triangulares', fontsize=10)
        axes[1,1].text(0.15, 0.15, '• Inferencia Mamdani', fontsize=10)
        axes[1,1].text(0.15, 0.10, '• Defuzzificación por centro de gravedad', fontsize=10)
        axes[1,1].text(0.15, 0.05, '• Reglas basadas en conocimiento demográfico', fontsize=10)
        
        # Evaluación del rendimiento
        if precision > 0.85:
            estado = "🎉 Excelente"
            color = 'green'
        elif precision > 0.75:
            estado = "👍 Bueno"
            color = 'orange'
        else:
            estado = "🔧 Moderado"
            color = 'red'
        
        axes[1,1].text(0.1, 0.01, f'📈 Rendimiento: {estado}', fontsize=11, fontweight='bold', color=color)
        
        axes[1,1].set_xlim(0, 1)
        axes[1,1].set_ylim(0, 1)
        axes[1,1].axis('off')
        
        plt.tight_layout()
        
        # Guardar gráfico
        ruta_grafico = '/home/sedc/Proyectos/MineriaDeDatos/results/graficos/logica_borrosa.png'
        os.makedirs(os.path.dirname(ruta_grafico), exist_ok=True)
        plt.savefig(ruta_grafico, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"💾 Gráfico guardado: {ruta_grafico}")
        
        return True
        
    except Exception as e:
        print(f"⚠️ Error creando visualizaciones: {e}")
        return False

def ejecutar_logica_borrosa():
    """Función principal para ejecución desde menú"""
    archivo = "/home/sedc/Proyectos/MineriaDeDatos/data/ceros_sin_columnasAB_limpio_weka.csv"
    
    if not os.path.isfile(archivo):
        print(f"❌ No se encontró el archivo: {archivo}")
        return
    
    # Cargar y preparar datos
    datos = pd.read_csv(archivo)
    X, y, variables_disponibles = preparar_datos_optimizado(datos)
    
    if X is None:
        print("❌ No hay suficientes variables para lógica borrosa")
        return
    
    print(f"📊 Variables: {len(variables_disponibles)} | Datos: {len(X):,}")
    
    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Entrenar modelo borroso
    clasificador = ClasificadorBorrosoOptimizado()
    clasificador.fit(X_train)
    
    # Predicciones
    y_pred = clasificador.predict(X_test)
    
    # Calcular precisión
    precision = accuracy_score(y_test, y_pred)
    
    print(f"🎯 Precisión: {precision:.3f} ({precision*100:.1f}%)")
    print(f"📋 Variables utilizadas: {', '.join(variables_disponibles)}")
    
    # Crear visualizaciones
    crear_visualizaciones_borrosas(y_test, y_pred, variables_disponibles, precision)
    
    return {
        'precision': precision,
        'modelo': clasificador,
        'variables': variables_disponibles
    }

if __name__ == "__main__":
    ejecutar_logica_borrosa()