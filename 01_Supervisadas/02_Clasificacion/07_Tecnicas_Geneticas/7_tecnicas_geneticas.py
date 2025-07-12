#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TÉCNICAS GENÉTICAS OPTIMIZADAS - CLASIFICACIÓN
Optimización evolutiva para seleccionar las mejores variables
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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

class AlgoritmoGeneticoOptimizado:
    """Algoritmo genético optimizado para selección de variables"""
    
    def __init__(self, tamaño_poblacion=20, n_generaciones=15, prob_mutacion=0.2):
        self.tamaño_poblacion = tamaño_poblacion
        self.n_generaciones = n_generaciones
        self.prob_mutacion = prob_mutacion
        self.historial_fitness = []
        self.mejor_individuo_global = None
        self.mejor_fitness_global = 0
        
    def crear_individuo(self, n_variables):
        """Crea un individuo (combinación de variables)"""
        # Crear individuo con 3-6 variables seleccionadas
        n_seleccionadas = np.random.randint(3, min(7, n_variables + 1))
        individuo = np.zeros(n_variables, dtype=int)
        indices = np.random.choice(n_variables, n_seleccionadas, replace=False)
        individuo[indices] = 1
        return individuo
    
    def calcular_fitness(self, individuo, X_train, X_test, y_train, y_test):
        """Calcula qué tan buena es una combinación de variables"""
        variables_seleccionadas = individuo == 1
        n_variables = np.sum(variables_seleccionadas)
        
        if n_variables < 2:
            return 0.0
        
        try:
            # Entrenar con variables seleccionadas
            X_train_sel = X_train[:, variables_seleccionadas]
            X_test_sel = X_test[:, variables_seleccionadas]
            
            clf = DecisionTreeClassifier(max_depth=6, random_state=42)
            clf.fit(X_train_sel, y_train)
            y_pred = clf.predict(X_test_sel)
            
            precision = accuracy_score(y_test, y_pred)
            
            # Bonus por usar pocas variables (simplicidad)
            bonus_simplicidad = (1 - n_variables / len(individuo)) * 0.05
            
            fitness = precision + bonus_simplicidad
            return min(1.0, fitness)
            
        except:
            return 0.0
    
    def seleccion_torneo(self, poblacion, fitness_scores):
        """Selecciona padres usando torneo"""
        indices = np.random.choice(len(poblacion), 3, replace=False)
        mejor_idx = indices[np.argmax([fitness_scores[i] for i in indices])]
        return poblacion[mejor_idx].copy()
    
    def cruzar(self, padre1, padre2):
        """Cruza dos padres para crear un hijo"""
        # Cruce uniforme
        hijo = np.zeros(len(padre1), dtype=int)
        for i in range(len(padre1)):
            if np.random.random() < 0.5:
                hijo[i] = padre1[i]
            else:
                hijo[i] = padre2[i]
        
        # Asegurar mínimo 2 variables
        if np.sum(hijo) < 2:
            indices = np.random.choice(len(hijo), 2, replace=False)
            hijo[indices] = 1
            
        return hijo
    
    def mutar(self, individuo):
        """Aplica mutación a un individuo"""
        individuo_mutado = individuo.copy()
        
        for i in range(len(individuo_mutado)):
            if np.random.random() < self.prob_mutacion:
                individuo_mutado[i] = 1 - individuo_mutado[i]
        
        # Asegurar mínimo 2 variables
        if np.sum(individuo_mutado) < 2:
            indices = np.random.choice(len(individuo_mutado), 2, replace=False)
            individuo_mutado[indices] = 1
            
        return individuo_mutado
    
    def evolucionar(self, X_train, X_test, y_train, y_test, nombres_variables):
        """Ejecuta la evolución genética"""
        n_variables = len(nombres_variables)
        
        # Crear población inicial
        poblacion = [self.crear_individuo(n_variables) for _ in range(self.tamaño_poblacion)]
        
        for generacion in range(self.n_generaciones):
            # Evaluar fitness de toda la población
            fitness_scores = [
                self.calcular_fitness(ind, X_train, X_test, y_train, y_test) 
                for ind in poblacion
            ]
            
            # Guardar el mejor
            mejor_idx = np.argmax(fitness_scores)
            mejor_fitness_actual = fitness_scores[mejor_idx]
            
            if mejor_fitness_actual > self.mejor_fitness_global:
                self.mejor_fitness_global = mejor_fitness_actual
                self.mejor_individuo_global = poblacion[mejor_idx].copy()
            
            self.historial_fitness.append(mejor_fitness_actual)
            
            # Crear nueva generación
            nueva_poblacion = []
            
            # Elitismo: conservar mejor individuo
            nueva_poblacion.append(poblacion[mejor_idx].copy())
            
            # Generar resto de la población
            while len(nueva_poblacion) < self.tamaño_poblacion:
                padre1 = self.seleccion_torneo(poblacion, fitness_scores)
                padre2 = self.seleccion_torneo(poblacion, fitness_scores)
                
                hijo = self.cruzar(padre1, padre2)
                hijo = self.mutar(hijo)
                
                nueva_poblacion.append(hijo)
            
            poblacion = nueva_poblacion
        
        return self.mejor_individuo_global, self.mejor_fitness_global

def preparar_datos_geneticos(datos):
    """Prepara datos para algoritmo genético"""
    variables = ['POBFEM', 'POBMAS', 'TOTHOG', 'VIVTOT', 'P_15YMAS', 'P_60YMAS', 'GRAPROES', 'PEA', 'POCUPADA']
    variables_disponibles = [v for v in variables if v in datos.columns]
    
    # Crear categorías
    categorizador = crear_categorias_poblacion_dinamica(datos)
    datos['CATEGORIA_POB'] = datos['POBTOT'].apply(categorizador)
    
    # Limpiar datos
    datos_limpios = datos[variables_disponibles + ['CATEGORIA_POB']].dropna()
    
    # Muestreo para eficiencia
    if len(datos_limpios) > 3000:
        datos_limpios = datos_limpios.sample(n=3000, random_state=42)
    
    X = datos_limpios[variables_disponibles]
    y = datos_limpios['CATEGORIA_POB']
    
    return X, y, variables_disponibles

def crear_visualizaciones_geneticas(resultado_genetico, y_test, y_pred_evolutivo, y_pred_base, variables_disponibles):
    """Crea visualizaciones para técnicas genéticas"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('🧬 TÉCNICAS GENÉTICAS - OPTIMIZACIÓN EVOLUTIVA', fontsize=16, fontweight='bold')
        
        # 1. Evolución del Fitness
        generaciones = range(1, len(resultado_genetico['historial_fitness']) + 1)
        axes[0,0].plot(generaciones, resultado_genetico['historial_fitness'], 'b-', linewidth=2, marker='o')
        axes[0,0].set_title('📈 Evolución del Fitness', fontweight='bold')
        axes[0,0].set_xlabel('Generación')
        axes[0,0].set_ylabel('Fitness (Aptitud)')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].set_ylim(0, 1)
        
        # Marcar mejor fitness
        mejor_gen = np.argmax(resultado_genetico['historial_fitness'])
        mejor_fitness = max(resultado_genetico['historial_fitness'])
        axes[0,0].scatter([mejor_gen + 1], [mejor_fitness], color='red', s=100, zorder=5)
        axes[0,0].annotate(f'Mejor: {mejor_fitness:.3f}', 
                          xy=(mejor_gen + 1, mejor_fitness), 
                          xytext=(10, 10), textcoords='offset points',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        # 2. Variables Seleccionadas vs Todas
        mejor_individuo = resultado_genetico['mejor_individuo']
        colores = ['green' if x == 1 else 'lightcoral' for x in mejor_individuo]
        
        axes[0,1].bar(range(len(variables_disponibles)), mejor_individuo, color=colores)
        axes[0,1].set_title('🔬 Variables Seleccionadas por Evolución', fontweight='bold')
        axes[0,1].set_xlabel('Variables')
        axes[0,1].set_ylabel('Seleccionada (1) / Descartada (0)')
        axes[0,1].set_xticks(range(len(variables_disponibles)))
        axes[0,1].set_xticklabels([v[:6] for v in variables_disponibles], rotation=45)
        
        # Añadir leyenda
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', label='Seleccionada'),
                          Patch(facecolor='lightcoral', label='Descartada')]
        axes[0,1].legend(handles=legend_elements)
        
        # 3. Comparación de Rendimiento
        modelos = ['Todas las Variables', 'Variables Evolutivas']
        precisiones = [resultado_genetico['precision_base'], resultado_genetico['precision_evolutiva']]
        n_vars = [len(variables_disponibles), np.sum(mejor_individuo)]
        
        x = np.arange(len(modelos))
        width = 0.35
        
        axes[1,0].bar(x - width/2, precisiones, width, label='Precisión', color='skyblue', alpha=0.8)
        axes[1,0].bar(x + width/2, [n/max(n_vars) for n in n_vars], width, 
                     label='Variables (normalizado)', color='orange', alpha=0.8)
        
        axes[1,0].set_title('⚖️ Comparación: Todas vs Evolutivas', fontweight='bold')
        axes[1,0].set_ylabel('Valor')
        axes[1,0].set_xticks(x)
        axes[1,0].set_xticklabels(modelos)
        axes[1,0].legend()
        axes[1,0].set_ylim(0, 1.1)
        
        # Añadir valores en las barras
        for i, (prec, n_var) in enumerate(zip(precisiones, n_vars)):
            axes[1,0].text(i - width/2, prec + 0.02, f'{prec:.3f}', ha='center', fontweight='bold')
            axes[1,0].text(i + width/2, (n_var/max(n_vars)) + 0.02, str(n_var), ha='center', fontweight='bold')
        
        # 4. Resumen y Resultados
        axes[1,1].text(0.1, 0.9, '🧬 ALGORITMO GENÉTICO', fontsize=14, fontweight='bold', color='darkblue')
        axes[1,1].text(0.1, 0.8, f'🎯 Fitness Final: {mejor_fitness:.3f}', fontsize=12)
        axes[1,1].text(0.1, 0.7, f'📊 Precisión Evolutiva: {resultado_genetico["precision_evolutiva"]:.3f}', fontsize=12)
        axes[1,1].text(0.1, 0.6, f'📊 Precisión Base: {resultado_genetico["precision_base"]:.3f}', fontsize=12)
        
        variables_seleccionadas = [var for var, sel in zip(variables_disponibles, mejor_individuo) if sel]
        axes[1,1].text(0.1, 0.5, f'🔬 Variables Seleccionadas: {len(variables_seleccionadas)}/{len(variables_disponibles)}', fontsize=12)
        
        # Mostrar variables seleccionadas
        vars_text = ', '.join(variables_seleccionadas)
        if len(vars_text) > 35:
            vars_lines = [vars_text[i:i+35] for i in range(0, len(vars_text), 35)]
            for i, line in enumerate(vars_lines[:2]):
                axes[1,1].text(0.15, 0.45 - i*0.05, line, fontsize=10)
        else:
            axes[1,1].text(0.15, 0.45, vars_text, fontsize=10)
        
        # Evaluación del resultado
        mejora = resultado_genetico['precision_evolutiva'] - resultado_genetico['precision_base']
        reduccion_vars = (1 - len(variables_seleccionadas)/len(variables_disponibles)) * 100
        
        axes[1,1].text(0.1, 0.3, f'📈 Reducción de Variables: {reduccion_vars:.1f}%', fontsize=11)
        
        if mejora >= 0:
            axes[1,1].text(0.1, 0.25, f'🎉 Mejora de Precisión: +{mejora:.3f}', fontsize=11, color='green')
            estado = "¡Optimización Exitosa!"
            color_estado = 'green'
        else:
            axes[1,1].text(0.1, 0.25, f'📊 Cambio de Precisión: {mejora:.3f}', fontsize=11, color='orange')
            estado = "Optimización de Eficiencia"
            color_estado = 'orange'
        
        axes[1,1].text(0.1, 0.15, f'✨ Principio: Evolución Natural', fontsize=11)
        axes[1,1].text(0.1, 0.10, f'⚙️ Método: Selección + Cruce + Mutación', fontsize=11)
        axes[1,1].text(0.1, 0.05, f'🏆 Resultado: {estado}', fontsize=11, fontweight='bold', color=color_estado)
        
        axes[1,1].set_xlim(0, 1)
        axes[1,1].set_ylim(0, 1)
        axes[1,1].axis('off')
        
        plt.tight_layout()
        
        # Guardar gráfico
        ruta_grafico = '/home/sedc/Proyectos/MineriaDeDatos/results/graficos/tecnicas_geneticas.png'
        os.makedirs(os.path.dirname(ruta_grafico), exist_ok=True)
        plt.savefig(ruta_grafico, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"💾 Gráfico guardado: {ruta_grafico}")
        
        return True
        
    except Exception as e:
        print(f"⚠️ Error creando visualizaciones: {e}")
        return False

def ejecutar_tecnicas_geneticas():
    """Función principal para ejecución desde menú"""
    archivo = '/home/sedc/Proyectos/MineriaDeDatos/data/ceros_sin_columnasAB_limpio_weka.csv'
    
    if not os.path.isfile(archivo):
        print(f"❌ No se encontró el archivo: {archivo}")
        return
    
    # Cargar y preparar datos
    datos = pd.read_csv(archivo)
    X, y, variables_disponibles = preparar_datos_geneticos(datos)
    
    print(f"📊 Variables: {len(variables_disponibles)} | Datos: {len(X):,}")
    
    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Escalar datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Ejecutar algoritmo genético
    algoritmo = AlgoritmoGeneticoOptimizado()
    mejor_individuo, mejor_fitness = algoritmo.evolucionar(
        X_train_scaled, X_test_scaled, y_train, y_test, variables_disponibles
    )
    
    # Evaluar modelos
    # 1. Modelo con variables evolutivas
    variables_seleccionadas = mejor_individuo == 1
    X_train_evolutivo = X_train_scaled[:, variables_seleccionadas]
    X_test_evolutivo = X_test_scaled[:, variables_seleccionadas]
    
    clf_evolutivo = DecisionTreeClassifier(max_depth=8, random_state=42)
    clf_evolutivo.fit(X_train_evolutivo, y_train)
    y_pred_evolutivo = clf_evolutivo.predict(X_test_evolutivo)
    precision_evolutiva = accuracy_score(y_test, y_pred_evolutivo)
    
    # 2. Modelo con todas las variables
    clf_base = DecisionTreeClassifier(max_depth=8, random_state=42)
    clf_base.fit(X_train_scaled, y_train)
    y_pred_base = clf_base.predict(X_test_scaled)
    precision_base = accuracy_score(y_test, y_pred_base)
    
    # Mostrar resultados
    variables_elegidas = [var for var, sel in zip(variables_disponibles, mejor_individuo) if sel]
    
    print(f"🎯 Precisión Evolutiva: {precision_evolutiva:.3f} ({precision_evolutiva*100:.1f}%)")
    print(f"📊 Precisión Base: {precision_base:.3f} ({precision_base*100:.1f}%)")
    print(f"🔬 Variables Seleccionadas: {', '.join(variables_elegidas)}")
    
    # Preparar resultado para visualización
    resultado_genetico = {
        'mejor_individuo': mejor_individuo,
        'historial_fitness': algoritmo.historial_fitness,
        'precision_evolutiva': precision_evolutiva,
        'precision_base': precision_base
    }
    
    # Crear visualizaciones
    crear_visualizaciones_geneticas(resultado_genetico, y_test, y_pred_evolutivo, y_pred_base, variables_disponibles)
    
    return {
        'precision_evolutiva': precision_evolutiva,
        'precision_base': precision_base,
        'variables_seleccionadas': variables_elegidas,
        'mejor_individuo': mejor_individuo
    }

if __name__ == "__main__":
    ejecutar_tecnicas_geneticas()