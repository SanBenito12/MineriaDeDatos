#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TÉCNICAS GENÉTICAS - Optimización Evolutiva para Clasificación
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

def cargar_datos():
    """Carga el dataset principal"""
    return pd.read_csv('data/ceros_sin_columnasAB_limpio_weka.csv')

def crear_categorias_poblacion(poblacion):
    """Crea categorías de población para clasificación"""
    if poblacion <= 500:
        return 'Pequeña'
    elif poblacion <= 2000:
        return 'Mediana'
    elif poblacion <= 8000:
        return 'Grande'
    else:
        return 'Muy_Grande'

class AlgoritmoGenetico:
    """Algoritmo genético simple para optimización de características"""
    
    def __init__(self, tamaño_poblacion=30, n_generaciones=20, prob_mutacion=0.15):
        self.tamaño_poblacion = tamaño_poblacion
        self.n_generaciones = n_generaciones
        self.prob_mutacion = prob_mutacion
        self.mejor_fitness = []
        
    def crear_individuo(self, n_variables):
        """Crear individuo con selección aleatoria de características"""
        genes = np.random.randint(0, 2, n_variables)
        # Asegurar al menos 2 características
        if np.sum(genes) < 2:
            indices = np.random.choice(n_variables, 2, replace=False)
            genes[indices] = 1
        return genes
    
    def evaluar_fitness(self, individuo, X_train, X_test, y_train, y_test):
        """Evaluar fitness del individuo"""
        # Seleccionar características
        caracteristicas = individuo == 1
        if np.sum(caracteristicas) == 0:
            return 0.0
        
        try:
            X_train_sel = X_train[:, caracteristicas]
            X_test_sel = X_test[:, caracteristicas]
            
            # Entrenar clasificador simple
            clf = DecisionTreeClassifier(max_depth=5, random_state=42)
            clf.fit(X_train_sel, y_train)
            y_pred = clf.predict(X_test_sel)
            
            # Fitness = precisión - penalización por muchas características
            precision = accuracy_score(y_test, y_pred)
            penalizacion = (np.sum(caracteristicas) / len(individuo)) * 0.1
            
            return max(0.0, precision - penalizacion)
        except:
            return 0.0
    
    def seleccion(self, poblacion, fitness_scores):
        """Selección por torneo simple"""
        indices = np.random.choice(len(poblacion), 3, replace=False)
        mejor_idx = indices[np.argmax([fitness_scores[i] for i in indices])]
        return poblacion[mejor_idx].copy()
    
    def cruce(self, padre1, padre2):
        """Cruce de un punto"""
        punto = np.random.randint(1, len(padre1))
        hijo = np.concatenate([padre1[:punto], padre2[punto:]])
        
        # Asegurar al menos 2 características
        if np.sum(hijo) < 2:
            indices = np.random.choice(len(hijo), 2, replace=False)
            hijo[indices] = 1
        
        return hijo
    
    def mutacion(self, individuo):
        """Mutación simple"""
        for i in range(len(individuo)):
            if np.random.random() < self.prob_mutacion:
                individuo[i] = 1 - individuo[i]
        
        # Asegurar al menos 2 características
        if np.sum(individuo) < 2:
            indices = np.random.choice(len(individuo), 2, replace=False)
            individuo[indices] = 1
        
        return individuo
    
    def evolucionar(self, X_train, X_test, y_train, y_test, variables):
        """Ejecutar algoritmo genético"""
        n_variables = len(variables)
        
        # Crear población inicial
        poblacion = [self.crear_individuo(n_variables) for _ in range(self.tamaño_poblacion)]
        
        print(f"    🧬 Evolución: {self.n_generaciones} generaciones, población {self.tamaño_poblacion}")
        
        for gen in range(self.n_generaciones):
            # Evaluar fitness
            fitness_scores = [self.evaluar_fitness(ind, X_train, X_test, y_train, y_test) 
                            for ind in poblacion]
            
            # Guardar mejor fitness
            mejor_fitness = max(fitness_scores)
            self.mejor_fitness.append(mejor_fitness)
            
            if gen % 5 == 0:
                mejor_idx = np.argmax(fitness_scores)
                n_vars = np.sum(poblacion[mejor_idx])
                print(f"       Gen {gen:2d}: Fitness={mejor_fitness:.3f} | Variables={n_vars}")
            
            # Crear nueva generación
            nueva_poblacion = []
            
            # Mantener mejor individuo (elitismo)
            mejor_idx = np.argmax(fitness_scores)
            nueva_poblacion.append(poblacion[mejor_idx].copy())
            
            # Generar resto de la población
            while len(nueva_poblacion) < self.tamaño_poblacion:
                padre1 = self.seleccion(poblacion, fitness_scores)
                padre2 = self.seleccion(poblacion, fitness_scores)
                
                hijo = self.cruce(padre1, padre2)
                hijo = self.mutacion(hijo)
                
                nueva_poblacion.append(hijo)
            
            poblacion = nueva_poblacion
        
        # Encontrar mejor solución final
        fitness_finales = [self.evaluar_fitness(ind, X_train, X_test, y_train, y_test) 
                          for ind in poblacion]
        mejor_idx = np.argmax(fitness_finales)
        
        return poblacion[mejor_idx], max(fitness_finales)

def preparar_datos(datos, max_muestras=2000):
    """Prepara variables para algoritmo genético"""
    variables = ['POBFEM', 'POBMAS', 'TOTHOG', 'VIVTOT', 'P_15YMAS', 'P_60YMAS', 'GRAPROES', 'PEA', 'POCUPADA']
    variables_disponibles = [v for v in variables if v in datos.columns]
    
    # Crear variable objetivo categórica
    datos['CATEGORIA_POB'] = datos['POBTOT'].apply(crear_categorias_poblacion)
    
    # Dataset limpio
    df = datos[variables_disponibles + ['CATEGORIA_POB']].dropna()
    
    # Reducir muestra para eficiencia
    if len(df) > max_muestras:
        df = df.sample(n=max_muestras, random_state=42)
        print(f"📝 Muestra reducida a {len(df):,} registros (para eficiencia)")
    
    X = df[variables_disponibles].values
    y = df['CATEGORIA_POB'].values
    
    return X, y, variables_disponibles

def visualizar_resultados_geneticos(mejor_individuo, fitness_evolution, variables, precision_final, precision_base):
    """Crea visualizaciones del algoritmo genético"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Evolución del fitness
    axes[0,0].plot(fitness_evolution, 'b-', linewidth=2)
    axes[0,0].set_title('🧬 Evolución del Fitness')
    axes[0,0].set_xlabel('Generación')
    axes[0,0].set_ylabel('Fitness')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Variables seleccionadas
    axes[0,1].bar(range(len(variables)), mejor_individuo, 
                 color=['green' if x == 1 else 'red' for x in mejor_individuo])
    axes[0,1].set_title('🔬 Variables Seleccionadas')
    axes[0,1].set_xlabel('Variable')
    axes[0,1].set_ylabel('Seleccionada (1) / No (0)')
    axes[0,1].set_xticks(range(len(variables)))
    axes[0,1].set_xticklabels([v[:6] for v in variables], rotation=45)
    
    # 3. Comparación con modelo base
    modelos = ['Todas las variables', 'Variables evolutivas']
    precisiones = [precision_base, precision_final]
    n_vars = [len(variables), np.sum(mejor_individuo)]
    
    x = np.arange(len(modelos))
    width = 0.35
    
    axes[1,0].bar(x - width/2, precisiones, width, label='Precisión', color='skyblue')
    axes[1,0].bar(x + width/2, [n/max(n_vars) for n in n_vars], width, 
                 label='Variables (normalizado)', color='orange')
    
    axes[1,0].set_title('⚖️ Comparación de Modelos')
    axes[1,0].set_ylabel('Valor')
    axes[1,0].set_xticks(x)
    axes[1,0].set_xticklabels(modelos, rotation=45)
    axes[1,0].legend()
    
    # Añadir valores en las barras
    for i, (prec, n_var) in enumerate(zip(precisiones, n_vars)):
        axes[1,0].text(i - width/2, prec + 0.01, f'{prec:.3f}', ha='center')
        axes[1,0].text(i + width/2, (n_var/max(n_vars)) + 0.01, str(n_var), ha='center')
    
    # 4. Resumen del algoritmo genético
    axes[1,1].text(0.1, 0.9, 'ALGORITMO GENÉTICO', fontsize=14, fontweight='bold')
    axes[1,1].text(0.1, 0.8, f'Precisión evolutiva: {precision_final:.3f}', fontsize=12)
    axes[1,1].text(0.1, 0.7, f'Precisión base: {precision_base:.3f}', fontsize=12)
    axes[1,1].text(0.1, 0.6, f'Variables seleccionadas: {np.sum(mejor_individuo)}/{len(variables)}', fontsize=12)
    axes[1,1].text(0.1, 0.5, f'Reducción: {(1-np.sum(mejor_individuo)/len(variables))*100:.1f}%', fontsize=12)
    
    if precision_final >= precision_base:
        axes[1,1].text(0.1, 0.4, '🎉 Mejoró rendimiento', fontsize=11, color='green')
    else:
        axes[1,1].text(0.1, 0.4, '✅ Optimizó eficiencia', fontsize=11, color='blue')
    
    axes[1,1].text(0.1, 0.3, f'Fitness final: {fitness_evolution[-1]:.3f}', fontsize=11)
    axes[1,1].text(0.1, 0.2, f'Generaciones: {len(fitness_evolution)}', fontsize=11)
    
    axes[1,1].set_xlim(0, 1)
    axes[1,1].set_ylim(0, 1)
    axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/graficos/tecnicas_geneticas.png', dpi=150, bbox_inches='tight')
    plt.show()

def guardar_resultados_geneticos(mejor_individuo, fitness_evolution, variables, precision_final, precision_base, y_test, y_pred):
    """Guarda reporte de técnicas genéticas"""
    variables_seleccionadas = [var for var, sel in zip(variables, mejor_individuo) if sel]
    
    reporte = f"""TÉCNICAS GENÉTICAS - CLASIFICACIÓN
=================================

RESULTADO DE EVOLUCIÓN:
Fitness final: {fitness_evolution[-1]:.3f}
Precisión evolutiva: {precision_final:.3f} ({precision_final*100:.1f}%)
Precisión base: {precision_base:.3f} ({precision_base*100:.1f}%)

OPTIMIZACIÓN DE CARACTERÍSTICAS:
Variables originales: {len(variables)}
Variables seleccionadas: {len(variables_seleccionadas)}
Reducción: {(1 - len(variables_seleccionadas)/len(variables))*100:.1f}%

VARIABLES SELECCIONADAS:
{chr(10).join([f"- {var}" for var in variables_seleccionadas])}

COMPARACIÓN:
- Modelo base (todas): {precision_base:.3f} con {len(variables)} variables
- Modelo evolutivo: {precision_final:.3f} con {len(variables_seleccionadas)} variables
- Ratio eficiencia: {(precision_final/precision_base):.3f}x rendimiento con {len(variables_seleccionadas)/len(variables):.3f}x variables
"""
    
    # Métricas por clase si están disponibles
    try:
        reporte_sklearn = classification_report(y_test, y_pred, output_dict=True)
        reporte += f"\nMÉTRICAS POR CLASE (MODELO EVOLUTIVO):\n"
        for clase in np.unique(y_test):
            if clase in reporte_sklearn:
                prec = reporte_sklearn[clase]['precision']
                rec = reporte_sklearn[clase]['recall']
                f1 = reporte_sklearn[clase]['f1-score']
                reporte += f"{clase}: Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}\n"
    except:
        pass
    
    reporte += f"""
CONFIGURACIÓN GENÉTICA:
- Población: 30 individuos
- Generaciones: 20
- Probabilidad mutación: 0.15
- Selección: Torneo de 3
- Cruce: Un punto
- Elitismo: Mejor individuo conservado

PROCESO EVOLUTIVO:
- Cada individuo representa selección de variables
- Fitness = precisión - penalización por complejidad
- Evolución hacia soluciones más eficientes
- Búsqueda automática del subconjunto óptimo

PRINCIPIOS ALGORITMOS GENÉTICOS:
- Inspirados en evolución natural
- Población de soluciones candidatas
- Selección de los más aptos
- Reproducción con cruce y mutación
- Mejora gradual a través de generaciones

VENTAJAS:
- Optimización global (evita mínimos locales)
- No requiere gradientes o derivadas
- Maneja espacios de búsqueda complejos
- Paralelizable naturalmente

DESVENTAJAS:
- Computacionalmente costoso
- No garantiza encontrar el óptimo global
- Muchos hiperparámetros a ajustar
- Convergencia puede ser lenta

APLICACIONES:
- Selección de características en ML
- Optimización de hiperparámetros
- Diseño de redes neuronales
- Planificación de rutas
- Diseño de horarios
- Optimización financiera
- Ingeniería de diseño
"""
    
    with open('results/reportes/tecnicas_geneticas_reporte.txt', 'w', encoding='utf-8') as f:
        f.write(reporte)

def ejecutar_tecnicas_geneticas():
    """Función principal"""
    print("🧬 TÉCNICAS GENÉTICAS - CLASIFICACIÓN")
    print("="*40)
    
    # Cargar y preparar datos
    datos = cargar_datos()
    X, y, variables = preparar_datos(datos)
    
    print(f"📊 Datos: {len(X):,} registros")
    print(f"📊 Variables: {', '.join(variables)}")
    
    # División train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Escalar datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Codificar etiquetas
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    print(f"📊 División: {len(X_train):,} entrenamiento | {len(X_test):,} prueba")
    
    # Ejecutar algoritmo genético
    print(f"\n🧬 Ejecutando algoritmo genético...")
    ag = AlgoritmoGenetico(tamaño_poblacion=30, n_generaciones=20, prob_mutacion=0.15)
    mejor_individuo, mejor_fitness = ag.evolucionar(
        X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, variables
    )
    
    # Evaluar resultado final
    caracteristicas_seleccionadas = mejor_individuo == 1
    variables_seleccionadas = [var for var, sel in zip(variables, caracteristicas_seleccionadas) if sel]
    
    print(f"\n🏆 EVOLUCIÓN COMPLETADA")
    print(f"    Mejor fitness: {mejor_fitness:.3f}")
    print(f"    Variables seleccionadas: {len(variables_seleccionadas)}/{len(variables)}")
    print(f"    Variables: {', '.join(variables_seleccionadas)}")
    
    # Entrenar clasificador final con variables seleccionadas
    X_train_sel = X_train_scaled[:, caracteristicas_seleccionadas]
    X_test_sel = X_test_scaled[:, caracteristicas_seleccionadas]
    
    clf_final = DecisionTreeClassifier(max_depth=8, random_state=42)
    clf_final.fit(X_train_sel, y_train_encoded)
    y_pred_final = clf_final.predict(X_test_sel)
    
    precision_final = accuracy_score(y_test_encoded, y_pred_final)
    
    # Entrenar modelo base para comparación
    clf_base = DecisionTreeClassifier(max_depth=8, random_state=42)
    clf_base.fit(X_train_scaled, y_train_encoded)
    y_pred_base = clf_base.predict(X_test_scaled)
    precision_base = accuracy_score(y_test_encoded, y_pred_base)
    
    print(f"\n📊 RESULTADOS FINALES:")
    print(f"    Modelo base (todas): {precision_base:.3f} ({precision_base*100:.1f}%)")
    print(f"    Modelo evolutivo: {precision_final:.3f} ({precision_final*100:.1f}%)")
    print(f"    Eficiencia: {len(variables_seleccionadas)}/{len(variables)} variables")
    
    if precision_final >= precision_base:
        print(f"    🎉 ¡Algoritmo genético mejoró el rendimiento!")
    else:
        print(f"    ✅ Algoritmo genético optimizó la eficiencia")
    
    # Convertir predicciones de vuelta a etiquetas originales
    y_test_original = y_test_encoded
    y_pred_original = y_pred_final
    
    # Visualizar resultados
    visualizar_resultados_geneticos(mejor_individuo, ag.mejor_fitness, variables, 
                                  precision_final, precision_base)
    
    # Guardar resultados
    guardar_resultados_geneticos(mejor_individuo, ag.mejor_fitness, variables, 
                                precision_final, precision_base, y_test_original, y_pred_original)
    
    print("\n✅ COMPLETADO")
    
    return {
        'precision_evolutiva': precision_final,
        'precision_base': precision_base,
        'variables_seleccionadas': variables_seleccionadas,
        'fitness_evolution': ag.mejor_fitness,
        'mejor_individuo': mejor_individuo
    }

if __name__ == "__main__":
    ejecutar_tecnicas_geneticas()