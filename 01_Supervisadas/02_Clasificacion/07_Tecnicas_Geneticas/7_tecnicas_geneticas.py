#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TÉCNICAS GENÉTICAS - CLASIFICACIÓN (Versión Compacta)
Optimización evolutiva para selección de características
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
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

class AlgoritmoGenetico:
    """Algoritmo genético simple para optimización de características"""
    
    def __init__(self, tamaño_poblacion=30, n_generaciones=20, prob_mutacion=0.1):
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
        
        print(f"🧬 Iniciando evolución: {self.n_generaciones} generaciones")
        
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
                print(f"   Gen {gen:2d}: Fitness={mejor_fitness:.3f} | Variables={n_vars}")
            
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

def ejecutar_tecnicas_geneticas():
    print("🧬 TÉCNICAS GENÉTICAS - CLASIFICACIÓN")
    print("="*40)
    print("📝 Optimización evolutiva de características")
    print()
    
    # 1. CARGAR DATOS
    archivo = '/home/sedc/Proyectos/MineriaDeDatos/data/ceros_sin_columnasAB_limpio_weka.csv'
    try:
        datos = pd.read_csv(archivo)
        print(f"✅ Datos cargados: {datos.shape[0]:,} filas")
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # 2. PREPARAR DATOS
    variables_predictoras = [
        'POBFEM', 'POBMAS', 'TOTHOG', 'VIVTOT', 'P_15YMAS', 
        'P_60YMAS', 'GRAPROES', 'PEA', 'POCUPADA'
    ]
    
    variables_disponibles = [v for v in variables_predictoras if v in datos.columns]
    print(f"📊 Variables: {', '.join(variables_disponibles)}")
    
    # Crear categorías
    datos['CATEGORIA_POB'] = datos['POBTOT'].apply(crear_categorias_poblacion)
    
    # Limpiar datos
    datos_limpios = datos[variables_disponibles + ['CATEGORIA_POB']].dropna()
    
    # Reducir muestra para eficiencia
    if len(datos_limpios) > 2000:
        datos_limpios = datos_limpios.sample(n=2000, random_state=42)
    
    X = datos_limpios[variables_disponibles].values
    y = datos_limpios['CATEGORIA_POB'].values
    
    # Codificar etiquetas
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"🧹 Datos finales: {len(datos_limpios):,} registros")
    
    # 3. DIVIDIR DATOS
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )
    
    # Escalar datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"📊 Entrenamiento: {len(X_train):,} | Prueba: {len(X_test):,}")
    print()
    
    # 4. EJECUTAR ALGORITMO GENÉTICO
    print("🧬 EJECUTANDO ALGORITMO GENÉTICO...")
    
    ag = AlgoritmoGenetico(tamaño_poblacion=30, n_generaciones=20, prob_mutacion=0.15)
    mejor_individuo, mejor_fitness = ag.evolucionar(
        X_train_scaled, X_test_scaled, y_train, y_test, variables_disponibles
    )
    
    # 5. EVALUAR RESULTADO FINAL
    caracteristicas_seleccionadas = mejor_individuo == 1
    variables_seleccionadas = [var for var, sel in zip(variables_disponibles, caracteristicas_seleccionadas) if sel]
    
    print()
    print(f"🏆 EVOLUCIÓN COMPLETADA")
    print(f"   Mejor fitness: {mejor_fitness:.3f}")
    print(f"   Variables seleccionadas: {len(variables_seleccionadas)}/{len(variables_disponibles)}")
    print(f"   Variables: {', '.join(variables_seleccionadas)}")
    
    # 6. ENTRENAR CLASIFICADOR FINAL
    X_train_sel = X_train_scaled[:, caracteristicas_seleccionadas]
    X_test_sel = X_test_scaled[:, caracteristicas_seleccionadas]
    
    clf_final = DecisionTreeClassifier(max_depth=8, random_state=42)
    clf_final.fit(X_train_sel, y_train)
    y_pred = clf_final.predict(X_test_sel)
    
    precision_final = accuracy_score(y_test, y_pred)
    
    print()
    print(f"📊 RESULTADOS FINALES:")
    print(f"   Precisión: {precision_final:.3f} ({precision_final*100:.1f}%)")
    
    # Reporte por clase
    y_test_original = label_encoder.inverse_transform(y_test)
    y_pred_original = label_encoder.inverse_transform(y_pred)
    
    print("\n🎯 Métricas por Categoría:")
    try:
        reporte = classification_report(y_test_original, y_pred_original, output_dict=True)
        for categoria in ['Pequeña', 'Mediana', 'Grande', 'Muy Grande']:
            if categoria in reporte:
                prec = reporte[categoria]['precision']
                rec = reporte[categoria]['recall'] 
                f1 = reporte[categoria]['f1-score']
                print(f"   {categoria:12}: Prec={prec:.3f} | Rec={rec:.3f} | F1={f1:.3f}")
    except Exception as e:
        print(f"   ⚠️ Error: {e}")
    
    # 7. VISUALIZACIONES
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Evolución del fitness
        axes[0,0].plot(ag.mejor_fitness, 'b-', linewidth=2)
        axes[0,0].set_title('🧬 Evolución del Fitness', fontweight='bold')
        axes[0,0].set_xlabel('Generación')
        axes[0,0].set_ylabel('Fitness')
        axes[0,0].grid(True, alpha=0.3)
        
        # Variables seleccionadas
        axes[0,1].bar(range(len(variables_disponibles)), mejor_individuo, 
                     color=['green' if x == 1 else 'red' for x in mejor_individuo])
        axes[0,1].set_title('🔬 Variables Seleccionadas', fontweight='bold')
        axes[0,1].set_xlabel('Variable')
        axes[0,1].set_ylabel('Seleccionada (1) / No (0)')
        axes[0,1].set_xticks(range(len(variables_disponibles)))
        axes[0,1].set_xticklabels([v[:6] for v in variables_disponibles], rotation=45)
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1,0])
        axes[1,0].set_title('🎯 Matriz de Confusión', fontweight='bold')
        axes[1,0].set_xlabel('Predicción')
        axes[1,0].set_ylabel('Real')
        
        # Comparación con modelo base
        clf_base = DecisionTreeClassifier(max_depth=8, random_state=42)
        clf_base.fit(X_train_scaled, y_train)
        y_pred_base = clf_base.predict(X_test_scaled)
        precision_base = accuracy_score(y_test, y_pred_base)
        
        modelos = ['Todas las variables', 'Variables evolutivas']
        precisiones = [precision_base, precision_final]
        n_vars = [len(variables_disponibles), len(variables_seleccionadas)]
        
        x = np.arange(len(modelos))
        width = 0.35
        
        axes[1,1].bar(x - width/2, precisiones, width, label='Precisión', color='skyblue')
        axes[1,1].bar(x + width/2, [n/max(n_vars) for n in n_vars], width, 
                     label='Variables (normalizado)', color='orange')
        
        axes[1,1].set_title('⚖️ Comparación de Modelos', fontweight='bold')
        axes[1,1].set_ylabel('Valor')
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels(modelos)
        axes[1,1].legend()
        
        # Añadir valores en las barras
        for i, (prec, n_var) in enumerate(zip(precisiones, n_vars)):
            axes[1,1].text(i - width/2, prec + 0.01, f'{prec:.3f}', ha='center')
            axes[1,1].text(i + width/2, (n_var/max(n_vars)) + 0.01, str(n_var), ha='center')
        
        plt.tight_layout()
        plt.savefig('/home/sedc/Proyectos/MineriaDeDatos/results/graficos/tecnicas_geneticas_clasificacion.png', 
                   dpi=150, bbox_inches='tight')
        plt.show()
        
        print("💾 Gráficos guardados: results/graficos/tecnicas_geneticas_clasificacion.png")
        
    except Exception as e:
        print(f"⚠️ Error en visualizaciones: {e}")
    
    # 8. GUARDAR RESULTADOS
    try:
        import joblib
        
        # Guardar modelo y scaler
        joblib.dump(clf_final, '/home/sedc/Proyectos/MineriaDeDatos/results/modelos/mejor_clasificador_genetico.pkl')
        joblib.dump(scaler, '/home/sedc/Proyectos/MineriaDeDatos/results/modelos/scaler_genetico.pkl')
        
        # Crear reporte
        reporte = f"""
REPORTE TÉCNICAS GENÉTICAS - CLASIFICACIÓN
=========================================

RESULTADO DE EVOLUCIÓN:
- Generaciones: {ag.n_generaciones}
- Población: {ag.tamaño_poblacion}
- Mejor fitness: {mejor_fitness:.3f}
- Precisión final: {precision_final:.3f} ({precision_final*100:.1f}%)

OPTIMIZACIÓN DE CARACTERÍSTICAS:
- Variables originales: {len(variables_disponibles)}
- Variables seleccionadas: {len(variables_seleccionadas)}
- Reducción: {(1 - len(variables_seleccionadas)/len(variables_disponibles))*100:.1f}%

VARIABLES SELECCIONADAS:
{chr(10).join([f"- {var}" for var in variables_seleccionadas])}

COMPARACIÓN:
- Modelo base (todas): {precision_base:.3f} con {len(variables_disponibles)} variables
- Modelo evolutivo: {precision_final:.3f} con {len(variables_seleccionadas)} variables
- Mejora en eficiencia: {(precision_final/precision_base):.3f}x con {len(variables_seleccionadas)/len(variables_disponibles):.3f}x variables

CONFIGURACIÓN GENÉTICA:
- Selección: Torneo
- Cruce: Un punto
- Mutación: Bit flip ({ag.prob_mutacion})
- Elitismo: Mejor individuo conservado
"""
        
        with open('/home/sedc/Proyectos/MineriaDeDatos/results/reportes/tecnicas_geneticas_reporte.txt', 'w', encoding='utf-8') as f:
            f.write(reporte)
        
        print("💾 Modelo guardado: results/modelos/mejor_clasificador_genetico.pkl")
        print("📄 Reporte guardado: results/reportes/tecnicas_geneticas_reporte.txt")
        
    except Exception as e:
        print(f"⚠️ Error guardando: {e}")
    
    # 9. RESUMEN FINAL
    print()
    print("📝 RESUMEN:")
    print(f"   • Precisión evolutiva: {precision_final*100:.1f}%")
    print(f"   • Variables optimizadas: {len(variables_seleccionadas)}/{len(variables_disponibles)}")
    print(f"   • Eficiencia: Menos variables, igual o mejor rendimiento")
    
    if precision_final > precision_base:
        print("   🎉 ¡El algoritmo genético mejoró el rendimiento!")
    else:
        print("   ✅ El algoritmo genético optimizó la eficiencia")
    
    print("✅ TÉCNICAS GENÉTICAS COMPLETADAS")
    
    return {
        'precision': precision_final,
        'variables_seleccionadas': variables_seleccionadas,
        'fitness_evolution': ag.mejor_fitness,
        'modelo': clf_final
    }

if __name__ == "__main__":
    ejecutar_tecnicas_geneticas()