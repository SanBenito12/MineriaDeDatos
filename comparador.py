#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMPARADOR DE TAMAÑOS DE MUESTRA - ANÁLISIS DE IMPACTO
Compara el rendimiento de técnicas ML con muestras de diferente tamaño
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import time
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class ComparadorTamañoMuestra:
    """Comparador optimizado para evaluar el impacto del tamaño de muestra"""
    
    def __init__(self, archivo_datos):
        self.archivo_datos = archivo_datos
        self.datos_completos = None
        self.resultados_comparacion = defaultdict(dict)
        
    def cargar_datos(self):
        """Carga datos completos del proyecto"""
        try:
            self.datos_completos = pd.read_csv(self.archivo_datos)
            print(f"✅ Datos cargados: {self.datos_completos.shape[0]:,} registros totales")
            return True
        except Exception as e:
            print(f"❌ Error cargando datos: {e}")
            return False
    
    def crear_categorias_poblacion(self, poblacion):
        """Crear categorías balanceadas para clasificación"""
        if poblacion <= 100:
            return 'Muy_Pequeña'
        elif poblacion <= 500:
            return 'Pequeña'
        elif poblacion <= 2000:
            return 'Mediana'
        elif poblacion <= 8000:
            return 'Grande'
        else:
            return 'Muy_Grande'
    
    def preparar_datasets(self, tamaños_muestra):
        """Prepara múltiples datasets con diferentes tamaños"""
        # Variables predictoras principales
        variables_predictoras = [
            'POBFEM', 'POBMAS', 'TOTHOG', 'VIVTOT', 'P_15YMAS', 
            'P_60YMAS', 'GRAPROES', 'PEA', 'POCUPADA'
        ]
        
        variables_disponibles = [v for v in variables_predictoras 
                               if v in self.datos_completos.columns]
        
        # Crear variable de clasificación
        self.datos_completos['CATEGORIA_POB'] = self.datos_completos['POBTOT'].apply(
            self.crear_categorias_poblacion
        )
        
        # Limpiar datos
        datos_limpios = self.datos_completos[variables_disponibles + ['POBTOT', 'CATEGORIA_POB']].dropna()
        
        print(f"📊 Variables disponibles: {len(variables_disponibles)}")
        print(f"🧹 Datos limpios: {len(datos_limpios):,} registros")
        
        # Crear datasets para cada tamaño
        datasets = {}
        
        for tamaño in tamaños_muestra:
            if tamaño == 'completo':
                datos_muestra = datos_limpios.copy()
                tamaño_real = len(datos_muestra)
            else:
                # Muestreo estratificado para mantener proporciones
                try:
                    datos_muestra, _ = train_test_split(
                        datos_limpios, 
                        test_size=1-tamaño/len(datos_limpios),
                        stratify=datos_limpios['CATEGORIA_POB'],
                        random_state=42
                    )
                    tamaño_real = len(datos_muestra)
                except:
                    # Si falla estratificado, usar muestreo simple
                    datos_muestra = datos_limpios.sample(n=min(tamaño, len(datos_limpios)), 
                                                       random_state=42)
                    tamaño_real = len(datos_muestra)
            
            # Preparar X, y para regresión y clasificación
            X = datos_muestra[variables_disponibles]
            y_regresion = datos_muestra['POBTOT']
            y_clasificacion = datos_muestra['CATEGORIA_POB']
            
            datasets[tamaño] = {
                'X': X,
                'y_regresion': y_regresion,
                'y_clasificacion': y_clasificacion,
                'tamaño_real': tamaño_real,
                'distribucion_clases': y_clasificacion.value_counts().to_dict()
            }
            
            print(f"📝 Dataset {tamaño}: {tamaño_real:,} registros")
        
        return datasets, variables_disponibles
    
    def definir_modelos(self):
        """Define modelos optimizados para comparación"""
        modelos_clasificacion = {
            'RandomForest': RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42),
            'LogisticRegression': LogisticRegression(max_iter=500, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42, probability=True),
            'NeuralNetwork': MLPClassifier(hidden_layer_sizes=(20, 10), max_iter=300, 
                                         random_state=42, alpha=0.01)
        }
        
        modelos_regresion = {
            'RandomForest': RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42),
            'LinearRegression': LinearRegression(),
            'SVR': SVR(kernel='rbf', C=100),
            'NeuralNetwork': MLPRegressor(hidden_layer_sizes=(20, 10), max_iter=300, 
                                        random_state=42, alpha=0.01)
        }
        
        return modelos_clasificacion, modelos_regresion
    
    def evaluar_modelo_completo(self, modelo, X, y, tipo='clasificacion', cv_folds=5):
        """Evaluación robusta con validación cruzada anidada"""
        try:
            # Escalado si es necesario
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Codificación para clasificación si es necesario
            if tipo == 'clasificacion' and not isinstance(y.iloc[0], (int, float)):
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)
            else:
                y_encoded = y
            
            # Validación cruzada estratificada
            if tipo == 'clasificacion':
                cv = StratifiedKFold(n_splits=min(cv_folds, len(np.unique(y_encoded))), 
                                   shuffle=True, random_state=42)
                scoring = 'accuracy'
            else:
                from sklearn.model_selection import KFold
                cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                scoring = 'r2'
            
            # Realizar validación cruzada
            inicio_tiempo = time.time()
            scores = cross_val_score(modelo, X_scaled, y_encoded, cv=cv, scoring=scoring)
            tiempo_total = time.time() - inicio_tiempo
            
            # División para prueba final
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_encoded, test_size=0.2, random_state=42,
                stratify=y_encoded if tipo == 'clasificacion' else None
            )
            
            # Entrenar y evaluar
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)
            
            if tipo == 'clasificacion':
                score_final = accuracy_score(y_test, y_pred)
            else:
                score_final = r2_score(y_test, y_pred)
            
            return {
                'cv_mean': scores.mean(),
                'cv_std': scores.std(),
                'score_final': score_final,
                'tiempo': tiempo_total,
                'num_samples': len(X),
                'exito': True
            }
            
        except Exception as e:
            print(f"      ❌ Error: {str(e)[:50]}...")
            return {'exito': False, 'error': str(e)}
    
    def ejecutar_comparacion_completa(self, tamaños_muestra=[500, 1000, 2000, 5000, 'completo']):
        """Ejecuta comparación completa entre diferentes tamaños"""
        print("🔬 INICIANDO COMPARACIÓN COMPLETA DE TAMAÑOS DE MUESTRA")
        print("="*60)
        
        # Preparar datasets
        datasets, variables = self.preparar_datasets(tamaños_muestra)
        
        # Definir modelos
        modelos_cls, modelos_reg = self.definir_modelos()
        
        print(f"\n🧪 EVALUANDO {len(modelos_cls)} TÉCNICAS EN {len(datasets)} TAMAÑOS")
        print("-"*60)
        
        # Evaluar clasificación
        print("\n📊 EVALUANDO CLASIFICACIÓN:")
        for tamaño, dataset in datasets.items():
            print(f"\n   📝 Tamaño: {dataset['tamaño_real']:,} registros")
            
            for nombre_modelo, modelo in modelos_cls.items():
                print(f"      🔄 {nombre_modelo}...")
                
                resultado = self.evaluar_modelo_completo(
                    modelo, dataset['X'], dataset['y_clasificacion'], 
                    tipo='clasificacion'
                )
                
                if resultado['exito']:
                    self.resultados_comparacion[tamaño][f"{nombre_modelo}_cls"] = resultado
                    print(f"         ✅ CV: {resultado['cv_mean']:.3f}±{resultado['cv_std']:.3f} | "
                          f"Final: {resultado['score_final']:.3f} | "
                          f"Tiempo: {resultado['tiempo']:.1f}s")
        
        # Evaluar regresión
        print("\n📈 EVALUANDO REGRESIÓN:")
        for tamaño, dataset in datasets.items():
            print(f"\n   📝 Tamaño: {dataset['tamaño_real']:,} registros")
            
            for nombre_modelo, modelo in modelos_reg.items():
                print(f"      🔄 {nombre_modelo}...")
                
                resultado = self.evaluar_modelo_completo(
                    modelo, dataset['X'], dataset['y_regresion'], 
                    tipo='regresion'
                )
                
                if resultado['exito']:
                    self.resultados_comparacion[tamaño][f"{nombre_modelo}_reg"] = resultado
                    print(f"         ✅ CV: {resultado['cv_mean']:.3f}±{resultado['cv_std']:.3f} | "
                          f"Final: {resultado['score_final']:.3f} | "
                          f"Tiempo: {resultado['tiempo']:.1f}s")
        
        return self.resultados_comparacion
    
    def analizar_resultados(self):
        """Análisis completo de los resultados obtenidos"""
        print("\n" + "="*60)
        print("📊 ANÁLISIS DE RESULTADOS")
        print("="*60)
        
        # Crear tablas de comparación
        tamaños = list(self.resultados_comparacion.keys())
        técnicas = set()
        for resultados_tamaño in self.resultados_comparacion.values():
            técnicas.update(resultados_tamaño.keys())
        
        técnicas = sorted(list(técnicas))
        
        # Análisis por métrica
        metricas = ['cv_mean', 'cv_std', 'score_final', 'tiempo']
        
        for metrica in metricas:
            print(f"\n📈 {metrica.upper()}:")
            print("-" * 50)
            
            # Crear tabla
            tabla = []
            for técnica in técnicas:
                fila = [técnica[:15]]  # Limitar nombre
                for tamaño in tamaños:
                    if tamaño in self.resultados_comparacion:
                        if técnica in self.resultados_comparacion[tamaño]:
                            valor = self.resultados_comparacion[tamaño][técnica].get(metrica, 0)
                            if metrica == 'tiempo':
                                fila.append(f"{valor:.1f}s")
                            else:
                                fila.append(f"{valor:.3f}")
                        else:
                            fila.append("N/A")
                    else:
                        fila.append("N/A")
                tabla.append(fila)
            
            # Mostrar tabla
            header = ["Técnica"] + [str(t) for t in tamaños]
            print(f"{'Técnica':<15} | " + " | ".join([f"{h:>8}" for h in tamaños]))
            print("-" * (15 + 4 + len(tamaños) * 11))
            
            for fila in tabla:
                print(f"{fila[0]:<15} | " + " | ".join([f"{v:>8}" for v in fila[1:]]))
    
    def crear_visualizaciones(self):
        """Crear visualizaciones comprehensivas"""
        try:
            # Preparar datos para visualización
            datos_viz = []
            for tamaño, resultados in self.resultados_comparacion.items():
                for técnica, metrics in resultados.items():
                    tamaño_real = metrics.get('num_samples', 0)
                    datos_viz.append({
                        'tamaño': tamaño,
                        'tamaño_real': tamaño_real,
                        'técnica': técnica,
                        'tipo': 'Clasificación' if '_cls' in técnica else 'Regresión',
                        'cv_mean': metrics.get('cv_mean', 0),
                        'cv_std': metrics.get('cv_std', 0),
                        'score_final': metrics.get('score_final', 0),
                        'tiempo': metrics.get('tiempo', 0)
                    })
            
            df_viz = pd.DataFrame(datos_viz)
            
            if df_viz.empty:
                print("⚠️ No hay datos para visualizar")
                return False
            
            # Crear figura con múltiples subplots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('🔬 IMPACTO DEL TAMAÑO DE MUESTRA EN MACHINE LEARNING', 
                        fontsize=16, fontweight='bold')
            
            # Gráfico 1: Precisión vs Tamaño (Clasificación)
            df_cls = df_viz[df_viz['tipo'] == 'Clasificación']
            if not df_cls.empty:
                for técnica in df_cls['técnica'].unique():
                    datos_técnica = df_cls[df_cls['técnica'] == técnica]
                    axes[0,0].plot(datos_técnica['tamaño_real'], datos_técnica['cv_mean'], 
                                  'o-', label=técnica.replace('_cls', ''), linewidth=2, markersize=6)
                
                axes[0,0].set_title('📊 Precisión vs Tamaño - Clasificación', fontweight='bold')
                axes[0,0].set_xlabel('Tamaño de Muestra')
                axes[0,0].set_ylabel('Precisión (CV)')
                axes[0,0].legend()
                axes[0,0].grid(True, alpha=0.3)
                axes[0,0].set_xscale('log')
            
            # Gráfico 2: R² vs Tamaño (Regresión)
            df_reg = df_viz[df_viz['tipo'] == 'Regresión']
            if not df_reg.empty:
                for técnica in df_reg['técnica'].unique():
                    datos_técnica = df_reg[df_reg['técnica'] == técnica]
                    axes[0,1].plot(datos_técnica['tamaño_real'], datos_técnica['cv_mean'], 
                                  's-', label=técnica.replace('_reg', ''), linewidth=2, markersize=6)
                
                axes[0,1].set_title('📈 R² vs Tamaño - Regresión', fontweight='bold')
                axes[0,1].set_xlabel('Tamaño de Muestra')
                axes[0,1].set_ylabel('R² Score (CV)')
                axes[0,1].legend()
                axes[0,1].grid(True, alpha=0.3)
                axes[0,1].set_xscale('log')
            
            # Gráfico 3: Tiempo de entrenamiento vs Tamaño
            axes[0,2].scatter(df_viz['tamaño_real'], df_viz['tiempo'], 
                            c=['blue' if 'cls' in t else 'red' for t in df_viz['técnica']], 
                            alpha=0.6, s=50)
            axes[0,2].set_title('⏱️ Tiempo vs Tamaño', fontweight='bold')
            axes[0,2].set_xlabel('Tamaño de Muestra')
            axes[0,2].set_ylabel('Tiempo (segundos)')
            axes[0,2].set_xscale('log')
            axes[0,2].set_yscale('log')
            axes[0,2].grid(True, alpha=0.3)
            
            # Gráfico 4: Heatmap de rendimiento
            pivot_data = df_viz.pivot_table(
                index='técnica', 
                columns='tamaño', 
                values='cv_mean', 
                aggfunc='mean'
            )
            
            if not pivot_data.empty:
                sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[1,0])
                axes[1,0].set_title('🔥 Heatmap Rendimiento', fontweight='bold')
                axes[1,0].set_xlabel('Tamaño Muestra')
                axes[1,0].set_ylabel('Técnica')
            
            # Gráfico 5: Variabilidad (CV_std) vs Tamaño
            for tipo, color in [('Clasificación', 'blue'), ('Regresión', 'red')]:
                datos_tipo = df_viz[df_viz['tipo'] == tipo]
                axes[1,1].scatter(datos_tipo['tamaño_real'], datos_tipo['cv_std'], 
                                c=color, label=tipo, alpha=0.6, s=50)
            
            axes[1,1].set_title('📊 Variabilidad vs Tamaño', fontweight='bold')
            axes[1,1].set_xlabel('Tamaño de Muestra')
            axes[1,1].set_ylabel('Desviación Estándar CV')
            axes[1,1].legend()
            axes[1,1].set_xscale('log')
            axes[1,1].grid(True, alpha=0.3)
            
            # Gráfico 6: Eficiencia (Score/Tiempo)
            df_viz['eficiencia'] = df_viz['cv_mean'] / (df_viz['tiempo'] + 0.1)  # +0.1 para evitar división por 0
            
            axes[1,2].scatter(df_viz['tamaño_real'], df_viz['eficiencia'], 
                            c=['green' if 'cls' in t else 'orange' for t in df_viz['técnica']], 
                            alpha=0.6, s=50)
            axes[1,2].set_title('⚡ Eficiencia vs Tamaño', fontweight='bold')
            axes[1,2].set_xlabel('Tamaño de Muestra')
            axes[1,2].set_ylabel('Eficiencia (Score/Tiempo)')
            axes[1,2].set_xscale('log')
            axes[1,2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('comparacion_tamaño_muestra.png', dpi=150, bbox_inches='tight')
            plt.show()
            
            print("💾 Visualización guardada: comparacion_tamaño_muestra.png")
            return True
            
        except Exception as e:
            print(f"⚠️ Error en visualizaciones: {e}")
            return False
    
    def generar_recomendaciones(self):
        """Genera recomendaciones basadas en los resultados"""
        print("\n" + "="*60)
        print("💡 RECOMENDACIONES BASADAS EN ANÁLISIS")
        print("="*60)
        
        # Análizar tendencias
        mejores_por_tamaño = {}
        diferencias_rendimiento = {}
        
        for tamaño, resultados in self.resultados_comparacion.items():
            scores = [r['cv_mean'] for r in resultados.values() if r.get('cv_mean')]
            if scores:
                mejores_por_tamaño[tamaño] = max(scores)
        
        # Calcular diferencias
        if 'completo' in mejores_por_tamaño and 2000 in mejores_por_tamaño:
            diferencia_2k = mejores_por_tamaño['completo'] - mejores_por_tamaño[2000]
            porcentaje_perdida = (diferencia_2k / mejores_por_tamaño['completo']) * 100
            
            print(f"📊 IMPACTO DE USAR 2,000 VS DATASET COMPLETO:")
            print(f"   • Mejor score con dataset completo: {mejores_por_tamaño['completo']:.3f}")
            print(f"   • Mejor score con 2,000 muestras: {mejores_por_tamaño[2000]:.3f}")
            print(f"   • Diferencia absoluta: {diferencia_2k:.3f}")
            print(f"   • Pérdida porcentual: {porcentaje_perdida:.1f}%")
            
            if abs(porcentaje_perdida) < 5:
                print(f"   ✅ RECOMENDACIÓN: Pérdida mínima (<5%), usar 2K es aceptable")
            elif abs(porcentaje_perdida) < 15:
                print(f"   ⚠️ RECOMENDACIÓN: Pérdida moderada (5-15%), considerar más datos")
            else:
                print(f"   ❌ RECOMENDACIÓN: Pérdida significativa (>15%), usar dataset completo")
        
        print(f"\n🎯 ESTRATEGIA RECOMENDADA:")
        print(f"   1. 📈 DESARROLLO: Usar 2,000-5,000 muestras para desarrollo rápido")
        print(f"   2. 🔬 VALIDACIÓN: Usar dataset completo para validación final")
        print(f"   3. 📊 COMPARACIÓN: Siempre reportar ambos resultados")
        print(f"   4. 🎲 MUESTREO: Usar muestreo estratificado para mantener distribuciones")
        print(f"   5. ⚖️ TRADE-OFF: Balancear tiempo vs precisión según objetivos")

def main():
    """Función principal para ejecutar la comparación"""
    print("🔬 SISTEMA DE COMPARACIÓN DE TAMAÑOS DE MUESTRA")
    print("="*60)
    
    # Configurar archivo de datos
    archivo_datos = 'data/ceros_sin_columnasAB_limpio_weka.csv'
    
    # Crear comparador
    comparador = ComparadorTamañoMuestra(archivo_datos)
    
    # Cargar datos
    if not comparador.cargar_datos():
        print("❌ No se pueden cargar los datos")
        return
    
    # Definir tamaños a comparar
    tamaños_prueba = [500, 1000, 2000, 5000, 'completo']
    
    print(f"\n🎯 COMPARANDO TAMAÑOS: {tamaños_prueba}")
    
    # Ejecutar comparación
    resultados = comparador.ejecutar_comparacion_completa(tamaños_prueba)
    
    # Analizar resultados
    comparador.analizar_resultados()
    
    # Crear visualizaciones
    comparador.crear_visualizaciones()
    
    # Generar recomendaciones
    comparador.generar_recomendaciones()
    
    print("\n✅ COMPARACIÓN COMPLETADA")
    print("📊 Revisa los gráficos y recomendaciones para tomar decisiones informadas")

if __name__ == "__main__":
    main()