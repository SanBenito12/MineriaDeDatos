#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ALGORITMO A PRIORI - REGLAS DE ASOCIACIÓN
Encuentra patrones de asociación entre características demográficas
"""

import pandas as pd
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

class AlgoritmoAPriori:
    """Implementación del algoritmo A Priori para reglas de asociación"""
    
    def __init__(self, min_support=0.1, min_confidence=0.5, min_lift=1.0):
        self.min_support = min_support
        self.min_confidence = min_confidence  
        self.min_lift = min_lift
        self.transacciones = []
        self.items_frecuentes = {}
        self.reglas_asociacion = []
        
    def discretizar_variables(self, datos, variables):
        """Discretiza variables continuas en categorías"""
        datos_discretos = datos.copy()
        
        for variable in variables:
            if variable in datos.columns:
                valores = datos[variable]
                
                # Calcular cuartiles para discretización
                q1 = valores.quantile(0.25)
                q2 = valores.quantile(0.50)
                q3 = valores.quantile(0.75)
                
                # Crear categorías descriptivas
                def categorizar(valor):
                    if pd.isna(valor):
                        return None
                    elif valor <= q1:
                        return f"{variable}_Bajo"
                    elif valor <= q2:
                        return f"{variable}_Medio_Bajo"
                    elif valor <= q3:
                        return f"{variable}_Medio_Alto"
                    else:
                        return f"{variable}_Alto"
                
                datos_discretos[f"{variable}_Cat"] = datos[variable].apply(categorizar)
        
        return datos_discretos
    
    def crear_transacciones(self, datos_discretos):
        """Convierte el dataset en formato de transacciones"""
        transacciones = []
        
        # Obtener columnas categóricas
        columnas_cat = [col for col in datos_discretos.columns if col.endswith('_Cat')]
        
        for idx, fila in datos_discretos.iterrows():
            transaccion = []
            for col in columnas_cat:
                if pd.notna(fila[col]):
                    transaccion.append(fila[col])
            
            if len(transaccion) > 0:
                transacciones.append(transaccion)
        
        self.transacciones = transacciones
        return transacciones
    
    def calcular_soporte(self, itemset):
        """Calcula el soporte de un conjunto de items"""
        if not self.transacciones:
            return 0.0
            
        count = 0
        for transaccion in self.transacciones:
            if all(item in transaccion for item in itemset):
                count += 1
        
        return count / len(self.transacciones)
    
    def generar_candidatos_1(self):
        """Genera candidatos de tamaño 1"""
        items = set()
        for transaccion in self.transacciones:
            for item in transaccion:
                items.add(item)
        
        candidatos_1 = []
        for item in items:
            soporte = self.calcular_soporte([item])
            if soporte >= self.min_support:
                candidatos_1.append(([item], soporte))
        
        return candidatos_1
    
    def generar_candidatos_k(self, items_frecuentes_k_minus_1, k):
        """Genera candidatos de tamaño k"""
        candidatos = []
        items_list = [itemset for itemset, _ in items_frecuentes_k_minus_1]
        
        for i in range(len(items_list)):
            for j in range(i + 1, len(items_list)):
                # Combinar itemsets si comparten k-2 elementos
                union = sorted(list(set(items_list[i]) | set(items_list[j])))
                
                if len(union) == k and union not in candidatos:
                    soporte = self.calcular_soporte(union)
                    if soporte >= self.min_support:
                        candidatos.append((union, soporte))
        
        return candidatos
    
    def encontrar_items_frecuentes(self):
        """Ejecuta el algoritmo A Priori para encontrar items frecuentes"""
        print("🔍 Ejecutando algoritmo A Priori...")
        
        # Candidatos de tamaño 1
        items_frecuentes_1 = self.generar_candidatos_1()
        print(f"   Items frecuentes de tamaño 1: {len(items_frecuentes_1)}")
        
        self.items_frecuentes[1] = items_frecuentes_1
        k = 2
        
        # Iterar para encontrar items de mayor tamaño
        while k <= 4 and self.items_frecuentes.get(k-1, []):  # Limitar a máximo 4 items
            items_frecuentes_k = self.generar_candidatos_k(self.items_frecuentes[k-1], k)
            
            if not items_frecuentes_k:
                break
                
            print(f"   Items frecuentes de tamaño {k}: {len(items_frecuentes_k)}")
            self.items_frecuentes[k] = items_frecuentes_k
            k += 1
        
        return self.items_frecuentes
    
    def generar_reglas_asociacion(self):
        """Genera reglas de asociación a partir de items frecuentes"""
        reglas = []
        
        # Para cada tamaño de itemset >= 2
        for k in range(2, len(self.items_frecuentes) + 1):
            if k not in self.items_frecuentes:
                continue
                
            for itemset, soporte_itemset in self.items_frecuentes[k]:
                # Generar todas las posibles reglas
                for r in range(1, len(itemset)):
                    for antecedente in combinations(itemset, r):
                        antecedente = list(antecedente)
                        consecuente = [item for item in itemset if item not in antecedente]
                        
                        # Calcular métricas
                        soporte_antecedente = self.calcular_soporte(antecedente)
                        
                        if soporte_antecedente > 0:
                            confidence = soporte_itemset / soporte_antecedente
                            
                            soporte_consecuente = self.calcular_soporte(consecuente)
                            if soporte_consecuente > 0:
                                lift = confidence / soporte_consecuente
                            else:
                                lift = 0
                            
                            # Filtrar por métricas mínimas
                            if (confidence >= self.min_confidence and 
                                lift >= self.min_lift):
                                
                                reglas.append({
                                    'antecedente': antecedente,
                                    'consecuente': consecuente,
                                    'soporte': soporte_itemset,
                                    'confidence': confidence,
                                    'lift': lift,
                                    'conviction': self._calcular_conviction(confidence, soporte_consecuente)
                                })
        
        # Ordenar por lift descendente
        reglas.sort(key=lambda x: x['lift'], reverse=True)
        self.reglas_asociacion = reglas
        return reglas
    
    def _calcular_conviction(self, confidence, soporte_consecuente):
        """Calcula la conviction de una regla"""
        if confidence == 1.0:
            return float('inf')
        if soporte_consecuente == 1.0:
            return 1.0
        return (1 - soporte_consecuente) / (1 - confidence)

def crear_categorias_poblacion(poblacion):
    """Categorizar población para reglas de asociación"""
    if poblacion <= 500:
        return 'Población_Muy_Pequeña'
    elif poblacion <= 2000:
        return 'Población_Pequeña'
    elif poblacion <= 8000:
        return 'Población_Mediana'
    else:
        return 'Población_Grande'

def ejecutar_apriori():
    print("🔗 ALGORITMO A PRIORI - REGLAS DE ASOCIACIÓN")
    print("="*50)
    print("📝 Objetivo: Encontrar patrones de asociación en datos demográficos")
    print()
    
    # 1. CARGAR DATOS
    archivo = '/home/sedc/Proyectos/MineriaDeDatos/data/ceros_sin_columnasAB_limpio_weka.csv'
    try:
        datos = pd.read_csv(archivo)
        print(f"✅ Datos cargados: {datos.shape[0]:,} filas, {datos.shape[1]} columnas")
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        return
    
    # 2. SELECCIONAR VARIABLES PARA ASOCIACIÓN
    variables_asociacion = [
        'POBFEM', 'POBMAS', 'TOTHOG', 'VIVTOT', 'P_15YMAS',
        'GRAPROES', 'PEA', 'POCUPADA'
    ]
    
    variables_disponibles = [v for v in variables_asociacion if v in datos.columns]
    print(f"📊 Variables para asociación: {', '.join(variables_disponibles)}")
    
    # 3. AGREGAR CATEGORÍA DE POBLACIÓN
    datos['CATEGORIA_POB'] = datos['POBTOT'].apply(crear_categorias_poblacion)
    variables_disponibles.append('CATEGORIA_POB')
    
    # 4. MUESTREO PARA EFICIENCIA
    if len(datos) > 5000:
        datos_muestra = datos.sample(n=5000, random_state=42)
        print(f"📝 Muestra tomada: {len(datos_muestra):,} registros (para eficiencia)")
    else:
        datos_muestra = datos.copy()
    
    # Limpiar datos
    datos_limpios = datos_muestra[variables_disponibles].dropna()
    print(f"🧹 Datos limpios: {len(datos_limpios):,} registros")
    
    # 5. INICIALIZAR ALGORITMO A PRIORI
    print()
    print("⚙️ CONFIGURACIÓN A PRIORI:")
    
    # Probar diferentes configuraciones
    configuraciones = {
        'Restrictiva': {'min_support': 0.2, 'min_confidence': 0.7, 'min_lift': 1.5},
        'Moderada': {'min_support': 0.15, 'min_confidence': 0.6, 'min_lift': 1.2},
        'Permisiva': {'min_support': 0.1, 'min_confidence': 0.5, 'min_lift': 1.0}
    }
    
    todos_resultados = {}
    
    for nombre_config, params in configuraciones.items():
        print(f"\n🔧 Configuración {nombre_config}:")
        print(f"   Soporte mínimo: {params['min_support']}")
        print(f"   Confianza mínima: {params['min_confidence']}")
        print(f"   Lift mínimo: {params['min_lift']}")
        
        try:
            # Crear instancia del algoritmo
            apriori = AlgoritmoAPriori(**params)
            
            # 6. DISCRETIZAR VARIABLES
            print(f"   📊 Discretizando variables...")
            datos_discretos = apriori.discretizar_variables(datos_limpios, variables_disponibles[:-1])
            
            # Agregar categoría de población directamente
            datos_discretos['CATEGORIA_POB_Cat'] = datos_discretos['CATEGORIA_POB']
            
            # 7. CREAR TRANSACCIONES
            print(f"   🔄 Creando transacciones...")
            transacciones = apriori.crear_transacciones(datos_discretos)
            print(f"   📦 Transacciones creadas: {len(transacciones)}")
            
            if len(transacciones) == 0:
                print(f"   ❌ No se pudieron crear transacciones para {nombre_config}")
                continue
            
            # 8. ENCONTRAR ITEMS FRECUENTES
            items_frecuentes = apriori.encontrar_items_frecuentes()
            total_items = sum(len(items) for items in items_frecuentes.values())
            
            # 9. GENERAR REGLAS DE ASOCIACIÓN
            print(f"   🔗 Generando reglas de asociación...")
            reglas = apriori.generar_reglas_asociacion()
            
            print(f"   ✅ Items frecuentes totales: {total_items}")
            print(f"   ✅ Reglas de asociación: {len(reglas)}")
            
            todos_resultados[nombre_config] = {
                'apriori': apriori,
                'items_frecuentes': items_frecuentes,
                'reglas': reglas,
                'transacciones': len(transacciones)
            }
            
        except Exception as e:
            print(f"   ❌ Error en configuración {nombre_config}: {e}")
    
    if not todos_resultados:
        print("❌ No se pudieron generar reglas con ninguna configuración")
        return
    
    # 10. SELECCIONAR MEJOR CONFIGURACIÓN
    mejor_config = max(todos_resultados.keys(), 
                      key=lambda x: len(todos_resultados[x]['reglas']))
    
    resultado_mejor = todos_resultados[mejor_config]
    apriori_mejor = resultado_mejor['apriori']
    reglas_mejor = resultado_mejor['reglas']
    
    print()
    print(f"🏆 MEJOR CONFIGURACIÓN: {mejor_config}")
    print(f"   Reglas encontradas: {len(reglas_mejor)}")
    print(f"   Transacciones: {resultado_mejor['transacciones']}")
    
    # 11. MOSTRAR TOP REGLAS
    print()
    print("🔝 TOP 10 REGLAS DE ASOCIACIÓN:")
    print("-" * 80)
    
    for i, regla in enumerate(reglas_mejor[:10], 1):
        antecedente_str = " Y ".join(regla['antecedente'])
        consecuente_str = " Y ".join(regla['consecuente'])
        
        print(f"\nRegla #{i}:")
        print(f"   SI {antecedente_str}")
        print(f"   ENTONCES {consecuente_str}")
        print(f"   📊 Soporte: {regla['soporte']:.3f} ({regla['soporte']*100:.1f}%)")
        print(f"   🎯 Confianza: {regla['confidence']:.3f} ({regla['confidence']*100:.1f}%)")
        print(f"   🚀 Lift: {regla['lift']:.3f}")
        print(f"   💪 Conviction: {regla['conviction']:.3f}")
    
    # 12. ANÁLISIS DE PATRONES
    print()
    print("📈 ANÁLISIS DE PATRONES:")
    
    # Análisis por categorías más frecuentes
    categorias_antecedentes = defaultdict(int)
    categorias_consecuentes = defaultdict(int)
    
    for regla in reglas_mejor:
        for item in regla['antecedente']:
            categoria = item.split('_')[0]
            categorias_antecedentes[categoria] += 1
        
        for item in regla['consecuente']:
            categoria = item.split('_')[0]
            categorias_consecuentes[categoria] += 1
    
    print("\n🔍 Variables más frecuentes en antecedentes:")
    for categoria, count in sorted(categorias_antecedentes.items(), 
                                  key=lambda x: x[1], reverse=True)[:5]:
        print(f"   {categoria:15}: {count} reglas")
    
    print("\n🎯 Variables más frecuentes en consecuentes:")
    for categoria, count in sorted(categorias_consecuentes.items(), 
                                  key=lambda x: x[1], reverse=True)[:5]:
        print(f"   {categoria:15}: {count} reglas")
    
    # 13. VISUALIZACIONES
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Gráfico 1: Distribución de métricas
        if reglas_mejor:
            soportes = [r['soporte'] for r in reglas_mejor]
            confidences = [r['confidence'] for r in reglas_mejor]
            lifts = [r['lift'] for r in reglas_mejor]
            
            axes[0,0].hist(soportes, bins=15, alpha=0.7, color='blue', edgecolor='black')
            axes[0,0].set_title('📊 Distribución de Soporte', fontweight='bold')
            axes[0,0].set_xlabel('Soporte')
            axes[0,0].set_ylabel('Frecuencia')
            
            axes[0,1].hist(confidences, bins=15, alpha=0.7, color='green', edgecolor='black')
            axes[0,1].set_title('🎯 Distribución de Confianza', fontweight='bold')
            axes[0,1].set_xlabel('Confianza')
            axes[0,1].set_ylabel('Frecuencia')
            
            axes[0,2].hist(lifts, bins=15, alpha=0.7, color='orange', edgecolor='black')
            axes[0,2].set_title('🚀 Distribución de Lift', fontweight='bold')
            axes[0,2].set_xlabel('Lift')
            axes[0,2].set_ylabel('Frecuencia')
        
        # Gráfico 2: Scatter plots de métricas
        if len(reglas_mejor) > 1:
            axes[1,0].scatter(soportes, confidences, alpha=0.6, s=50)
            axes[1,0].set_xlabel('Soporte')
            axes[1,0].set_ylabel('Confianza')
            axes[1,0].set_title('📈 Soporte vs Confianza', fontweight='bold')
            
            axes[1,1].scatter(confidences, lifts, alpha=0.6, s=50, color='red')
            axes[1,1].set_xlabel('Confianza')
            axes[1,1].set_ylabel('Lift')
            axes[1,1].set_title('🎯 Confianza vs Lift', fontweight='bold')
        
        # Gráfico 3: Top variables en reglas
        if categorias_antecedentes:
            top_vars = list(categorias_antecedentes.keys())[:6]
            counts = [categorias_antecedentes[var] for var in top_vars]
            
            axes[1,2].barh(top_vars, counts, color='purple')
            axes[1,2].set_xlabel('Frecuencia en Reglas')
            axes[1,2].set_title('🔝 Variables Más Frecuentes', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('/home/sedc/Proyectos/MineriaDeDatos/results/graficos/apriori_asociacion.png', 
                   dpi=150, bbox_inches='tight')
        plt.show()
        
        print("💾 Gráficos guardados: results/graficos/apriori_asociacion.png")
        
    except Exception as e:
        print(f"⚠️ Error en visualizaciones: {e}")
    
    # 14. GUARDAR RESULTADOS
    try:
        import pickle
        
        # Guardar modelo y reglas
        with open('/home/sedc/Proyectos/MineriaDeDatos/results/modelos/apriori_modelo.pkl', 'wb') as f:
            pickle.dump(apriori_mejor, f)
        
        # Crear reporte detallado
        reporte_texto = f"""
REPORTE ALGORITMO A PRIORI - REGLAS DE ASOCIACIÓN
===============================================

CONFIGURACIÓN UTILIZADA: {mejor_config}
- Soporte mínimo: {apriori_mejor.min_support}
- Confianza mínima: {apriori_mejor.min_confidence}  
- Lift mínimo: {apriori_mejor.min_lift}

RESULTADOS:
- Transacciones procesadas: {resultado_mejor['transacciones']:,}
- Items frecuentes encontrados: {sum(len(items) for items in resultado_mejor['items_frecuentes'].values())}
- Reglas de asociación: {len(reglas_mejor)}

TOP 5 REGLAS POR LIFT:
"""
        
        for i, regla in enumerate(reglas_mejor[:5], 1):
            antecedente_str = " Y ".join(regla['antecedente'])
            consecuente_str = " Y ".join(regla['consecuente'])
            
            reporte_texto += f"""
Regla {i}:
- Antecedente: {antecedente_str}
- Consecuente: {consecuente_str}
- Soporte: {regla['soporte']:.3f}
- Confianza: {regla['confidence']:.3f}
- Lift: {regla['lift']:.3f}
- Conviction: {regla['conviction']:.3f}
"""
        
        reporte_texto += f"""

VARIABLES ANALIZADAS:
{', '.join(variables_disponibles)}

INTERPRETACIÓN:
- Lift > 1: Asociación positiva (más probable que por azar)
- Lift = 1: Independencia estadística
- Lift < 1: Asociación negativa

ANÁLISIS POR CONFIGURACIÓN:
"""
        for config, resultado in todos_resultados.items():
            reporte_texto += f"""
{config}:
- Items frecuentes: {sum(len(items) for items in resultado['items_frecuentes'].values())}
- Reglas generadas: {len(resultado['reglas'])}
"""
        
        with open('/home/sedc/Proyectos/MineriaDeDatos/results/reportes/apriori_reporte.txt', 'w', encoding='utf-8') as f:
            f.write(reporte_texto)
        
        print("💾 Modelo guardado: results/modelos/apriori_modelo.pkl")
        print("📄 Reporte guardado: results/reportes/apriori_reporte.txt")
        
    except Exception as e:
        print(f"⚠️ Error guardando archivos: {e}")
    
    # 15. RESUMEN FINAL
    print()
    print("📝 RESUMEN A PRIORI:")
    print(f"   • Configuración óptima: {mejor_config}")
    print(f"   • Reglas descubiertas: {len(reglas_mejor)}")
    print(f"   • Transacciones analizadas: {resultado_mejor['transacciones']:,}")
    
    if len(reglas_mejor) > 0:
        lift_promedio = np.mean([r['lift'] for r in reglas_mejor])
        confidence_promedio = np.mean([r['confidence'] for r in reglas_mejor])
        
        print(f"   • Lift promedio: {lift_promedio:.3f}")
        print(f"   • Confianza promedio: {confidence_promedio:.3f}")
        
        if len(reglas_mejor) >= 10:
            print("   🎉 ¡Excelente! Muchos patrones descubiertos")
        elif len(reglas_mejor) >= 5:
            print("   👍 Buenos patrones de asociación encontrados")
        else:
            print("   🔧 Pocos patrones, considerar relajar parámetros")
    else:
        print("   ⚠️ No se encontraron reglas, revisar configuración")
    
    print("   • Utilidad: Identificar relaciones ocultas entre variables")
    print("   • Aplicación: Segmentación y análisis demográfico")
    
    print("✅ ALGORITMO A PRIORI COMPLETADO")
    
    return {
        'mejor_configuracion': mejor_config,
        'reglas': reglas_mejor,
        'apriori': apriori_mejor,
        'todos_resultados': todos_resultados
    }

if __name__ == "__main__":
    ejecutar_apriori()