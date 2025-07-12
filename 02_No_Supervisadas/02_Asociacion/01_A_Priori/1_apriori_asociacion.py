#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A PRIORI RÁPIDO - REGLAS DE ASOCIACIÓN
Versión ultra-optimizada para ejecución rápida con +80% calidad
"""

import pandas as pd
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class APrioriRapido:
    """A Priori optimizado para velocidad y calidad"""
    
    def __init__(self, min_support=0.2, min_confidence=0.8, min_lift=1.5):
        self.min_support = min_support
        self.min_confidence = min_confidence  
        self.min_lift = min_lift
        self.transacciones = []
        self.reglas = []
        
    def discretizar_rapido(self, datos, variables):
        """Discretización simple y rápida"""
        datos_disc = datos.copy()
        
        for var in variables:
            if var in datos.columns:
                # Solo 3 categorías para mayor velocidad
                mediana = datos[var].median()
                q75 = datos[var].quantile(0.75)
                
                def cat_simple(val):
                    if pd.isna(val):
                        return None
                    elif val <= mediana:
                        return f"{var}_Bajo"
                    elif val <= q75:
                        return f"{var}_Medio"
                    else:
                        return f"{var}_Alto"
                
                datos_disc[f"{var}_Cat"] = datos[var].apply(cat_simple)
        
        return datos_disc
    
    def crear_transacciones_rapido(self, datos_disc):
        """Creación ultra-rápida de transacciones"""
        cols_cat = [c for c in datos_disc.columns if c.endswith('_Cat')]
        
        # Vectorización para máxima velocidad
        transacciones = []
        for _, fila in datos_disc[cols_cat].iterrows():
            trans = [item for item in fila.dropna() if item is not None]
            if len(trans) >= 2:
                transacciones.append(trans)
        
        self.transacciones = transacciones
        return len(transacciones)
    
    def encontrar_items_frecuentes_rapido(self):
        """Búsqueda rápida de items frecuentes"""
        # Solo items individuales y pares para velocidad
        todos_items = []
        for trans in self.transacciones:
            todos_items.extend(trans)
        
        # Items frecuentes de tamaño 1
        contador_items = Counter(todos_items)
        n_trans = len(self.transacciones)
        
        items_1 = []
        for item, count in contador_items.items():
            soporte = count / n_trans
            if soporte >= self.min_support:
                items_1.append((item, soporte))
        
        # Solo pares para velocidad (no itemsets más grandes)
        items_2 = []
        items_freq = [item for item, _ in items_1]
        
        for i, item1 in enumerate(items_freq):
            for item2 in items_freq[i+1:]:
                count_par = sum(1 for trans in self.transacciones 
                              if item1 in trans and item2 in trans)
                soporte = count_par / n_trans
                if soporte >= self.min_support:
                    items_2.append(([item1, item2], soporte))
        
        return items_1, items_2
    
    def generar_reglas_rapido(self, items_2):
        """Generación rápida de reglas de calidad"""
        reglas = []
        
        for itemset, soporte_itemset in items_2:
            item1, item2 = itemset
            
            # Calcular soportes individuales
            count1 = sum(1 for trans in self.transacciones if item1 in trans)
            count2 = sum(1 for trans in self.transacciones if item2 in trans)
            
            soporte1 = count1 / len(self.transacciones)
            soporte2 = count2 / len(self.transacciones)
            
            # Regla: item1 -> item2
            if soporte1 > 0 and soporte2 > 0:
                conf1 = soporte_itemset / soporte1
                lift1 = conf1 / soporte2
                
                if conf1 >= self.min_confidence and lift1 >= self.min_lift:
                    reglas.append({
                        'antecedente': [item1],
                        'consecuente': [item2],
                        'confidence': conf1,
                        'lift': lift1,
                        'soporte': soporte_itemset
                    })
                
                # Regla: item2 -> item1
                conf2 = soporte_itemset / soporte2
                lift2 = conf2 / soporte1
                
                if conf2 >= self.min_confidence and lift2 >= self.min_lift:
                    reglas.append({
                        'antecedente': [item2],
                        'consecuente': [item1],
                        'confidence': conf2,
                        'lift': lift2,
                        'soporte': soporte_itemset
                    })
        
        # Ordenar por confianza descendente
        reglas.sort(key=lambda x: x['confidence'], reverse=True)
        self.reglas = reglas
        return reglas

def preparar_datos_rapido(datos):
    """Preparación ultra-rápida de datos"""
    # Solo variables más importantes para velocidad
    vars_clave = ['POBFEM', 'POBMAS', 'TOTHOG', 'VIVTOT', 'P_15YMAS']
    vars_disponibles = [v for v in vars_clave if v in datos.columns]
    
    # Categoría de población simple
    mediana_pob = datos['POBTOT'].median()
    datos['CAT_POB'] = datos['POBTOT'].apply(
        lambda x: 'Población_Grande' if x > mediana_pob else 'Población_Pequeña'
    )
    vars_disponibles.append('CAT_POB')
    
    # Muestra pequeña para velocidad máxima
    if len(datos) > 1500:
        datos_muestra = datos.sample(n=1500, random_state=42)
    else:
        datos_muestra = datos.copy()
    
    datos_limpios = datos_muestra[vars_disponibles].dropna()
    return datos_limpios, vars_disponibles

def crear_visualizacion_rapida(reglas, vars_disponibles):
    """Visualización rápida y efectiva"""
    try:
        if len(reglas) == 0:
            print("⚠️ No hay reglas para visualizar")
            return False
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('🔗 A PRIORI RÁPIDO - RESULTADOS', fontsize=14, fontweight='bold')
        
        # 1. Top reglas por confianza
        top_reglas = reglas[:8]
        confidences = [r['confidence'] for r in top_reglas]
        labels = [f"R{i+1}" for i in range(len(top_reglas))]
        
        axes[0].barh(labels, confidences, color='skyblue')
        axes[0].set_xlabel('Confianza')
        axes[0].set_title('🎯 Top Reglas por Confianza')
        axes[0].set_xlim(0, 1)
        
        # 2. Distribución Confianza vs Lift
        if len(reglas) > 3:
            confs = [r['confidence'] for r in reglas]
            lifts = [r['lift'] for r in reglas]
            
            axes[1].scatter(confs, lifts, alpha=0.7, color='orange', s=60)
            axes[1].set_xlabel('Confianza')
            axes[1].set_ylabel('Lift')
            axes[1].set_title('📊 Confianza vs Lift')
            axes[1].grid(True, alpha=0.3)
        
        # 3. Variables más frecuentes
        todas_vars = []
        for regla in reglas:
            todas_vars.extend([item.split('_')[0] for item in regla['antecedente']])
            todas_vars.extend([item.split('_')[0] for item in regla['consecuente']])
        
        contador_vars = Counter(todas_vars)
        top_vars = contador_vars.most_common(5)
        
        if top_vars:
            vars_nombres = [v[0] for v in top_vars]
            vars_counts = [v[1] for v in top_vars]
            
            axes[2].bar(vars_nombres, vars_counts, color='lightgreen')
            axes[2].set_ylabel('Frecuencia')
            axes[2].set_title('📈 Variables Más Usadas')
            axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('/home/sedc/Proyectos/MineriaDeDatos/results/graficos/apriori_rapido.png', 
                   dpi=150, bbox_inches='tight')
        plt.show()
        
        return True
        
    except Exception as e:
        print(f"⚠️ Error en visualización: {e}")
        return False

def ejecutar_apriori():
    """Función principal ultra-rápida"""
    print("🔗 A PRIORI RÁPIDO - REGLAS DE ASOCIACIÓN")
    
    # 1. Cargar datos
    archivo = '/home/sedc/Proyectos/MineriaDeDatos/data/ceros_sin_columnasAB_limpio_weka.csv'
    try:
        datos = pd.read_csv(archivo)
        print(f"✅ Datos: {datos.shape[0]:,} registros")
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # 2. Preparar datos rápidamente
    datos_limpios, vars_disponibles = preparar_datos_rapido(datos)
    print(f"📊 Variables: {len(vars_disponibles)-1} + categoría población")
    print(f"🧹 Muestra: {len(datos_limpios):,} registros")
    
    # 3. Configuraciones rápidas (solo 3 para velocidad)
    configuraciones = [
        {'min_support': 0.25, 'min_confidence': 0.8, 'min_lift': 1.5},
        {'min_support': 0.2, 'min_confidence': 0.85, 'min_lift': 1.8},
        {'min_support': 0.3, 'min_confidence': 0.75, 'min_lift': 2.0}
    ]
    
    mejor_resultado = None
    mejor_score = 0
    
    for i, config in enumerate(configuraciones):
        try:
            apriori = APrioriRapido(**config)
            
            # Discretizar
            datos_disc = apriori.discretizar_rapido(datos_limpios, vars_disponibles[:-1])
            datos_disc['CAT_POB_Cat'] = datos_disc['CAT_POB']
            
            # Procesar
            n_trans = apriori.crear_transacciones_rapido(datos_disc)
            if n_trans < 20:
                continue
            
            items_1, items_2 = apriori.encontrar_items_frecuentes_rapido()
            reglas = apriori.generar_reglas_rapido(items_2)
            
            if len(reglas) > 0:
                conf_promedio = np.mean([r['confidence'] for r in reglas])
                score = conf_promedio * len(reglas)  # Score simple
                
                if score > mejor_score:
                    mejor_score = score
                    mejor_resultado = {
                        'reglas': reglas,
                        'config': config,
                        'conf_promedio': conf_promedio,
                        'transacciones': n_trans
                    }
        except:
            continue
    
    if mejor_resultado is None:
        print("❌ No se encontraron reglas válidas")
        return
    
    # 4. Mostrar resultados
    reglas = mejor_resultado['reglas']
    conf_prom = mejor_resultado['conf_promedio']
    
    print(f"🏆 Configuración: {mejor_resultado['config']}")
    print(f"🔗 Reglas: {len(reglas)}")
    print(f"🎯 Confianza promedio: {conf_prom:.3f} ({conf_prom*100:.1f}%)")
    print(f"📦 Transacciones: {mejor_resultado['transacciones']}")
    
    # 5. Top 5 reglas
    print("\n🔝 TOP 5 REGLAS:")
    for i, regla in enumerate(reglas[:5], 1):
        ant = " Y ".join([item.replace('_', ' ') for item in regla['antecedente']])
        con = " Y ".join([item.replace('_', ' ') for item in regla['consecuente']])
        print(f"{i}. SI {ant} → {con}")
        print(f"   Confianza: {regla['confidence']:.3f} | Lift: {regla['lift']:.3f}")
    
    # 6. Visualización
    crear_visualizacion_rapida(reglas, vars_disponibles)
    
    # 7. Evaluación
    reglas_calidad = len([r for r in reglas if r['confidence'] >= 0.8])
    porcentaje_calidad = reglas_calidad / len(reglas) * 100 if reglas else 0
    
    print(f"\n📊 Evaluación:")
    print(f"⭐ Reglas alta calidad (≥80%): {reglas_calidad}/{len(reglas)} ({porcentaje_calidad:.1f}%)")
    
    if conf_prom >= 0.8:
        print("✅ OBJETIVO CUMPLIDO: Confianza promedio ≥80%")
    elif porcentaje_calidad >= 80:
        print("✅ OBJETIVO CUMPLIDO: ≥80% reglas de alta calidad")
    else:
        print(f"⚠️ Confianza: {conf_prom*100:.1f}% (objetivo: ≥80%)")
    
    print("✅ A PRIORI RÁPIDO COMPLETADO")
    
    return {
        'reglas': reglas,
        'confianza_promedio': conf_prom,
        'cumple_objetivo': conf_prom >= 0.8 or porcentaje_calidad >= 80
    }

if __name__ == "__main__":
    ejecutar_apriori()