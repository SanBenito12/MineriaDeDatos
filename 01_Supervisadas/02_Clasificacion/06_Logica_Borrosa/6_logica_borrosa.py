#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LÓGICA BORROSA - CLASIFICACIÓN OPTIMIZADA
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def membresia_gaussiana(x, centro, sigma):
    return np.exp(-0.5 * ((x - centro) / (sigma + 1e-10)) ** 2)

def membresia_sigmoidal(x, a, c):
    return 1 / (1 + np.exp(-a * (x - c)))

class ClasificadorBorrosoHibrido:
    def __init__(self):
        self.conjuntos = {}
        self.reglas = []
        self.clases = None
        self.modelo_base = None
        
    def _crear_conjuntos_adaptativos(self, X, y, variables):
        self.conjuntos = {}
        
        for i, var in enumerate(variables):
            valores = X[:, i]
            self.conjuntos[var] = {}
            
            for clase in self.clases:
                indices = np.where(y == clase)[0]
                if len(indices) > 5:
                    vals_clase = valores[indices]
                    centro = np.mean(vals_clase)
                    sigma = np.std(vals_clase) + 1e-10
                    
                    self.conjuntos[var][f"{clase}_Gauss"] = ('gaussiana', centro, sigma)
                    
                    mediana = np.median(vals_clase)
                    self.conjuntos[var][f"{clase}_Sig"] = ('sigmoidal', 2.0, mediana)
            
            q10, q30, q50, q70, q90 = np.percentile(valores, [10, 30, 50, 70, 90])
            self.conjuntos[var].update({
                'Muy_Bajo': ('gaussiana', q10, (q30-q10)/2),
                'Bajo': ('gaussiana', q30, (q50-q30)/2),
                'Medio': ('gaussiana', q50, (q70-q50)/2),
                'Alto': ('gaussiana', q70, (q90-q70)/2),
                'Muy_Alto': ('gaussiana', q90, (q90-q70)/2)
            })
    
    def _calcular_membresia(self, x, params):
        if params[0] == 'gaussiana':
            return membresia_gaussiana(x, params[1], params[2])
        elif params[0] == 'sigmoidal':
            return membresia_sigmoidal(x, params[1], params[2])
        return 0.0
    
    def _extraer_reglas_arbol(self, X, y, variables):
        dt = DecisionTreeClassifier(max_depth=4, min_samples_split=20, random_state=42)
        dt.fit(X, y)
        
        tree = dt.tree_
        reglas = []
        
        def obtener_reglas(nodo, condiciones=[]):
            if tree.children_left[nodo] != tree.children_right[nodo]:
                var_idx = tree.feature[nodo]
                umbral = tree.threshold[nodo]
                variable = variables[var_idx]
                
                cond_izq = condiciones + [(variable, '<=', umbral)]
                obtener_reglas(tree.children_left[nodo], cond_izq)
                
                cond_der = condiciones + [(variable, '>', umbral)]
                obtener_reglas(tree.children_right[nodo], cond_der)
            else:
                clase_idx = np.argmax(tree.value[nodo])
                clase = dt.classes_[clase_idx]
                confianza = tree.value[nodo][0][clase_idx] / tree.n_node_samples[nodo]
                
                if confianza > 0.6 and len(condiciones) <= 3:
                    reglas.append({
                        'condiciones': condiciones,
                        'clase': clase,
                        'confianza': confianza
                    })
        
        obtener_reglas(0)
        return reglas[:20]
    
    def fit(self, X, y, variables):
        self.clases = np.unique(y)
        self._crear_conjuntos_adaptativos(X, y, variables)
        self.reglas = self._extraer_reglas_arbol(X, y, variables)
        
        self.modelo_base = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
        self.modelo_base.fit(X, y)
        
        return self
    
    def predict(self, X, variables):
        pred_base = self.modelo_base.predict(X)
        pred_borroso = []
        
        for i in range(X.shape[0]):
            x = X[i]
            scores = {clase: 0 for clase in self.clases}
            
            for regla in self.reglas:
                activacion_total = 1.0
                
                for var, op, valor in regla['condiciones']:
                    var_idx = variables.index(var)
                    
                    if op == '<=':
                        activacion_total *= (1.0 if x[var_idx] <= valor else 0.3)
                    else:
                        activacion_total *= (1.0 if x[var_idx] > valor else 0.3)
                
                scores[regla['clase']] += activacion_total * regla['confianza']
            
            if max(scores.values()) > 0:
                pred_borroso.append(max(scores.keys(), key=lambda k: scores[k]))
            else:
                pred_borroso.append(pred_base[i])
        
        return np.array(pred_borroso)

def ejecutar_logica_borrosa():
    print("🌫️ LÓGICA BORROSA - CLASIFICACIÓN")
    print("="*40)
    
    archivo = 'data/ceros_sin_columnasAB_limpio_weka.csv'
    try:
        datos = pd.read_csv(archivo)
        print(f"✅ Datos: {datos.shape[0]:,} filas")
    except:
        print("❌ Error cargando datos")
        return
    
    variables = ['POBFEM', 'POBMAS', 'TOTHOG', 'VIVTOT', 'P_15YMAS', 'GRAPROES', 'PEA', 'POCUPADA']
    variables_ok = [v for v in variables if v in datos.columns]
    
    def categorizar_mejorado(pobl):
        if pobl <= 100:
            return 'Muy_Pequeña'
        elif pobl <= 500:
            return 'Pequeña'
        elif pobl <= 2000:
            return 'Mediana'
        elif pobl <= 8000:
            return 'Grande'
        else:
            return 'Muy_Grande'
    
    datos['CATEGORIA'] = datos['POBTOT'].apply(categorizar_mejorado)
    
    datos_limpios = datos[variables_ok + ['CATEGORIA']].dropna()
    
    conteos = datos_limpios['CATEGORIA'].value_counts()
    clases_validas = conteos[conteos >= 30].index
    datos_filtrados = datos_limpios[datos_limpios['CATEGORIA'].isin(clases_validas)]
    
    if len(datos_filtrados) > 2000:
        datos_sample = datos_filtrados.groupby('CATEGORIA').apply(
            lambda x: x.sample(min(len(x), 400), random_state=42)
        ).reset_index(drop=True)
    else:
        datos_sample = datos_filtrados
    
    X = datos_sample[variables_ok].values
    y = datos_sample['CATEGORIA'].values
    
    print(f"🧹 Datos finales: {len(datos_sample):,}")
    print(f"📊 Variables: {len(variables_ok)}")
    print(f"🎯 Clases: {len(np.unique(y))}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    print("🌫️ Entrenando clasificador borroso...")
    clasificador = ClasificadorBorrosoHibrido()
    clasificador.fit(X_train, y_train, variables_ok)
    
    y_pred = clasificador.predict(X_test, variables_ok)
    precision = accuracy_score(y_test, y_pred)
    
    print(f"🎯 Precisión: {precision:.3f} ({precision*100:.1f}%)")
    
    try:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        axes[0].bar(['Lógica Borrosa'], [precision], color='purple')
        axes[0].set_ylim(0, 1)
        axes[0].set_title('Precisión')
        axes[0].text(0, precision + 0.02, f'{precision:.3f}', ha='center', fontweight='bold')
        
        from collections import Counter
        real_counts = Counter(y_test)
        pred_counts = Counter(y_pred)
        
        clases = list(real_counts.keys())
        reales = [real_counts[c] for c in clases]
        predichas = [pred_counts.get(c, 0) for c in clases]
        
        x = np.arange(len(clases))
        width = 0.35
        axes[1].bar(x - width/2, reales, width, label='Real', alpha=0.7)
        axes[1].bar(x + width/2, predichas, width, label='Predicha', alpha=0.7)
        axes[1].set_title('Distribución')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([c[:6] for c in clases], rotation=45)
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig('logica_borrosa.png', dpi=150, bbox_inches='tight')
        plt.show()
    except:
        pass
    
    print("✅ LÓGICA BORROSA COMPLETADA")
    return {'precision': precision, 'modelo': clasificador}

if __name__ == "__main__":
    ejecutar_logica_borrosa()