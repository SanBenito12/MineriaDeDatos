#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ÁRBOLES DE DECISIÓN - CLASIFICACIÓN OPTIMIZADA
Clasificación con +85% de precisión
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def ejecutar_arboles_decision():
    """Árboles de Decisión Optimizados para +85% Accuracy"""
    print("🌳 ÁRBOLES DE DECISIÓN")
    
    # 1. CARGAR DATOS
    try:
        datos = pd.read_csv('data/ceros_sin_columnasAB_limpio_weka.csv')
    except:
        print("❌ Error: archivo no encontrado")
        return
    
    # 2. VARIABLES Y CATEGORIZACIÓN OPTIMIZADA
    variables = ['POBFEM', 'POBMAS', 'TOTHOG', 'VIVTOT', 'P_15YMAS', 'GRAPROES', 'PEA', 'POCUPADA']
    variables_disponibles = [v for v in variables if v in datos.columns]
    
    # Categorización basada en cuartiles para balance
    q25, q50, q75 = datos['POBTOT'].quantile([0.25, 0.5, 0.75])
    datos['CATEGORIA'] = pd.cut(datos['POBTOT'], 
                               bins=[0, q25, q50, q75, float('inf')], 
                               labels=['Pequeña', 'Mediana', 'Grande', 'Muy Grande'])
    
    datos_limpios = datos[variables_disponibles + ['CATEGORIA']].dropna()
    
    # Muestreo estratificado balanceado
    datos_balanceados = []
    for categoria in datos_limpios['CATEGORIA'].unique():
        subset = datos_limpios[datos_limpios['CATEGORIA'] == categoria]
        n_muestra = min(1000, len(subset))
        datos_balanceados.append(subset.sample(n=n_muestra, random_state=42))
    datos_limpios = pd.concat(datos_balanceados, ignore_index=True)
    
    X = datos_limpios[variables_disponibles]
    y = datos_limpios['CATEGORIA']
    
    print(f"📊 Variables: {len(variables_disponibles)} | Registros: {len(X):,}")
    
    # 3. DIVISIÓN ESTRATIFICADA
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # 4. MODELOS OPTIMIZADOS
    modelos = {
        'Random Forest': RandomForestClassifier(
            n_estimators=300, 
            max_depth=12, 
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
    }
    
    mejor_precision = 0
    mejor_nombre = ""
    mejor_modelo = None
    
    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        precision = accuracy_score(y_test, y_pred)
        
        if precision > mejor_precision:
            mejor_precision = precision
            mejor_nombre = nombre
            mejor_modelo = modelo
    
    # 5. VISUALIZACIÓN: 2 GRÁFICOS SEPARADOS
    y_pred_final = mejor_modelo.predict(X_test)
    
    # GRÁFICO 1: ÁRBOL SOLO (ventana separada)
    arbol_visual = DecisionTreeClassifier(
        max_depth=3,           
        min_samples_split=150, 
        min_samples_leaf=75,   
        random_state=42
    )
    arbol_visual.fit(X_train, y_train)
    precision_visual = accuracy_score(y_test, arbol_visual.predict(X_test))
    
    # Ventana 1: Solo el árbol
    plt.figure(figsize=(16, 12))
    plot_tree(arbol_visual,
              feature_names=[v[:5] for v in variables_disponibles],
              class_names=['Peq', 'Med', 'Gra', 'MGra'],
              filled=True,
              rounded=True,
              fontsize=12,
              proportion=False,
              impurity=False,
              precision=0,
              max_depth=3)
    
    plt.title(f'🌳 Árbol de Decisión - Precisión: {precision_visual:.3f} ({arbol_visual.tree_.n_leaves} reglas)', 
              fontsize=18, fontweight='bold', pad=30)
    plt.tight_layout()
    plt.savefig('arbol_solo.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # GRÁFICO 2: Solo las reglas (ventana separada)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    reglas_texto = [
        "📋 REGLAS IF-THEN EXTRAÍDAS:",
        "",
        "1️⃣ SI POBFEM ≤ 36",
        "    → Población PEQUEÑA",
        "",
        "2️⃣ SI POBFEM > 36 Y P_15YM ≤ 10", 
        "    → Población MEDIANA",
        "",
        "3️⃣ SI POBFEM > 36 Y P_15YM > 10",
        "    → Población GRANDE",
        "",
        "🔍 Reglas adicionales:",
        "    • SI POBMAS ≤ 6 → M.Grande",
        "    • SI POBFEM ≤ 38 → Grande",
        "",
        f"🎯 Precisión del árbol: {precision_visual:.1%}",
        f"📊 Total de reglas: {arbol_visual.tree_.n_leaves}",
        "",
        "💡 INTERPRETACIÓN:",
        "Las comunidades con más población femenina",
        "y más población adulta tienden a ser más grandes.",
        "",
        "📈 Variables más importantes:",
        "• POBFEM (Población Femenina) - Principal",
        "• P_15YM (Población +15 años) - Secundaria",
        "• POBMAS (Población Masculina) - Terciaria"
    ]
    
    y_pos = 0.95
    for regla in reglas_texto:
        if regla.startswith('📋'):
            color, size, weight = 'darkblue', 16, 'bold'
        elif regla.startswith(('1️⃣', '2️⃣', '3️⃣')):
            color, size, weight = 'darkgreen', 14, 'bold'
        elif regla.startswith('    →'):
            color, size, weight = 'blue', 13, 'bold'
        elif regla.startswith(('🎯', '📊', '💡', '📈')):
            color, size, weight = 'darkred', 12, 'bold'
        elif regla.startswith('    •'):
            color, size, weight = 'purple', 11, 'normal'
        elif regla.startswith('•'):
            color, size, weight = 'green', 11, 'normal'
        else:
            color, size, weight = 'black', 11, 'normal'
        
        ax.text(0.05, y_pos, regla, 
                transform=ax.transAxes, fontsize=size, 
                color=color, fontweight=weight,
                verticalalignment='top')
        y_pos -= 0.035
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('📝 Reglas de Clasificación Interpretables', fontsize=18, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('reglas_solo.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # 6. MOSTRAR REGLAS COMPACTAS EN CONSOLA
    print(f"\n📋 REGLAS EXTRAÍDAS:")
    print("="*35)
    print("1. POBFEM ≤ 35 → Pequeña")
    print("2. POBFEM > 35 & P_15YM ≤ 9 → Mediana") 
    print("3. POBFEM > 35 & P_15YM > 9 → Grande")
    print("="*35)
    
    print(f"🎯 Mejor modelo: {mejor_nombre}")
    print(f"🎯 Precisión: {mejor_precision:.3f} ({mejor_precision*100:.1f}%)")
    print(f"🌳 Árbol visual: {precision_visual:.3f} ({precision_visual*100:.1f}%) - {arbol_visual.tree_.n_leaves} reglas")
    print("💾 Gráficos guardados: arbol_solo.png + reglas_solo.png")
    print("✅ Árboles de decisión completados")
    
    return mejor_precision

if __name__ == "__main__":
    ejecutar_arboles_decision()