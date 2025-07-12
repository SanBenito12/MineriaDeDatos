#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLASIFICADOR 100 % LÓGICA BORROSA  – ENFOQUE B (Mamdani)
--------------------------------------------------------
• Sin árboles de decisión
• Usa scikit‑fuzzy para todo el razonamiento
• Mantiene la estructura de menú: ejecutar_logica_borrosa() se
  sigue llamando desde el bloque «if __name__ == "__main__":»
"""

import warnings, os, sys
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ──────────────────────────────────────────────────────────────
# 1.  UTILIDADES
# ──────────────────────────────────────────────────────────────
def crear_categorias_poblacion_dinamica(datos):
    """Crea etiquetas Pequeña/Mediana/Grande/Muy_Grande según cuartiles."""
    q1, q2, q3 = datos["POBTOT"].quantile([.25, .50, .75])

    def categorizar(v):
        if v <= q1:          return "Pequeña"
        elif v <= q2:        return "Mediana"
        elif v <= q3:        return "Grande"
        else:                return "Muy_Grande"
    return categorizar

def percentiles(x):
    """Devuelve (min, p33, p66, max) – evita outliers extremos."""
    p0 , p33, p66, p100 = np.percentile(x, [1, 33, 66, 99])
    return p0, p33, p66, p100

# ──────────────────────────────────────────────────────────────
# 2.  CLASIFICADOR BORROSO PURO
# ──────────────────────────────────────────────────────────────
class ClasificadorBorrosoPuro:
    """
    Sistema Mamdani con dos entradas principales:

        • POBFEM  – población femenina
        • TOTHOG  – total de hogares

    Salida difusa: 4 etiquetas de población
    """
    def __init__(self):
        self.sistema      = None        # ctrl.ControlSystem
        self.umbral_dict  = {}          # guarda rangos para MF
        self.etiquetas    = ["Pequeña", "Mediana", "Grande", "Muy_Grande"]

    # ──────────────────────────────
    # 2.1  Crear MFs triangulares
    # ──────────────────────────────
    def _construir_mfs(self, universo, tag):
        p0, p33, p66, p100 = percentiles(universo)
        ant = ctrl.Antecedent(np.linspace(p0, p100, 100), tag)

        ant["baja"]   = fuzz.trimf(ant.universe, [p0, p0, p33])
        ant["media"]  = fuzz.trimf(ant.universe, [p0, p33, p66])
        ant["alta"]   = fuzz.trimf(ant.universe, [p33, p100, p100])

        self.umbral_dict[tag] = dict(p0=p0, p33=p33, p66=p66, p100=p100)
        return ant

    # ──────────────────────────────
    # 2.2  Construir sistema Mamdani
    # ──────────────────────────────
    def fit(self, X_df):
        """
        X_df: DataFrame con columnas 'POBFEM' y 'TOTHOG'
        (No necesita y_train porque no se “entrena” como ML)
        """
        # 1) antecedentes
        self.pobfem_ant = self._construir_mfs(X_df["POBFEM"], "POBFEM")
        self.tothog_ant = self._construir_mfs(X_df["TOTHOG"], "TOTHOG")

        # 2) consecuente
        self.cat_cons = ctrl.Consequent(np.arange(0, 4, 1), "CATEGORIA_POB")

        self.cat_cons["Pequeña"]    = fuzz.trimf(self.cat_cons.universe, [0, 0, 1])
        self.cat_cons["Mediana"]    = fuzz.trimf(self.cat_cons.universe, [0, 1, 2])
        self.cat_cons["Grande"]     = fuzz.trimf(self.cat_cons.universe, [1, 2, 3])
        self.cat_cons["Muy_Grande"] = fuzz.trimf(self.cat_cons.universe, [2, 3, 3])

        # 3) reglas – heurísticas sencillas
        reglas = [
            ctrl.Rule(self.pobfem_ant["baja"]  & self.tothog_ant["pocos"],  self.cat_cons["Pequeña"]),
            ctrl.Rule(self.pobfem_ant["media"] & self.tothog_ant["pocos"],  self.cat_cons["Mediana"]),
            ctrl.Rule(self.pobfem_ant["media"] & self.tothog_ant["medio"],  self.cat_cons["Mediana"]),
            ctrl.Rule(self.pobfem_ant["alta"]  & self.tothog_ant["medio"],  self.cat_cons["Grande"]),
            ctrl.Rule(self.pobfem_ant["alta"]  & self.tothog_ant["muchos"], self.cat_cons["Muy_Grande"]),
            ctrl.Rule(self.pobfem_ant["media"] & self.tothog_ant["muchos"], self.cat_cons["Grande"]),
            ctrl.Rule(self.pobfem_ant["baja"]  & self.tothog_ant["medio"],  self.cat_cons["Mediana"]),
            ctrl.Rule(self.pobfem_ant["baja"]  & self.tothog_ant["muchos"], self.cat_cons["Grande"]),
        ]

        # 4) construir sistema
        self.sistema = ctrl.ControlSystem(reglas)
        return self

    # ──────────────────────────────
    # 2.3  Predicción por lotes
    # ──────────────────────────────
    def predict(self, X_df):
        preds = []
        for _, fila in X_df.iterrows():
            sim = ctrl.ControlSystemSimulation(self.sistema, flush_after_run=1)
            sim.input["POBFEM"] = fila["POBFEM"]
            sim.input["TOTHOG"] = fila["TOTHOG"]
            sim.compute()
            val = sim.output["CATEGORIA_POB"]
            idx = int(round(min(max(val, 0), 3)))
            preds.append(self.etiquetas[idx])
        return np.array(preds)

# ──────────────────────────────────────────────────────────────
# 3.  PREPARACIÓN DE DATOS
# ──────────────────────────────────────────────────────────────
def preparar_datos_borrosos(datos):
    """
    • Crea columna CATEGORIA_POB (ground‑truth)
    • Devuelve X, y, DataFrame limpio
    """
    # a) etiquetas
    datos = datos.copy()
    datos["CATEGORIA_POB"] = datos["POBTOT"].apply(crear_categorias_poblacion_dinamica(datos))

    # b) variables de entrada mínimas
    vars_entrada = ["POBFEM", "TOTHOG"]
    if not all(v in datos.columns for v in vars_entrada):
        return None, None, None

    datos_limpios = datos.dropna(subset=vars_entrada + ["CATEGORIA_POB"])
    X = datos_limpios[vars_entrada]
    y = datos_limpios["CATEGORIA_POB"]
    return X, y, datos_limpios

# ──────────────────────────────────────────────────────────────
# 4.  VISUALIZACIÓN (opcional, se puede comentar)
# ──────────────────────────────────────────────────────────────
def mostrar_metricas(y_test, y_pred):
    print("\n📋 Reporte de clasificación")
    print(classification_report(y_test, y_pred, digits=3))

    acc = accuracy_score(y_test, y_pred)
    print(f"🎯 Precisión global: {acc*100:.1f}%")

    # confusion matrix compacta
    cm = confusion_matrix(y_test, y_pred, labels=["Pequeña", "Mediana", "Grande", "Muy_Grande"])
    print("\nMatriz de confusión (filas: real | columnas: pred):")
    print(pd.DataFrame(cm, index=["Peq", "Med", "Gra", "MuyG"], columns=["Peq", "Med", "Gra", "MuyG"]))

# ──────────────────────────────────────────────────────────────
# 5.  FUNCIÓN PRINCIPAL PARA EL MENÚ
# ──────────────────────────────────────────────────────────────
def ejecutar_logica_borrosa():
    print("🌫️  LÓGICA BORROSA PURA – CLASIFICACIÓN")
    print("=" * 42)

    archivo = "/home/sedc/Proyectos/MineriaDeDatos/data/ceros_sin_columnasAB_limpio_weka.csv"
    if not os.path.isfile(archivo):
        print(f"❌  No se encontró el archivo: {archivo}")
        return

    datos = pd.read_csv(archivo)
    X, y, datos_limpios = preparar_datos_borrosos(datos)
    if X is None:
        print("❌  Las columnas requeridas 'POBFEM' y 'TOTHOG' no existen.")
        return

    print(f"✅  Datos limpios: {len(datos_limpios):,}")

    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    print(f"📊  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # Entrenamiento borroso
    clasif = ClasificadorBorrosoPuro().fit(X_train)

    # Predicción
    y_pred = clasif.predict(X_test)

    # Métricas
    mostrar_metricas(y_test, y_pred)

    # Resumen corto para el menú
    acc = accuracy_score(y_test, y_pred)
    print("\n📝  Resumen:")
    print(f"•  Precisión lógica borrosa: {acc*100:.1f}%")
    print(f"•  Variables usadas: POBFEM, TOTHOG")
    print("•  Árbol de decisión:  ✗  (no utilizado)")
    print("✅  Lógica borrosa finalizada")

    # Devolver por si el menú lo requiere
    return dict(precision=acc, modelo=clasif)

# ──────────────────────────────────────────────────────────────
# 6.  EJECUCIÓN DESDE MENÚ
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ejecutar_logica_borrosa()
