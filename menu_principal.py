#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MENÚ PRINCIPAL OPTIMIZADO - SISTEMA COMPLETO DE MINERÍA DE DATOS
Sistema integral con todas las técnicas supervisadas y no supervisadas
"""

import os
import sys
import time
import importlib.util
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable

# ═══════════════════════════════════════════════════════════════════
# 🔧 CONFIGURACIÓN OPTIMIZADA DEL SISTEMA
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ConfiguracionSistema:
    """Configuración centralizada del sistema de minería de datos"""
    
    # Rutas base del proyecto
    RUTA_BASE: str = '/home/sedc/Proyectos/MineriaDeDatos'
    RUTA_DATOS: str = '/home/sedc/Proyectos/MineriaDeDatos/data/ceros_sin_columnasAB_limpio_weka.csv'
    RUTA_RESULTADOS: str = '/home/sedc/Proyectos/MineriaDeDatos/results/'
    
    # Configuración de visualización
    ANCHO_BANNER: int = 80
    TIEMPO_PAUSA: float = 2.0
    MAX_REINTENTOS: int = 3

# Colores optimizados para la consola
class Colores:
    """Clase de colores con métodos optimizados"""
    AZUL = '\033[94m'
    VERDE = '\033[92m'
    AMARILLO = '\033[93m'
    ROJO = '\033[91m'
    MORADO = '\033[95m'
    CYAN = '\033[96m'
    BLANCO = '\033[97m'
    NEGRITA = '\033[1m'
    FIN = '\033[0m'
    
    @classmethod
    def aplicar(cls, texto: str, color: str) -> str:
        """Aplica color a un texto de manera segura"""
        return f"{color}{texto}{cls.FIN}"

# ═══════════════════════════════════════════════════════════════════
# 🔧 ESTRUCTURA COMPLETA DE TÉCNICAS
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TecnicaInfo:
    """Información detallada de cada técnica"""
    id: int
    nombre: str
    descripcion: str
    categoria: str
    subcategoria: str
    ruta_archivo: str
    nombre_funcion: str
    emoji: str
    dificultad: str
    tiempo_estimado: str

class RegistroTecnicas:
    """Registro completo y optimizado de todas las técnicas"""
    
    def __init__(self):
        self.config = ConfiguracionSistema()
        self.tecnicas = self._cargar_tecnicas_completas()
    
    def _cargar_tecnicas_completas(self) -> Dict[int, TecnicaInfo]:
        """Carga todas las técnicas del sistema de manera optimizada"""
        
        base = self.config.RUTA_BASE
        
        tecnicas = {
            # ═══════════════════════════════════════════════════════════════════
            # 📊 TÉCNICAS SUPERVISADAS - PREDICCIÓN
            # ═══════════════════════════════════════════════════════════════════
            1: TecnicaInfo(
                id=1,
                nombre="🔵 Regresión Lineal",
                descripcion="Predicción lineal de población usando relaciones estadísticas",
                categoria="Supervisada",
                subcategoria="Predicción",
                ruta_archivo=f"{base}/01_Supervisadas/01_Prediccion/01_Regresion/1_regresion_lineal.py",
                nombre_funcion="ejecutar_regresion",
                emoji="📈",
                dificultad="Básico",
                tiempo_estimado="2-3 min"
            ),
            
            2: TecnicaInfo(
                id=2,
                nombre="🌳 Árboles de Predicción",
                descripcion="Predicción usando árboles de decisión interpretables",
                categoria="Supervisada",
                subcategoria="Predicción",
                ruta_archivo=f"{base}/01_Supervisadas/01_Prediccion/02_Arboles_Prediccion/2_arboles_prediccion.py",
                nombre_funcion="ejecutar_arboles",
                emoji="🌲",
                dificultad="Intermedio",
                tiempo_estimado="3-4 min"
            ),
            
            3: TecnicaInfo(
                id=3,
                nombre="🔬 Estimadores de Núcleos",
                descripcion="Predicción avanzada con SVR y K-NN para patrones complejos",
                categoria="Supervisada",
                subcategoria="Predicción",
                ruta_archivo=f"{base}/01_Supervisadas/01_Prediccion/03_Estimador_Nucleos/3_estimador_nucleos.py",
                nombre_funcion="ejecutar_nucleos",
                emoji="⚛️",
                dificultad="Avanzado",
                tiempo_estimado="4-6 min"
            ),
            
            # ═══════════════════════════════════════════════════════════════════
            # 📊 TÉCNICAS SUPERVISADAS - CLASIFICACIÓN
            # ═══════════════════════════════════════════════════════════════════
            4: TecnicaInfo(
                id=4,
                nombre="🌳 Árboles de Decisión",
                descripcion="Clasificación interpretable con reglas jerárquicas",
                categoria="Supervisada",
                subcategoria="Clasificación",
                ruta_archivo=f"{base}/01_Supervisadas/02_Clasificacion/01_Arboles_Decision/1_arboles_decision.py",
                nombre_funcion="ejecutar_arboles_decision",
                emoji="🎯",
                dificultad="Básico",
                tiempo_estimado="3-4 min"
            ),
            
            5: TecnicaInfo(
                id=5,
                nombre="📏 Inducción de Reglas",
                descripcion="Generación automática de reglas IF-THEN explicativas",
                categoria="Supervisada",
                subcategoria="Clasificación",
                ruta_archivo=f"{base}/01_Supervisadas/02_Clasificacion/02_Induccion_Reglas/2_induccion_reglas.py",
                nombre_funcion="ejecutar_induccion_reglas",
                emoji="📋",
                dificultad="Intermedio",
                tiempo_estimado="4-5 min"
            ),
            
            6: TecnicaInfo(
                id=6,
                nombre="🎲 Clasificación Bayesiana",
                descripcion="Clasificación probabilística usando teorema de Bayes",
                categoria="Supervisada",
                subcategoria="Clasificación",
                ruta_archivo=f"{base}/01_Supervisadas/02_Clasificacion/03_Bayesiana/3_clasificacion_bayesiana.py",
                nombre_funcion="ejecutar_clasificacion_bayesiana",
                emoji="🎯",
                dificultad="Intermedio",
                tiempo_estimado="3-4 min"
            ),
            
            7: TecnicaInfo(
                id=7,
                nombre="👥 Basado en Ejemplares (K-NN)",
                descripcion="Clasificación por similitud con vecinos cercanos",
                categoria="Supervisada",
                subcategoria="Clasificación",
                ruta_archivo=f"{base}/01_Supervisadas/02_Clasificacion/04_Basado_Ejemplares/4_basado_ejemplares.py",
                nombre_funcion="ejecutar_clasificacion_ejemplares",
                emoji="👥",
                dificultad="Básico",
                tiempo_estimado="3-4 min"
            ),
            
            8: TecnicaInfo(
                id=8,
                nombre="🧠 Redes de Neuronas",
                descripcion="Aprendizaje profundo con múltiples arquitecturas",
                categoria="Supervisada",
                subcategoria="Clasificación",
                ruta_archivo=f"{base}/01_Supervisadas/02_Clasificacion/05_Redes_Neuronas/5_redes_neuronas.py",
                nombre_funcion="ejecutar_redes_neuronas",
                emoji="🧠",
                dificultad="Avanzado",
                tiempo_estimado="5-8 min"
            ),
            
            9: TecnicaInfo(
                id=9,
                nombre="🌫️ Lógica Borrosa",
                descripcion="Clasificación con conjuntos difusos y reglas borrosas",
                categoria="Supervisada",
                subcategoria="Clasificación",
                ruta_archivo=f"{base}/01_Supervisadas/02_Clasificacion/06_Logica_Borrosa/6_logica_borrosa.py",
                nombre_funcion="ejecutar_logica_borrosa",
                emoji="🌀",
                dificultad="Avanzado",
                tiempo_estimado="4-6 min"
            ),
            
            10: TecnicaInfo(
                id=10,
                nombre="🧬 Técnicas Genéticas",
                descripcion="Optimización evolutiva de características e hiperparámetros",
                categoria="Supervisada",
                subcategoria="Clasificación",
                ruta_archivo=f"{base}/01_Supervisadas/02_Clasificacion/07_Tecnicas_Geneticas/7_tecnicas_geneticas.py",
                nombre_funcion="ejecutar_tecnicas_geneticas",
                emoji="🧬",
                dificultad="Avanzado",
                tiempo_estimado="6-8 min"
            ),
            
            # ═══════════════════════════════════════════════════════════════════
            # 📊 TÉCNICAS NO SUPERVISADAS - CLUSTERING
            # ═══════════════════════════════════════════════════════════════════
            11: TecnicaInfo(
                id=11,
                nombre="📊 Clustering Numérico",
                descripcion="Agrupación K-Means por similitud numérica",
                categoria="No Supervisada",
                subcategoria="Clustering",
                ruta_archivo=f"{base}/02_No_Supervisadas/01_Clustering/01_Numerico/1_clustering_numerico.py",
                nombre_funcion="ejecutar_clustering_numerico",
                emoji="📊",
                dificultad="Básico",
                tiempo_estimado="3-4 min"
            ),
            
            12: TecnicaInfo(
                id=12,
                nombre="🎯 Clustering Conceptual",
                descripcion="Agrupación basada en conceptos y características",
                categoria="No Supervisada",
                subcategoria="Clustering",
                ruta_archivo=f"{base}/02_No_Supervisadas/01_Clustering/02_Conceptual/2_clustering_conceptual.py",
                nombre_funcion="ejecutar_clustering_conceptual",
                emoji="🎯",
                dificultad="Intermedio",
                tiempo_estimado="4-5 min"
            ),
            
            13: TecnicaInfo(
                id=13,
                nombre="🎲 Clustering Probabilístico",
                descripcion="Agrupación EM con modelos probabilísticos",
                categoria="No Supervisada",
                subcategoria="Clustering",
                ruta_archivo=f"{base}/02_No_Supervisadas/01_Clustering/03_Probabilistico/3_clustering_probabilistico.py",
                nombre_funcion="ejecutar_clustering_probabilistico",
                emoji="🎲",
                dificultad="Avanzado",
                tiempo_estimado="4-6 min"
            ),
            
            # ═══════════════════════════════════════════════════════════════════
            # 📊 TÉCNICAS NO SUPERVISADAS - ASOCIACIÓN
            # ═══════════════════════════════════════════════════════════════════
            14: TecnicaInfo(
                id=14,
                nombre="🔗 A Priori (Reglas de Asociación)",
                descripcion="Patrones 'si A entonces B' en datos demográficos",
                categoria="No Supervisada",
                subcategoria="Asociación",
                ruta_archivo=f"{base}/02_No_Supervisadas/02_Asociacion/01_A_Priori/1_apriori_asociacion.py",
                nombre_funcion="ejecutar_apriori",
                emoji="🔗",
                dificultad="Intermedio",
                tiempo_estimado="5-7 min"
            )
        }
        
        return tecnicas
    
    def obtener_tecnica(self, id_tecnica: int) -> Optional[TecnicaInfo]:
        """Obtiene información de una técnica específica"""
        return self.tecnicas.get(id_tecnica)
    
    def obtener_por_categoria(self, categoria: str) -> List[TecnicaInfo]:
        """Obtiene técnicas por categoría"""
        return [t for t in self.tecnicas.values() if t.categoria == categoria]
    
    def obtener_supervisadas(self) -> List[TecnicaInfo]:
        """Obtiene todas las técnicas supervisadas"""
        return self.obtener_por_categoria("Supervisada")
    
    def obtener_no_supervisadas(self) -> List[TecnicaInfo]:
        """Obtiene todas las técnicas no supervisadas"""
        return self.obtener_por_categoria("No Supervisada")

# ═══════════════════════════════════════════════════════════════════
# 🔧 UTILIDADES OPTIMIZADAS
# ═══════════════════════════════════════════════════════════════════

def limpiar_pantalla():
    """Limpia la pantalla de manera optimizada"""
    os.system('cls' if os.name == 'nt' else 'clear')

def pausar_sistema(mensaje: str = "Presiona ENTER para continuar...", tiempo: float = 0):
    """Pausa optimizada del sistema con timeout opcional"""
    try:
        if tiempo > 0:
            print(f"\n{Colores.AMARILLO}{mensaje} (auto-continúa en {tiempo}s){Colores.FIN}")
            time.sleep(tiempo)
        else:
            input(f"\n{Colores.AMARILLO}{mensaje}{Colores.FIN}")
    except KeyboardInterrupt:
        print(f"\n{Colores.AMARILLO}⚠️ Operación interrumpida{Colores.FIN}")

def mostrar_banner_optimizado():
    """Banner principal optimizado con información dinámica"""
    config = ConfiguracionSistema()
    registro = RegistroTecnicas()
    
    total_tecnicas = len(registro.tecnicas)
    supervisadas = len(registro.obtener_supervisadas())
    no_supervisadas = len(registro.obtener_no_supervisadas())
    
    banner = f"""
{Colores.CYAN}╔════════════════════════════════════════════════════════════════════════════════╗
║                                                                                ║
║  {Colores.NEGRITA}🧠 SISTEMA INTEGRAL DE MINERÍA DE DATOS - IA AVANZADA 🧠{Colores.FIN}{Colores.CYAN}              ║
║                                                                                ║
║  {Colores.BLANCO}Universidad: Universidad Tecnológica de Puebla (UTP){Colores.CYAN}                       ║
║  {Colores.BLANCO}Proyecto: Análisis Demográfico Integral Michoacán{Colores.CYAN}                          ║
║  {Colores.BLANCO}Dataset: Censo Poblacional INEGI - 69K+ registros{Colores.CYAN}                          ║
║  {Colores.BLANCO}Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}{Colores.CYAN}                                                     ║
║  {Colores.BLANCO}Técnicas: {total_tecnicas} Total ({supervisadas} Supervisadas + {no_supervisadas} No Supervisadas){Colores.CYAN}                      ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝{Colores.FIN}

{Colores.AMARILLO}🎯 SISTEMA COMPLETO: {total_tecnicas} Técnicas de IA y Machine Learning Implementadas{Colores.FIN}
{Colores.VERDE}📊 OBJETIVO: Análisis integral de datos demográficos con todas las técnicas principales{Colores.FIN}
"""
    print(banner)

def mostrar_menu_principal_optimizado():
    """Menú principal optimizado con estructura mejorada"""
    registro = RegistroTecnicas()
    
    menu = f"""
{Colores.NEGRITA}{Colores.AZUL}═══════════════════════════════════════════════════════════════════════════════
                    🔬 TÉCNICAS SUPERVISADAS (PREDICCIÓN)
═══════════════════════════════════════════════════════════════════════════════{Colores.FIN}"""
    
    # Técnicas de Predicción (1-3)
    for i in range(1, 4):
        tecnica = registro.obtener_tecnica(i)
        if tecnica:
            menu += f"\n{Colores.VERDE}{i}.{Colores.FIN} {Colores.NEGRITA}{tecnica.nombre}{Colores.FIN} - {tecnica.descripcion}"
            menu += f"\n   {Colores.AMARILLO}⏱️ {tecnica.tiempo_estimado} | 📊 {tecnica.dificultad}{Colores.FIN}"
    
    menu += f"""

{Colores.NEGRITA}{Colores.AZUL}═══════════════════════════════════════════════════════════════════════════════
                    🎯 TÉCNICAS SUPERVISADAS (CLASIFICACIÓN)
═══════════════════════════════════════════════════════════════════════════════{Colores.FIN}"""
    
    # Técnicas de Clasificación (4-10)
    for i in range(4, 11):
        tecnica = registro.obtener_tecnica(i)
        if tecnica:
            menu += f"\n{Colores.VERDE}{i}.{Colores.FIN} {Colores.NEGRITA}{tecnica.nombre}{Colores.FIN} - {tecnica.descripcion}"
            menu += f"\n   {Colores.AMARILLO}⏱️ {tecnica.tiempo_estimado} | 📊 {tecnica.dificultad}{Colores.FIN}"
    
    menu += f"""

{Colores.NEGRITA}{Colores.MORADO}═══════════════════════════════════════════════════════════════════════════════
                    🔍 TÉCNICAS NO SUPERVISADAS (DESCUBRIMIENTO)
═══════════════════════════════════════════════════════════════════════════════{Colores.FIN}"""
    
    # Técnicas No Supervisadas (11-14)
    for i in range(11, 15):
        tecnica = registro.obtener_tecnica(i)
        if tecnica:
            menu += f"\n{Colores.CYAN}{i}.{Colores.FIN} {Colores.NEGRITA}{tecnica.nombre}{Colores.FIN} - {tecnica.descripcion}"
            menu += f"\n   {Colores.AMARILLO}⏱️ {tecnica.tiempo_estimado} | 📊 {tecnica.dificultad}{Colores.FIN}"
    
    menu += f"""

{Colores.NEGRITA}{Colores.AMARILLO}═══════════════════════════════════════════════════════════════════════════════
                              🚀 OPCIONES DEL SISTEMA
═══════════════════════════════════════════════════════════════════════════════{Colores.FIN}

{Colores.ROJO}15.{Colores.FIN} {Colores.NEGRITA}🔥 EJECUTAR TODAS LAS TÉCNICAS{Colores.FIN} - Sistema completo ({len(registro.tecnicas)} técnicas)
{Colores.ROJO}16.{Colores.FIN} {Colores.NEGRITA}⚖️ EJECUTAR SOLO SUPERVISADAS{Colores.FIN} - Las {len(registro.obtener_supervisadas())} técnicas de predicción/clasificación
{Colores.ROJO}17.{Colores.FIN} {Colores.NEGRITA}🔍 EJECUTAR SOLO NO SUPERVISADAS{Colores.FIN} - Las {len(registro.obtener_no_supervisadas())} técnicas de descubrimiento

{Colores.VERDE}18.{Colores.FIN} {Colores.NEGRITA}📊 Ver Resultados y Comparar{Colores.FIN} - Revisar reportes, gráficos y comparaciones
{Colores.VERDE}19.{Colores.FIN} {Colores.NEGRITA}🏆 Ranking de Técnicas{Colores.FIN} - Comparación de rendimiento por precisión
{Colores.VERDE}20.{Colores.FIN} {Colores.NEGRITA}📈 Dashboard Ejecutivo{Colores.FIN} - Resumen visual de todos los resultados

{Colores.AMARILLO}21.{Colores.FIN} {Colores.NEGRITA}🔧 Configuración y Diagnóstico{Colores.FIN} - Verificar sistema, rutas y dependencias
{Colores.AMARILLO}22.{Colores.FIN} {Colores.NEGRITA}🧹 Limpiar Resultados{Colores.FIN} - Borrar modelos y reportes anteriores
{Colores.AMARILLO}23.{Colores.FIN} {Colores.NEGRITA}💾 Exportar Proyecto{Colores.FIN} - Crear backup completo del proyecto

{Colores.ROJO}0.{Colores.FIN} {Colores.NEGRITA}❌ Salir del Sistema{Colores.FIN}

{Colores.AZUL}═══════════════════════════════════════════════════════════════════════════════{Colores.FIN}
"""
    print(menu)

# ═══════════════════════════════════════════════════════════════════
# 🔧 SISTEMA DE VERIFICACIÓN OPTIMIZADO
# ═══════════════════════════════════════════════════════════════════

def verificar_sistema_completo() -> Tuple[bool, int, int]:
    """Verificación completa y optimizada del sistema"""
    print(f"{Colores.AMARILLO}🔍 Verificando sistema completo...{Colores.FIN}")
    
    config = ConfiguracionSistema()
    registro = RegistroTecnicas()
    
    # Verificar archivo de datos
    datos_ok = Path(config.RUTA_DATOS).exists()
    print(f"  {'✅' if datos_ok else '❌'} Datos principales: {Colores.VERDE if datos_ok else Colores.ROJO}{'OK' if datos_ok else 'FALTA'}{Colores.FIN}")
    
    # Verificar directorios de resultados
    directorios_resultados = ['graficos', 'modelos', 'reportes']
    for directorio in directorios_resultados:
        ruta = Path(config.RUTA_RESULTADOS) / directorio
        if not ruta.exists():
            ruta.mkdir(parents=True, exist_ok=True)
            print(f"  📁 Carpeta creada: {ruta}")
    
    # Verificar técnicas
    supervisadas_disponibles = 0
    no_supervisadas_disponibles = 0
    
    print(f"\n{Colores.AMARILLO}🔍 Verificando técnicas implementadas...{Colores.FIN}")
    
    for tecnica in registro.tecnicas.values():
        if Path(tecnica.ruta_archivo).exists():
            if tecnica.categoria == "Supervisada":
                supervisadas_disponibles += 1
            else:
                no_supervisadas_disponibles += 1
            print(f"  ✅ {tecnica.nombre}: {Colores.VERDE}OK{Colores.FIN}")
        else:
            print(f"  ❌ {tecnica.nombre}: {Colores.ROJO}NO ENCONTRADO{Colores.FIN}")
    
    total_disponibles = supervisadas_disponibles + no_supervisadas_disponibles
    total_tecnicas = len(registro.tecnicas)
    
    print(f"\n{Colores.CYAN}📊 Resumen del sistema:")
    print(f"  Datos principales: {'✅' if datos_ok else '❌'}")
    print(f"  Técnicas supervisadas: {supervisadas_disponibles}/{len(registro.obtener_supervisadas())}")
    print(f"  Técnicas no supervisadas: {no_supervisadas_disponibles}/{len(registro.obtener_no_supervisadas())}")
    print(f"  TOTAL técnicas disponibles: {total_disponibles}/{total_tecnicas}{Colores.FIN}")
    
    return datos_ok, supervisadas_disponibles, no_supervisadas_disponibles

# ═══════════════════════════════════════════════════════════════════
# 🔧 SISTEMA DE EJECUCIÓN OPTIMIZADO
# ═══════════════════════════════════════════════════════════════════

def importar_modulo_dinamico(ruta_archivo: str, nombre_funcion: str) -> Optional[Callable]:
    """Importación dinámica optimizada con manejo de errores"""
    try:
        if not Path(ruta_archivo).exists():
            print(f"  ❌ Archivo no encontrado: {ruta_archivo}")
            return None
        
        spec = importlib.util.spec_from_file_location("modulo_temporal", ruta_archivo)
        if spec is None or spec.loader is None:
            print(f"  ❌ No se pudo cargar el módulo: {ruta_archivo}")
            return None
        
        modulo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(modulo)
        
        if hasattr(modulo, nombre_funcion):
            return getattr(modulo, nombre_funcion)
        else:
            print(f"  ⚠️ Función '{nombre_funcion}' no encontrada en {Path(ruta_archivo).name}")
            return None
            
    except Exception as e:
        print(f"  ❌ Error importando {Path(ruta_archivo).name}: {e}")
        return None

def ejecutar_tecnica_optimizada(id_tecnica: int) -> bool:
    """Ejecución optimizada de técnicas individuales"""
    registro = RegistroTecnicas()
    tecnica = registro.obtener_tecnica(id_tecnica)
    
    if not tecnica:
        print(f"{Colores.ROJO}❌ Técnica no encontrada: ID {id_tecnica}{Colores.FIN}")
        return False
    
    print(f"\n{Colores.CYAN}{'='*80}")
    print(f"🚀 EJECUTANDO: {tecnica.nombre}")
    print(f"📝 {tecnica.descripcion}")
    print(f"⏱️ Tiempo estimado: {tecnica.tiempo_estimado} | 📊 Dificultad: {tecnica.dificultad}")
    print(f"📂 Categoría: {tecnica.categoria} → {tecnica.subcategoria}")
    print(f"{'='*80}{Colores.FIN}\n")
    
    inicio = time.time()
    
    try:
        # Importar función dinámicamente
        funcion = importar_modulo_dinamico(tecnica.ruta_archivo, tecnica.nombre_funcion)
        
        if funcion is None:
            print(f"{Colores.ROJO}❌ No se pudo importar la función {tecnica.nombre_funcion}{Colores.FIN}")
            return False
        
        # Ejecutar la técnica
        print(f"{Colores.VERDE}✅ Función importada correctamente. Iniciando ejecución...{Colores.FIN}\n")
        resultado = funcion()
        
        fin = time.time()
        duracion = fin - inicio
        
        print(f"\n{Colores.VERDE}{'='*60}")
        print(f"✅ TÉCNICA COMPLETADA EXITOSAMENTE")
        print(f"⏱️ Tiempo de ejecución: {duracion:.1f}s ({duracion/60:.1f} min)")
        print(f"📊 Resultados guardados en: /results/")
        print(f"{'='*60}{Colores.FIN}")
        
        return True
        
    except KeyboardInterrupt:
        print(f"\n{Colores.AMARILLO}⚠️ Ejecución interrumpida por el usuario{Colores.FIN}")
        return False
    except Exception as e:
        fin = time.time()
        duracion = fin - inicio
        
        print(f"\n{Colores.ROJO}{'='*60}")
        print(f"❌ ERROR EN LA EJECUCIÓN:")
        print(f"   {str(e)}")
        print(f"⏱️ Tiempo transcurrido: {duracion:.1f}s")
        print(f"{'='*60}{Colores.FIN}")
        return False

def ejecutar_conjunto_tecnicas(tecnicas: List[TecnicaInfo], nombre_conjunto: str) -> Dict[str, bool]:
    """Ejecuta un conjunto de técnicas de manera optimizada"""
    print(f"\n{Colores.MORADO}{Colores.NEGRITA}🚀 EJECUTANDO {nombre_conjunto.upper()}{Colores.FIN}")
    print(f"{Colores.AMARILLO}📊 Total técnicas: {len(tecnicas)}")
    
    # Calcular tiempo estimado total
    tiempo_total_estimado = sum([
        int(t.tiempo_estimado.split('-')[1].split()[0]) 
        for t in tecnicas 
        if '-' in t.tiempo_estimado and t.tiempo_estimado.split('-')[1].split()[0].isdigit()
    ])
    
    print(f"⏱️ Tiempo estimado total: {tiempo_total_estimado}-{tiempo_total_estimado + 10} minutos{Colores.FIN}")
    
    # Confirmar ejecución
    confirmacion = input(f"\n{Colores.AMARILLO}¿Ejecutar {nombre_conjunto}? (s/N): {Colores.FIN}").strip().lower()
    if confirmacion not in ['s', 'si', 'sí', 'y', 'yes']:
        print(f"{Colores.AMARILLO}❌ Ejecución cancelada{Colores.FIN}")
        return {}
    
    resultados = {}
    tiempo_inicio = time.time()
    exitosas = 0
    
    for i, tecnica in enumerate(tecnicas, 1):
        print(f"\n{Colores.CYAN}{'─'*80}")
        print(f"[{i}/{len(tecnicas)}] {tecnica.nombre}")
        print(f"📝 {tecnica.descripcion}")
        print(f"⏱️ {tecnica.tiempo_estimado} | 📊 {tecnica.dificultad}")
        print(f"{'─'*80}{Colores.FIN}")
        
        exito = ejecutar_tecnica_optimizada(tecnica.id)
        resultados[tecnica.nombre] = exito
        
        if exito:
            exitosas += 1
            print(f"{Colores.VERDE}✅ {tecnica.nombre} completada{Colores.FIN}")
        else:
            print(f"{Colores.ROJO}❌ {tecnica.nombre} falló{Colores.FIN}")
        
        # Pausa entre técnicas (excepto la última)
        if i < len(tecnicas):
            print(f"\n{Colores.AMARILLO}⏳ Preparando siguiente técnica...{Colores.FIN}")
            time.sleep(2)
    
    duracion = (time.time() - tiempo_inicio) / 60
    
    print(f"\n{Colores.MORADO}{'='*60}")
    print(f"📊 RESUMEN {nombre_conjunto.upper()}:")
    print(f"✅ Exitosas: {exitosas}/{len(tecnicas)} ({exitosas/len(tecnicas)*100:.1f}%)")
    print(f"⏱️ Tiempo total: {duracion:.1f} minutos")
    print(f"{'='*60}{Colores.FIN}")
    
    return resultados

# ═══════════════════════════════════════════════════════════════════
# 🔧 FUNCIONES DE ANÁLISIS Y GESTIÓN
# ═══════════════════════════════════════════════════════════════════

def ver_resultados_guardados():
    """Análisis optimizado de resultados guardados"""
    print(f"\n{Colores.CYAN}📊 ANÁLISIS DE RESULTADOS GUARDADOS...{Colores.FIN}\n")
    
    config = ConfiguracionSistema()
    rutas = {
        'Gráficos': Path(config.RUTA_RESULTADOS) / 'graficos',
        'Modelos': Path(config.RUTA_RESULTADOS) / 'modelos',
        'Reportes': Path(config.RUTA_RESULTADOS) / 'reportes'
    }
    
    total_archivos = 0
    total_tamaño = 0
    tecnicas_completadas = set()
    
    for categoria, ruta in rutas.items():
        print(f"{Colores.NEGRITA}{categoria}:{Colores.FIN}")
        
        if ruta.exists():
            archivos = list(ruta.glob('*'))
            if archivos:
                # Ordenar por fecha de modificación
                archivos.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                
                for archivo in archivos[:10]:  # Mostrar solo los 10 más recientes
                    if archivo.is_file():
                        tamaño = archivo.stat().st_size
                        fecha = datetime.fromtimestamp(archivo.stat().st_mtime)
                        
                        # Iconos según el tipo de archivo
                        iconos = {
                            '.png': "🖼️", '.jpg': "🖼️", '.jpeg': "🖼️",
                            '.pkl': "🤖", '.joblib': "🤖",
                            '.txt': "📄", '.md': "📄",
                            '.csv': "📊", '.json': "📋"
                        }
                        icono = iconos.get(archivo.suffix.lower(), "📁")
                        
                        print(f"  {icono} {archivo.name}")
                        print(f"      📏 {tamaño/1024:.1f} KB | 📅 {fecha.strftime('%Y-%m-%d %H:%M')}")
                        
                        total_archivos += 1
                        total_tamaño += tamaño
                        
                        # Detectar técnicas completadas
                        nombre_sin_ext = archivo.stem.lower()
                        if any(keyword in nombre_sin_ext for keyword in 
                              ['regresion', 'arboles', 'nucleos', 'clasificacion', 'bayesian', 
                               'knn', 'redes', 'borrosa', 'genetica', 'clustering', 'apriori']):
                            tecnicas_completadas.add(nombre_sin_ext.replace('_', ' ').title())
                
                if len(archivos) > 10:
                    print(f"  {Colores.AMARILLO}... y {len(archivos) - 10} archivos más{Colores.FIN}")
            else:
                print(f"  {Colores.AMARILLO}📭 Carpeta vacía{Colores.FIN}")
        else:
            print(f"  {Colores.ROJO}❌ Carpeta no existe{Colores.FIN}")
        print()
    
    # Resumen total
    if total_archivos > 0:
        print(f"{Colores.CYAN}📊 RESUMEN TOTAL:")
        print(f"  📁 Total archivos: {total_archivos}")
        print(f"  💾 Espacio utilizado: {total_tamaño/1024/1024:.1f} MB")
        print(f"  🔬 Técnicas detectadas: {len(tecnicas_completadas)}/14{Colores.FIN}")
        
        if tecnicas_completadas:
            print(f"\n{Colores.VERDE}✅ Técnicas completadas:")
            for tecnica in sorted(tecnicas_completadas)[:10]:
                print(f"  🎯 {tecnica}{Colores.FIN}")

def crear_ranking_tecnicas():
    """Crear ranking optimizado de técnicas por rendimiento"""
    print(f"\n{Colores.CYAN}🏆 GENERANDO RANKING DE TÉCNICAS...{Colores.FIN}\n")
    
    config = ConfiguracionSistema()
    reportes_path = Path(config.RUTA_RESULTADOS) / 'reportes'
    
    if not reportes_path.exists():
        print(f"{Colores.ROJO}❌ No se encontraron reportes{Colores.FIN}")
        return
    
    tecnicas_rendimiento = []
    
    # Buscar archivos de reporte
    archivos_reporte = list(reportes_path.glob('*_reporte.txt'))
    
    for archivo in archivos_reporte:
        try:
            contenido = archivo.read_text(encoding='utf-8')
            
            # Extraer precisión del contenido usando regex
            import re
            matches = re.findall(r'[Pp]recisión:?\s*(\d+\.\d+)', contenido)
            
            if matches:
                precision = float(matches[0])
                tecnica_nombre = archivo.stem.replace('_reporte', '').replace('_', ' ').title()
                
                # Determinar categoría
                categoria = "Supervisada" if any(x in archivo.name.lower() for x in 
                          ['clasificacion', 'arboles', 'bayesian', 'knn', 'redes', 'borrosa', 
                           'genetica', 'regresion', 'nucleos']) else "No Supervisada"
                
                tecnicas_rendimiento.append({
                    'nombre': tecnica_nombre,
                    'precision': precision,
                    'categoria': categoria,
                    'archivo': archivo.name
                })
                
        except Exception as e:
            print(f"  ⚠️ Error leyendo {archivo.name}: {e}")
    
    if not tecnicas_rendimiento:
        print(f"{Colores.AMARILLO}⚠️ No se encontraron métricas de rendimiento{Colores.FIN}")
        return
    
    # Ordenar por precisión
    tecnicas_rendimiento.sort(key=lambda x: x['precision'], reverse=True)
    
    print(f"{Colores.NEGRITA}🏆 RANKING DE TÉCNICAS POR PRECISIÓN:{Colores.FIN}")
    print("=" * 70)
    
    for i, tecnica in enumerate(tecnicas_rendimiento, 1):
        if i == 1:
            emoji = "🥇"
            color = Colores.AMARILLO
        elif i == 2:
            emoji = "🥈"
            color = Colores.BLANCO
        elif i == 3:
            emoji = "🥉"
            color = Colores.AMARILLO
        else:
            emoji = f"{i:2d}."
            color = Colores.VERDE
        
        categoria_emoji = "⚖️" if tecnica['categoria'] == "Supervisada" else "🔍"
        
        print(f"{color}{emoji} {tecnica['nombre']:30} | {categoria_emoji} {tecnica['categoria']:13} | 🎯 {tecnica['precision']:.3f}{Colores.FIN}")
    
    # Estadísticas
    supervisadas = [t for t in tecnicas_rendimiento if t['categoria'] == "Supervisada"]
    no_supervisadas = [t for t in tecnicas_rendimiento if t['categoria'] == "No Supervisada"]
    
    print(f"\n{Colores.CYAN}📊 ESTADÍSTICAS:")
    if supervisadas:
        precision_sup = sum(t['precision'] for t in supervisadas) / len(supervisadas)
        print(f"  ⚖️ Precisión promedio supervisadas: {precision_sup:.3f}")
    if no_supervisadas:
        precision_no_sup = sum(t['precision'] for t in no_supervisadas) / len(no_supervisadas)
        print(f"  🔍 Precisión promedio no supervisadas: {precision_no_sup:.3f}")
    
    precision_total = sum(t['precision'] for t in tecnicas_rendimiento) / len(tecnicas_rendimiento)
    print(f"  🎯 Precisión promedio total: {precision_total:.3f}{Colores.FIN}")

def limpiar_resultados():
    """Limpieza optimizada de resultados"""
    print(f"\n{Colores.AMARILLO}🧹 LIMPIEZA DE RESULTADOS ANTERIORES{Colores.FIN}")
    
    confirmacion = input(f"\n{Colores.ROJO}⚠️ ¿Borrar TODOS los resultados? (s/N): {Colores.FIN}").strip().lower()
    if confirmacion not in ['s', 'si', 'sí', 'y', 'yes']:
        print(f"{Colores.AMARILLO}❌ Limpieza cancelada{Colores.FIN}")
        return
    
    config = ConfiguracionSistema()
    rutas_limpiar = [
        Path(config.RUTA_RESULTADOS) / 'graficos',
        Path(config.RUTA_RESULTADOS) / 'modelos',
        Path(config.RUTA_RESULTADOS) / 'reportes'
    ]
    
    archivos_borrados = 0
    
    for ruta in rutas_limpiar:
        if ruta.exists():
            for archivo in ruta.glob('*'):
                if archivo.is_file():
                    try:
                        archivo.unlink()
                        archivos_borrados += 1
                    except Exception as e:
                        print(f"  ❌ Error borrando {archivo.name}: {e}")
    
    print(f"\n{Colores.VERDE}✅ Limpieza completada")
    print(f"  🗑️ Archivos borrados: {archivos_borrados}")
    print(f"  📁 Carpetas mantenidas para nuevos resultados{Colores.FIN}")

# ═══════════════════════════════════════════════════════════════════
# 🔧 FUNCIÓN PRINCIPAL OPTIMIZADA
# ═══════════════════════════════════════════════════════════════════

def main():
    """Función principal optimizada del menú"""
    registro = RegistroTecnicas()
    
    while True:
        try:
            limpiar_pantalla()
            mostrar_banner_optimizado()
            mostrar_menu_principal_optimizado()
            
            opcion = input(f"{Colores.AMARILLO}👉 Selecciona una opción (0-23): {Colores.FIN}").strip()
            
            if opcion == '0':
                print(f"\n{Colores.VERDE}👋 ¡Gracias por usar el Sistema Integral de Minería de Datos!")
                print(f"🎓 ¡Éxito en tu proyecto de IA y Análisis de Datos!")
                print(f"📧 Revisa los reportes generados para documentar tu trabajo{Colores.FIN}\n")
                break
            
            # Técnicas individuales (1-14)
            elif opcion in [str(i) for i in range(1, 15)]:
                tecnica_id = int(opcion)
                ejecutar_tecnica_optimizada(tecnica_id)
            
            # Ejecutar todas las técnicas
            elif opcion == '15':
                todas_tecnicas = list(registro.tecnicas.values())
                ejecutar_conjunto_tecnicas(todas_tecnicas, "SISTEMA COMPLETO")
            
            # Ejecutar solo supervisadas
            elif opcion == '16':
                supervisadas = registro.obtener_supervisadas()
                ejecutar_conjunto_tecnicas(supervisadas, "TÉCNICAS SUPERVISADAS")
            
            # Ejecutar solo no supervisadas
            elif opcion == '17':
                no_supervisadas = registro.obtener_no_supervisadas()
                ejecutar_conjunto_tecnicas(no_supervisadas, "TÉCNICAS NO SUPERVISADAS")
            
            # Ver resultados
            elif opcion == '18':
                ver_resultados_guardados()
            
            # Ranking de técnicas
            elif opcion == '19':
                crear_ranking_tecnicas()
            
            # Dashboard ejecutivo
            elif opcion == '20':
                print(f"{Colores.AMARILLO}📈 Dashboard ejecutivo en desarrollo...{Colores.FIN}")
                # TODO: Implementar dashboard con matplotlib
            
            # Configuración y diagnóstico
            elif opcion == '21':
                verificar_sistema_completo()
            
            # Limpiar resultados
            elif opcion == '22':
                limpiar_resultados()
            
            # Exportar proyecto
            elif opcion == '23':
                print(f"{Colores.AMARILLO}💾 Exportación de proyecto en desarrollo...{Colores.FIN}")
                # TODO: Implementar exportación con shutil
            
            else:
                print(f"{Colores.ROJO}❌ Opción inválida. Selecciona un número del 0 al 23.{Colores.FIN}")
                time.sleep(2)
            
            if opcion != '0':
                pausar_sistema("📎 Presiona ENTER para volver al menú principal...")
                
        except KeyboardInterrupt:
            print(f"\n\n{Colores.AMARILLO}⚠️ Operación cancelada por el usuario")
            confirmacion = input(f"¿Deseas salir del sistema? (s/N): {Colores.FIN}").strip().lower()
            if confirmacion in ['s', 'si', 'sí', 'y', 'yes']:
                print(f"{Colores.VERDE}👋 ¡Hasta luego!{Colores.FIN}\n")
                break
        except Exception as e:
            print(f"\n{Colores.ROJO}❌ Error inesperado: {e}{Colores.FIN}")
            pausar_sistema("📎 Presiona ENTER para continuar...")

# ═══════════════════════════════════════════════════════════════════
# 🔧 PUNTO DE ENTRADA OPTIMIZADO
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Verificar sistema antes de iniciar
    print(f"{Colores.CYAN}🔍 Inicializando Sistema Integral de Minería de Datos...{Colores.FIN}")
    time.sleep(1)
    
    try:
        datos_ok, supervisadas_disponibles, no_supervisadas_disponibles = verificar_sistema_completo()
        total_tecnicas = supervisadas_disponibles + no_supervisadas_disponibles
        
        if datos_ok and total_tecnicas >= 1:
            print(f"\n{Colores.VERDE}✅ Sistema inicializado correctamente")
            print(f"📊 Datos: OK | Técnicas: {total_tecnicas}/14 ({supervisadas_disponibles} sup. + {no_supervisadas_disponibles} no sup.)")
            
            if total_tecnicas == 14:
                print(f"🎉 ¡Sistema COMPLETO disponible!{Colores.FIN}")
            elif total_tecnicas >= 10:
                print(f"👍 Sistema casi completo{Colores.FIN}")
            else:
                print(f"🔧 Sistema en desarrollo{Colores.FIN}")
                
            time.sleep(2)
            main()
        else:
            print(f"\n{Colores.ROJO}❌ Sistema no está completamente configurado")
            print(f"📊 Datos: {'OK' if datos_ok else 'FALTA'} | Técnicas: {total_tecnicas}/14{Colores.FIN}")
            
            continuar = input(f"\n{Colores.AMARILLO}¿Deseas continuar de todas formas? (s/N): {Colores.FIN}").strip().lower()
            if continuar in ['s', 'si', 'sí', 'y', 'yes']:
                main()
            else:
                print(f"{Colores.AMARILLO}💡 Configura el sistema y vuelve a intentar")
                print(f"📋 Usa la opción 21 del menú para más detalles{Colores.FIN}")
                
    except Exception as e:
        print(f"{Colores.ROJO}❌ Error crítico en inicialización: {e}{Colores.FIN}")
        print(f"{Colores.AMARILLO}💡 Verifica la instalación del sistema{Colores.FIN}")