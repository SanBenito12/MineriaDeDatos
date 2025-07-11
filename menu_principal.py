#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MENÚ PRINCIPAL SIMPLIFICADO - SISTEMA DE MINERÍA DE DATOS
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
# 🔧 CONFIGURACIÓN DEL SISTEMA
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ConfiguracionSistema:
    """Configuración centralizada del sistema de minería de datos"""
    
    # Rutas base del proyecto
    RUTA_BASE: str = '/home/sedc/Proyectos/MineriaDeDatos'
    RUTA_DATOS: str = '/home/sedc/Proyectos/MineriaDeDatos/data/ceros_sin_columnasAB_limpio_weka.csv'
    RUTA_RESULTADOS: str = '/home/sedc/Proyectos/MineriaDeDatos/results/'

# Colores para la consola
class Colores:
    """Clase de colores"""
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
        """Aplica color a un texto"""
        return f"{color}{texto}{cls.FIN}"

# ═══════════════════════════════════════════════════════════════════
# 🔧 ESTRUCTURA DE TÉCNICAS
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TecnicaInfo:
    """Información de cada técnica"""
    id: int
    nombre: str
    descripcion: str
    categoria: str
    ruta_archivo: str
    nombre_funcion: str
    tiempo_estimado: str

class RegistroTecnicas:
    """Registro de todas las técnicas"""
    
    def __init__(self):
        self.config = ConfiguracionSistema()
        self.tecnicas = self._cargar_tecnicas()
    
    def _cargar_tecnicas(self) -> Dict[int, TecnicaInfo]:
        """Carga todas las técnicas del sistema"""
        
        base = self.config.RUTA_BASE
        
        tecnicas = {
            # TÉCNICAS SUPERVISADAS - PREDICCIÓN
            1: TecnicaInfo(
                id=1,
                nombre="🔵 Regresión Lineal",
                descripcion="Predicción lineal de población usando relaciones estadísticas",
                categoria="Supervisada",
                ruta_archivo=f"{base}/01_Supervisadas/01_Prediccion/01_Regresion/1_regresion_lineal.py",
                nombre_funcion="ejecutar_regresion",
                tiempo_estimado="2-3 min"
            ),
            
            2: TecnicaInfo(
                id=2,
                nombre="🌳 Árboles de Predicción",
                descripcion="Predicción usando árboles de decisión interpretables",
                categoria="Supervisada",
                ruta_archivo=f"{base}/01_Supervisadas/01_Prediccion/02_Arboles_Prediccion/2_arboles_prediccion.py",
                nombre_funcion="ejecutar_arboles",
                tiempo_estimado="3-4 min"
            ),
            
            3: TecnicaInfo(
                id=3,
                nombre="🔬 Estimadores de Núcleos",
                descripcion="Predicción avanzada con SVR y K-NN para patrones complejos",
                categoria="Supervisada",
                ruta_archivo=f"{base}/01_Supervisadas/01_Prediccion/03_Estimador_Nucleos/3_estimador_nucleos.py",
                nombre_funcion="ejecutar_nucleos",
                tiempo_estimado="4-6 min"
            ),
            
            # TÉCNICAS SUPERVISADAS - CLASIFICACIÓN
            4: TecnicaInfo(
                id=4,
                nombre="🌳 Árboles de Decisión",
                descripcion="Clasificación interpretable con reglas jerárquicas",
                categoria="Supervisada",
                ruta_archivo=f"{base}/01_Supervisadas/02_Clasificacion/01_Arboles_Decision/1_arboles_decision.py",
                nombre_funcion="ejecutar_arboles_decision",
                tiempo_estimado="3-4 min"
            ),
            
            5: TecnicaInfo(
                id=5,
                nombre="📏 Inducción de Reglas",
                descripcion="Generación automática de reglas IF-THEN explicativas",
                categoria="Supervisada",
                ruta_archivo=f"{base}/01_Supervisadas/02_Clasificacion/02_Induccion_Reglas/2_induccion_reglas.py",
                nombre_funcion="ejecutar_induccion_reglas",
                tiempo_estimado="4-5 min"
            ),
            
            6: TecnicaInfo(
                id=6,
                nombre="🎲 Clasificación Bayesiana",
                descripcion="Clasificación probabilística usando teorema de Bayes",
                categoria="Supervisada",
                ruta_archivo=f"{base}/01_Supervisadas/02_Clasificacion/03_Bayesiana/3_clasificacion_bayesiana.py",
                nombre_funcion="ejecutar_clasificacion_bayesiana",
                tiempo_estimado="3-4 min"
            ),
            
            7: TecnicaInfo(
                id=7,
                nombre="👥 Basado en Ejemplares (K-NN)",
                descripcion="Clasificación por similitud con vecinos cercanos",
                categoria="Supervisada",
                ruta_archivo=f"{base}/01_Supervisadas/02_Clasificacion/04_Basado_Ejemplares/4_basado_ejemplares.py",
                nombre_funcion="ejecutar_clasificacion_ejemplares",
                tiempo_estimado="3-4 min"
            ),
            
            8: TecnicaInfo(
                id=8,
                nombre="🧠 Redes de Neuronas",
                descripcion="Aprendizaje profundo con múltiples arquitecturas",
                categoria="Supervisada",
                ruta_archivo=f"{base}/01_Supervisadas/02_Clasificacion/05_Redes_Neuronas/5_redes_neuronas.py",
                nombre_funcion="ejecutar_redes_neuronas",
                tiempo_estimado="5-8 min"
            ),
            
            9: TecnicaInfo(
                id=9,
                nombre="🌫️ Lógica Borrosa",
                descripcion="Clasificación con conjuntos difusos y reglas borrosas",
                categoria="Supervisada",
                ruta_archivo=f"{base}/01_Supervisadas/02_Clasificacion/06_Logica_Borrosa/6_logica_borrosa.py",
                nombre_funcion="ejecutar_logica_borrosa",
                tiempo_estimado="4-6 min"
            ),
            
            10: TecnicaInfo(
                id=10,
                nombre="🧬 Técnicas Genéticas",
                descripcion="Optimización evolutiva de características e hiperparámetros",
                categoria="Supervisada",
                ruta_archivo=f"{base}/01_Supervisadas/02_Clasificacion/07_Tecnicas_Geneticas/7_tecnicas_geneticas.py",
                nombre_funcion="ejecutar_tecnicas_geneticas",
                tiempo_estimado="6-8 min"
            ),
            
            # TÉCNICAS NO SUPERVISADAS - CLUSTERING
            11: TecnicaInfo(
                id=11,
                nombre="📊 Clustering Numérico",
                descripcion="Agrupación K-Means por similitud numérica",
                categoria="No Supervisada",
                ruta_archivo=f"{base}/02_No_Supervisadas/01_Clustering/01_Numerico/1_clustering_numerico.py",
                nombre_funcion="ejecutar_clustering_numerico",
                tiempo_estimado="3-4 min"
            ),
            
            12: TecnicaInfo(
                id=12,
                nombre="🎯 Clustering Conceptual",
                descripcion="Agrupación basada en conceptos y características",
                categoria="No Supervisada",
                ruta_archivo=f"{base}/02_No_Supervisadas/01_Clustering/02_Conceptual/2_clustering_conceptual.py",
                nombre_funcion="ejecutar_clustering_conceptual",
                tiempo_estimado="4-5 min"
            ),
            
            13: TecnicaInfo(
                id=13,
                nombre="🎲 Clustering Probabilístico",
                descripcion="Agrupación EM con modelos probabilísticos",
                categoria="No Supervisada",
                ruta_archivo=f"{base}/02_No_Supervisadas/01_Clustering/03_Probabilistico/3_clustering_probabilistico.py",
                nombre_funcion="ejecutar_clustering_probabilistico",
                tiempo_estimado="4-6 min"
            ),
            
            # TÉCNICAS NO SUPERVISADAS - ASOCIACIÓN
            14: TecnicaInfo(
                id=14,
                nombre="🔗 A Priori (Reglas de Asociación)",
                descripcion="Patrones 'si A entonces B' en datos demográficos",
                categoria="No Supervisada",
                ruta_archivo=f"{base}/02_No_Supervisadas/02_Asociacion/01_A_Priori/1_apriori_asociacion.py",
                nombre_funcion="ejecutar_apriori",
                tiempo_estimado="5-7 min"
            )
        }
        
        return tecnicas
    
    def obtener_tecnica(self, id_tecnica: int) -> Optional[TecnicaInfo]:
        """Obtiene información de una técnica específica"""
        return self.tecnicas.get(id_tecnica)
    
    def obtener_supervisadas(self) -> List[TecnicaInfo]:
        """Obtiene todas las técnicas supervisadas"""
        return [t for t in self.tecnicas.values() if t.categoria == "Supervisada"]
    
    def obtener_no_supervisadas(self) -> List[TecnicaInfo]:
        """Obtiene todas las técnicas no supervisadas"""
        return [t for t in self.tecnicas.values() if t.categoria == "No Supervisada"]

# ═══════════════════════════════════════════════════════════════════
# 🔧 UTILIDADES
# ═══════════════════════════════════════════════════════════════════

def limpiar_pantalla():
    """Limpia la pantalla"""
    os.system('cls' if os.name == 'nt' else 'clear')

def pausar_sistema(mensaje: str = "Presiona ENTER para continuar..."):
    """Pausa del sistema"""
    try:
        input(f"\n{Colores.AMARILLO}{mensaje}{Colores.FIN}")
    except KeyboardInterrupt:
        print(f"\n{Colores.AMARILLO}⚠️ Operación interrumpida{Colores.FIN}")

def mostrar_banner():
    """Banner principal simplificado"""
    registro = RegistroTecnicas()
    total_tecnicas = len(registro.tecnicas)
    
    banner = f"""
{Colores.CYAN}╔════════════════════════════════════════════════════════════════════════════════╗
║                                                                                ║
║  {Colores.NEGRITA}🧠 SISTEMA DE MINERÍA DE DATOS - IA AVANZADA 🧠{Colores.FIN}{Colores.CYAN}                      ║
║                                                                                ║
║  {Colores.BLANCO}Universidad: Universidad Tecnológica de Puebla (UTP){Colores.CYAN}                       ║
║  {Colores.BLANCO}Proyecto: Análisis Demográfico Integral Michoacán{Colores.CYAN}                          ║
║  {Colores.BLANCO}Técnicas Disponibles: {total_tecnicas} técnicas implementadas{Colores.CYAN}                              ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝{Colores.FIN}
"""
    print(banner)

def mostrar_menu_principal():
    """Menú principal simplificado"""
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
    
    menu += f"""

{Colores.NEGRITA}{Colores.AZUL}═══════════════════════════════════════════════════════════════════════════════
                    🎯 TÉCNICAS SUPERVISADAS (CLASIFICACIÓN)
═══════════════════════════════════════════════════════════════════════════════{Colores.FIN}"""
    
    # Técnicas de Clasificación (4-10)
    for i in range(4, 11):
        tecnica = registro.obtener_tecnica(i)
        if tecnica:
            menu += f"\n{Colores.VERDE}{i}.{Colores.FIN} {Colores.NEGRITA}{tecnica.nombre}{Colores.FIN} - {tecnica.descripcion}"
    
    menu += f"""

{Colores.NEGRITA}{Colores.MORADO}═══════════════════════════════════════════════════════════════════════════════
                    🔍 TÉCNICAS NO SUPERVISADAS (DESCUBRIMIENTO)
═══════════════════════════════════════════════════════════════════════════════{Colores.FIN}"""
    
    # Técnicas No Supervisadas (11-14)
    for i in range(11, 15):
        tecnica = registro.obtener_tecnica(i)
        if tecnica:
            menu += f"\n{Colores.CYAN}{i}.{Colores.FIN} {Colores.NEGRITA}{tecnica.nombre}{Colores.FIN} - {tecnica.descripcion}"
    
    menu += f"""

{Colores.NEGRITA}{Colores.AMARILLO}═══════════════════════════════════════════════════════════════════════════════
                              🚀 OPCIONES DEL SISTEMA
═══════════════════════════════════════════════════════════════════════════════{Colores.FIN}

{Colores.ROJO}15.{Colores.FIN} {Colores.NEGRITA}🔥 EJECUTAR TODAS LAS TÉCNICAS{Colores.FIN} - Sistema completo ({len(registro.tecnicas)} técnicas)
{Colores.ROJO}16.{Colores.FIN} {Colores.NEGRITA}⚖️ EJECUTAR SOLO SUPERVISADAS{Colores.FIN} - Las {len(registro.obtener_supervisadas())} técnicas de predicción/clasificación
{Colores.ROJO}17.{Colores.FIN} {Colores.NEGRITA}🔍 EJECUTAR SOLO NO SUPERVISADAS{Colores.FIN} - Las {len(registro.obtener_no_supervisadas())} técnicas de descubrimiento

{Colores.AMARILLO}18.{Colores.FIN} {Colores.NEGRITA}🧹 Limpiar Resultados{Colores.FIN} - Borrar modelos y reportes anteriores

{Colores.ROJO}0.{Colores.FIN} {Colores.NEGRITA}❌ Salir del Sistema{Colores.FIN}

{Colores.AZUL}═══════════════════════════════════════════════════════════════════════════════{Colores.FIN}
"""
    print(menu)

# ═══════════════════════════════════════════════════════════════════
# 🔧 SISTEMA DE EJECUCIÓN
# ═══════════════════════════════════════════════════════════════════

def importar_modulo_dinamico(ruta_archivo: str, nombre_funcion: str) -> Optional[Callable]:
    """Importación dinámica con manejo de errores"""
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

def ejecutar_tecnica(id_tecnica: int) -> bool:
    """Ejecución de técnicas individuales"""
    registro = RegistroTecnicas()
    tecnica = registro.obtener_tecnica(id_tecnica)
    
    if not tecnica:
        print(f"{Colores.ROJO}❌ Técnica no encontrada: ID {id_tecnica}{Colores.FIN}")
        return False
    
    print(f"\n{Colores.CYAN}{'='*80}")
    print(f"🚀 EJECUTANDO: {tecnica.nombre}")
    print(f"📝 {tecnica.descripcion}")
    print(f"⏱️ Tiempo estimado: {tecnica.tiempo_estimado}")
    print(f"{'='*80}{Colores.FIN}\n")
    
    inicio = time.time()
    
    try:
        # Importar función dinámicamente
        funcion = importar_modulo_dinamico(tecnica.ruta_archivo, tecnica.nombre_funcion)
        
        if funcion is None:
            return False
        
        # Ejecutar la técnica
        print(f"{Colores.VERDE}✅ Función importada correctamente. Iniciando ejecución...{Colores.FIN}\n")
        resultado = funcion()
        
        fin = time.time()
        duracion = fin - inicio
        
        print(f"\n{Colores.VERDE}{'='*60}")
        print(f"✅ TÉCNICA COMPLETADA EXITOSAMENTE")
        print(f"⏱️ Tiempo de ejecución: {duracion:.1f}s")
        print(f"{'='*60}{Colores.FIN}")
        
        return True
        
    except KeyboardInterrupt:
        print(f"\n{Colores.AMARILLO}⚠️ Ejecución interrumpida por el usuario{Colores.FIN}")
        return False
    except Exception as e:
        print(f"\n{Colores.ROJO}❌ ERROR EN LA EJECUCIÓN: {str(e)}{Colores.FIN}")
        return False

def ejecutar_conjunto_tecnicas(tecnicas: List[TecnicaInfo], nombre_conjunto: str):
    """Ejecuta un conjunto de técnicas"""
    print(f"\n{Colores.MORADO}{Colores.NEGRITA}🚀 EJECUTANDO {nombre_conjunto.upper()}{Colores.FIN}")
    print(f"{Colores.AMARILLO}📊 Total técnicas: {len(tecnicas)}{Colores.FIN}")
    
    confirmacion = input(f"\n{Colores.AMARILLO}¿Ejecutar {nombre_conjunto}? (s/N): {Colores.FIN}").strip().lower()
    if confirmacion not in ['s', 'si', 'sí', 'y', 'yes']:
        print(f"{Colores.AMARILLO}❌ Ejecución cancelada{Colores.FIN}")
        return
    
    exitosas = 0
    tiempo_inicio = time.time()
    
    for i, tecnica in enumerate(tecnicas, 1):
        print(f"\n{Colores.CYAN}[{i}/{len(tecnicas)}] {tecnica.nombre}{Colores.FIN}")
        
        exito = ejecutar_tecnica(tecnica.id)
        if exito:
            exitosas += 1
        
        # Pausa entre técnicas (excepto la última)
        if i < len(tecnicas):
            time.sleep(1)
    
    duracion = (time.time() - tiempo_inicio) / 60
    
    print(f"\n{Colores.MORADO}{'='*60}")
    print(f"📊 RESUMEN {nombre_conjunto.upper()}:")
    print(f"✅ Exitosas: {exitosas}/{len(tecnicas)} ({exitosas/len(tecnicas)*100:.1f}%)")
    print(f"⏱️ Tiempo total: {duracion:.1f} minutos")
    print(f"{'='*60}{Colores.FIN}")

def limpiar_resultados():
    """Limpieza de resultados"""
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
    print(f"  🗑️ Archivos borrados: {archivos_borrados}{Colores.FIN}")

# ═══════════════════════════════════════════════════════════════════
# 🔧 FUNCIÓN PRINCIPAL
# ═══════════════════════════════════════════════════════════════════

def main():
    """Función principal del menú"""
    registro = RegistroTecnicas()
    
    while True:
        try:
            limpiar_pantalla()
            mostrar_banner()
            mostrar_menu_principal()
            
            opcion = input(f"{Colores.AMARILLO}👉 Selecciona una opción (0-18): {Colores.FIN}").strip()
            
            if opcion == '0':
                print(f"\n{Colores.VERDE}👋 ¡Gracias por usar el Sistema de Minería de Datos!{Colores.FIN}\n")
                break
            
            # Técnicas individuales (1-14)
            elif opcion in [str(i) for i in range(1, 15)]:
                tecnica_id = int(opcion)
                ejecutar_tecnica(tecnica_id)
            
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
            
            # Limpiar resultados
            elif opcion == '18':
                limpiar_resultados()
            
            else:
                print(f"{Colores.ROJO}❌ Opción inválida. Selecciona un número del 0 al 18.{Colores.FIN}")
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
# 🔧 PUNTO DE ENTRADA
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"{Colores.CYAN}🚀 Iniciando Sistema de Minería de Datos...{Colores.FIN}")
    time.sleep(1)
    main()