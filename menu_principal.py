#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MENÚ PRINCIPAL SIMPLIFICADO - SISTEMA DE MINERÍA DE DATOS
Sistema integral con todas las técnicas supervisadas y no supervisadas
Versión optimizada sin verificaciones complejas al inicio
"""

import os
import sys
import time
import importlib.util
from datetime import datetime
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════
# 🎨 COLORES Y CONFIGURACIÓN BÁSICA
# ═══════════════════════════════════════════════════════════════════

class Color:
    """Colores simplificados para consola"""
    AZUL = '\033[94m'
    VERDE = '\033[92m'
    AMARILLO = '\033[93m'
    ROJO = '\033[91m'
    MORADO = '\033[95m'
    CYAN = '\033[96m'
    NEGRITA = '\033[1m'
    FIN = '\033[0m'

def limpiar_pantalla():
    """Limpia la pantalla"""
    os.system('cls' if os.name == 'nt' else 'clear')

def pausar():
    """Pausa simple"""
    input(f"\n{Color.AMARILLO}📎 Presiona ENTER para continuar...{Color.FIN}")

# ═══════════════════════════════════════════════════════════════════
# 🗂️ REGISTRO SIMPLIFICADO DE TÉCNICAS
# ═══════════════════════════════════════════════════════════════════

class TecnicaSimple:
    """Información básica de cada técnica"""
    def __init__(self, id, nombre, descripcion, archivo, funcion, emoji="🔬"):
        self.id = id
        self.nombre = nombre
        self.descripcion = descripcion
        self.archivo = archivo
        self.funcion = funcion
        self.emoji = emoji

def obtener_tecnicas():
    """Lista completa de técnicas disponibles"""
    ruta_base = '/home/sedc/Proyectos/MineriaDeDatos'
    
    return {
        # TÉCNICAS SUPERVISADAS - PREDICCIÓN
        1: TecnicaSimple(1, "🔵 Regresión Lineal", "Predicción lineal de población", 
                        f"{ruta_base}/01_Supervisadas/01_Prediccion/01_Regresion/1_regresion_lineal.py", 
                        "ejecutar_regresion", "📈"),
        
        2: TecnicaSimple(2, "🌳 Árboles de Predicción", "Predicción con árboles de decisión", 
                        f"{ruta_base}/01_Supervisadas/01_Prediccion/02_Arboles_Prediccion/2_arboles_prediccion.py", 
                        "ejecutar_arboles", "🌲"),
        
        3: TecnicaSimple(3, "🔬 Estimadores de Núcleos", "SVR y K-NN para patrones complejos", 
                        f"{ruta_base}/01_Supervisadas/01_Prediccion/03_Estimador_Nucleos/3_estimador_nucleos.py", 
                        "ejecutar_nucleos", "⚛️"),
        
        # TÉCNICAS SUPERVISADAS - CLASIFICACIÓN
        4: TecnicaSimple(4, "🌳 Árboles de Decisión", "Clasificación con reglas jerárquicas", 
                        f"{ruta_base}/01_Supervisadas/02_Clasificacion/01_Arboles_Decision/1_arboles_decision.py", 
                        "ejecutar_arboles_decision", "🎯"),
        
        5: TecnicaSimple(5, "📏 Inducción de Reglas", "Reglas IF-THEN automáticas", 
                        f"{ruta_base}/01_Supervisadas/02_Clasificacion/02_Induccion_Reglas/2_induccion_reglas.py", 
                        "ejecutar_induccion_reglas", "📋"),
        
        6: TecnicaSimple(6, "🎲 Clasificación Bayesiana", "Clasificación probabilística", 
                        f"{ruta_base}/01_Supervisadas/02_Clasificacion/03_Bayesiana/3_clasificacion_bayesiana.py", 
                        "ejecutar_clasificacion_bayesiana", "🎯"),
        
        7: TecnicaSimple(7, "👥 Basado en Ejemplares (K-NN)", "Clasificación por vecinos cercanos", 
                        f"{ruta_base}/01_Supervisadas/02_Clasificacion/04_Basado_Ejemplares/4_basado_ejemplares.py", 
                        "ejecutar_clasificacion_ejemplares", "👥"),
        
        8: TecnicaSimple(8, "🧠 Redes de Neuronas", "Aprendizaje con redes neuronales", 
                        f"{ruta_base}/01_Supervisadas/02_Clasificacion/05_Redes_Neuronas/5_redes_neuronas.py", 
                        "ejecutar_redes_neuronas", "🧠"),
        
        9: TecnicaSimple(9, "🌫️ Lógica Borrosa", "Clasificación con conjuntos difusos", 
                        f"{ruta_base}/01_Supervisadas/02_Clasificacion/06_Logica_Borrosa/6_logica_borrosa.py", 
                        "ejecutar_logica_borrosa", "🌀"),
        
        10: TecnicaSimple(10, "🧬 Técnicas Genéticas", "Optimización evolutiva", 
                         f"{ruta_base}/01_Supervisadas/02_Clasificacion/07_Tecnicas_Geneticas/7_tecnicas_geneticas.py", 
                         "ejecutar_tecnicas_geneticas", "🧬"),
        
        # TÉCNICAS NO SUPERVISADAS
        11: TecnicaSimple(11, "📊 Clustering Numérico", "Agrupación K-Means", 
                         f"{ruta_base}/02_No_Supervisadas/01_Clustering/01_Numerico/1_clustering_numerico.py", 
                         "ejecutar_clustering_numerico", "📊"),
        
        12: TecnicaSimple(12, "🎯 Clustering Conceptual", "Agrupación por conceptos", 
                         f"{ruta_base}/02_No_Supervisadas/01_Clustering/02_Conceptual/2_clustering_conceptual.py", 
                         "ejecutar_clustering_conceptual", "🎯"),
        
        13: TecnicaSimple(13, "🎲 Clustering Probabilístico", "Agrupación con modelos probabilísticos", 
                         f"{ruta_base}/02_No_Supervisadas/01_Clustering/03_Probabilistico/3_clustering_probabilistico.py", 
                         "ejecutar_clustering_probabilistico", "🎲"),
        
        14: TecnicaSimple(14, "🔗 A Priori (Reglas de Asociación)", "Patrones 'si A entonces B'", 
                         f"{ruta_base}/02_No_Supervisadas/02_Asociacion/01_A_Priori/1_apriori_asociacion.py", 
                         "ejecutar_apriori", "🔗")
    }

# ═══════════════════════════════════════════════════════════════════
# 🎨 INTERFAZ SIMPLIFICADA
# ═══════════════════════════════════════════════════════════════════

def mostrar_banner():
    """Banner principal simplificado"""
    fecha_hora = datetime.now().strftime('%Y-%m-%d %H:%M')
    print(f"""
{Color.CYAN}╔════════════════════════════════════════════════════════════════════════════════╗
║                                                                                ║
║  {Color.NEGRITA}🧠 SISTEMA INTEGRAL DE MINERÍA DE DATOS - IA AVANZADA 🧠{Color.FIN}{Color.CYAN}               ║
║                                                                                ║
║  {Color.AMARILLO}📊 Universidad Tecnológica de Puebla (UTP){Color.CYAN}                                 ║
║  {Color.AMARILLO}🎯 Análisis Demográfico Integral Michoacán{Color.CYAN}                                 ║
║  {Color.AMARILLO}📈 14 Técnicas de IA y Machine Learning{Color.CYAN}                                    ║
║  {Color.AMARILLO}⏰ {fecha_hora}{Color.CYAN}                                                      ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝{Color.FIN}
""")

def mostrar_menu():
    """Menú principal simplificado"""
    tecnicas = obtener_tecnicas()
    
    print(f"""
{Color.NEGRITA}{Color.AZUL}═══════════════════════════════════════════════════════════════════════════════
                    🔬 TÉCNICAS SUPERVISADAS (PREDICCIÓN)
═══════════════════════════════════════════════════════════════════════════════{Color.FIN}""")
    
    # Técnicas de Predicción (1-3)
    for i in range(1, 4):
        t = tecnicas[i]
        print(f"{Color.VERDE}{i}.{Color.FIN} {Color.NEGRITA}{t.nombre}{Color.FIN}")
        print(f"   📝 {t.descripcion}")
    
    print(f"""
{Color.NEGRITA}{Color.AZUL}═══════════════════════════════════════════════════════════════════════════════
                    🎯 TÉCNICAS SUPERVISADAS (CLASIFICACIÓN)
═══════════════════════════════════════════════════════════════════════════════{Color.FIN}""")
    
    # Técnicas de Clasificación (4-10)
    for i in range(4, 11):
        t = tecnicas[i]
        print(f"{Color.VERDE}{i}.{Color.FIN} {Color.NEGRITA}{t.nombre}{Color.FIN}")
        print(f"   📝 {t.descripcion}")
    
    print(f"""
{Color.NEGRITA}{Color.MORADO}═══════════════════════════════════════════════════════════════════════════════
                    🔍 TÉCNICAS NO SUPERVISADAS (DESCUBRIMIENTO)
═══════════════════════════════════════════════════════════════════════════════{Color.FIN}""")
    
    # Técnicas No Supervisadas (11-14)
    for i in range(11, 15):
        t = tecnicas[i]
        print(f"{Color.CYAN}{i}.{Color.FIN} {Color.NEGRITA}{t.nombre}{Color.FIN}")
        print(f"   📝 {t.descripcion}")
    
    print(f"""
{Color.NEGRITA}{Color.AMARILLO}═══════════════════════════════════════════════════════════════════════════════
                              🚀 OPCIONES AUTOMÁTICAS
═══════════════════════════════════════════════════════════════════════════════{Color.FIN}

{Color.ROJO}15.{Color.FIN} {Color.NEGRITA}🔥 EJECUTAR TODAS LAS TÉCNICAS{Color.FIN} - Sistema completo (14 técnicas)
{Color.ROJO}16.{Color.FIN} {Color.NEGRITA}⚖️ EJECUTAR SOLO SUPERVISADAS{Color.FIN} - Las 10 técnicas supervisadas
{Color.ROJO}17.{Color.FIN} {Color.NEGRITA}🔍 EJECUTAR SOLO NO SUPERVISADAS{Color.FIN} - Las 4 técnicas no supervisadas

{Color.VERDE}18.{Color.FIN} {Color.NEGRITA}📊 Ver Resultados{Color.FIN} - Revisar archivos generados
{Color.VERDE}19.{Color.FIN} {Color.NEGRITA}🧹 Limpiar Resultados{Color.FIN} - Borrar archivos anteriores

{Color.ROJO}0.{Color.FIN} {Color.NEGRITA}❌ Salir{Color.FIN}

{Color.AZUL}═══════════════════════════════════════════════════════════════════════════════{Color.FIN}
""")

# ═══════════════════════════════════════════════════════════════════
# ⚙️ SISTEMA DE EJECUCIÓN SIMPLIFICADO
# ═══════════════════════════════════════════════════════════════════

def importar_y_ejecutar(tecnica):
    """Importa y ejecuta una técnica de manera simplificada"""
    try:
        # Verificar si existe el archivo
        if not Path(tecnica.archivo).exists():
            print(f"{Color.ROJO}❌ Archivo no encontrado: {Path(tecnica.archivo).name}{Color.FIN}")
            return False
        
        # Importar módulo dinámicamente
        spec = importlib.util.spec_from_file_location("modulo_temp", tecnica.archivo)
        modulo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(modulo)
        
        # Obtener función
        if hasattr(modulo, tecnica.funcion):
            funcion = getattr(modulo, tecnica.funcion)
            
            # Ejecutar técnica
            print(f"\n{Color.CYAN}{'='*60}")
            print(f"🚀 EJECUTANDO: {tecnica.nombre}")
            print(f"📝 {tecnica.descripcion}")
            print(f"{'='*60}{Color.FIN}\n")
            
            inicio = time.time()
            resultado = funcion()
            fin = time.time()
            
            print(f"\n{Color.VERDE}{'='*50}")
            print(f"✅ TÉCNICA COMPLETADA")
            print(f"⏱️ Tiempo: {fin-inicio:.1f}s")
            print(f"{'='*50}{Color.FIN}")
            
            return True
        else:
            print(f"{Color.ROJO}❌ Función '{tecnica.funcion}' no encontrada{Color.FIN}")
            return False
            
    except Exception as e:
        print(f"{Color.ROJO}❌ Error ejecutando {tecnica.nombre}: {e}{Color.FIN}")
        return False

def ejecutar_conjunto(ids_tecnicas, nombre_conjunto):
    """Ejecuta un conjunto de técnicas"""
    tecnicas = obtener_tecnicas()
    
    print(f"\n{Color.MORADO}{Color.NEGRITA}🚀 EJECUTANDO {nombre_conjunto.upper()}{Color.FIN}")
    print(f"{Color.AMARILLO}📊 Total técnicas: {len(ids_tecnicas)}{Color.FIN}")
    
    # Confirmar ejecución
    respuesta = input(f"\n{Color.AMARILLO}¿Continuar con la ejecución? (s/N): {Color.FIN}").strip().lower()
    if respuesta not in ['s', 'si', 'sí', 'y', 'yes']:
        print(f"{Color.AMARILLO}❌ Ejecución cancelada{Color.FIN}")
        return
    
    exitosas = 0
    inicio_total = time.time()
    
    for i, id_tecnica in enumerate(ids_tecnicas, 1):
        print(f"\n{Color.CYAN}[{i}/{len(ids_tecnicas)}] Procesando técnica {id_tecnica}...{Color.FIN}")
        
        if importar_y_ejecutar(tecnicas[id_tecnica]):
            exitosas += 1
        
        # Pausa entre técnicas
        if i < len(ids_tecnicas):
            time.sleep(1)
    
    duracion_total = (time.time() - inicio_total) / 60
    
    print(f"\n{Color.MORADO}{'='*50}")
    print(f"📊 RESUMEN {nombre_conjunto.upper()}:")
    print(f"✅ Exitosas: {exitosas}/{len(ids_tecnicas)} ({exitosas/len(ids_tecnicas)*100:.1f}%)")
    print(f"⏱️ Tiempo total: {duracion_total:.1f} minutos")
    print(f"{'='*50}{Color.FIN}")

def ver_resultados():
    """Ver resultados de manera simplificada"""
    print(f"\n{Color.CYAN}📊 VERIFICANDO RESULTADOS...{Color.FIN}\n")
    
    rutas = {
        'Gráficos': Path('/home/sedc/Proyectos/MineriaDeDatos/results/graficos'),
        'Modelos': Path('/home/sedc/Proyectos/MineriaDeDatos/results/modelos'),
        'Reportes': Path('/home/sedc/Proyectos/MineriaDeDatos/results/reportes')
    }
    
    total_archivos = 0
    
    for categoria, ruta in rutas.items():
        print(f"{Color.NEGRITA}{categoria}:{Color.FIN}")
        
        if ruta.exists():
            archivos = list(ruta.glob('*'))
            if archivos:
                for archivo in archivos[:5]:  # Mostrar solo los primeros 5
                    if archivo.is_file():
                        tamaño = archivo.stat().st_size / 1024  # KB
                        fecha = datetime.fromtimestamp(archivo.stat().st_mtime)
                        print(f"  📄 {archivo.name} ({tamaño:.1f} KB) - {fecha.strftime('%d/%m %H:%M')}")
                        total_archivos += 1
                
                if len(archivos) > 5:
                    print(f"  {Color.AMARILLO}... y {len(archivos) - 5} archivos más{Color.FIN}")
            else:
                print(f"  {Color.AMARILLO}📭 Sin archivos{Color.FIN}")
        else:
            print(f"  {Color.ROJO}❌ Carpeta no existe{Color.FIN}")
        print()
    
    print(f"{Color.CYAN}📊 Total archivos encontrados: {total_archivos}{Color.FIN}")

def limpiar_resultados():
    """Limpiar resultados de manera simplificada"""
    print(f"\n{Color.AMARILLO}🧹 LIMPIEZA DE RESULTADOS{Color.FIN}")
    
    respuesta = input(f"\n{Color.ROJO}⚠️ ¿Borrar TODOS los resultados? (s/N): {Color.FIN}").strip().lower()
    if respuesta not in ['s', 'si', 'sí', 'y', 'yes']:
        print(f"{Color.AMARILLO}❌ Limpieza cancelada{Color.FIN}")
        return
    
    rutas = [
        Path('/home/sedc/Proyectos/MineriaDeDatos/results/graficos'),
        Path('/home/sedc/Proyectos/MineriaDeDatos/results/modelos'),
        Path('/home/sedc/Proyectos/MineriaDeDatos/results/reportes')
    ]
    
    archivos_borrados = 0
    
    for ruta in rutas:
        if ruta.exists():
            for archivo in ruta.glob('*'):
                if archivo.is_file():
                    try:
                        archivo.unlink()
                        archivos_borrados += 1
                    except:
                        pass
    
    print(f"\n{Color.VERDE}✅ Limpieza completada")
    print(f"🗑️ Archivos borrados: {archivos_borrados}{Color.FIN}")

# ═══════════════════════════════════════════════════════════════════
# 🎯 FUNCIÓN PRINCIPAL SIMPLIFICADA
# ═══════════════════════════════════════════════════════════════════

def main():
    """Función principal simplificada"""
    tecnicas = obtener_tecnicas()
    
    while True:
        try:
            limpiar_pantalla()
            mostrar_banner()
            mostrar_menu()
            
            opcion = input(f"{Color.AMARILLO}👉 Selecciona una opción (0-19): {Color.FIN}").strip()
            
            if opcion == '0':
                print(f"\n{Color.VERDE}👋 ¡Gracias por usar el Sistema de Minería de Datos!")
                print(f"🎓 ¡Éxito en tu proyecto!{Color.FIN}\n")
                break
            
            # Técnicas individuales (1-14)
            elif opcion in [str(i) for i in range(1, 15)]:
                id_tecnica = int(opcion)
                if id_tecnica in tecnicas:
                    importar_y_ejecutar(tecnicas[id_tecnica])
                else:
                    print(f"{Color.ROJO}❌ Técnica no disponible{Color.FIN}")
            
            # Ejecutar todas las técnicas
            elif opcion == '15':
                ejecutar_conjunto(list(range(1, 15)), "SISTEMA COMPLETO")
            
            # Ejecutar solo supervisadas
            elif opcion == '16':
                ejecutar_conjunto(list(range(1, 11)), "TÉCNICAS SUPERVISADAS")
            
            # Ejecutar solo no supervisadas
            elif opcion == '17':
                ejecutar_conjunto(list(range(11, 15)), "TÉCNICAS NO SUPERVISADAS")
            
            # Ver resultados
            elif opcion == '18':
                ver_resultados()
            
            # Limpiar resultados
            elif opcion == '19':
                limpiar_resultados()
            
            else:
                print(f"{Color.ROJO}❌ Opción inválida. Usa números del 0 al 19.{Color.FIN}")
                time.sleep(2)
            
            if opcion != '0':
                pausar()
                
        except KeyboardInterrupt:
            print(f"\n\n{Color.AMARILLO}⚠️ Operación cancelada")
            respuesta = input(f"¿Salir del sistema? (s/N): {Color.FIN}").strip().lower()
            if respuesta in ['s', 'si', 'sí', 'y', 'yes']:
                print(f"{Color.VERDE}👋 ¡Hasta luego!{Color.FIN}\n")
                break
        except Exception as e:
            print(f"\n{Color.ROJO}❌ Error: {e}{Color.FIN}")
            pausar()

# ═══════════════════════════════════════════════════════════════════
# 🚀 PUNTO DE ENTRADA SIMPLIFICADO
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Crear carpetas básicas si no existen
    carpetas = [
        '/home/sedc/Proyectos/MineriaDeDatos/results/graficos',
        '/home/sedc/Proyectos/MineriaDeDatos/results/modelos', 
        '/home/sedc/Proyectos/MineriaDeDatos/results/reportes'
    ]
    
    for carpeta in carpetas:
        Path(carpeta).mkdir(parents=True, exist_ok=True)
    
    print(f"{Color.CYAN}🚀 Iniciando Sistema de Minería de Datos...{Color.FIN}")
    time.sleep(1)
    main()#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MENÚ PRINCIPAL SIMPLIFICADO - SISTEMA DE MINERÍA DE DATOS
Sistema integral con todas las técnicas supervisadas y no supervisadas
Versión optimizada sin verificaciones complejas al inicio
"""

import os
import sys
import time
import importlib.util
from datetime import datetime
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════
# 🎨 COLORES Y CONFIGURACIÓN BÁSICA
# ═══════════════════════════════════════════════════════════════════

class Color:
    """Colores simplificados para consola"""
    AZUL = '\033[94m'
    VERDE = '\033[92m'
    AMARILLO = '\033[93m'
    ROJO = '\033[91m'
    MORADO = '\033[95m'
    CYAN = '\033[96m'
    NEGRITA = '\033[1m'
    FIN = '\033[0m'

def limpiar_pantalla():
    """Limpia la pantalla"""
    os.system('cls' if os.name == 'nt' else 'clear')

def pausar():
    """Pausa simple"""
    input(f"\n{Color.AMARILLO}📎 Presiona ENTER para continuar...{Color.FIN}")

# ═══════════════════════════════════════════════════════════════════
# 🗂️ REGISTRO SIMPLIFICADO DE TÉCNICAS
# ═══════════════════════════════════════════════════════════════════

class TecnicaSimple:
    """Información básica de cada técnica"""
    def __init__(self, id, nombre, descripcion, archivo, funcion, emoji="🔬"):
        self.id = id
        self.nombre = nombre
        self.descripcion = descripcion
        self.archivo = archivo
        self.funcion = funcion
        self.emoji = emoji

def obtener_tecnicas():
    """Lista completa de técnicas disponibles"""
    ruta_base = '/home/sedc/Proyectos/MineriaDeDatos'
    
    return {
        # TÉCNICAS SUPERVISADAS - PREDICCIÓN
        1: TecnicaSimple(1, "🔵 Regresión Lineal", "Predicción lineal de población", 
                        f"{ruta_base}/01_Supervisadas/01_Prediccion/01_Regresion/1_regresion_lineal.py", 
                        "ejecutar_regresion", "📈"),
        
        2: TecnicaSimple(2, "🌳 Árboles de Predicción", "Predicción con árboles de decisión", 
                        f"{ruta_base}/01_Supervisadas/01_Prediccion/02_Arboles_Prediccion/2_arboles_prediccion.py", 
                        "ejecutar_arboles", "🌲"),
        
        3: TecnicaSimple(3, "🔬 Estimadores de Núcleos", "SVR y K-NN para patrones complejos", 
                        f"{ruta_base}/01_Supervisadas/01_Prediccion/03_Estimador_Nucleos/3_estimador_nucleos.py", 
                        "ejecutar_nucleos", "⚛️"),
        
        # TÉCNICAS SUPERVISADAS - CLASIFICACIÓN
        4: TecnicaSimple(4, "🌳 Árboles de Decisión", "Clasificación con reglas jerárquicas", 
                        f"{ruta_base}/01_Supervisadas/02_Clasificacion/01_Arboles_Decision/1_arboles_decision.py", 
                        "ejecutar_arboles_decision", "🎯"),
        
        5: TecnicaSimple(5, "📏 Inducción de Reglas", "Reglas IF-THEN automáticas", 
                        f"{ruta_base}/01_Supervisadas/02_Clasificacion/02_Induccion_Reglas/2_induccion_reglas.py", 
                        "ejecutar_induccion_reglas", "📋"),
        
        6: TecnicaSimple(6, "🎲 Clasificación Bayesiana", "Clasificación probabilística", 
                        f"{ruta_base}/01_Supervisadas/02_Clasificacion/03_Bayesiana/3_clasificacion_bayesiana.py", 
                        "ejecutar_clasificacion_bayesiana", "🎯"),
        
        7: TecnicaSimple(7, "👥 Basado en Ejemplares (K-NN)", "Clasificación por vecinos cercanos", 
                        f"{ruta_base}/01_Supervisadas/02_Clasificacion/04_Basado_Ejemplares/4_basado_ejemplares.py", 
                        "ejecutar_clasificacion_ejemplares", "👥"),
        
        8: TecnicaSimple(8, "🧠 Redes de Neuronas", "Aprendizaje con redes neuronales", 
                        f"{ruta_base}/01_Supervisadas/02_Clasificacion/05_Redes_Neuronas/5_redes_neuronas.py", 
                        "ejecutar_redes_neuronas", "🧠"),
        
        9: TecnicaSimple(9, "🌫️ Lógica Borrosa", "Clasificación con conjuntos difusos", 
                        f"{ruta_base}/01_Supervisadas/02_Clasificacion/06_Logica_Borrosa/6_logica_borrosa.py", 
                        "ejecutar_logica_borrosa", "🌀"),
        
        10: TecnicaSimple(10, "🧬 Técnicas Genéticas", "Optimización evolutiva", 
                         f"{ruta_base}/01_Supervisadas/02_Clasificacion/07_Tecnicas_Geneticas/7_tecnicas_geneticas.py", 
                         "ejecutar_tecnicas_geneticas", "🧬"),
        
        # TÉCNICAS NO SUPERVISADAS
        11: TecnicaSimple(11, "📊 Clustering Numérico", "Agrupación K-Means", 
                         f"{ruta_base}/02_No_Supervisadas/01_Clustering/01_Numerico/1_clustering_numerico.py", 
                         "ejecutar_clustering_numerico", "📊"),
        
        12: TecnicaSimple(12, "🎯 Clustering Conceptual", "Agrupación por conceptos", 
                         f"{ruta_base}/02_No_Supervisadas/01_Clustering/02_Conceptual/2_clustering_conceptual.py", 
                         "ejecutar_clustering_conceptual", "🎯"),
        
        13: TecnicaSimple(13, "🎲 Clustering Probabilístico", "Agrupación con modelos probabilísticos", 
                         f"{ruta_base}/02_No_Supervisadas/01_Clustering/03_Probabilistico/3_clustering_probabilistico.py", 
                         "ejecutar_clustering_probabilistico", "🎲"),
        
        14: TecnicaSimple(14, "🔗 A Priori (Reglas de Asociación)", "Patrones 'si A entonces B'", 
                         f"{ruta_base}/02_No_Supervisadas/02_Asociacion/01_A_Priori/1_apriori_asociacion.py", 
                         "ejecutar_apriori", "🔗")
    }

# ═══════════════════════════════════════════════════════════════════
# 🎨 INTERFAZ SIMPLIFICADA
# ═══════════════════════════════════════════════════════════════════

def mostrar_banner():
    """Banner principal simplificado"""
    fecha_hora = datetime.now().strftime('%Y-%m-%d %H:%M')
    print(f"""
{Color.CYAN}╔════════════════════════════════════════════════════════════════════════════════╗
║                                                                                ║
║  {Color.NEGRITA}🧠 SISTEMA INTEGRAL DE MINERÍA DE DATOS - IA AVANZADA 🧠{Color.FIN}{Color.CYAN}               ║
║                                                                                ║
║  {Color.AMARILLO}📊 Universidad Tecnológica de Puebla (UTP){Color.CYAN}                                 ║
║  {Color.AMARILLO}🎯 Análisis Demográfico Integral Michoacán{Color.CYAN}                                 ║
║  {Color.AMARILLO}📈 14 Técnicas de IA y Machine Learning{Color.CYAN}                                    ║
║  {Color.AMARILLO}⏰ {fecha_hora}{Color.CYAN}                                                      ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝{Color.FIN}
""")

def mostrar_menu():
    """Menú principal simplificado"""
    tecnicas = obtener_tecnicas()
    
    print(f"""
{Color.NEGRITA}{Color.AZUL}═══════════════════════════════════════════════════════════════════════════════
                    🔬 TÉCNICAS SUPERVISADAS (PREDICCIÓN)
═══════════════════════════════════════════════════════════════════════════════{Color.FIN}""")
    
    # Técnicas de Predicción (1-3)
    for i in range(1, 4):
        t = tecnicas[i]
        print(f"{Color.VERDE}{i}.{Color.FIN} {Color.NEGRITA}{t.nombre}{Color.FIN}")
        print(f"   📝 {t.descripcion}")
    
    print(f"""
{Color.NEGRITA}{Color.AZUL}═══════════════════════════════════════════════════════════════════════════════
                    🎯 TÉCNICAS SUPERVISADAS (CLASIFICACIÓN)
═══════════════════════════════════════════════════════════════════════════════{Color.FIN}""")
    
    # Técnicas de Clasificación (4-10)
    for i in range(4, 11):
        t = tecnicas[i]
        print(f"{Color.VERDE}{i}.{Color.FIN} {Color.NEGRITA}{t.nombre}{Color.FIN}")
        print(f"   📝 {t.descripcion}")
    
    print(f"""
{Color.NEGRITA}{Color.MORADO}═══════════════════════════════════════════════════════════════════════════════
                    🔍 TÉCNICAS NO SUPERVISADAS (DESCUBRIMIENTO)
═══════════════════════════════════════════════════════════════════════════════{Color.FIN}""")
    
    # Técnicas No Supervisadas (11-14)
    for i in range(11, 15):
        t = tecnicas[i]
        print(f"{Color.CYAN}{i}.{Color.FIN} {Color.NEGRITA}{t.nombre}{Color.FIN}")
        print(f"   📝 {t.descripcion}")
    
    print(f"""
{Color.NEGRITA}{Color.AMARILLO}═══════════════════════════════════════════════════════════════════════════════
                              🚀 OPCIONES AUTOMÁTICAS
═══════════════════════════════════════════════════════════════════════════════{Color.FIN}

{Color.ROJO}15.{Color.FIN} {Color.NEGRITA}🔥 EJECUTAR TODAS LAS TÉCNICAS{Color.FIN} - Sistema completo (14 técnicas)
{Color.ROJO}16.{Color.FIN} {Color.NEGRITA}⚖️ EJECUTAR SOLO SUPERVISADAS{Color.FIN} - Las 10 técnicas supervisadas
{Color.ROJO}17.{Color.FIN} {Color.NEGRITA}🔍 EJECUTAR SOLO NO SUPERVISADAS{Color.FIN} - Las 4 técnicas no supervisadas

{Color.VERDE}18.{Color.FIN} {Color.NEGRITA}📊 Ver Resultados{Color.FIN} - Revisar archivos generados
{Color.VERDE}19.{Color.FIN} {Color.NEGRITA}🧹 Limpiar Resultados{Color.FIN} - Borrar archivos anteriores

{Color.ROJO}0.{Color.FIN} {Color.NEGRITA}❌ Salir{Color.FIN}

{Color.AZUL}═══════════════════════════════════════════════════════════════════════════════{Color.FIN}
""")

# ═══════════════════════════════════════════════════════════════════
# ⚙️ SISTEMA DE EJECUCIÓN SIMPLIFICADO
# ═══════════════════════════════════════════════════════════════════

def importar_y_ejecutar(tecnica):
    """Importa y ejecuta una técnica de manera simplificada"""
    try:
        # Verificar si existe el archivo
        if not Path(tecnica.archivo).exists():
            print(f"{Color.ROJO}❌ Archivo no encontrado: {Path(tecnica.archivo).name}{Color.FIN}")
            return False
        
        # Importar módulo dinámicamente
        spec = importlib.util.spec_from_file_location("modulo_temp", tecnica.archivo)
        modulo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(modulo)
        
        # Obtener función
        if hasattr(modulo, tecnica.funcion):
            funcion = getattr(modulo, tecnica.funcion)
            
            # Ejecutar técnica
            print(f"\n{Color.CYAN}{'='*60}")
            print(f"🚀 EJECUTANDO: {tecnica.nombre}")
            print(f"📝 {tecnica.descripcion}")
            print(f"{'='*60}{Color.FIN}\n")
            
            inicio = time.time()
            resultado = funcion()
            fin = time.time()
            
            print(f"\n{Color.VERDE}{'='*50}")
            print(f"✅ TÉCNICA COMPLETADA")
            print(f"⏱️ Tiempo: {fin-inicio:.1f}s")
            print(f"{'='*50}{Color.FIN}")
            
            return True
        else:
            print(f"{Color.ROJO}❌ Función '{tecnica.funcion}' no encontrada{Color.FIN}")
            return False
            
    except Exception as e:
        print(f"{Color.ROJO}❌ Error ejecutando {tecnica.nombre}: {e}{Color.FIN}")
        return False

def ejecutar_conjunto(ids_tecnicas, nombre_conjunto):
    """Ejecuta un conjunto de técnicas"""
    tecnicas = obtener_tecnicas()
    
    print(f"\n{Color.MORADO}{Color.NEGRITA}🚀 EJECUTANDO {nombre_conjunto.upper()}{Color.FIN}")
    print(f"{Color.AMARILLO}📊 Total técnicas: {len(ids_tecnicas)}{Color.FIN}")
    
    # Confirmar ejecución
    respuesta = input(f"\n{Color.AMARILLO}¿Continuar con la ejecución? (s/N): {Color.FIN}").strip().lower()
    if respuesta not in ['s', 'si', 'sí', 'y', 'yes']:
        print(f"{Color.AMARILLO}❌ Ejecución cancelada{Color.FIN}")
        return
    
    exitosas = 0
    inicio_total = time.time()
    
    for i, id_tecnica in enumerate(ids_tecnicas, 1):
        print(f"\n{Color.CYAN}[{i}/{len(ids_tecnicas)}] Procesando técnica {id_tecnica}...{Color.FIN}")
        
        if importar_y_ejecutar(tecnicas[id_tecnica]):
            exitosas += 1
        
        # Pausa entre técnicas
        if i < len(ids_tecnicas):
            time.sleep(1)
    
    duracion_total = (time.time() - inicio_total) / 60
    
    print(f"\n{Color.MORADO}{'='*50}")
    print(f"📊 RESUMEN {nombre_conjunto.upper()}:")
    print(f"✅ Exitosas: {exitosas}/{len(ids_tecnicas)} ({exitosas/len(ids_tecnicas)*100:.1f}%)")
    print(f"⏱️ Tiempo total: {duracion_total:.1f} minutos")
    print(f"{'='*50}{Color.FIN}")

def ver_resultados():
    """Ver resultados de manera simplificada"""
    print(f"\n{Color.CYAN}📊 VERIFICANDO RESULTADOS...{Color.FIN}\n")
    
    rutas = {
        'Gráficos': Path('/home/sedc/Proyectos/MineriaDeDatos/results/graficos'),
        'Modelos': Path('/home/sedc/Proyectos/MineriaDeDatos/results/modelos'),
        'Reportes': Path('/home/sedc/Proyectos/MineriaDeDatos/results/reportes')
    }
    
    total_archivos = 0
    
    for categoria, ruta in rutas.items():
        print(f"{Color.NEGRITA}{categoria}:{Color.FIN}")
        
        if ruta.exists():
            archivos = list(ruta.glob('*'))
            if archivos:
                for archivo in archivos[:5]:  # Mostrar solo los primeros 5
                    if archivo.is_file():
                        tamaño = archivo.stat().st_size / 1024  # KB
                        fecha = datetime.fromtimestamp(archivo.stat().st_mtime)
                        print(f"  📄 {archivo.name} ({tamaño:.1f} KB) - {fecha.strftime('%d/%m %H:%M')}")
                        total_archivos += 1
                
                if len(archivos) > 5:
                    print(f"  {Color.AMARILLO}... y {len(archivos) - 5} archivos más{Color.FIN}")
            else:
                print(f"  {Color.AMARILLO}📭 Sin archivos{Color.FIN}")
        else:
            print(f"  {Color.ROJO}❌ Carpeta no existe{Color.FIN}")
        print()
    
    print(f"{Color.CYAN}📊 Total archivos encontrados: {total_archivos}{Color.FIN}")

def limpiar_resultados():
    """Limpiar resultados de manera simplificada"""
    print(f"\n{Color.AMARILLO}🧹 LIMPIEZA DE RESULTADOS{Color.FIN}")
    
    respuesta = input(f"\n{Color.ROJO}⚠️ ¿Borrar TODOS los resultados? (s/N): {Color.FIN}").strip().lower()
    if respuesta not in ['s', 'si', 'sí', 'y', 'yes']:
        print(f"{Color.AMARILLO}❌ Limpieza cancelada{Color.FIN}")
        return
    
    rutas = [
        Path('/home/sedc/Proyectos/MineriaDeDatos/results/graficos'),
        Path('/home/sedc/Proyectos/MineriaDeDatos/results/modelos'),
        Path('/home/sedc/Proyectos/MineriaDeDatos/results/reportes')
    ]
    
    archivos_borrados = 0
    
    for ruta in rutas:
        if ruta.exists():
            for archivo in ruta.glob('*'):
                if archivo.is_file():
                    try:
                        archivo.unlink()
                        archivos_borrados += 1
                    except:
                        pass
    
    print(f"\n{Color.VERDE}✅ Limpieza completada")
    print(f"🗑️ Archivos borrados: {archivos_borrados}{Color.FIN}")

# ═══════════════════════════════════════════════════════════════════
# 🎯 FUNCIÓN PRINCIPAL SIMPLIFICADA
# ═══════════════════════════════════════════════════════════════════

def main():
    """Función principal simplificada"""
    tecnicas = obtener_tecnicas()
    
    while True:
        try:
            limpiar_pantalla()
            mostrar_banner()
            mostrar_menu()
            
            opcion = input(f"{Color.AMARILLO}👉 Selecciona una opción (0-19): {Color.FIN}").strip()
            
            if opcion == '0':
                print(f"\n{Color.VERDE}👋 ¡Gracias por usar el Sistema de Minería de Datos!")
                print(f"🎓 ¡Éxito en tu proyecto!{Color.FIN}\n")
                break
            
            # Técnicas individuales (1-14)
            elif opcion in [str(i) for i in range(1, 15)]:
                id_tecnica = int(opcion)
                if id_tecnica in tecnicas:
                    importar_y_ejecutar(tecnicas[id_tecnica])
                else:
                    print(f"{Color.ROJO}❌ Técnica no disponible{Color.FIN}")
            
            # Ejecutar todas las técnicas
            elif opcion == '15':
                ejecutar_conjunto(list(range(1, 15)), "SISTEMA COMPLETO")
            
            # Ejecutar solo supervisadas
            elif opcion == '16':
                ejecutar_conjunto(list(range(1, 11)), "TÉCNICAS SUPERVISADAS")
            
            # Ejecutar solo no supervisadas
            elif opcion == '17':
                ejecutar_conjunto(list(range(11, 15)), "TÉCNICAS NO SUPERVISADAS")
            
            # Ver resultados
            elif opcion == '18':
                ver_resultados()
            
            # Limpiar resultados
            elif opcion == '19':
                limpiar_resultados()
            
            else:
                print(f"{Color.ROJO}❌ Opción inválida. Usa números del 0 al 19.{Color.FIN}")
                time.sleep(2)
            
            if opcion != '0':
                pausar()
                
        except KeyboardInterrupt:
            print(f"\n\n{Color.AMARILLO}⚠️ Operación cancelada")
            respuesta = input(f"¿Salir del sistema? (s/N): {Color.FIN}").strip().lower()
            if respuesta in ['s', 'si', 'sí', 'y', 'yes']:
                print(f"{Color.VERDE}👋 ¡Hasta luego!{Color.FIN}\n")
                break
        except Exception as e:
            print(f"\n{Color.ROJO}❌ Error: {e}{Color.FIN}")
            pausar()

# ═══════════════════════════════════════════════════════════════════
# 🚀 PUNTO DE ENTRADA SIMPLIFICADO
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Crear carpetas básicas si no existen
    carpetas = [
        '/home/sedc/Proyectos/MineriaDeDatos/results/graficos',
        '/home/sedc/Proyectos/MineriaDeDatos/results/modelos', 
        '/home/sedc/Proyectos/MineriaDeDatos/results/reportes'
    ]
    
    for carpeta in carpetas:
        Path(carpeta).mkdir(parents=True, exist_ok=True)
    
    print(f"{Color.CYAN}🚀 Iniciando Sistema de Minería de Datos...{Color.FIN}")
    time.sleep(1)
    main()