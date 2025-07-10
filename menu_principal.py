#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MENÚ PRINCIPAL ACTUALIZADO - SISTEMA COMPLETO DE MINERÍA DE DATOS
Incluye técnicas supervisadas y no supervisadas
"""

import os
import sys
import time
import importlib.util
from datetime import datetime

sys.path.append('/home/sedc/Proyectos/MineriaDeDatos')

# Colores para la consola
class Colores:
    AZUL = '\033[94m'
    VERDE = '\033[92m'
    AMARILLO = '\033[93m'
    ROJO = '\033[91m'
    MORADO = '\033[95m'
    CYAN = '\033[96m'
    BLANCO = '\033[97m'
    NEGRITA = '\033[1m'
    FIN = '\033[0m'

def limpiar_pantalla():
    os.system('cls' if os.name == 'nt' else 'clear')

def mostrar_banner():
    """Mostrar banner principal del sistema"""
    banner = f"""
{Colores.CYAN}╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  {Colores.NEGRITA}🧠 SISTEMA COMPLETO DE MINERÍA DE DATOS - IA AVANZADA 🧠{Colores.FIN}{Colores.CYAN}            ║
║                                                                              ║
║  {Colores.BLANCO}Universidad: [UTP]{Colores.CYAN}                                            ║
║  {Colores.BLANCO}Proyecto: Análisis Demográfico Integral Michoacan{Colores.CYAN}                          ║
║  {Colores.BLANCO}Dataset: Censo Poblacional INEGI - 69K+ registros{Colores.CYAN}                      ║
║  {Colores.BLANCO}Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}{Colores.CYAN}                                             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝{Colores.FIN}

{Colores.AMARILLO}📊 TÉCNICAS IMPLEMENTADAS: Supervisadas (7) + No Supervisadas (4) = 11 TOTAL{Colores.FIN}
{Colores.VERDE}🎯 OBJETIVO: Sistema integral de minería de datos con todas las técnicas principales{Colores.FIN}
"""
    print(banner)

def mostrar_menu_principal():
    """Mostrar el menú principal con todas las opciones"""
    menu = f"""
{Colores.NEGRITA}{Colores.AZUL}═══════════════════════════════════════════════════════════════════════════════
                    🔬 TÉCNICAS SUPERVISADAS (CLASIFICACIÓN)
═══════════════════════════════════════════════════════════════════════════════{Colores.FIN}

{Colores.VERDE}1.{Colores.FIN} {Colores.NEGRITA}🌳 Árboles de Decisión{Colores.FIN} - Clasificación interpretable con reglas jerárquicas
{Colores.VERDE}2.{Colores.FIN} {Colores.NEGRITA}📏 Inducción de Reglas{Colores.FIN} - Generación automática de reglas IF-THEN explicativas  
{Colores.VERDE}3.{Colores.FIN} {Colores.NEGRITA}🎲 Clasificación Bayesiana{Colores.FIN} - Clasificación probabilística (Naive Bayes)
{Colores.VERDE}4.{Colores.FIN} {Colores.NEGRITA}👥 Basado en Ejemplares (K-NN){Colores.FIN} - Clasificación por similitud con vecinos
{Colores.VERDE}5.{Colores.FIN} {Colores.NEGRITA}🧠 Redes de Neuronas{Colores.FIN} - Aprendizaje profundo con múltiples arquitecturas
{Colores.VERDE}6.{Colores.FIN} {Colores.NEGRITA}🌫️  Lógica Borrosa (Fuzzy Logic){Colores.FIN} - Clasificación con conjuntos difusos
{Colores.VERDE}7.{Colores.FIN} {Colores.NEGRITA}🧬 Técnicas Genéticas{Colores.FIN} - Optimización evolutiva de características

{Colores.NEGRITA}{Colores.MORADO}═══════════════════════════════════════════════════════════════════════════════
                    🔍 TÉCNICAS NO SUPERVISADAS (DESCUBRIMIENTO)
═══════════════════════════════════════════════════════════════════════════════{Colores.FIN}

{Colores.CYAN}8.{Colores.FIN} {Colores.NEGRITA}🔗 Reglas de Asociación (A Priori){Colores.FIN} - Patrones "si A entonces B"
{Colores.CYAN}9.{Colores.FIN} {Colores.NEGRITA}📊 Clustering Numérico (K-Means){Colores.FIN} - Agrupación por similitud numérica
{Colores.CYAN}10.{Colores.FIN} {Colores.NEGRITA}🎯 Clustering Conceptual{Colores.FIN} - Agrupación basada en conceptos
{Colores.CYAN}11.{Colores.FIN} {Colores.NEGRITA}🎲 Clustering Probabilístico (EM){Colores.FIN} - Agrupación con modelos probabilísticos

{Colores.NEGRITA}{Colores.AMARILLO}═══════════════════════════════════════════════════════════════════════════════
                              🚀 OPCIONES DEL SISTEMA
═══════════════════════════════════════════════════════════════════════════════{Colores.FIN}

{Colores.ROJO}12.{Colores.FIN} {Colores.NEGRITA}🔥 EJECUTAR TODAS LAS TÉCNICAS{Colores.FIN} - Sistema completo (11 técnicas)
{Colores.ROJO}13.{Colores.FIN} {Colores.NEGRITA}⚖️  EJECUTAR SOLO SUPERVISADAS{Colores.FIN} - Las 7 técnicas de clasificación
{Colores.ROJO}14.{Colores.FIN} {Colores.NEGRITA}🔍 EJECUTAR SOLO NO SUPERVISADAS{Colores.FIN} - Las 4 técnicas de descubrimiento

{Colores.VERDE}15.{Colores.FIN} {Colores.NEGRITA}📊 Ver Resultados y Comparar{Colores.FIN} - Revisar reportes, gráficos y comparaciones
{Colores.VERDE}16.{Colores.FIN} {Colores.NEGRITA}🏆 Ranking de Técnicas{Colores.FIN} - Comparación de rendimiento por precisión
{Colores.VERDE}17.{Colores.FIN} {Colores.NEGRITA}📈 Dashboard Ejecutivo{Colores.FIN} - Resumen visual de todos los resultados

{Colores.AMARILLO}18.{Colores.FIN} {Colores.NEGRITA}🔧 Configuración y Diagnóstico{Colores.FIN} - Verificar sistema, rutas y dependencias
{Colores.AMARILLO}19.{Colores.FIN} {Colores.NEGRITA}🧹 Limpiar Resultados{Colores.FIN} - Borrar modelos y reportes anteriores
{Colores.AMARILLO}20.{Colores.FIN} {Colores.NEGRITA}💾 Exportar Proyecto{Colores.FIN} - Crear backup completo del proyecto

{Colores.ROJO}0.{Colores.FIN} {Colores.NEGRITA}❌ Salir del Sistema{Colores.FIN}

{Colores.AZUL}═══════════════════════════════════════════════════════════════════════════════{Colores.FIN}
"""
    print(menu)

def importar_modulo_dinamico(ruta_archivo, nombre_funcion):
    """Importar dinámicamente un módulo y obtener una función específica"""
    try:
        spec = importlib.util.spec_from_file_location("modulo_temporal", ruta_archivo)
        modulo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(modulo)
        
        if hasattr(modulo, nombre_funcion):
            return getattr(modulo, nombre_funcion)
        else:
            print(f"  ⚠️ Función '{nombre_funcion}' no encontrada en {ruta_archivo}")
            return None
    except Exception as e:
        print(f"  ❌ Error importando {ruta_archivo}: {e}")
        return None

def verificar_archivos():
    """Verificar que existan los archivos necesarios"""
    rutas_requeridas = {
        'datos': '/home/sedc/Proyectos/MineriaDeDatos/data/ceros_sin_columnasAB_limpio_weka.csv',
        'resultados': '/home/sedc/Proyectos/MineriaDeDatos/results/',
        'graficos': '/home/sedc/Proyectos/MineriaDeDatos/results/graficos/',
        'modelos': '/home/sedc/Proyectos/MineriaDeDatos/results/modelos/',
        'reportes': '/home/sedc/Proyectos/MineriaDeDatos/results/reportes/'
    }
    
    # Técnicas supervisadas
    archivos_supervisadas = {
        'Árboles de Decisión': '/home/sedc/Proyectos/MineriaDeDatos/01_Supervisadas/02_Clasificacion/01_Arboles_Decision/1_arboles_decision.py',
        'Inducción de Reglas': '/home/sedc/Proyectos/MineriaDeDatos/01_Supervisadas/02_Clasificacion/02_Induccion_Reglas/2_induccion_reglas.py',
        'Clasificación Bayesiana': '/home/sedc/Proyectos/MineriaDeDatos/01_Supervisadas/02_Clasificacion/03_Bayesiana/3_clasificacion_bayesiana.py',
        'Basado en Ejemplares': '/home/sedc/Proyectos/MineriaDeDatos/01_Supervisadas/02_Clasificacion/04_Basado_Ejemplares/4_basado_ejemplares.py',
        'Redes de Neuronas': '/home/sedc/Proyectos/MineriaDeDatos/01_Supervisadas/02_Clasificacion/05_Redes_Neuronas/5_redes_neuronas.py',
        'Lógica Borrosa': '/home/sedc/Proyectos/MineriaDeDatos/01_Supervisadas/02_Clasificacion/06_Logica_Borrosa/6_logica_borrosa.py',
        'Técnicas Genéticas': '/home/sedc/Proyectos/MineriaDeDatos/01_Supervisadas/02_Clasificacion/07_Tecnicas_Geneticas/7_tecnicas_geneticas.py'
    }
    
    # Técnicas no supervisadas
    archivos_no_supervisadas = {
        'A Priori': '/home/sedc/Proyectos/MineriaDeDatos/02_No_Supervisadas/02_Asociacion/01_A_Priori/1_apriori_asociacion.py',
        'Clustering Numérico': '/home/sedc/Proyectos/MineriaDeDatos/02_No_Supervisadas/01_Clustering/01_Numerico/1_clustering_numerico.py',
        'Clustering Conceptual': '/home/sedc/Proyectos/MineriaDeDatos/02_No_Supervisadas/01_Clustering/02_Conceptual/2_clustering_conceptual.py',
        'Clustering Probabilístico': '/home/sedc/Proyectos/MineriaDeDatos/02_No_Supervisadas/01_Clustering/03_Probabilistico/3_clustering_probabilistico.py'
    }
    
    print(f"{Colores.AMARILLO}🔍 Verificando sistema completo...{Colores.FIN}")
    
    # Verificar directorios
    for nombre, ruta in rutas_requeridas.items():
        if os.path.exists(ruta):
            print(f"  ✅ {nombre.capitalize()}: {Colores.VERDE}OK{Colores.FIN}")
        else:
            print(f"  ❌ {nombre.capitalize()}: {Colores.ROJO}NO ENCONTRADO{Colores.FIN}")
            if nombre != 'datos':
                try:
                    os.makedirs(ruta, exist_ok=True)
                    print(f"     {Colores.AMARILLO}📁 Carpeta creada: {ruta}{Colores.FIN}")
                except Exception as e:
                    print(f"     {Colores.ROJO}❌ Error creando carpeta: {e}{Colores.FIN}")
    
    # Verificar técnicas supervisadas
    print(f"\n{Colores.AMARILLO}🔍 Verificando técnicas supervisadas...{Colores.FIN}")
    supervisadas_disponibles = 0
    for nombre, ruta in archivos_supervisadas.items():
        if os.path.exists(ruta):
            print(f"  ✅ {nombre}: {Colores.VERDE}OK{Colores.FIN}")
            supervisadas_disponibles += 1
        else:
            print(f"  ❌ {nombre}: {Colores.ROJO}NO ENCONTRADO{Colores.FIN}")
    
    # Verificar técnicas no supervisadas
    print(f"\n{Colores.AMARILLO}🔍 Verificando técnicas no supervisadas...{Colores.FIN}")
    no_supervisadas_disponibles = 0
    for nombre, ruta in archivos_no_supervisadas.items():
        if os.path.exists(ruta):
            print(f"  ✅ {nombre}: {Colores.VERDE}OK{Colores.FIN}")
            no_supervisadas_disponibles += 1
        else:
            print(f"  ❌ {nombre}: {Colores.ROJO}NO ENCONTRADO{Colores.FIN}")
            print(f"     {Colores.AMARILLO}Esperado en: {ruta}{Colores.FIN}")
    
    datos_ok = os.path.exists(rutas_requeridas['datos'])
    total_tecnicas = supervisadas_disponibles + no_supervisadas_disponibles
    
    print(f"\n{Colores.CYAN}📊 Resumen del sistema:")
    print(f"  Datos principales: {'✅' if datos_ok else '❌'}")
    print(f"  Técnicas supervisadas: {supervisadas_disponibles}/7")
    print(f"  Técnicas no supervisadas: {no_supervisadas_disponibles}/4")
    print(f"  TOTAL técnicas disponibles: {total_tecnicas}/11{Colores.FIN}")
    
    return datos_ok, supervisadas_disponibles, no_supervisadas_disponibles

def ejecutar_tecnica(numero, nombre_tecnica, descripcion):
    """Ejecutar una técnica específica"""
    print(f"\n{Colores.CYAN}{'='*80}")
    print(f"🚀 EJECUTANDO: {nombre_tecnica}")
    print(f"📝 {descripcion}")
    print(f"{'='*80}{Colores.FIN}\n")
    
    # Mapeo completo de técnicas del powerpoint
    archivos_tecnicas = {
        # Supervisadas
        1: ('/home/sedc/Proyectos/MineriaDeDatos/01_Supervisadas/02_Clasificacion/01_Arboles_Decision/1_arboles_decision.py', 'ejecutar_arboles_decision'),
        2: ('/home/sedc/Proyectos/MineriaDeDatos/01_Supervisadas/02_Clasificacion/02_Induccion_Reglas/2_induccion_reglas.py', 'ejecutar_induccion_reglas'),
        3: ('/home/sedc/Proyectos/MineriaDeDatos/01_Supervisadas/02_Clasificacion/03_Bayesiana/3_clasificacion_bayesiana.py', 'ejecutar_clasificacion_bayesiana'),
        4: ('/home/sedc/Proyectos/MineriaDeDatos/01_Supervisadas/02_Clasificacion/04_Basado_Ejemplares/4_basado_ejemplares.py', 'ejecutar_clasificacion_ejemplares'),
        5: ('/home/sedc/Proyectos/MineriaDeDatos/01_Supervisadas/02_Clasificacion/05_Redes_Neuronas/5_redes_neuronas.py', 'ejecutar_redes_neuronas'),
        6: ('/home/sedc/Proyectos/MineriaDeDatos/01_Supervisadas/02_Clasificacion/06_Logica_Borrosa/6_logica_borrosa.py', 'ejecutar_logica_borrosa'),
        7: ('/home/sedc/Proyectos/MineriaDeDatos/01_Supervisadas/02_Clasificacion/07_Tecnicas_Geneticas/7_tecnicas_geneticas.py', 'ejecutar_tecnicas_geneticas'),
        # No supervisadas
        8: ('/home/sedc/Proyectos/MineriaDeDatos/02_No_Supervisadas/02_Asociacion/01_A_Priori/1_apriori_asociacion.py', 'ejecutar_apriori'),
        9: ('/home/sedc/Proyectos/MineriaDeDatos/02_No_Supervisadas/01_Clustering/01_Numerico/1_clustering_numerico.py', 'ejecutar_clustering_numerico'),
        10: ('/home/sedc/Proyectos/MineriaDeDatos/02_No_Supervisadas/01_Clustering/02_Conceptual/2_clustering_conceptual.py', 'ejecutar_clustering_conceptual'),
        11: ('/home/sedc/Proyectos/MineriaDeDatos/02_No_Supervisadas/01_Clustering/03_Probabilistico/3_clustering_probabilistico.py', 'ejecutar_clustering_probabilistico')
    }
    
    inicio = time.time()
    
    try:
        if numero in archivos_tecnicas:
            archivo_path, nombre_funcion = archivos_tecnicas[numero]
            
            # Verificar que el archivo existe
            if not os.path.exists(archivo_path):
                print(f"{Colores.ROJO}❌ Archivo no encontrado: {archivo_path}{Colores.FIN}")
                return False
            
            # Importar dinámicamente la función
            funcion = importar_modulo_dinamico(archivo_path, nombre_funcion)
            
            if funcion is None:
                print(f"{Colores.ROJO}❌ No se pudo importar la función {nombre_funcion}{Colores.FIN}")
                return False
            
            # Ejecutar la técnica
            print(f"{Colores.VERDE}✅ Función importada correctamente. Iniciando ejecución...{Colores.FIN}\n")
            resultado = funcion()
            
            fin = time.time()
            duracion = fin - inicio
            
            print(f"\n{Colores.VERDE}{'='*60}")
            print(f"✅ TÉCNICA COMPLETADA EXITOSAMENTE")
            print(f"⏱️  Tiempo de ejecución: {duracion:.1f} segundos ({duracion/60:.1f} minutos)")
            print(f"📊 Resultados guardados en /results/")
            print(f"{'='*60}{Colores.FIN}")
            
            return True
        else:
            print(f"{Colores.ROJO}❌ Número de técnica inválido: {numero}{Colores.FIN}")
            return False
            
    except KeyboardInterrupt:
        print(f"\n{Colores.AMARILLO}⚠️  Ejecución interrumpida por el usuario{Colores.FIN}")
        return False
    except Exception as e:
        fin = time.time()
        duracion = fin - inicio
        
        print(f"\n{Colores.ROJO}{'='*60}")
        print(f"❌ ERROR EN LA EJECUCIÓN:")
        print(f"   {str(e)}")
        print(f"⏱️  Tiempo transcurrido: {duracion:.1f} segundos")
        print(f"{'='*60}{Colores.FIN}")
        return False

def ejecutar_todas_las_tecnicas():
    """Ejecutar las 11 técnicas completas"""
    tecnicas_info = [
        # Supervisadas
        (1, "🌳 Árboles de Decisión", "Clasificación con reglas interpretables"),
        (2, "📏 Inducción de Reglas", "Generación de reglas IF-THEN"),
        (3, "🎲 Clasificación Bayesiana", "Clasificación probabilística"),
        (4, "👥 Basado en Ejemplares", "Clasificación por similitud (K-NN)"),
        (5, "🧠 Redes de Neuronas", "Aprendizaje con redes neuronales"),
        (6, "🌫️ Lógica Borrosa", "Clasificación con conjuntos difusos"),
        (7, "🧬 Técnicas Genéticas", "Optimización evolutiva"),
        # No supervisadas
        (8, "🔗 A Priori", "Reglas de asociación"),
        (9, "📊 Clustering Numérico", "Agrupación K-Means"),
        (10, "🎯 Clustering Conceptual", "Agrupación por conceptos"),
        (11, "🎲 Clustering Probabilístico", "Agrupación EM")
    ]
    
    print(f"\n{Colores.MORADO}{Colores.NEGRITA}🚀 EJECUTANDO SISTEMA COMPLETO DE MINERÍA DE DATOS{Colores.FIN}")
    print(f"{Colores.AMARILLO}⏱️  Sistema integral: 11 técnicas (7 supervisadas + 4 no supervisadas)")
    print(f"   Tiempo estimado: 30-60 minutos dependiendo de tu hardware...{Colores.FIN}")
    
    # Confirmar ejecución
    confirmacion = input(f"\n{Colores.AMARILLO}¿Ejecutar el sistema completo? (s/N): {Colores.FIN}").strip().lower()
    if confirmacion not in ['s', 'si', 'sí', 'y', 'yes']:
        print(f"{Colores.AMARILLO}❌ Ejecución cancelada por el usuario{Colores.FIN}")
        return
    
    print(f"\n{Colores.VERDE}✅ Iniciando sistema integral de minería de datos...{Colores.FIN}\n")
    
    resultados = {}
    tiempo_total_inicio = time.time()
    supervisadas_exitosas = 0
    no_supervisadas_exitosas = 0
    
    for i, (num, nombre, desc) in enumerate(tecnicas_info, 1):
        print(f"{Colores.CYAN}{'─'*80}")
        print(f"[{i}/11] Ejecutando: {nombre}")
        if i <= 7:
            print(f"        Categoría: SUPERVISADA (Clasificación)")
        else:
            print(f"        Categoría: NO SUPERVISADA (Descubrimiento)")
        print(f"{'─'*80}{Colores.FIN}")
        
        exito = ejecutar_tecnica(num, nombre, desc)
        resultados[nombre] = exito
        
        if exito:
            print(f"{Colores.VERDE}✅ {nombre} completado exitosamente{Colores.FIN}")
            if i <= 7:
                supervisadas_exitosas += 1
            else:
                no_supervisadas_exitosas += 1
        else:
            print(f"{Colores.ROJO}❌ {nombre} falló durante la ejecución{Colores.FIN}")
        
        # Pausa entre técnicas
        if i < len(tecnicas_info):
            print(f"\n{Colores.AMARILLO}⏳ Preparando siguiente técnica en 3 segundos...{Colores.FIN}")
            time.sleep(3)
    
    tiempo_total_fin = time.time()
    duracion_total = tiempo_total_fin - tiempo_total_inicio
    
    # Resumen final
    print(f"\n{Colores.MORADO}{Colores.NEGRITA}{'='*80}")
    print("📊 RESUMEN DEL SISTEMA INTEGRAL DE MINERÍA DE DATOS")
    print(f"{'='*80}{Colores.FIN}")
    
    total_exitosas = supervisadas_exitosas + no_supervisadas_exitosas
    total_fallidas = 11 - total_exitosas
    
    print(f"✅ Técnicas exitosas: {Colores.VERDE}{total_exitosas}/11{Colores.FIN}")
    print(f"   └─ Supervisadas: {Colores.VERDE}{supervisadas_exitosas}/7{Colores.FIN}")
    print(f"   └─ No supervisadas: {Colores.VERDE}{no_supervisadas_exitosas}/4{Colores.FIN}")
    print(f"❌ Técnicas fallidas: {Colores.ROJO}{total_fallidas}/11{Colores.FIN}")
    print(f"⏱️  Tiempo total: {Colores.AMARILLO}{duracion_total/60:.1f} minutos{Colores.FIN}")
    
    print(f"\n{Colores.CYAN}📁 ARCHIVOS GENERADOS:{Colores.FIN}")
    print("   📊 Gráficos: /results/graficos/")
    print("   🤖 Modelos: /results/modelos/")
    print("   📄 Reportes: /results/reportes/")
    
    if total_exitosas == 11:
        print(f"\n{Colores.VERDE}🎉 ¡SISTEMA COMPLETO EJECUTADO EXITOSAMENTE!")
        print(f"🏆 Tu proyecto integral de minería de datos está completo")
        print(f"📈 Tienes implementadas TODAS las técnicas principales de IA{Colores.FIN}")
    elif total_exitosas >= 8:
        print(f"\n{Colores.AMARILLO}⚠️  Sistema mayormente completado ({total_exitosas}/11)")
        print(f"💡 Excelente progreso, revisa las técnicas fallidas{Colores.FIN}")
    elif total_exitosas >= 5:
        print(f"\n{Colores.AMARILLO}⚠️  Sistema parcialmente completado ({total_exitosas}/11)")
        print(f"💡 Buen progreso, pero revisa configuración del sistema{Colores.FIN}")
    else:
        print(f"\n{Colores.ROJO}❌ Sistema no completado exitosamente")
        print(f"🔧 Revisa la configuración del sistema y dependencias{Colores.FIN}")

def ejecutar_solo_supervisadas():
    """Ejecutar solo las 7 técnicas supervisadas"""
    tecnicas_supervisadas = [
        (1, "🌳 Árboles de Decisión", "Clasificación interpretable"),
        (2, "📏 Inducción de Reglas", "Reglas IF-THEN"),
        (3, "🎲 Clasificación Bayesiana", "Naive Bayes"),
        (4, "👥 Basado en Ejemplares", "K-NN"),
        (5, "🧠 Redes de Neuronas", "Deep Learning"),
        (6, "🌫️ Lógica Borrosa", "Fuzzy Logic"),
        (7, "🧬 Técnicas Genéticas", "Algoritmos Evolutivos")
    ]
    
    print(f"\n{Colores.VERDE}{Colores.NEGRITA}⚖️  EJECUTANDO TÉCNICAS SUPERVISADAS{Colores.FIN}")
    print(f"{Colores.AMARILLO}🎯 Enfoque: Clasificación de poblaciones por tamaño")
    print(f"⏱️  Tiempo estimado: 20-35 minutos{Colores.FIN}")
    
    confirmacion = input(f"\n{Colores.AMARILLO}¿Ejecutar las 7 técnicas supervisadas? (s/N): {Colores.FIN}").strip().lower()
    if confirmacion not in ['s', 'si', 'sí', 'y', 'yes']:
        return
    
    _ejecutar_conjunto_tecnicas(tecnicas_supervisadas, "SUPERVISADAS")

def ejecutar_solo_no_supervisadas():
    """Ejecutar solo las 4 técnicas no supervisadas"""
    tecnicas_no_supervisadas = [
        (8, "🔗 A Priori", "Reglas de asociación"),
        (9, "📊 Clustering Numérico", "K-Means"),
        (10, "🎯 Clustering Conceptual", "Agrupación conceptual"),
        (11, "🎲 Clustering Probabilístico", "EM Algorithm")
    ]
    
    print(f"\n{Colores.CYAN}{Colores.NEGRITA}🔍 EJECUTANDO TÉCNICAS NO SUPERVISADAS{Colores.FIN}")
    print(f"{Colores.AMARILLO}🎯 Enfoque: Descubrimiento de patrones ocultos")
    print(f"⏱️  Tiempo estimado: 15-25 minutos{Colores.FIN}")
    
    confirmacion = input(f"\n{Colores.AMARILLO}¿Ejecutar las 4 técnicas no supervisadas? (s/N): {Colores.FIN}").strip().lower()
    if confirmacion not in ['s', 'si', 'sí', 'y', 'yes']:
        return
    
    _ejecutar_conjunto_tecnicas(tecnicas_no_supervisadas, "NO SUPERVISADAS")

def _ejecutar_conjunto_tecnicas(tecnicas, tipo):
    """Función auxiliar para ejecutar un conjunto de técnicas"""
    resultados = {}
    tiempo_inicio = time.time()
    exitosas = 0
    
    for i, (num, nombre, desc) in enumerate(tecnicas, 1):
        print(f"{Colores.CYAN}{'─'*60}")
        print(f"[{i}/{len(tecnicas)}] {nombre}")
        print(f"{'─'*60}{Colores.FIN}")
        
        exito = ejecutar_tecnica(num, nombre, desc)
        resultados[nombre] = exito
        
        if exito:
            exitosas += 1
            print(f"{Colores.VERDE}✅ Completado{Colores.FIN}")
        else:
            print(f"{Colores.ROJO}❌ Falló{Colores.FIN}")
        
        if i < len(tecnicas):
            time.sleep(2)
    
    duracion = (time.time() - tiempo_inicio) / 60
    
    print(f"\n{Colores.MORADO}📊 RESUMEN {tipo}:")
    print(f"✅ Exitosas: {exitosas}/{len(tecnicas)}")
    print(f"⏱️  Tiempo: {duracion:.1f} minutos{Colores.FIN}")

def ver_resultados_guardados():
    """Mostrar información detallada sobre resultados guardados"""
    print(f"\n{Colores.CYAN}📊 ANÁLISIS DE RESULTADOS GUARDADOS...{Colores.FIN}\n")
    
    rutas = {
        'Gráficos': '/home/sedc/Proyectos/MineriaDeDatos/results/graficos/',
        'Modelos': '/home/sedc/Proyectos/MineriaDeDatos/results/modelos/',
        'Reportes': '/home/sedc/Proyectos/MineriaDeDatos/results/reportes/'
    }
    
    total_archivos = 0
    total_tamaño = 0
    tecnicas_completadas = []
    
    for categoria, ruta in rutas.items():
        print(f"{Colores.NEGRITA}{categoria}:{Colores.FIN}")
        
        if os.path.exists(ruta):
            archivos = os.listdir(ruta)
            if archivos:
                archivos_ordenados = sorted(archivos, key=lambda x: os.path.getmtime(os.path.join(ruta, x)), reverse=True)
                
                for archivo in archivos_ordenados:
                    ruta_completa = os.path.join(ruta, archivo)
                    tamaño = os.path.getsize(ruta_completa)
                    fecha = datetime.fromtimestamp(os.path.getmtime(ruta_completa))
                    
                    # Iconos según el tipo de archivo
                    if archivo.endswith('.png'):
                        icono = "🖼️"
                    elif archivo.endswith('.pkl'):
                        icono = "🤖"
                    elif archivo.endswith('.txt'):
                        icono = "📄"
                    else:
                        icono = "📁"
                    
                    print(f"  {icono} {archivo}")
                    print(f"      📏 {tamaño/1024:.1f} KB | 📅 {fecha.strftime('%Y-%m-%d %H:%M')}")
                    
                    total_archivos += 1
                    total_tamaño += tamaño
                    
                    # Detectar técnicas completadas
                    if '_clasificacion.png' in archivo or '_asociacion.png' in archivo or '_clustering.png' in archivo:
                        tecnica = archivo.replace('_clasificacion.png', '').replace('_asociacion.png', '').replace('_clustering.png', '').replace('_', ' ').title()
                        if tecnica not in tecnicas_completadas:
                            tecnicas_completadas.append(tecnica)
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
        print(f"  🔬 Técnicas detectadas: {len(tecnicas_completadas)}/11{Colores.FIN}")
        
        if tecnicas_completadas:
            print(f"\n{Colores.VERDE}✅ Técnicas completadas:")
            for tecnica in sorted(tecnicas_completadas):
                print(f"  🎯 {tecnica}{Colores.FIN}")

def crear_ranking_tecnicas():
    """Crear ranking de técnicas por rendimiento"""
    print(f"\n{Colores.CYAN}🏆 GENERANDO RANKING DE TÉCNICAS...{Colores.FIN}\n")
    
    # Buscar reportes de técnicas
    reportes_path = '/home/sedc/Proyectos/MineriaDeDatos/results/reportes/'
    
    if not os.path.exists(reportes_path):
        print(f"{Colores.ROJO}❌ No se encontraron reportes{Colores.FIN}")
        return
    
    tecnicas_rendimiento = []
    
    # Buscar archivos de reporte
    archivos_reporte = [f for f in os.listdir(reportes_path) if f.endswith('_reporte.txt')]
    
    for archivo in archivos_reporte:
        try:
            with open(os.path.join(reportes_path, archivo), 'r', encoding='utf-8') as f:
                contenido = f.read()
                
                # Extraer precisión del contenido
                if 'Precisión:' in contenido:
                    lineas = contenido.split('\n')
                    for linea in lineas:
                        if 'Precisión:' in linea or 'precisión:' in linea:
                            # Buscar valor numérico
                            import re
                            match = re.search(r'(\d+\.\d+)', linea)
                            if match:
                                precision = float(match.group(1))
                                tecnica_nombre = archivo.replace('_reporte.txt', '').replace('_', ' ').title()
                                
                                # Determinar categoría
                                categoria = "Supervisada" if any(x in archivo for x in ['clasificacion', 'arboles', 'bayesian', 'knn', 'redes', 'borrosa', 'genetica']) else "No Supervisada"
                                
                                tecnicas_rendimiento.append({
                                    'nombre': tecnica_nombre,
                                    'precision': precision,
                                    'categoria': categoria,
                                    'archivo': archivo
                                })
                                break
        except Exception as e:
            print(f"  ⚠️ Error leyendo {archivo}: {e}")
    
    if not tecnicas_rendimiento:
        print(f"{Colores.AMARILLO}⚠️ No se encontraron métricas de rendimiento{Colores.FIN}")
        return
    
    # Ordenar por precisión
    tecnicas_rendimiento.sort(key=lambda x: x['precision'], reverse=True)
    
    print(f"{Colores.NEGRITA}🏆 RANKING DE TÉCNICAS POR PRECISIÓN:{Colores.FIN}")
    print("=" * 60)
    
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
        
        print(f"{color}{emoji} {tecnica['nombre']:25} | {categoria_emoji} {tecnica['categoria']:13} | 🎯 {tecnica['precision']:.3f}{Colores.FIN}")
    
    # Estadísticas
    supervisadas = [t for t in tecnicas_rendimiento if t['categoria'] == "Supervisada"]
    no_supervisadas = [t for t in tecnicas_rendimiento if t['categoria'] == "No Supervisada"]
    
    print(f"\n{Colores.CYAN}📊 ESTADÍSTICAS:")
    if supervisadas:
        precision_sup = np.mean([t['precision'] for t in supervisadas])
        print(f"  ⚖️  Precisión promedio supervisadas: {precision_sup:.3f}")
    if no_supervisadas:
        precision_no_sup = np.mean([t['precision'] for t in no_supervisadas])
        print(f"  🔍 Precisión promedio no supervisadas: {precision_no_sup:.3f}")
    
    precision_total = np.mean([t['precision'] for t in tecnicas_rendimiento])
    print(f"  🎯 Precisión promedio total: {precision_total:.3f}{Colores.FIN}")

def crear_dashboard_ejecutivo():
    """Crear dashboard ejecutivo con resumen visual"""
    print(f"\n{Colores.CYAN}📈 GENERANDO DASHBOARD EJECUTIVO...{Colores.FIN}\n")
    
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Datos del sistema
        datos_ok, supervisadas_disponibles, no_supervisadas_disponibles = verificar_archivos()
        
        # Crear figura del dashboard
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('📈 DASHBOARD EJECUTIVO - SISTEMA DE MINERÍA DE DATOS', fontsize=16, fontweight='bold')
        
        # Gráfico 1: Estado del sistema
        categorias = ['Supervisadas', 'No Supervisadas']
        disponibles = [supervisadas_disponibles, no_supervisadas_disponibles]
        totales = [7, 4]
        
        x = np.arange(len(categorias))
        width = 0.35
        
        axes[0,0].bar(x - width/2, disponibles, width, label='Disponibles', color='lightgreen')
        axes[0,0].bar(x + width/2, totales, width, label='Total', color='lightblue', alpha=0.7)
        axes[0,0].set_xlabel('Categoría de Técnicas')
        axes[0,0].set_ylabel('Cantidad')
        axes[0,0].set_title('🔬 Estado de Implementación')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels(categorias)
        axes[0,0].legend()
        
        # Añadir etiquetas
        for i, (disp, total) in enumerate(zip(disponibles, totales)):
            axes[0,0].text(i - width/2, disp + 0.1, str(disp), ha='center', fontweight='bold')
            axes[0,0].text(i + width/2, total + 0.1, str(total), ha='center', fontweight='bold')
        
        # Gráfico 2: Técnicas por categoría (pie chart)
        total_implementadas = sum(disponibles)
        labels = ['Supervisadas', 'No Supervisadas']
        sizes = disponibles
        colors = ['#ff9999', '#66b3ff']
        
        axes[0,1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[0,1].set_title('📊 Distribución de Técnicas Implementadas')
        
        # Gráfico 3: Progreso del proyecto
        etapas = ['Análisis', 'Diseño', 'Implementación', 'Pruebas', 'Documentación']
        progreso = [100, 100, (total_implementadas/11)*100, 80, 90]  # Porcentajes estimados
        
        axes[1,0].barh(etapas, progreso, color=['green', 'green', 'orange', 'yellow', 'lightblue'])
        axes[1,0].set_xlabel('Progreso (%)')
        axes[1,0].set_title('🚀 Progreso del Proyecto')
        axes[1,0].set_xlim(0, 100)
        
        for i, v in enumerate(progreso):
            axes[1,0].text(v + 1, i, f'{v:.0f}%', va='center', fontweight='bold')
        
        # Gráfico 4: Resumen de archivos generados
        reportes_path = '/home/sedc/Proyectos/MineriaDeDatos/results/'
        tipos_archivos = {'Gráficos': 0, 'Modelos': 0, 'Reportes': 0}
        
        for tipo, path in [('Gráficos', 'graficos/'), ('Modelos', 'modelos/'), ('Reportes', 'reportes/')]:
            full_path = os.path.join(reportes_path, path)
            if os.path.exists(full_path):
                tipos_archivos[tipo] = len([f for f in os.listdir(full_path) if os.path.isfile(os.path.join(full_path, f))])
        
        axes[1,1].bar(tipos_archivos.keys(), tipos_archivos.values(), color=['purple', 'orange', 'green'])
        axes[1,1].set_ylabel('Cantidad de Archivos')
        axes[1,1].set_title('💾 Archivos Generados')
        
        for i, (tipo, cantidad) in enumerate(tipos_archivos.items()):
            axes[1,1].text(i, cantidad + 0.5, str(cantidad), ha='center', fontweight='bold')
        
        plt.tight_layout()
        dashboard_path = '/home/sedc/Proyectos/MineriaDeDatos/results/graficos/dashboard_ejecutivo.png'
        plt.savefig(dashboard_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"💾 Dashboard guardado: {dashboard_path}")
        
        # Resumen textual
        print(f"\n{Colores.CYAN}📋 RESUMEN EJECUTIVO:")
        print(f"  🔬 Técnicas implementadas: {total_implementadas}/11 ({(total_implementadas/11)*100:.1f}%)")
        print(f"  📁 Archivos generados: {sum(tipos_archivos.values())}")
        print(f"  📊 Sistema {'COMPLETO' if total_implementadas == 11 else 'EN DESARROLLO'}")
        
        if total_implementadas >= 8:
            print(f"  🎉 Proyecto en excelente estado")
        elif total_implementadas >= 5:
            print(f"  👍 Proyecto en buen progreso")
        else:
            print(f"  🔧 Proyecto requiere más desarrollo{Colores.FIN}")
        
    except ImportError:
        print(f"{Colores.ROJO}❌ Matplotlib no disponible para generar dashboard{Colores.FIN}")
    except Exception as e:
        print(f"{Colores.ROJO}❌ Error generando dashboard: {e}{Colores.FIN}")

def limpiar_resultados():
    """Limpiar todos los resultados anteriores"""
    print(f"\n{Colores.AMARILLO}🧹 LIMPIEZA DE RESULTADOS ANTERIORES{Colores.FIN}")
    
    confirmacion = input(f"\n{Colores.ROJO}⚠️ ¿Estás seguro de borrar TODOS los resultados? (s/N): {Colores.FIN}").strip().lower()
    if confirmacion not in ['s', 'si', 'sí', 'y', 'yes']:
        print(f"{Colores.AMARILLO}❌ Limpieza cancelada{Colores.FIN}")
        return
    
    rutas_limpiar = [
        '/home/sedc/Proyectos/MineriaDeDatos/results/graficos/',
        '/home/sedc/Proyectos/MineriaDeDatos/results/modelos/',
        '/home/sedc/Proyectos/MineriaDeDatos/results/reportes/'
    ]
    
    archivos_borrados = 0
    
    for ruta in rutas_limpiar:
        if os.path.exists(ruta):
            for archivo in os.listdir(ruta):
                archivo_path = os.path.join(ruta, archivo)
                try:
                    os.remove(archivo_path)
                    archivos_borrados += 1
                except Exception as e:
                    print(f"  ❌ Error borrando {archivo}: {e}")
    
    print(f"\n{Colores.VERDE}✅ Limpieza completada")
    print(f"  🗑️ Archivos borrados: {archivos_borrados}")
    print(f"  📁 Carpetas mantenidas para nuevos resultados{Colores.FIN}")

def exportar_proyecto():
    """Crear backup completo del proyecto"""
    print(f"\n{Colores.CYAN}💾 EXPORTANDO PROYECTO COMPLETO...{Colores.FIN}")
    
    try:
        import shutil
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"MineriaDeDatos_Backup_{timestamp}"
        backup_path = f"/home/sedc/Proyectos/{backup_name}"
        
        print(f"📦 Creando backup en: {backup_path}")
        
        # Copiar todo el proyecto
        shutil.copytree('/home/sedc/Proyectos/MineriaDeDatos', backup_path)
        
        # Crear archivo comprimido
        archivo_zip = f"{backup_path}.zip"
        shutil.make_archive(backup_path, 'zip', '/home/sedc/Proyectos/', backup_name)
        
        # Borrar carpeta temporal
        shutil.rmtree(backup_path)
        
        tamaño = os.path.getsize(archivo_zip) / 1024 / 1024
        
        print(f"\n{Colores.VERDE}✅ Backup creado exitosamente")
        print(f"  📦 Archivo: {archivo_zip}")
        print(f"  📏 Tamaño: {tamaño:.1f} MB")
        print(f"  📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colores.FIN}")
        
    except Exception as e:
        print(f"{Colores.ROJO}❌ Error creando backup: {e}{Colores.FIN}")

def configuracion_sistema():
    """Mostrar configuración detallada del sistema"""
    print(f"\n{Colores.CYAN}🔧 CONFIGURACIÓN Y DIAGNÓSTICO COMPLETO{Colores.FIN}\n")
    
    # Información del sistema
    print(f"{Colores.NEGRITA}💻 Información del Sistema:{Colores.FIN}")
    try:
        import platform
        print(f"  OS: {platform.system()} {platform.release()}")
        print(f"  Arquitectura: {platform.machine()}")
        print(f"  Python: {sys.version.split()[0]}")
        print(f"  Directorio actual: {os.getcwd()}")
    except Exception as e:
        print(f"  ⚠️ Error obteniendo info del sistema: {e}")
    
    # Verificar librerías
    print(f"\n{Colores.NEGRITA}🐍 Librerías de Python:{Colores.FIN}")
    librerias_requeridas = {
        'pandas': 'Manipulación de datos',
        'numpy': 'Computación numérica',
        'sklearn': 'Machine Learning',
        'matplotlib': 'Gráficos básicos',
        'seaborn': 'Gráficos estadísticos',
        'scipy': 'Computación científica'
    }
    
    for lib, descripcion in librerias_requeridas.items():
        try:
            modulo = __import__(lib)
            version = getattr(modulo, '__version__', 'N/A')
            print(f"  ✅ {lib} ({version}): {Colores.VERDE}{descripcion}{Colores.FIN}")
        except ImportError:
            print(f"  ❌ {lib}: {Colores.ROJO}NO instalado - {descripcion}{Colores.FIN}")
    
    # Verificar estructura completa
    print(f"\n{Colores.NEGRITA}📁 Verificación Completa del Sistema:{Colores.FIN}")
    datos_ok, supervisadas_disponibles, no_supervisadas_disponibles = verificar_archivos()
    
    # Recomendaciones
    print(f"\n{Colores.NEGRITA}💡 Diagnóstico y Recomendaciones:{Colores.FIN}")
    
    total_tecnicas = supervisadas_disponibles + no_supervisadas_disponibles
    
    if not datos_ok:
        print(f"  {Colores.ROJO}❌ CRÍTICO: Archivo de datos no encontrado{Colores.FIN}")
        print(f"    💡 Verifica que el archivo CSV esté en: /home/sedc/Proyectos/MineriaDeDatos/data/")
    
    if total_tecnicas < 11:
        print(f"  {Colores.AMARILLO}⚠️ ADVERTENCIA: Solo {total_tecnicas}/11 técnicas disponibles{Colores.FIN}")
        print(f"    💡 Faltan {11-total_tecnicas} técnicas por implementar")
    
    if total_tecnicas == 11 and datos_ok:
        print(f"  {Colores.VERDE}✅ EXCELENTE: Sistema completo configurado y listo{Colores.FIN}")
        print(f"    🚀 Puedes ejecutar cualquier técnica sin problemas")
        print(f"    🎯 Sistema integral de minería de datos operativo")

def main():
    """Función principal del menú actualizado"""
    while True:
        limpiar_pantalla()
        mostrar_banner()
        mostrar_menu_principal()
        
        try:
            opcion = input(f"{Colores.AMARILLO}👉 Selecciona una opción (0-20): {Colores.FIN}").strip()
            
            if opcion == '0':
                print(f"\n{Colores.VERDE}👋 ¡Gracias por usar el Sistema Integral de Minería de Datos!")
                print(f"🎓 ¡Éxito en tu proyecto de IA y Análisis de Datos!")
                print(f"📧 Revisa los reportes generados para documentar tu trabajo{Colores.FIN}\n")
                break
                
            elif opcion in [str(i) for i in range(1, 12)]:  # Técnicas individuales 1-11
                tecnicas_nombres = {
                    '1': ("🌳 Árboles de Decisión", "Clasificación interpretable con reglas de decisión"),
                    '2': ("📏 Inducción de Reglas", "Generación automática de reglas IF-THEN"),
                    '3': ("🎲 Clasificación Bayesiana", "Clasificación probabilística usando teorema de Bayes"),
                    '4': ("👥 Basado en Ejemplares (K-NN)", "Clasificación por similitud con vecinos cercanos"),
                    '5': ("🧠 Redes de Neuronas", "Aprendizaje profundo con múltiples arquitecturas"),
                    '6': ("🌫️ Lógica Borrosa", "Clasificación con conjuntos difusos y reglas borrosas"),
                    '7': ("🧬 Técnicas Genéticas", "Optimización evolutiva de características e hiperparámetros"),
                    '8': ("🔗 A Priori", "Reglas de asociación - patrones si A entonces B"),
                    '9': ("📊 Clustering Numérico", "Agrupación K-Means por similitud numérica"),
                    '10': ("🎯 Clustering Conceptual", "Agrupación basada en conceptos y características"),
                    '11': ("🎲 Clustering Probabilístico", "Agrupación EM con modelos probabilísticos")
                }
                
                nombre, desc = tecnicas_nombres[opcion]
                ejecutar_tecnica(int(opcion), nombre, desc)
                
            elif opcion == '12':
                ejecutar_todas_las_tecnicas()
                
            elif opcion == '13':
                ejecutar_solo_supervisadas()
                
            elif opcion == '14':
                ejecutar_solo_no_supervisadas()
                
            elif opcion == '15':
                ver_resultados_guardados()
                
            elif opcion == '16':
                crear_ranking_tecnicas()
                
            elif opcion == '17':
                crear_dashboard_ejecutivo()
                
            elif opcion == '18':
                configuracion_sistema()
                
            elif opcion == '19':
                limpiar_resultados()
                
            elif opcion == '20':
                exportar_proyecto()
                
            else:
                print(f"{Colores.ROJO}❌ Opción inválida. Por favor selecciona un número del 0 al 20.{Colores.FIN}")
                time.sleep(2)
            
            if opcion != '0':
                input(f"\n{Colores.AMARILLO}📎 Presiona ENTER para volver al menú principal...{Colores.FIN}")
                
        except KeyboardInterrupt:
            print(f"\n\n{Colores.AMARILLO}⚠️  Operación cancelada por el usuario")
            confirmacion = input(f"¿Deseas salir del sistema? (s/N): {Colores.FIN}").strip().lower()
            if confirmacion in ['s', 'si', 'sí', 'y', 'yes']:
                print(f"{Colores.VERDE}👋 ¡Hasta luego!{Colores.FIN}\n")
                break
        except Exception as e:
            print(f"\n{Colores.ROJO}❌ Error inesperado: {e}{Colores.FIN}")
            input(f"{Colores.AMARILLO}📎 Presiona ENTER para continuar...{Colores.FIN}")

if __name__ == "__main__":
    # Verificar sistema antes de iniciar
    print(f"{Colores.CYAN}🔍 Inicializando Sistema Integral de Minería de Datos...{Colores.FIN}")
    time.sleep(1)
    
    datos_ok, supervisadas_disponibles, no_supervisadas_disponibles = verificar_archivos()
    total_tecnicas = supervisadas_disponibles + no_supervisadas_disponibles
    
    if datos_ok and total_tecnicas >= 1:
        print(f"\n{Colores.VERDE}✅ Sistema inicializado correctamente")
        print(f"📊 Datos: OK | Técnicas: {total_tecnicas}/11 ({supervisadas_disponibles} sup. + {no_supervisadas_disponibles} no sup.)")
        
        if total_tecnicas == 11:
            print(f"🎉 ¡Sistema COMPLETO disponible!{Colores.FIN}")
        elif total_tecnicas >= 8:
            print(f"👍 Sistema casi completo{Colores.FIN}")
        else:
            print(f"🔧 Sistema en desarrollo{Colores.FIN}")
            
        time.sleep(2)
        main()
    else:
        print(f"\n{Colores.ROJO}❌ Sistema no está completamente configurado")
        print(f"📊 Datos: {'OK' if datos_ok else 'FALTA'} | Técnicas: {total_tecnicas}/11{Colores.FIN}")
        
        continuar = input(f"\n{Colores.AMARILLO}¿Deseas continuar de todas formas? (s/N): {Colores.FIN}").strip().lower()
        if continuar in ['s', 'si', 'sí', 'y', 'yes']:
            main()
        else:
            print(f"{Colores.AMARILLO}💡 Por favor configura el sistema y vuelve a intentar")
            print(f"📋 Usa la opción 18 del menú para más detalles de configuración{Colores.FIN}")