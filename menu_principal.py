#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import importlib.util
from datetime import datetime

# Añadir el directorio del proyecto al path
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
    """Limpiar la pantalla según el sistema operativo"""
    os.system('cls' if os.name == 'nt' else 'clear')

def mostrar_banner():
    """Mostrar banner principal del sistema"""
    banner = f"""
{Colores.CYAN}╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  {Colores.NEGRITA}🧠 SISTEMA COMPLETO DE MINERÍA DE DATOS - IA AVANZADA 🧠{Colores.FIN}{Colores.CYAN}              ║
║                                                                              ║
║  {Colores.BLANCO}Universidad: [Tu Universidad]{Colores.CYAN}                                            ║
║  {Colores.BLANCO}Proyecto: Análisis Demográfico con Técnicas de IA{Colores.CYAN}                      ║
║  {Colores.BLANCO}Dataset: Censo Poblacional INEGI{Colores.CYAN}                                        ║
║  {Colores.BLANCO}Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}{Colores.CYAN}                                             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝{Colores.FIN}

{Colores.AMARILLO}📊 SISTEMA COMPLETO: Supervisadas + No Supervisadas{Colores.FIN}
{Colores.VERDE}🎯 CLASIFICACIÓN: 7 técnicas | CLUSTERING: 3 técnicas | PREDICCIÓN: 3 técnicas{Colores.FIN}
"""
    print(banner)

def mostrar_menu_principal():
    """Mostrar el menú principal con todas las opciones actualizadas"""
    menu = f"""
{Colores.NEGRITA}{Colores.AZUL}═══════════════════════════════════════════════════════════════════════════════
                           🔬 TÉCNICAS DISPONIBLES
═══════════════════════════════════════════════════════════════════════════════{Colores.FIN}

{Colores.VERDE}{Colores.NEGRITA}📚 TÉCNICAS SUPERVISADAS (CLASIFICACIÓN):{Colores.FIN}
{Colores.VERDE}1.{Colores.FIN} {Colores.NEGRITA}🌳 Árboles de Decisión{Colores.FIN}
   └─ Clasificación interpretable con reglas de decisión jerárquicas

{Colores.VERDE}2.{Colores.FIN} {Colores.NEGRITA}📏 Inducción de Reglas{Colores.FIN}
   └─ Generación automática de reglas IF-THEN legibles y explicativas

{Colores.VERDE}3.{Colores.FIN} {Colores.NEGRITA}🎲 Clasificación Bayesiana{Colores.FIN}
   └─ Clasificación probabilística usando teorema de Bayes (Naive Bayes)

{Colores.VERDE}4.{Colores.FIN} {Colores.NEGRITA}👥 Basado en Ejemplares (K-NN){Colores.FIN}
   └─ Clasificación por similitud con vecinos más cercanos

{Colores.VERDE}5.{Colores.FIN} {Colores.NEGRITA}🧠 Redes de Neuronas{Colores.FIN}
   └─ Aprendizaje profundo con múltiples arquitecturas neuronales

{Colores.VERDE}6.{Colores.FIN} {Colores.NEGRITA}🌫️  Lógica Borrosa (Fuzzy Logic){Colores.FIN}
   └─ Clasificación con conjuntos difusos y reglas borrosas

{Colores.VERDE}7.{Colores.FIN} {Colores.NEGRITA}🧬 Técnicas Genéticas{Colores.FIN}
   └─ Optimización evolutiva de características e hiperparámetros

{Colores.CYAN}{Colores.NEGRITA}🔍 TÉCNICAS NO SUPERVISADAS (CLUSTERING):{Colores.FIN}
{Colores.CYAN}8.{Colores.FIN} {Colores.NEGRITA}📈 Clustering Numérico{Colores.FIN}
   └─ K-Means, Clustering Jerárquico y DBSCAN para grupos similares

{Colores.CYAN}9.{Colores.FIN} {Colores.NEGRITA}🧠 Clustering Conceptual{Colores.FIN}
   └─ Agrupación interpretable basada en conceptos demográficos

{Colores.CYAN}10.{Colores.FIN} {Colores.NEGRITA}🎯 Clustering Probabilístico{Colores.FIN}
    └─ Gaussian Mixture Models con análisis de incertidumbre

{Colores.MORADO}{Colores.NEGRITA}🚀 EJECUCIÓN MASIVA:{Colores.FIN}
{Colores.MORADO}11.{Colores.FIN} {Colores.NEGRITA}🔥 Ejecutar TODAS las Supervisadas{Colores.FIN}
    └─ Ejecución completa de las 7 técnicas de clasificación

{Colores.MORADO}12.{Colores.FIN} {Colores.NEGRITA}🌟 Ejecutar TODOS los Clustering{Colores.FIN}
    └─ Ejecución completa de las 3 técnicas de clustering

{Colores.MORADO}13.{Colores.FIN} {Colores.NEGRITA}💥 Ejecutar TODO EL SISTEMA{Colores.FIN}
    └─ Ejecución completa: 7 Supervisadas + 3 Clustering (10 técnicas)

{Colores.AMARILLO}{Colores.NEGRITA}📊 ANÁLISIS Y GESTIÓN:{Colores.FIN}
{Colores.AMARILLO}14.{Colores.FIN} {Colores.NEGRITA}📊 Ver Resultados y Comparar{Colores.FIN}
    └─ Revisar reportes, gráficos y comparación de rendimiento

{Colores.AMARILLO}15.{Colores.FIN} {Colores.NEGRITA}🔧 Configuración y Diagnóstico{Colores.FIN}
    └─ Verificar sistema, rutas, datos y dependencias

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
    
    # Técnicas Supervisadas
    archivos_supervisadas = {
        'Árboles de Decisión': '/home/sedc/Proyectos/MineriaDeDatos/01_Supervisadas/02_Clasificacion/01_Arboles_Decision/1_arboles_decision.py',
        'Inducción de Reglas': '/home/sedc/Proyectos/MineriaDeDatos/01_Supervisadas/02_Clasificacion/02_Induccion_Reglas/2_induccion_reglas.py',
        'Clasificación Bayesiana': '/home/sedc/Proyectos/MineriaDeDatos/01_Supervisadas/02_Clasificacion/03_Bayesiana/3_clasificacion_bayesiana.py',
        'Basado en Ejemplares': '/home/sedc/Proyectos/MineriaDeDatos/01_Supervisadas/02_Clasificacion/04_Basado_Ejemplares/4_basado_ejemplares.py',
        'Redes de Neuronas': '/home/sedc/Proyectos/MineriaDeDatos/01_Supervisadas/02_Clasificacion/05_Redes_Neuronas/5_redes_neuronas.py',
        'Lógica Borrosa': '/home/sedc/Proyectos/MineriaDeDatos/01_Supervisadas/02_Clasificacion/06_Logica_Borrosa/6_logica_borrosa.py',
        'Técnicas Genéticas': '/home/sedc/Proyectos/MineriaDeDatos/01_Supervisadas/02_Clasificacion/07_Tecnicas_Geneticas/7_tecnicas_geneticas.py'
    }
    
    # Técnicas No Supervisadas
    archivos_clustering = {
        'Clustering Numérico': '/home/sedc/Proyectos/MineriaDeDatos/02_No_Supervisadas/01_Clustering/01_Numerico/clustering_numerico.py',
        'Clustering Conceptual': '/home/sedc/Proyectos/MineriaDeDatos/02_No_Supervisadas/01_Clustering/02_Conceptual/clustering_conceptual.py',
        'Clustering Probabilístico': '/home/sedc/Proyectos/MineriaDeDatos/02_No_Supervisadas/01_Clustering/03_Probabilistico/clustering_probabilistico.py'
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
    
    # Verificar técnicas de clustering
    print(f"\n{Colores.AMARILLO}🔍 Verificando técnicas de clustering...{Colores.FIN}")
    clustering_disponibles = 0
    for nombre, ruta in archivos_clustering.items():
        if os.path.exists(ruta):
            print(f"  ✅ {nombre}: {Colores.VERDE}OK{Colores.FIN}")
            clustering_disponibles += 1
        else:
            print(f"  ❌ {nombre}: {Colores.ROJO}NO ENCONTRADO{Colores.FIN}")
    
    datos_ok = os.path.exists(rutas_requeridas['datos'])
    
    print(f"\n{Colores.CYAN}📊 Resumen del sistema:")
    print(f"  Datos principales: {'✅' if datos_ok else '❌'}")
    print(f"  Técnicas supervisadas: {supervisadas_disponibles}/7")
    print(f"  Técnicas clustering: {clustering_disponibles}/3")
    print(f"  Total técnicas: {supervisadas_disponibles + clustering_disponibles}/10{Colores.FIN}")
    
    return datos_ok, supervisadas_disponibles, clustering_disponibles

def ejecutar_tecnica_supervisada(numero, nombre_tecnica, descripcion):
    """Ejecutar una técnica supervisada específica"""
    print(f"\n{Colores.CYAN}{'='*80}")
    print(f"🚀 EJECUTANDO TÉCNICA SUPERVISADA: {nombre_tecnica}")
    print(f"📝 {descripcion}")
    print(f"{'='*80}{Colores.FIN}\n")
    
    # Mapeo de técnicas supervisadas
    archivos_supervisadas = {
        1: ('/home/sedc/Proyectos/MineriaDeDatos/01_Supervisadas/02_Clasificacion/01_Arboles_Decision/1_arboles_decision.py', 'ejecutar_arboles_decision'),
        2: ('/home/sedc/Proyectos/MineriaDeDatos/01_Supervisadas/02_Clasificacion/02_Induccion_Reglas/2_induccion_reglas.py', 'ejecutar_induccion_reglas'),
        3: ('/home/sedc/Proyectos/MineriaDeDatos/01_Supervisadas/02_Clasificacion/03_Bayesiana/3_clasificacion_bayesiana.py', 'ejecutar_clasificacion_bayesiana'),
        4: ('/home/sedc/Proyectos/MineriaDeDatos/01_Supervisadas/02_Clasificacion/04_Basado_Ejemplares/4_basado_ejemplares.py', 'ejecutar_clasificacion_ejemplares'),
        5: ('/home/sedc/Proyectos/MineriaDeDatos/01_Supervisadas/02_Clasificacion/05_Redes_Neuronas/5_redes_neuronas.py', 'ejecutar_redes_neuronas'),
        6: ('/home/sedc/Proyectos/MineriaDeDatos/01_Supervisadas/02_Clasificacion/06_Logica_Borrosa/6_logica_borrosa.py', 'ejecutar_logica_borrosa'),
        7: ('/home/sedc/Proyectos/MineriaDeDatos/01_Supervisadas/02_Clasificacion/07_Tecnicas_Geneticas/7_tecnicas_geneticas.py', 'ejecutar_tecnicas_geneticas')
    }
    
    return ejecutar_tecnica_generica(numero, archivos_supervisadas, nombre_tecnica)

def ejecutar_tecnica_clustering(numero, nombre_tecnica, descripcion):
    """Ejecutar una técnica de clustering específica"""
    print(f"\n{Colores.CYAN}{'='*80}")
    print(f"🔍 EJECUTANDO TÉCNICA DE CLUSTERING: {nombre_tecnica}")
    print(f"📝 {descripcion}")
    print(f"{'='*80}{Colores.FIN}\n")
    
    # Mapeo de técnicas de clustering (ajustar números)
    archivos_clustering = {
        8: ('/home/sedc/Proyectos/MineriaDeDatos/02_No_Supervisadas/01_Clustering/01_Numerico/clustering_numerico.py', 'ejecutar_clustering_numerico'),
        9: ('/home/sedc/Proyectos/MineriaDeDatos/02_No_Supervisadas/01_Clustering/02_Conceptual/clustering_conceptual.py', 'ejecutar_clustering_conceptual'),
        10: ('/home/sedc/Proyectos/MineriaDeDatos/02_No_Supervisadas/01_Clustering/03_Probabilistico/clustering_probabilistico.py', 'ejecutar_clustering_probabilistico')
    }
    
    return ejecutar_tecnica_generica(numero, archivos_clustering, nombre_tecnica)

def ejecutar_tecnica_generica(numero, archivos_dict, nombre_tecnica):
    """Función genérica para ejecutar cualquier técnica"""
    inicio = time.time()
    
    try:
        if numero in archivos_dict:
            archivo_path, nombre_funcion = archivos_dict[numero]
            
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

def ejecutar_todas_supervisadas():
    """Ejecutar todas las técnicas supervisadas"""
    tecnicas_supervisadas = [
        (1, "🌳 Árboles de Decisión", "Clasificación con reglas interpretables"),
        (2, "📏 Inducción de Reglas", "Generación de reglas IF-THEN"),
        (3, "🎲 Clasificación Bayesiana", "Clasificación probabilística"),
        (4, "👥 Basado en Ejemplares", "Clasificación por similitud (K-NN)"),
        (5, "🧠 Redes de Neuronas", "Aprendizaje con redes neuronales"),
        (6, "🌫️ Lógica Borrosa", "Clasificación con conjuntos difusos"),
        (7, "🧬 Técnicas Genéticas", "Optimización evolutiva")
    ]
    
    return ejecutar_conjunto_tecnicas(tecnicas_supervisadas, "SUPERVISADAS", ejecutar_tecnica_supervisada)

def ejecutar_todos_clustering():
    """Ejecutar todas las técnicas de clustering"""
    tecnicas_clustering = [
        (8, "📈 Clustering Numérico", "K-Means, Jerárquico, DBSCAN"),
        (9, "🧠 Clustering Conceptual", "Agrupación interpretable"),
        (10, "🎯 Clustering Probabilístico", "Gaussian Mixture Models")
    ]
    
    return ejecutar_conjunto_tecnicas(tecnicas_clustering, "CLUSTERING", ejecutar_tecnica_clustering)

def ejecutar_sistema_completo():
    """Ejecutar todo el sistema: supervisadas + clustering"""
    print(f"\n{Colores.MORADO}{Colores.NEGRITA}💥 EJECUTANDO TODO EL SISTEMA DE MINERÍA DE DATOS{Colores.FIN}")
    print(f"{Colores.AMARILLO}⏱️  Esto puede tomar 45-60 minutos dependiendo de tu hardware...{Colores.FIN}")
    print(f"{Colores.CYAN}📊 Total: 7 Supervisadas + 3 Clustering = 10 técnicas{Colores.FIN}")
    
    # Confirmar ejecución
    confirmacion = input(f"\n{Colores.AMARILLO}¿Deseas continuar con la ejecución COMPLETA? (s/N): {Colores.FIN}").strip().lower()
    if confirmacion not in ['s', 'si', 'sí', 'y', 'yes']:
        print(f"{Colores.AMARILLO}❌ Ejecución cancelada por el usuario{Colores.FIN}")
        return
    
    tiempo_total_inicio = time.time()
    
    # Ejecutar supervisadas
    print(f"\n{Colores.VERDE}{'='*80}")
    print(f"🚀 FASE 1: TÉCNICAS SUPERVISADAS (7/10)")
    print(f"{'='*80}{Colores.FIN}")
    
    resultado_supervisadas = ejecutar_todas_supervisadas()
    
    # Pausa entre fases
    print(f"\n{Colores.AMARILLO}⏳ Preparando fase de clustering en 5 segundos...{Colores.FIN}")
    time.sleep(5)
    
    # Ejecutar clustering
    print(f"\n{Colores.CYAN}{'='*80}")
    print(f"🔍 FASE 2: TÉCNICAS DE CLUSTERING (3/10)")
    print(f"{'='*80}{Colores.FIN}")
    
    resultado_clustering = ejecutar_todos_clustering()
    
    # Resumen final
    tiempo_total_fin = time.time()
    duracion_total = tiempo_total_fin - tiempo_total_inicio
    
    exitosas_supervisadas = sum(resultado_supervisadas.values()) if resultado_supervisadas else 0
    exitosas_clustering = sum(resultado_clustering.values()) if resultado_clustering else 0
    total_exitosas = exitosas_supervisadas + exitosas_clustering
    
    print(f"\n{Colores.MORADO}{Colores.NEGRITA}{'='*80}")
    print("💥 RESUMEN SISTEMA COMPLETO DE MINERÍA DE DATOS")
    print(f"{'='*80}{Colores.FIN}")
    
    print(f"✅ Supervisadas exitosas: {exitosas_supervisadas}/7")
    print(f"✅ Clustering exitosas: {exitosas_clustering}/3")
    print(f"🏆 Total exitosas: {total_exitosas}/10")
    print(f"⏱️ Tiempo total: {Colores.AMARILLO}{duracion_total/60:.1f} minutos{Colores.FIN}")
    
    if total_exitosas == 10:
        print(f"\n{Colores.VERDE}🎉 ¡SISTEMA COMPLETO EJECUTADO AL 100%!")
        print(f"🏆 Las 10 técnicas de IA completadas exitosamente")
        print(f"💼 Tu proyecto de minería de datos está COMPLETO{Colores.FIN}")
    elif total_exitosas >= 7:
        print(f"\n{Colores.AMARILLO}⚠️ Sistema mayormente completado ({total_exitosas}/10)")
        print(f"💡 Revisa los errores y ejecuta las técnicas faltantes{Colores.FIN}")
    else:
        print(f"\n{Colores.ROJO}❌ Sistema parcialmente completado ({total_exitosas}/10)")
        print(f"🔧 Revisa la configuración y vuelve a intentar{Colores.FIN}")

def ejecutar_conjunto_tecnicas(tecnicas_lista, tipo_nombre, funcion_ejecutar):
    """Función genérica para ejecutar un conjunto de técnicas"""
    print(f"\n{Colores.MORADO}{Colores.NEGRITA}🚀 EJECUTANDO TODAS LAS TÉCNICAS {tipo_nombre}{Colores.FIN}")
    print(f"{Colores.AMARILLO}⏱️  Esto puede tomar 15-30 minutos dependiendo de tu hardware...{Colores.FIN}")
    
    # Confirmar ejecución
    confirmacion = input(f"\n{Colores.AMARILLO}¿Deseas continuar? (s/N): {Colores.FIN}").strip().lower()
    if confirmacion not in ['s', 'si', 'sí', 'y', 'yes']:
        print(f"{Colores.AMARILLO}❌ Ejecución cancelada por el usuario{Colores.FIN}")
        return
    
    print(f"\n{Colores.VERDE}✅ Iniciando ejecución de {len(tecnicas_lista)} técnicas...{Colores.FIN}\n")
    
    resultados = {}
    tiempo_total_inicio = time.time()
    
    for i, (num, nombre, desc) in enumerate(tecnicas_lista, 1):
        print(f"{Colores.CYAN}{'─'*80}")
        print(f"[{i}/{len(tecnicas_lista)}] Ejecutando: {nombre}")
        print(f"{'─'*80}{Colores.FIN}")
        
        exito = funcion_ejecutar(num, nombre, desc)
        resultados[nombre] = exito
        
        if exito:
            print(f"{Colores.VERDE}✅ {nombre} completado exitosamente{Colores.FIN}")
        else:
            print(f"{Colores.ROJO}❌ {nombre} falló durante la ejecución{Colores.FIN}")
        
        # Pausa entre técnicas
        if i < len(tecnicas_lista):
            print(f"\n{Colores.AMARILLO}⏳ Preparando siguiente técnica en 3 segundos...{Colores.FIN}")
            time.sleep(3)
    
    tiempo_total_fin = time.time()
    duracion_total = tiempo_total_fin - tiempo_total_inicio
    
    # Resumen
    exitosas = sum(resultados.values())
    fallidas = len(resultados) - exitosas
    
    print(f"\n{Colores.MORADO}{Colores.NEGRITA}{'='*80}")
    print(f"📊 RESUMEN TÉCNICAS {tipo_nombre}")
    print(f"{'='*80}{Colores.FIN}")
    
    print(f"✅ Técnicas exitosas: {Colores.VERDE}{exitosas}/{len(tecnicas_lista)}{Colores.FIN}")
    print(f"❌ Técnicas fallidas: {Colores.ROJO}{fallidas}/{len(tecnicas_lista)}{Colores.FIN}")
    print(f"⏱️  Tiempo total: {Colores.AMARILLO}{duracion_total/60:.1f} minutos{Colores.FIN}")
    
    if exitosas == len(tecnicas_lista):
        print(f"\n{Colores.VERDE}🎉 ¡TODAS LAS TÉCNICAS {tipo_nombre} COMPLETADAS!")
        print(f"🏆 Ejecución perfecta{Colores.FIN}")
    elif exitosas > 0:
        print(f"\n{Colores.AMARILLO}⚠️ Ejecución parcial ({exitosas}/{len(tecnicas_lista)})")
        print(f"💡 Revisa los errores y vuelve a ejecutar las fallidas{Colores.FIN}")
    
    return resultados

def ver_resultados_guardados():
    """Mostrar información detallada sobre resultados guardados"""
    print(f"\n{Colores.CYAN}📊 REVISANDO RESULTADOS GUARDADOS...{Colores.FIN}\n")
    
    rutas = {
        'Gráficos': '/home/sedc/Proyectos/MineriaDeDatos/results/graficos/',
        'Modelos': '/home/sedc/Proyectos/MineriaDeDatos/results/modelos/',
        'Reportes': '/home/sedc/Proyectos/MineriaDeDatos/results/reportes/'
    }
    
    total_archivos = 0
    total_tamaño = 0
    
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
            else:
                print(f"  {Colores.AMARILLO}📭 Carpeta vacía{Colores.FIN}")
        else:
            print(f"  {Colores.ROJO}❌ Carpeta no existe{Colores.FIN}")
        print()
    
    # Resumen total
    if total_archivos > 0:
        print(f"{Colores.CYAN}📊 RESUMEN TOTAL:")
        print(f"  📁 Total archivos: {total_archivos}")
        print(f"  💾 Espacio utilizado: {total_tamaño/1024/1024:.1f} MB{Colores.FIN}")
        
        # Detectar técnicas completadas
        print(f"\n{Colores.VERDE}✅ TÉCNICAS COMPLETADAS DETECTADAS:")
        
        # Supervisadas
        tecnicas_supervisadas_completadas = []
        tecnicas_clustering_completadas = []
        
        graficos_path = rutas['Gráficos']
        if os.path.exists(graficos_path):
            archivos_graficos = os.listdir(graficos_path)
            
            # Detectar supervisadas
            patrones_supervisadas = {
                'arboles_decision_clasificacion.png': '🌳 Árboles de Decisión',
                'induccion_reglas.png': '📏 Inducción de Reglas',
                'clasificacion_bayesiana.png': '🎲 Clasificación Bayesiana',
                'clasificacion_knn.png': '👥 Basado en Ejemplares',
                'redes_neuronas_clasificacion.png': '🧠 Redes de Neuronas',
                'logica_borrosa_clasificacion.png': '🌫️ Lógica Borrosa',
                'tecnicas_geneticas_clasificacion.png': '🧬 Técnicas Genéticas'
            }
            
            # Detectar clustering
            patrones_clustering = {
                'clustering_numerico.png': '📈 Clustering Numérico',
                'clustering_conceptual.png': '🧠 Clustering Conceptual',
                'clustering_probabilistico.png': '🎯 Clustering Probabilístico'
            }
            
            for archivo in archivos_graficos:
                if archivo in patrones_supervisadas:
                    tecnicas_supervisadas_completadas.append(patrones_supervisadas[archivo])
                elif archivo in patrones_clustering:
                    tecnicas_clustering_completadas.append(patrones_clustering[archivo])
        
        # Mostrar técnicas completadas
        print(f"  {Colores.VERDE}📚 Supervisadas ({len(tecnicas_supervisadas_completadas)}/7):{Colores.FIN}")
        for tecnica in tecnicas_supervisadas_completadas:
            print(f"    ✅ {tecnica}")
        
        print(f"  {Colores.CYAN}🔍 Clustering ({len(tecnicas_clustering_completadas)}/3):{Colores.FIN}")
        for tecnica in tecnicas_clustering_completadas:
            print(f"    ✅ {tecnica}")
        
        total_completadas = len(tecnicas_supervisadas_completadas) + len(tecnicas_clustering_completadas)
        print(f"\n  {Colores.MORADO}🏆 Total completadas: {total_completadas}/10 técnicas{Colores.FIN}")
        
        if total_completadas == 10:
            print(f"  {Colores.VERDE}🎉 ¡SISTEMA COMPLETO AL 100%!{Colores.FIN}")
        elif total_completadas >= 7:
            print(f"  {Colores.AMARILLO}⚠️ Sistema mayormente completo{Colores.FIN}")
        elif total_completadas >= 3:
            print(f"  {Colores.CYAN}🔧 Sistema parcialmente completo{Colores.FIN}")
        else:
            print(f"  {Colores.ROJO}📝 Pocas técnicas completadas{Colores.FIN}")

def configuracion_sistema():
    """Mostrar configuración detallada del sistema"""
    print(f"\n{Colores.CYAN}🔧 CONFIGURACIÓN Y DIAGNÓSTICO DEL SISTEMA{Colores.FIN}\n")
    
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
        'scipy': 'Computación científica',
        'joblib': 'Serialización de modelos'
    }
    
    for lib, descripcion in librerias_requeridas.items():
        try:
            modulo = __import__(lib)
            version = getattr(modulo, '__version__', 'N/A')
            print(f"  ✅ {lib} ({version}): {Colores.VERDE}{descripcion}{Colores.FIN}")
        except ImportError:
            print(f"  ❌ {lib}: {Colores.ROJO}NO instalado - {descripcion}{Colores.FIN}")
    
    # Verificar estructura completa
    print(f"\n{Colores.NEGRITA}📁 Estructura del Sistema:{Colores.FIN}")
    datos_ok, supervisadas_ok, clustering_ok = verificar_archivos()
    
    # Verificar espacio en disco
    print(f"\n{Colores.NEGRITA}💾 Espacio en Disco:{Colores.FIN}")
    try:
        import shutil
        total, usado, libre = shutil.disk_usage("/home/sedc/Proyectos/MineriaDeDatos/")
        print(f"  📊 Total: {total/1024**3:.1f} GB")
        print(f"  📈 Usado: {usado/1024**3:.1f} GB ({usado/total*100:.1f}%)")
        print(f"  📉 Libre: {libre/1024**3:.1f} GB ({libre/total*100:.1f}%)")
        
        if libre/1024**3 < 1:
            print(f"  {Colores.AMARILLO}⚠️ Advertencia: Poco espacio libre disponible{Colores.FIN}")
    except Exception as e:
        print(f"  ⚠️ Error obteniendo info de disco: {e}")
    
    # Diagnóstico y recomendaciones
    print(f"\n{Colores.NEGRITA}💡 Diagnóstico y Recomendaciones:{Colores.FIN}")
    
    if not datos_ok:
        print(f"  {Colores.ROJO}❌ CRÍTICO: Archivo de datos no encontrado{Colores.FIN}")
        print(f"    💡 Verifica que el archivo CSV esté en: /home/sedc/Proyectos/MineriaDeDatos/data/")
    
    if supervisadas_ok < 7:
        print(f"  {Colores.AMARILLO}⚠️ ADVERTENCIA: Solo {supervisadas_ok}/7 técnicas supervisadas disponibles{Colores.FIN}")
        print(f"    💡 Completa los archivos faltantes en /01_Supervisadas/02_Clasificacion/")
    
    if clustering_ok < 3:
        print(f"  {Colores.AMARILLO}⚠️ ADVERTENCIA: Solo {clustering_ok}/3 técnicas de clustering disponibles{Colores.FIN}")
        print(f"    💡 Completa los archivos faltantes en /02_No_Supervisadas/01_Clustering/")
    
    total_tecnicas = supervisadas_ok + clustering_ok
    if total_tecnicas == 10 and datos_ok:
        print(f"  {Colores.VERDE}✅ EXCELENTE: Sistema completamente configurado y listo{Colores.FIN}")
        print(f"    🚀 Puedes ejecutar cualquier técnica sin problemas")
        print(f"    💥 Sistema completo: 10/10 técnicas disponibles")
    elif total_tecnicas >= 7:
        print(f"  {Colores.AMARILLO}👍 BUENO: Sistema mayormente configurado ({total_tecnicas}/10){Colores.FIN}")
        print(f"    🔧 Completa las técnicas faltantes para funcionalidad total")
    else:
        print(f"  {Colores.ROJO}🔧 INCOMPLETO: Sistema requiere configuración ({total_tecnicas}/10){Colores.FIN}")
        print(f"    📋 Revisa los archivos faltantes y completa la instalación")

def main():
    """Función principal del menú actualizado"""
    while True:
        limpiar_pantalla()
        mostrar_banner()
        mostrar_menu_principal()
        
        try:
            opcion = input(f"{Colores.AMARILLO}👉 Selecciona una opción (0-15): {Colores.FIN}").strip()
            
            if opcion == '0':
                print(f"\n{Colores.VERDE}👋 ¡Gracias por usar el Sistema Completo de Minería de Datos!")
                print(f"🎓 ¡Éxito en tu proyecto de IA y Machine Learning!")
                print(f"📧 Revisa los reportes y gráficos generados para tu análisis{Colores.FIN}\n")
                break
                
            # Técnicas supervisadas (1-7)
            elif opcion in ['1', '2', '3', '4', '5', '6', '7']:
                tecnicas_supervisadas = {
                    '1': ("🌳 Árboles de Decisión", "Clasificación interpretable con reglas de decisión"),
                    '2': ("📏 Inducción de Reglas", "Generación automática de reglas IF-THEN"),
                    '3': ("🎲 Clasificación Bayesiana", "Clasificación probabilística usando teorema de Bayes"),
                    '4': ("👥 Basado en Ejemplares (K-NN)", "Clasificación por similitud con vecinos cercanos"),
                    '5': ("🧠 Redes de Neuronas", "Aprendizaje profundo con múltiples arquitecturas"),
                    '6': ("🌫️ Lógica Borrosa", "Clasificación con conjuntos difusos y reglas borrosas"),
                    '7': ("🧬 Técnicas Genéticas", "Optimización evolutiva de características e hiperparámetros")
                }
                
                nombre, desc = tecnicas_supervisadas[opcion]
                ejecutar_tecnica_supervisada(int(opcion), nombre, desc)
            
            # Técnicas de clustering (8-10)
            elif opcion in ['8', '9', '10']:
                tecnicas_clustering = {
                    '8': ("📈 Clustering Numérico", "K-Means, Clustering Jerárquico y DBSCAN"),
                    '9': ("🧠 Clustering Conceptual", "Agrupación interpretable basada en conceptos demográficos"),
                    '10': ("🎯 Clustering Probabilístico", "Gaussian Mixture Models con análisis de incertidumbre")
                }
                
                nombre, desc = tecnicas_clustering[opcion]
                ejecutar_tecnica_clustering(int(opcion), nombre, desc)
            
            # Ejecución masiva
            elif opcion == '11':
                ejecutar_todas_supervisadas()
                
            elif opcion == '12':
                ejecutar_todos_clustering()
                
            elif opcion == '13':
                ejecutar_sistema_completo()
                
            # Análisis y gestión
            elif opcion == '14':
                ver_resultados_guardados()
                
            elif opcion == '15':
                configuracion_sistema()
                
            else:
                print(f"{Colores.ROJO}❌ Opción inválida. Por favor selecciona un número del 0 al 15.{Colores.FIN}")
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
    print(f"{Colores.CYAN}🔍 Inicializando Sistema Completo de Minería de Datos...{Colores.FIN}")
    time.sleep(1)
    
    datos_ok, supervisadas_ok, clustering_ok = verificar_archivos()
    total_tecnicas = supervisadas_ok + clustering_ok
    
    if datos_ok and total_tecnicas >= 1:
        print(f"\n{Colores.VERDE}✅ Sistema inicializado correctamente")
        print(f"📊 Datos: OK | Supervisadas: {supervisadas_ok}/7 | Clustering: {clustering_ok}/3")
        print(f"🏆 Total técnicas: {total_tecnicas}/10{Colores.FIN}")
        time.sleep(2)
        main()
    else:
        print(f"\n{Colores.ROJO}❌ Sistema no está completamente configurado")
        print(f"📊 Datos: {'OK' if datos_ok else 'FALTA'} | Técnicas: {total_tecnicas}/10{Colores.FIN}")
        
        continuar = input(f"\n{Colores.AMARILLO}¿Deseas continuar de todas formas? (s/N): {Colores.FIN}").strip().lower()
        if continuar in ['s', 'si', 'sí', 'y', 'yes']:
            main()
        else:
            print(f"{Colores.AMARILLO}💡 Por favor configura el sistema y vuelve a intentar{Colores.FIN}")
            print(f"📋 Usa la opción 15 del menú para más detalles de configuración")