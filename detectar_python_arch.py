#!/usr/bin/env python3
"""
DETECTOR DE CONFIGURACIÓN PYTHON EN ARCH LINUX
Detecta dónde están instaladas las librerías y cómo configurar VS Code
"""

import sys
import os
import subprocess

def ejecutar_comando(comando):
    """Ejecutar comando y devolver resultado"""
    try:
        resultado = subprocess.run(comando, shell=True, capture_output=True, text=True)
        return resultado.stdout.strip()
    except:
        return None

def encontrar_site_packages():
    """Encontrar todas las rutas de site-packages"""
    rutas = []
    
    # Rutas estándar del sistema
    for version in ['3.12', '3.11', '3.10', '3.9']:
        ruta_sistema = f"/usr/lib/python{version}/site-packages"
        if os.path.exists(ruta_sistema):
            rutas.append(ruta_sistema)
        
        # Ruta de usuario
        home = os.path.expanduser("~")
        ruta_usuario = f"{home}/.local/lib/python{version}/site-packages"
        if os.path.exists(ruta_usuario):
            rutas.append(ruta_usuario)
    
    return rutas

def verificar_libreria_en_ruta(libreria, ruta):
    """Verificar si una librería está en una ruta específica"""
    posibles_nombres = [libreria, libreria.replace('-', '_')]
    
    for nombre in posibles_nombres:
        ruta_lib = os.path.join(ruta, nombre)
        if os.path.exists(ruta_lib):
            return True
        
        # Buscar archivos .dist-info
        for item in os.listdir(ruta):
            if item.startswith(nombre) and ('.dist-info' in item or '.egg-info' in item):
                return True
    return False

def main():
    print("🔍 DETECTOR DE CONFIGURACIÓN PYTHON EN ARCH LINUX")
    print("=" * 55)
    
    # Información básica
    print(f"\n🐍 INFORMACIÓN DE PYTHON:")
    print(f"   Versión: {sys.version}")
    print(f"   Ejecutable: {sys.executable}")
    print(f"   Plataforma: {sys.platform}")
    
    # Rutas de Python
    print(f"\n📁 RUTAS DE PYTHON:")
    for i, ruta in enumerate(sys.path):
        if ruta:
            print(f"   {i+1}. {ruta}")
    
    # Encontrar site-packages
    print(f"\n📦 DIRECTORIOS SITE-PACKAGES:")
    rutas_site_packages = encontrar_site_packages()
    
    for ruta in rutas_site_packages:
        print(f"   📁 {ruta}")
        if os.path.exists(ruta):
            try:
                archivos = len(os.listdir(ruta))
                print(f"      ({archivos} paquetes)")
            except:
                print("      (sin acceso)")
        else:
            print("      (no existe)")
    
    # Verificar librerías específicas
    librerias_requeridas = {
        'pandas': 'pandas',
        'numpy': 'numpy', 
        'sklearn': 'scikit_learn',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'scipy': 'scipy',
        'joblib': 'joblib'
    }
    
    print(f"\n🔍 VERIFICACIÓN DE LIBRERÍAS:")
    configuracion_rutas = []
    
    for lib_import, lib_package in librerias_requeridas.items():
        print(f"\n   📚 {lib_import} ({lib_package}):")
        
        # Intentar importar
        try:
            modulo = __import__(lib_import)
            version = getattr(modulo, '__version__', 'N/A')
            ruta_modulo = getattr(modulo, '__file__', 'N/A')
            print(f"      ✅ Importación exitosa - Versión: {version}")
            print(f"      📍 Ubicación: {ruta_modulo}")
            
            # Extraer directorio padre para configuración
            if ruta_modulo != 'N/A':
                directorio = os.path.dirname(os.path.dirname(ruta_modulo))
                if directorio not in configuracion_rutas:
                    configuracion_rutas.append(directorio)
                    
        except ImportError as e:
            print(f"      ❌ Error de importación: {e}")
            
            # Buscar en rutas conocidas
            encontrada = False
            for ruta in rutas_site_packages:
                if verificar_libreria_en_ruta(lib_package, ruta):
                    print(f"      📦 Encontrada en: {ruta}")
                    if ruta not in configuracion_rutas:
                        configuracion_rutas.append(ruta)
                    encontrada = True
                    break
            
            if not encontrada:
                print(f"      ❌ No encontrada en rutas conocidas")
    
    # Generar configuración para VS Code
    print(f"\n⚙️ CONFIGURACIÓN RECOMENDADA PARA VS CODE:")
    print("   Crear archivo .vscode/settings.json con:")
    
    configuracion_json = {
        "python.defaultInterpreterPath": sys.executable,
        "python.analysis.extraPaths": configuracion_rutas,
        "python.analysis.autoSearchPaths": True,
        "python.analysis.useLibraryCodeForTypes": True,
        "python.analysis.diagnosticMode": "workspace"
    }
    
    import json
    print(json.dumps(configuracion_json, indent=4))
    
    # Comandos para instalar faltantes
    print(f"\n📦 COMANDOS DE INSTALACIÓN PARA ARCH:")
    print("   # Paquetes oficiales:")
    print("   sudo pacman -S python-pandas python-numpy python-scikit-learn python-matplotlib python-seaborn python-scipy")
    print("   ")
    print("   # Con pip (alternativa):")
    print("   pip install --user pandas numpy scikit-learn matplotlib seaborn scipy joblib")
    
    # Información adicional
    print(f"\n💡 INFORMACIÓN ADICIONAL:")
    print(f"   • Versión Python detectada: {sys.version_info.major}.{sys.version_info.minor}")
    print(f"   • Pip disponible: {ejecutar_comando('which pip') or 'No encontrado'}")
    print(f"   • Pacman disponible: {ejecutar_comando('which pacman') or 'No encontrado'}")
    
    # Comando para VS Code
    print(f"\n🔧 PASOS PARA VS CODE:")
    print("   1. Crear carpeta .vscode en tu proyecto")
    print("   2. Copiar la configuración JSON de arriba a .vscode/settings.json")
    print("   3. Reiniciar VS Code")
    print("   4. Ctrl+Shift+P → 'Python: Select Interpreter'")
    print(f"   5. Seleccionar: {sys.executable}")

if __name__ == "__main__":
    main()