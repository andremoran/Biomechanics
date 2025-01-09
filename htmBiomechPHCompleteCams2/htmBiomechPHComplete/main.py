# Ejemplo básico de código Python
print("Hello, Python in Visual Studio!")
import subprocess
import sys

# Instalar opencv-python-headless
subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless<4.3"])

# Instalar cellpose
subprocess.check_call([sys.executable, "-m", "pip", "install", "cellpose"])



import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from cellpose import core, utils, io, models, metrics
from glob import glob

# Verificar si la GPU está activada
use_GPU = core.use_gpu()
yn = ['NO', 'YES']
print(f'>>> GPU activated? {yn[use_GPU]}')
