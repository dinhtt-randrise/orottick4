```
              _   _   _    _   _ _  
  ___ _ _ ___| |_| |_(_)__| |_| | | 
 / _ \ '_/ _ \  _|  _| / _| / /_  _|
 \___/_| \___/\__|\__|_\__|_\_\ |_| 
------------------------------------
 Oregon Lottery - Pick 4 Predictor
------------------------------------

====================================
              ABOUT
  -------------------------------

Orottick4 (Oregon Lottery - Pick 4 Predictor) predicts
Pick 4 (Oregon Lottery) draw by simulating computerized
lottery drawing.


====================================
              LINKS
  -------------------------------

+ Kaggle: https://orottick4.com/kaggle

+ GitHub: https://orottick4.com/github

+ Lottery: https://orottick4.com/lotte


====================================
            HOW TO USE
  -------------------------------

#----------#

import json
import time
from datetime import datetime
from datetime import timedelta
import pandas as pd
import numpy as np
import os
import warnings 
warnings.filterwarnings('ignore')

OROTTICK4P_DIR = "/kaggle/buffers/orottick4p"

os.system(f'mkdir -p "{OROTTICK4P_DIR}"')
os.system(f'cd "{OROTTICK4P_DIR}" && git clone https://github.com/dinhtt-randrise/orottick4.git')

import sys 
sys.path.append(os.path.abspath(OROTTICK4P_DIR))
import orottick4.orottick4p as vok4p

#----------#

BUY_DATE = '2025.01.02'
LAST_BUY_DATE = '2025.01.01'
FIRST_BUY_DATE = '2015.01.01'
BUFFER_DIR = '/kaggle/buffers/orottick4p'
LOTTE_KIND = 'p4a'
DATA_DF = None
HAS_STEP_LOG = True
RESULT_DIR = '/kaggle/working'
LOAD_CACHE_DIR = '/kaggle/working'
SAVE_CACHE_DIR = '/kaggle/working'
CACHE_RUN = True
DEBUG_SEED_CNT = 2
SEED_CNT = 10000
SEED_CNT_2 = 2000
USE_GITHUB = True
TRACK_DIR = None
TRACK_BUY_DATE = LAST_BUY_DATE
CACHE_DEBUG_ON = True
RUNTIME = 60 * 60 * 11
START_TIME = time.time()

MP_COST = 1.3
MP_PRIZE = 5000 * (1 - 0.24 - 0.08)
MP_RATE = 15
MIN_TRY_NO = 1
STEP_TRY_NO = 100000
MAX_TRY_NO = MIN_TRY_NO + STEP_TRY_NO
MB_0_RATE = 1.5
CACHE_DIR = LOAD_CACHE_DIR

MODEL_FILE = None
SEED_LOOP_MAX = 10
PICKED_BUY_DATE = None

#METHOD = 'download'
METHOD = 'cache'
#METHOD = 'train'
#METHOD = 'analyze'
#METHOD = 'predict'

#----------#

options = {'BUY_DATE': BUY_DATE, 'BUFFER_DIR': BUFFER_DIR, 'LOTTE_KIND': LOTTE_KIND, 'DATA_DF': DATA_DF, 'HAS_STEP_LOG': HAS_STEP_LOG, 'RESULT_DIR': RESULT_DIR, 'LOAD_CACHE_DIR': LOAD_CACHE_DIR, 'SAVE_CACHE_DIR': SAVE_CACHE_DIR, 'CACHE_RUN': CACHE_RUN, 'DEBUG_SEED_CNT': DEBUG_SEED_CNT, 'SEED_CNT': SEED_CNT, 'SEED_CNT_2': SEED_CNT_2, 'USE_GITHUB': USE_GITHUB, 'METHOD': METHOD, 'LAST_BUY_DATE': LAST_BUY_DATE, 'FIRST_BUY_DATE': FIRST_BUY_DATE, 'TRACK_DIR': TRACK_DIR, 'TRACK_BUY_DATE': TRACK_BUY_DATE, 'CACHE_DEBUG_ON': CACHE_DEBUG_ON, 'RUNTIME': RUNTIME, 'START_TIME': START_TIME, 'MP_COST': MP_COST, 'MP_PRIZE': MP_PRIZE, 'MP_RATE': MP_RATE, 'MIN_TRY_NO': MIN_TRY_NO, 'STEP_TRY_NO': STEP_TRY_NO, 'MAX_TRY_NO': MAX_TRY_NO, 'MB_0_RATE': MB_0_RATE, 'CACHE_DIR': CACHE_DIR, 'MODEL_FILE': MODEL_FILE, 'SEED_LOOP_MAX': SEED_LOOP_MAX, 'PICKED_BUY_DATE': PICKED_BUY_DATE}
vok4p.Orottick4PSimulator.run(options, vok4p, None)

#----------#


====================================
             CACHES
  -------------------------------


  -------------------------------
         PICK 4 (1PM)
  -------------------------------

+ Notebook 1: https://www.kaggle.com/code/dinhttrandrise/orottick4pc-cache-p4a-1-2025-01-01

```
