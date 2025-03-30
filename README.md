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

OROTTICK4_DIR = "/kaggle/buffers/orottick4"

os.system(f'mkdir -p "{OROTTICK4_DIR}"')
os.system(f'cd "{OROTTICK4_DIR}" && git clone https://github.com/dinhtt-randrise/orottick4.git')

import sys 
sys.path.append(os.path.abspath(OROTTICK4_DIR))
import orottick4.orottick4 as vok4

#----------#

BUY_DATE = '2025.03.27'
BUFFER_DIR = '/kaggle/buffers/orottick4'
LOTTE_KIND = 'p4a'
DATA_DF = None
DATE_CNT = 56 * 5
O_DATE_CNT = 7
TCK_CNT = 56 * 5
F_TCK_CNT = 250
RUNTIME = 60 * 60 * 11.5
PRD_SORT_ORDER = 'B'
HAS_STEP_LOG = True
RANGE_CNT = 52
M4P_OBS = False
M4P_CNT = 3
M4P_VRY = False
M4P_ONE = True
RESULT_DIR = '/kaggle/working'
LOAD_CACHE_DIR = '/kaggle/working'
SAVE_CACHE_DIR = '/kaggle/working'
CACHE_CNT = -1
USE_GITHUB = True

METHOD = 'simulate'
#METHOD = 'observe'
#METHOD = 'observe_range'
#METHOD = 'download'
#METHOD = 'build_cache'

#----------#

options = {'BUY_DATE': BUY_DATE, 'BUFFER_DIR': BUFFER_DIR, 'LOTTE_KIND': LOTTE_KIND, 'DATA_DF': DATA_DF, 'DATE_CNT': DATE_CNT, 'O_DATE_CNT': O_DATE_CNT, 'TCK_CNT': TCK_CNT, 'F_TCK_CNT': F_TCK_CNT, 'RUNTIME': RUNTIME, 'PRD_SORT_ORDER': PRD_SORT_ORDER, 'HAS_STEP_LOG': HAS_STEP_LOG, 'RANGE_CNT': RANGE_CNT, 'M4P_OBS': M4P_OBS, 'M4P_CNT': M4P_CNT, 'M4P_VRY': M4P_VRY, 'M4P_ONE': M4P_ONE, 'RESULT_DIR': RESULT_DIR, 'LOAD_CACHE_DIR': LOAD_CACHE_DIR, 'SAVE_CACHE_DIR': SAVE_CACHE_DIR, 'CACHE_CNT': CACHE_CNT, 'USE_GITHUB': USE_GITHUB, 'METHOD': METHOD}

vok4.Orottick4Simulator.run(options, vok4, None)

#----------#


====================================
             CACHES
  -------------------------------


  -------------------------------
         FORWARD CACHES
  -------------------------------

[ 2025.03.30 ]

+ Notebook: https://www.kaggle.com/code/dinhttrandrise/orottick4-cache-p4a-f-2025-03-30

[ 2025.03.23 ]

+ Notebook 1: https://www.kaggle.com/code/dinhttrandrise/orottick4-cache-p4a-f-2025-03-23

+ Notebook 2: https://www.kaggle.com/code/dinhttrandrise/orottick4-cache-p4a-f-2-2025-03-23

[ 2024.03.24 ]

+ Notebook 1: https://www.kaggle.com/code/dinhttrandrise/orottick4-cache-p4a-f-2024-03-24

+ Notebook 2: https://www.kaggle.com/code/dinhttrandrise/orottick4-cache-p4a-f-2-2024-03-24


====================================
           HOW TO INVEST
  -------------------------------

With Orottick4, we can invest in lottery (Pick 4) as following research:


  -------------------------------
    RESEARCH PROJECT A - PLAN B
  -------------------------------

+ Invest Period: 364 days

+ Invest Cost: $24,375

+ Invest Cost (w/o broker): $18,750

+ Invest Return: $5,625

+ Invest Return (w/o broker): $11,250

+ Invest ROI: 23.1%

+ Invest ROI (w/o broker, w/ cost): 46.2%

+ Invest ROI (w/o broker, w/ no broker cost): 60%

+ Research project A: https://github.com/dinhtt-randrise/orottick4/tree/main/research/rsp-a

+ Lottery Investment A Project: https://github.com/dinhtt-randrise/orottick4/tree/main/invest/lip-a

```

![](https://github.com/dinhtt-randrise/orottick4/blob/ad930f2d91e56434d272099e722f575296f3f2f3/research/rsp-a/orottick4-rsp-a-analyze-b.png)


