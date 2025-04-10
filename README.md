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

BUY_DATE = '2025.04.06'
BUFFER_DIR = '/kaggle/buffers/orottick4'
LOTTE_KIND = 'p4a'
DATA_DF = None
DATE_CNT = 56 * 5
O_DATE_CNT = 7
TCK_CNT = 56 * 5
F_TCK_CNT = 56 * 5
RUNTIME = 60 * 60 * 11.5
PRD_SORT_ORDER = 'B'
HAS_STEP_LOG = True
RANGE_CNT = 52
M4P_OBS = False 
M4P_CNT = 10
M4P_VRY = False
M4P_ONE = False
RESULT_DIR = '/kaggle/working'
LOAD_CACHE_DIR = '/kaggle/input/orottick4-research-a-1-2025-04-06'
SAVE_CACHE_DIR = '/kaggle/working'
CACHE_CNT = -1
USE_GITHUB = True
M4P_COLLECT_DATA_DIRS = []
M4P_COLLECT_SAVE_DIR = '/kaggle/working'
M4P_PREPARE_DATA_DIR = '/kaggle/working'
M4P_PREPARE_SAVE_DIR = '/kaggle/working'
M4P_TRAIN_DATA_DIR = '/kaggle/working'
M4P_TRAIN_SAVE_DIR = '/kaggle/working'
M4P_MODEL_DIR = '/kaggle/input/orottick4-m4pm-rsp-a-k-2025-03-23'
M4PC_MODEL_DIR = '/kaggle/input/orottick4-m4pcm-rsp-a-o-2025-04-06'
M4P_RANKER_ONLY = True
M4P_MAX = 5
M4PL_MAX = 2.5
M4PL_MIN = -2.5
M4PL_STEP = 0.0225
M4PC_TRAIN_DATA_DIR = '/kaggle/input/orottick4-research-b-1-2025-04-06'
M4PC_TRAIN_SAVE_DIR = '/kaggle/working'
TCK_PRIZE = 5000
BRK_COST = 0.3
PREDICT_NOTEBOOK = 'https://www.kaggle.com/code/dinhttrandrise/orottick4-predict-rsp-a-o-2025-04-06'
PERIOD_NO = 1
DAY_NO = 1
REAL_BUY_TIMES = 1

METHOD = 'simulate'
#METHOD = 'observe'
#METHOD = 'observe_range'
#METHOD = 'download'
#METHOD = 'build_cache'
#METHOD = 'm4p_collect'
#METHOD = 'm4p_prepare'
#METHOD = 'm4p_train'
#METHOD = 'research_a'
#METHOD = 'm4pc_prepare'
#METHOD = 'm4pc_train'

#----------#

options = {'BUY_DATE': BUY_DATE, 'BUFFER_DIR': BUFFER_DIR, 'LOTTE_KIND': LOTTE_KIND, 'DATA_DF': DATA_DF, 'DATE_CNT': DATE_CNT, 'O_DATE_CNT': O_DATE_CNT, 'TCK_CNT': TCK_CNT, 'F_TCK_CNT': F_TCK_CNT, 'RUNTIME': RUNTIME, 'PRD_SORT_ORDER': PRD_SORT_ORDER, 'HAS_STEP_LOG': HAS_STEP_LOG, 'RANGE_CNT': RANGE_CNT, 'M4P_OBS': M4P_OBS, 'M4P_CNT': M4P_CNT, 'M4P_VRY': M4P_VRY, 'M4P_ONE': M4P_ONE, 'RESULT_DIR': RESULT_DIR, 'LOAD_CACHE_DIR': LOAD_CACHE_DIR, 'SAVE_CACHE_DIR': SAVE_CACHE_DIR, 'CACHE_CNT': CACHE_CNT, 'USE_GITHUB': USE_GITHUB, 'METHOD': METHOD, 'M4P_COLLECT_DATA_DIRS': M4P_COLLECT_DATA_DIRS, 'M4P_COLLECT_SAVE_DIR': M4P_COLLECT_SAVE_DIR, 'M4P_PREPARE_DATA_DIR': M4P_PREPARE_DATA_DIR, 'M4P_PREPARE_SAVE_DIR': M4P_PREPARE_SAVE_DIR, 'M4P_TRAIN_DATA_DIR': M4P_TRAIN_DATA_DIR, 'M4P_TRAIN_SAVE_DIR': M4P_TRAIN_SAVE_DIR, 'M4PC_MODEL_DIR': M4PC_MODEL_DIR, 'M4P_MODEL_DIR': M4P_MODEL_DIR, 'M4P_RANKER_ONLY': M4P_RANKER_ONLY, 'M4P_MAX': M4P_MAX, 'M4PL_MAX': M4PL_MAX, 'M4PL_MIN': M4PL_MIN, 'M4PL_STEP': M4PL_STEP, 'M4PC_TRAIN_DATA_DIR': M4PC_TRAIN_DATA_DIR, 'M4PC_TRAIN_SAVE_DIR': M4PC_TRAIN_SAVE_DIR, 'TCK_PRIZE': TCK_PRIZE, 'BRK_COST': BRK_COST, 'PREDICT_NOTEBOOK': PREDICT_NOTEBOOK, 'PERIOD_NO': PERIOD_NO, 'DAY_NO': DAY_NO, 'REAL_BUY_TIMES': REAL_BUY_TIMES}

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

[ 2023.03.26 ]

+ Notebook 1: https://www.kaggle.com/code/dinhttrandrise/orottick4-cache-p4a-f-2023-03-26

+ Notebook 2: https://www.kaggle.com/code/dinhttrandrise/orottick4-cache-p4a-f-2-2023-03-26


====================================
           M4P MODELS
  -------------------------------

[ RSP A, Plan G]

+ Notebook: https://www.kaggle.com/code/dinhttrandrise/orottick4-m4pm-rsp-a-g-2025-03-23

[ RSP A, Plan K]

+ Notebook: https://www.kaggle.com/code/dinhttrandrise/orottick4-m4pm-rsp-a-k-2025-03-23


====================================
           HOW TO INVEST
  -------------------------------

With Orottick4, we can invest in lottery (Pick 4) as following research:


  -------------------------------
    RESEARCH PROJECT A - PLAN O
  -------------------------------

+ Invest Period: 364 days

+ Invest Cost: $6,188

+ Invest Cost (w/o broker): $4,760

+ Invest Return: $58,812

+ Invest Return (w/o broker): $60,240

+ Invest ROI: 950.4%

+ Invest ROI (w/o broker, w/ cost): 973.5%

+ Invest ROI (w/o broker, w/ no broker cost): 1065.5%

+ Research project A: https://github.com/dinhtt-randrise/orottick4/tree/main/research/rsp-a

+ Lottery Investment A Project: https://github.com/dinhtt-randrise/orottick4/tree/main/invest/lip-a

```

```
  ----------- Plan O ------------
```

![](https://github.com/dinhtt-randrise/orottick4/blob/fed4e897e4927ee3e30e79a616ddd2b1ab4b1b37/research/rsp-a/orottick4-rsp-a-analyze-o.png)



