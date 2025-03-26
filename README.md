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

BUY_DATE = '2025.03.25'
BUFFER_DIR = '/kaggle/buffers/orottick4'
LOTTE_KIND = 'p4a'
DATA_DF = None
DATE_CNT = 56 * 5
O_DATE_CNT = 7
TCK_CNT = 56 * 5
RUNTIME = 60 * 60 * 11.5
PRD_SORT_ORDER = 'B'
HAS_STEP_LOG = True
RANGE_CNT = 52
M4P_OBS = True
M4P_CNT = 3
M4P_VRY = True
LOAD_CACHE_DIR = '/kaggle/working'
SAVE_CACHE_DIR = '/kaggle/working'
CACHE_CNT = -1

METHOD = 'simulate'
#METHOD = 'observe'
#METHOD = 'observe_range'
#METHOD = 'download'
#METHOD = 'build_cache'

#----------#

ok4s = vok4.Orottick4Simulator(PRD_SORT_ORDER, HAS_STEP_LOG, M4P_OBS, M4P_CNT, M4P_VRY, LOAD_CACHE_DIR, SAVE_CACHE_DIR)

if METHOD == 'build_cache':
    cdf = ok4s.build_cache(BUY_DATE, CACHE_CNT, BUFFER_DIR, LOTTE_KIND, DATA_DF, RUNTIME)
    if cdf is not None:
        try:
            cdf.to_csv(f'/kaggle/working/{LOTTE_KIND}-cache-{BUY_DATE}.csv', index=False)
        except Exception as e:
            msg = str(e)
            print(f'=> [E] {msg}')
    
if METHOD == 'simulate':
    zdf, json_pred, pdf = ok4s.simulate(BUY_DATE, BUFFER_DIR, LOTTE_KIND, DATA_DF, DATE_CNT, TCK_CNT, RUNTIME)
    if zdf is not None:
        zdf.to_csv(f'/kaggle/working/{LOTTE_KIND}-sim-{BUY_DATE}.csv', index=False)
    if pdf is not None:
        pdf.to_csv(f'/kaggle/working/{LOTTE_KIND}-pick-{BUY_DATE}.csv', index=False)
    if json_pred is not None:
        with open(f'/kaggle/working/{LOTTE_KIND}-pred-{BUY_DATE}.json', 'w') as f:
            json.dump(json_pred, f)

        text = '''
====================================
    PREDICT: [__LK__] __BD__
  -------------------------------

+ Predicted Numbers: __RS__

+ M4P Numbers: __M4__

+ Win Number:

+ Result: 

+ M4P Result:

+ Predict Notebook:


  -------------------------------
              MONEY
  -------------------------------

+ Period No: 

+ Day No: 

+ Tickets: 280

+ Cost: $364

+ Total Cost: $364

+ Prize: $0

+ Total Prize: $0

+ Current ROI: 0.0  


  -------------------------------
            REAL BUY
  -------------------------------

+ Buy Number: __M4__

+ Confirmation Number: 

+ Cost: $2.6

+ Total Cost: $2.6

+ Prize: $0

+ Total Prize: $0

+ Current ROI: 0.0

        '''
        text = text.replace('__LK__', str(LOTTE_KIND)).replace('__BD__', str(BUY_DATE)).replace('__RS__', str(json_pred['pred'])).replace('__M4__', str(json_pred['m4_pred']))
        with open(f'/kaggle/working/{LOTTE_KIND}-pred-{BUY_DATE}.txt', 'w') as f:
            f.write(text)
        print(text)

if METHOD == 'observe':
    odf, more = ok4s.observe(LOTTE_KIND, BUY_DATE, TCK_CNT, O_DATE_CNT, RUNTIME, DATE_CNT, BUFFER_DIR, DATA_DF)

    if odf is not None and more is not None and len(odf) > 0:
        odf.to_csv(f'/kaggle/working/{LOTTE_KIND}-observe-{BUY_DATE}.csv', index=False)
        qdf = odf[odf['m4'] > 0]
        if len(qdf) > 0:
            for ri in range(len(qdf)):
                t_buy_date = qdf['buy_date'].iloc[ri]

                key = 'pred_' + t_buy_date    
                if key in more:
                    json_pred = more[key]
                    if json_pred is not None:
                        with open(f'/kaggle/working/{LOTTE_KIND}-pred-{t_buy_date}.json', 'w') as f:
                            json.dump(json_pred, f)

                key = 'sim_' + t_buy_date                
                if key in more:
                    xdf = more[key]
                    if xdf is not None:
                        xdf.to_csv(f'/kaggle/working/{LOTTE_KIND}-sim-{t_buy_date}.csv', index=False)

                key = 'pick_' + t_buy_date                
                if key in more:
                    xdf = more[key]
                    if xdf is not None:
                        xdf.to_csv(f'/kaggle/working/{LOTTE_KIND}-pick-{t_buy_date}.csv', index=False)

if METHOD == 'observe_range':
    start_time = time.time()
    range_idx = 0
    while range_idx < RANGE_CNT:
        if time.time() - start_time > RUNTIME:
            break
            
        d1 = datetime.strptime(BUY_DATE, "%Y.%m.%d")
        g = -(range_idx * O_DATE_CNT)
        d2 = d1 + timedelta(minutes=int(+(g*(60 * 24))))
        v_buy_date = d2.strftime('%Y.%m.%d')
        o_overtime = time.time() - start_time
        v_runtime = RUNTIME - o_overtime

        odf, more = ok4s.observe(LOTTE_KIND, v_buy_date, TCK_CNT, O_DATE_CNT, v_runtime, DATE_CNT, BUFFER_DIR, DATA_DF)
    
        if odf is not None and more is not None and len(odf) > 0:
            odf.to_csv(f'/kaggle/working/{LOTTE_KIND}-observe-{v_buy_date}.csv', index=False)
            qdf = odf[odf['m4'] > 0]
            if len(qdf) > 0:
                for ri in range(len(qdf)):
                    t_buy_date = qdf['buy_date'].iloc[ri]
    
                    key = 'pred_' + t_buy_date    
                    if key in more:
                        json_pred = more[key]
                        if json_pred is not None:
                            with open(f'/kaggle/working/{LOTTE_KIND}-pred-{t_buy_date}.json', 'w') as f:
                                json.dump(json_pred, f)
    
                    key = 'sim_' + t_buy_date                
                    if key in more:
                        xdf = more[key]
                        if xdf is not None:
                            xdf.to_csv(f'/kaggle/working/{LOTTE_KIND}-sim-{t_buy_date}.csv', index=False)
    
                    key = 'pick_' + t_buy_date                
                    if key in more:
                        xdf = more[key]
                        if xdf is not None:
                            xdf.to_csv(f'/kaggle/working/{LOTTE_KIND}-pick-{t_buy_date}.csv', index=False)

        range_idx += 1


if METHOD == 'download':  
    d1 = datetime.strptime(BUY_DATE, "%Y.%m.%d")
    g = -1
    d2 = d1 + timedelta(minutes=int(+(g*(60 * 24))))
    v_date = d2.strftime('%Y.%m.%d')

    data_df = ok4s.download_drawing(BUFFER_DIR, LOTTE_KIND, v_date)

    if data_df is not None:
        data_df.to_csv(f'/kaggle/working/{LOTTE_KIND}-{BUY_DATE}.csv', index=False)

#----------#


====================================
             CACHES
  -------------------------------


  -------------------------------
         FORWARD CACHES
  -------------------------------

[ 2025.03.23 ]

+ Notebook: https://www.kaggle.com/code/dinhttrandrise/orottick4-cache-p4a-f-2025-03-23


  -------------------------------
         BACKWARD CACHES
  -------------------------------

[ 2025.03.23 ]

+ Notebook: 


====================================
            EXAMPLES
  -------------------------------

+ Predict: https://www.kaggle.com/code/dinhttrandrise/orottick4-predict-p4a-2025-03-23

+ Observe (One Period): https://www.kaggle.com/code/dinhttrandrise/orottick4-observe-p4a-2025-03-23

+ Observe (Multiple Periods): https://www.kaggle.com/code/dinhttrandrise/orottick4-observe-range-p4a-2025-03-23


```
