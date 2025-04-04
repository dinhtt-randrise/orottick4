# ------------------------------------------------------------ #
#              _   _   _    _   _ _  
#  ___ _ _ ___| |_| |_(_)__| |_| | | 
# / _ \ '_/ _ \  _|  _| / _| / /_  _|
# \___/_| \___/\__|\__|_\__|_\_\ |_| 
#------------------------------------
# Oregon Lottery - Pick 4 Predictor
#------------------------------------
#
#====================================
#            LINKS
#  -------------------------------
#
# + Kaggle: https://orottick4.com/kaggle
#
# + GitHub: https://orottick4.com/github
#
# + Lottery: https://orottick4.com/lotte
#
#
#====================================
#            Copyright
#  -------------------------------
#
# "Oregon Lottery - Pick 4 Predictor" is written and copyrighted
# by Dinh Thoai Tran <dinhtt@randrise.com> [https://dinhtt.randrise.com]
#
#
#====================================
#            License
#  -------------------------------
#
# "Oregon Lottery - Pick 4 Predictor" is distributed under Apache-2.0 license
# [ https://github.com/dinhtt-randrise/orottick4/blob/main/LICENSE ]
#
# ------------------------------------------------------------ #

import random
import os
import json
import time
from datetime import datetime
from datetime import timedelta
import pandas as pd
import numpy as np
import pickle
import glob

# ------------------------------------------------------------ #

class Orottick4Simulator:
    def __init__(self, prd_sort_order = 'A', has_step_log = True, m4p_obs = False, m4p_cnt = -1, m4p_vry = True, load_cache_dir = '/kaggle/working', save_cache_dir = '/kaggle/working', heading_printed = False):
        self.min_num = 0
        self.max_num = 9999

        self.baseset = {0: 1000, 1: 100, 2: 10, 3: 1}

        self.heading_printed = heading_printed

        if prd_sort_order not in ['A', 'B', 'C']:
            prd_sort_order = 'A'
        self.prd_sort_order = prd_sort_order

        self.has_step_log = has_step_log
        self.m4p_obs = m4p_obs
        self.m4p_cnt = m4p_cnt

        if m4p_cnt < 2:
            m4p_vry = False
        self.m4p_vry = m4p_vry

        self.cache_capture_seed = {}
        self.cache_reproduce_one = {}
        self.cache_capture = {}

        self.debug_on = False
        self.debug_gn_on = False
        self.debug_cs_on = False

        os.system(f'mkdir -p "{load_cache_dir}"')
        self.load_cache_dir = load_cache_dir

        os.system(f'mkdir -p "{save_cache_dir}"')
        self.save_cache_dir = save_cache_dir

        self.load_cache()

        self.lotte_kind = 'p4a'

    def save_cache(self):
        cdir = self.save_cache_dir

        fn = f'{cdir}/cache_capture_seed.pkl'
        with open(fn, 'wb') as f:
            pickle.dump(self.cache_capture_seed, f)

        fn = f'{cdir}/cache_reproduce_one.pkl'
        with open(fn, 'wb') as f:
            pickle.dump(self.cache_reproduce_one, f)

        fn = f'{cdir}/cache_capture.pkl'
        with open(fn, 'wb') as f:
            pickle.dump(self.cache_capture, f)

    def load_cache(self):
        cdir = self.load_cache_dir
        
        fn = f'{cdir}/cache_capture_seed.pkl'
        if os.path.exists(fn):
            with open(fn, 'rb') as f:
                self.cache_capture_seed = pickle.load(f)
                print('=> [cache_capture_seed] Loaded')

        fn = f'{cdir}/cache_reproduce_one.pkl'
        if os.path.exists(fn):
            with open(fn, 'rb') as f:
                self.cache_reproduce_one = pickle.load(f)
                print('=> [cache_reproduce_one] Loaded')

        fn = f'{cdir}/cache_capture.pkl'
        if os.path.exists(fn):
            with open(fn, 'rb') as f:
                self.cache_capture = pickle.load(f)
                print('=> [cache_capture] Loaded')

    def print_heading(self):
        if self.heading_printed:
            return
        self.heading_printed = True
            
        text = '''
              _   _   _    _   _ _  
  ___ _ _ ___| |_| |_(_)__| |_| | | 
 / _ \ '_/ _ \  _|  _| / _| / /_  _|
 \___/_| \___/\__|\__|_\__|_\_\ |_| 
------------------------------------
 Oregon Lottery - Pick 4 Predictor
------------------------------------

        '''
        print(text)
        
    def gen_num(self):
        return random.randint(self.min_num, self.max_num)

    def capture_seed(self, sim_cnt, n):
        key = f'{sim_cnt}_{n}'
        if key in self.cache_capture_seed:
            return self.cache_capture_seed[key]
            
        sim_seed = 0
        p = self.reproduce_one(sim_seed, sim_cnt)

        if self.debug_cs_on:
            print(f'=> [CS1] {sim_seed}, {sim_cnt}, {n}, {p}')
            
        while not self.match(n, p):
            sim_seed += 1
            p = self.reproduce_one(sim_seed, sim_cnt)

            if self.debug_cs_on:
                if sim_seed % 100000 == 0:
                    print(f'=> [CS2] {sim_seed}, {sim_cnt}, {n}, {p}')

        if self.debug_cs_on:
            print(f'=> [CS3] {sim_seed}, {sim_cnt}, {n}, {p}')

        self.debug_cs_on = False

        self.cache_capture_seed[key] = sim_seed
        
        return sim_seed

    def capture(self, w, n):
        key = f'{w}_{n}'
        if key in self.cache_capture:
            return self.cache_capture[key][0], self.cache_capture[key][1]
            
        if self.debug_on:
            print(f'=> [C1] {w}, {n}')
            
        sim_seed = self.capture_seed(1, n)

        if self.debug_on:
            print(f'=> [C2] {sim_seed}, {w}, {n}')

        random.seed(sim_seed)
        
        sim_cnt = 0
        p = self.gen_num()
        sim_cnt += 1

        if self.debug_on:
            print(f'=> [C3] {sim_seed}, {sim_cnt}, {w}, {n}')

        while not self.match(w, p):
            p = self.gen_num()
            sim_cnt += 1

            if self.debug_on:
                if sim_cnt % 100000 == 0:
                    print(f'=> [C4] {sim_seed}, {sim_cnt}, {w}, {n}')

        pn = self.reproduce_one(sim_seed, 1)
        pw = self.reproduce_one(sim_seed, sim_cnt)

        if self.debug_on:
            print(f'=> [C5] {sim_seed}, {sim_cnt}, {w}, {n}, {pw}, {pn}')

        self.debug_on = False
        
        if pn == n and pw == w:
            self.cache_capture[key] = [sim_seed, sim_cnt]
            return sim_seed, sim_cnt
        else:
            self.cache_capture[key] = [-1, -1]
            return -1, -1
            
    def reproduce_one(self, sim_seed, sim_cnt):
        key = f'{sim_seed}_{sim_cnt}'
        if key in self.cache_reproduce_one:
            return self.cache_reproduce_one[key]
            
        random.seed(sim_seed)
        n = -1
        for si in range(sim_cnt):
            n = self.gen_num()

        self.cache_reproduce_one[key] = n
        
        return n

    def match(self, w, p, match_kind = 'm4'):
        if match_kind == 'm4':
            if w < 0:
                return False
            elif w == p:
                return True
            else:
                return False
        elif match_kind == 'm3f':
            wa = self.n2a(w)
            pa = self.n2a(p)
            if w < 0:
                return False
            elif wa[0] == pa[0] and wa[1] == pa[1] and wa[2] == pa[2]:
                return True
            else:
                return False
        elif match_kind == 'm3l':
            wa = self.n2a(w)
            pa = self.n2a(p)
            if w < 0:
                return False
            elif wa[1] == pa[1] and wa[2] == pa[2] and wa[3] == pa[3]:
                return True
            else:
                return False
        elif match_kind == 'm3':
            wa = self.n2a(w)
            pa = self.n2a(p)
            if w < 0:
                return False
            elif (wa[0] == pa[0] and wa[1] == pa[1] and wa[2] == pa[2]) or (wa[1] == pa[1] and wa[2] == pa[2] and wa[3] == pa[3]):
                return True
            else:
                return False
        elif match_kind == 'm2':
            wa = self.n2a(w)
            pa = self.n2a(p)
            if w < 0:
                return False
            elif (wa[0] == pa[0] and wa[1] == pa[1]) or (wa[1] == pa[1] and wa[2] == pa[2]) or (wa[2] == pa[2] and wa[3] == pa[3]):
                return True
            else:
                return False
        else:
            return False

    def a2n(self, a):
        if a is None:
            return None
        if len(a) < 4:
            return None
        
        n = 0
        for ni in range(4):
            n += (a[ni]) * self.baseset[ni]
        return n
    
    def n2a(self, n):
        try:
            if n is None:
                return None

            a = []
            for ni in range(4):
                b = self.baseset[ni]
                c = int((n - (n % b)) // b)
                a.append(c)
                n = n - (c * b)

            return a
        except Exception as e:
            return None

    def capture_m4p_p_1(self, pdf):
        xdf = pdf.sort_values(by=['buy_date'], ascending=[False])
        xdf = xdf[(xdf['m4'] == 0)&(xdf['m3f'] == 0)&(xdf['m3l'] == 0)&(xdf['m3'] == 0)&(xdf['m2'] == 1)&(xdf['a_m4'] == 0)&(xdf['a_m3f'] == 0)&(xdf['a_m3l'] == 0)&(xdf['a_m3'] == 0)&(xdf['a_m2'] > 0)&(xdf['m4_cnt'] > 0)&(xdf['m4_cnt'] < 10)]
        if len(xdf) == 0:
            return None
        else:
            for ri in range(len(xdf)):
                if xdf['a_m2'].iloc[ri] * 2 == xdf['m4_cnt'].iloc[ri]:
                    buy_date = xdf['buy_date'].iloc[ri]
                    zdf = pdf[pdf['buy_date'] > buy_date]
                    if len(zdf) <= 60 and len(zdf) > 30:
                        return buy_date
                    else:
                        return None
            return None

    def capture_m4p_p_2(self, pdf):
        xdf = pdf.sort_values(by=['buy_date'], ascending=[False])
        xdf1 = xdf[(xdf['m4'] == 0)&(xdf['m3f'] == 0)&(xdf['m3l'] == 0)&(xdf['m2'] == 0)&(xdf['a_m4'] == 0)&(xdf['a_m3f'] == 1)&(xdf['a_m3l'] == 0)]
        if len(xdf1) == 0:
            return None
        else:
            buy_date = xdf1['date'].iloc[0]
            xdf2 = xdf[xdf['buy_date'] == buy_date]
            if len(xdf2) == 0:
                return None
            else:
                buy_date = xdf2['buy_date'].iloc[0]
                zdf = pdf[pdf['buy_date'] > buy_date]
                if len(zdf) <= 28 and len(zdf) > 14:
                    return buy_date
                else:
                    return None

    def capture_m4p_p_3(self, pdf):
        xdf = pdf.sort_values(by=['buy_date'], ascending=[False])
        xdf1 = xdf[(xdf['m4'] == 0)&(xdf['m3f'] == 0)&(xdf['m3l'] == 0)&(xdf['m3'] == 0)&(xdf['m2'] == 1)&(xdf['a_m2'] > 0)&(xdf['m4_cnt'] > 0)]
        if len(xdf1) == 0:
            return None
        else:
            xdf2 = xdf[xdf['m4'] == 1]
            if len(xdf2) == 0:
                return None
            m4_buy_date = xdf2['buy_date'].iloc[0]
            xdf1 = xdf1.sort_values(by=['buy_date'], ascending=[False])
            for ri in range(len(xdf1)):
                buy_date = xdf1['buy_date'].iloc[ri]
                a = xdf1['a_m2'].iloc[ri]
                b = 2 * (xdf1['m4_cnt'].iloc[ri] / 3)
                if a == b and buy_date < m4_buy_date: 
                    zdf = pdf[pdf['buy_date'] > buy_date]
                    if len(zdf) <= 36 and len(zdf) > 18:
                        return buy_date
                    else:
                        return None
            return None
            
    def join_m4p(self, pdf, adf, l_buy_date, buy_date):
        if buy_date is not None:
            if buy_date not in l_buy_date:
                l_buy_date.append(buy_date)
                df = pdf[pdf['buy_date'] == buy_date]
                df['m4p_no'] = len(l_buy_date)
                if adf is None:
                    adf = df
                else:
                    adf = pd.concat([adf, df])
                adf = adf.sort_values(by=['m4p_no'], ascending=[True])
        return l_buy_date, adf
        
    def capture_m4p(self, pdf, x_sim_seed):
        l_pred = []
        l_buy_date = []
        adf = None

        buy_date = self.capture_m4p_p_1(pdf)
        l_buy_date, adf = self.join_m4p(pdf, adf, l_buy_date, buy_date)

        sz = 0
        if adf is not None:
            sz = len(adf)
        print(f'=> [M4PC-1] {l_buy_date} -> {sz}')

        buy_date = self.capture_m4p_p_2(pdf)
        l_buy_date, adf = self.join_m4p(pdf, adf, l_buy_date, buy_date)

        sz = 0
        if adf is not None:
            sz = len(adf)
        print(f'=> [M4PC-2] {l_buy_date} -> {sz}')

        buy_date = self.capture_m4p_p_3(pdf)
        l_buy_date, adf = self.join_m4p(pdf, adf, l_buy_date, buy_date)

        sz = 0
        if adf is not None:
            sz = len(adf)
        print(f'=> [M4PC-3] {l_buy_date} -> {sz}')
        
        if adf is None or len(l_buy_date) == 0 or len(adf) == 0:
            return l_pred

        adf = adf.sort_values(by=['m4p_no'], ascending=[True])
        for ri in range(len(adf)):
            x_sim_cnt = adf['sim_cnt'].iloc[ri]
            p = self.reproduce_one(x_sim_seed, x_sim_cnt)
            l_pred.append(p)

        return l_pred

    def download_drawing(self, buffer_dir, lotte_kind, v_date):
        self.print_heading()

        text = '''
====================================
        DOWNLOAD DRAWING
  -------------------------------
        '''
        print(text) 

        text = '''
  -------------------------------
           PARAMETERS
  -------------------------------
        '''
        print(text) 

        print(f'[BUFFER_DIR] {buffer_dir}')
        print(f'[LOTTE_KIND] {lotte_kind}')
        print(f'[DATE] {v_date}')

        text = '''
  -------------------------------
        '''
        print(text) 

        tmp_dir = buffer_dir
        work_dir = f'{tmp_dir}/data-' + str(random.randint(1, 1000000))  
        os.system(f'mkdir -p {work_dir}')

        curl_file = f'{work_dir}/curl.txt'
        fds = v_date.split('.')
        END_DATE = str(int(fds[1])) + '/' + str(int(fds[2])) + '/' + str(int(fds[0]))[2:]
        cmd = "curl 'https://api2.oregonlottery.org/drawresults/ByDrawDate?gameSelector=p4&startingDate=01/01/1984&endingDate=" + END_DATE + "&pageSize=50000&includeOpen=False' -H 'Accept: application/json, text/javascript, */*; q=0.01' -H 'Accept-Language: en-US,en;q=0.9' -H 'Cache-Control: no-cache' -H 'Connection: keep-alive' -H 'Ocp-Apim-Subscription-Key: 683ab88d339c4b22b2b276e3c2713809' -H 'Origin: https://www.oregonlottery.org' -H 'Pragma: no-cache' -H 'Referer: https://www.oregonlottery.org/' -H 'Sec-Fetch-Dest: empty' -H 'Sec-Fetch-Mode: cors' -H 'Sec-Fetch-Site: same-site' -H 'User-Agent: Mozilla/5.0 (X11; CrOS x86_64 14541.0.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36' -H 'sec-ch-ua: " + '"' + "Not/A)Brand" + '"' + ";v=" + '"' + "8" + '"' + ", " + '"' + "Chromium" + '"' + ";v=" + '"' + "126" + '"' + ", " + '"' + "Google Chrome" + '"' + ";v=" + '"' + "126" + '"' + "' -H 'sec-ch-ua-mobile: ?0' -H 'sec-ch-ua-platform: " + '"' + "Chrome OS" + '"' + "' > " + curl_file
        print(cmd)
        os.system(cmd)

        T = 'T13:00:00'
        if lotte_kind == 'p4a':
            T = 'T13:00:00'
        if lotte_kind == 'p4b':
            T = 'T16:00:00'
        if lotte_kind == 'p4c':
            T = 'T19:00:00'
        if lotte_kind == 'p4d':
            T = 'T22:00:00'
            
        FIND = v_date.replace('.', '-') + T
        data = []
        if os.path.exists(curl_file):
            with open(curl_file, 'r') as f:
                data = json.load(f)
        line = ''
        for di in range(len(data)):
            et = data[di]
            if et['DrawDateTime'] == FIND:
                line = 'yes'
                break
        
        if line == '':
            os.system(f'rm -rf {work_dir}')
            print(f'== [Error] ==> Drawing is not found!')
            return None

        rows = []
        for di in range(len(data)):
            et = data[di]
            if T not in et['DrawDateTime']:
                continue
            date = et['DrawDateTime'].split('T')[0].replace('-', '.')  
            sl = et['WinningNumbers']
            n = (sl[0] * 1000) + (sl[1] * 100) + (sl[2] * 10) + (sl[3] * 1)
            rw = {'date': date, 'w': -1, 'n': n}
            rows.append(rw)
        ddf = pd.DataFrame(rows)    

        rows = []
        date_list = ddf['date'].unique()
        for today in date_list:
            d1 = datetime.strptime(today, "%Y.%m.%d")
            d2 = d1 + timedelta(minutes=int(+(1) * (60 * 24)))
            buy_date = d2.strftime('%Y.%m.%d')   
            tdf = ddf[ddf['date'] == today]
            bdf = ddf[ddf['date'] == buy_date]
            next_date = buy_date
            n = tdf['n'].iloc[0]
            if len(bdf) == 0:
                w = -1
            else:
                w = bdf['n'].iloc[0]
            rw = {'date': today, 'buy_date': buy_date, 'next_date': next_date, 'w': w, 'n': n}
            rows.append(rw)
        df = pd.DataFrame(rows)
        df = df.sort_values(by=['date'], ascending=[False])
        os.system(f'rm -rf {work_dir}')

        sz = len(df)
        print(f'== [Success] ==> Drawing data is downloaded. It contains {sz} rows.')
        
        text = '''
  -------------------------------
        DOWNLOAD DRAWING
====================================
        '''
        print(text) 

        return df

    def is_observe_good(self, odf, o_cnt, o_ma_field = 'm4'):
        if len(odf) == o_cnt:
            df = odf[odf[o_ma_field] > 0]
            if len(df) > 0:
                return True
            return False
        else:
            return False   

    def is_pick_good(self, odf, p_cnt):
        if len(odf) == p_cnt:
            return True
        else:
            return False

    def copy_file(self, src_fn, tag_fn):
        os.system(f'cp -f "{src_fn}" "{tag_fn}"')
        
    def m4p_collect_observe_glob(self):
        lotte_kind = self.lotte_kind
        return f'{lotte_kind}-observe-*.*.*.csv'

    def m4p_collect_observe_file(self, obs_fn):
        lotte_kind = self.lotte_kind
        fn2 = obs_fn.split('/')[-1]
        fn3 = fn2.replace(f'{lotte_kind}-observe-', '').replace(f'.csv', '')
        return f'{lotte_kind}-observe-{fn3}.csv'

    def m4p_collect_pick_file(self, odf, ri):
        lotte_kind = self.lotte_kind
        buy_date = odf['buy_date'].iloc[ri]
        return f'{lotte_kind}-pick-{buy_date}.csv'

    def m4p_collect_pred_file(self, odf, ri):
        lotte_kind = self.lotte_kind
        buy_date = odf['buy_date'].iloc[ri]
        return f'{lotte_kind}-pred-{buy_date}.json'

    def m4p_collect(self, lotte_kind, o_cnt, p_cnt, data_dirs, save_dir):
        self.print_heading()

        text = '''
====================================
           M4P COLLECT
  -------------------------------
        '''
        print(text) 

        text = '''
  -------------------------------
           PARAMETERS
  -------------------------------
        '''
        print(text) 

        print(f'LOTTE_KIND: {lotte_kind}')
        print(f'O_CNT: {o_cnt}')
        print(f'P_CNT: {p_cnt}')
        print(f'DATA_DIRS: {data_dirs}')
        print(f'SAVE_DIR: {save_dir}')

        text = '''
  -------------------------------
        '''
        print(text) 
        
        self.lotte_kind = lotte_kind
        ma_field = 'm4'
        fn_observe_glob = self.m4p_collect_observe_glob
        fn_observe_file = self.m4p_collect_observe_file
        fn_pick_file = self.m4p_collect_pick_file
        fn_pred_file = self.m4p_collect_pred_file
        
        for data_dir in data_dirs:
            obs_glob = fn_observe_glob()
            lg_obs = glob.glob(f'{data_dir}/{obs_glob}')
            for fn_obs in lg_obs:
                odf = pd.read_csv(fn_obs)
                if not self.is_observe_good(odf, o_cnt, ma_field):
                    continue
                observe_fn = fn_observe_file(fn_obs)
                self.copy_file(fn_obs, f'{save_dir}/{observe_fn}')
                print(f'== [Copy] ==> {observe_fn}')
                for ri in range(len(odf)):
                    if odf[ma_field].iloc[ri] > 0:
                        pick_fn = fn_pick_file(odf, ri)
                        pred_fn = fn_pred_file(odf, ri)
                        pdf = pd.read_csv(f'{data_dir}/{pick_fn}')
                        if not self.is_pick_good(pdf, p_cnt):
                            continue
                        self.copy_file(f'{data_dir}/{pick_fn}', f'{save_dir}/{pick_fn}')
                        print(f'== [Copy] ==> {pick_fn}')
                        self.copy_file(f'{data_dir}/{pred_fn}', f'{save_dir}/{pred_fn}')
                        print(f'== [Copy] ==> {pred_fn}')
        
        text = '''
  -------------------------------
           M4P COLLECT
====================================
        '''
        print(text) 

    def simulate(self, v_buy_date, buffer_dir = '/kaggle/buffers/orottick4', lotte_kind = 'p4a', data_df = None, v_date_cnt = 56, tck_cnt = 2, runtime = None):
        self.print_heading()

        pso = self.prd_sort_order
        if pso == 'C':
            tck_cnt = -1
            pso = 'B'
            
        text = '''
====================================
       ANALYZE SIMULATION
  -------------------------------
        '''
        print(text) 

        text = '''
  -------------------------------
           PARAMETERS
  -------------------------------
        '''
        print(text) 

        v_data_df_is_none = False
        if data_df is None:
            v_data_df_is_none = True
            
        print(f'[BUFFER_DIR] {buffer_dir}')
        print(f'[LOTTE_KIND] {lotte_kind}')
        print(f'[DATA_DF_IS_NONE] {v_data_df_is_none}')
        print(f'[BUY_DATE] {v_buy_date}')
        print(f'[DATE_CNT] {v_date_cnt}')
        print(f'[TCK_CNT] {tck_cnt}')
        print(f'[RUNTIME] {runtime}')

        text = '''
  -------------------------------
        '''
        print(text) 

        if data_df is None:
            d1 = datetime.strptime(v_buy_date, "%Y.%m.%d")
            g = -1
            d2 = d1 + timedelta(minutes=int(+(g*(60 * 24))))
            v_date = d2.strftime('%Y.%m.%d')
    
            data_df = self.download_drawing(buffer_dir, lotte_kind, v_date)
            if data_df is None:
                return None, None, None
                
        rwcnt = v_date_cnt * 2 + 1
        start_time = time.time()
        ddf = data_df[data_df['buy_date'] <= v_buy_date]
        ddf = ddf.sort_values(by=['date'], ascending=[False])
        if v_date_cnt > 0:
            if len(ddf) > rwcnt:
                ddf = ddf[:rwcnt]
        ddf = ddf.sort_values(by=['date'], ascending=[True])
        rows = []
        sz = len(ddf)
        db_cnt = int(round(sz / 100.0))
        if db_cnt <= 0:
            db_cnt = 1
        m4_cnt = 0
        m3f_cnt = 0
        m3l_cnt = 0
        m3_cnt = 0
        m2_cnt = 0
        for ri in range(len(ddf)):
            if runtime is not None:
                if time.time() - start_time > runtime:
                    break
            if self.has_step_log:
                if ri % db_cnt == 0:
                    print(f'== [S1] ==> {ri} / {sz}')
                
            t_date = ddf['date'].iloc[ri]
            t_buy_date = ddf['buy_date'].iloc[ri]
            t_next_date = ddf['next_date'].iloc[ri]
            t_n = ddf['n'].iloc[ri]
            t_w = ddf['w'].iloc[ri]
            sim_seed = -1
            sim_cnt = -1
            t_p = -1
            m4 = 0
            m3f = 0
            m3l = 0
            m3 = 0
            m2 = 0
            a_m4 = 0
            a_m3f = 0
            a_m3l = 0
            a_m3 = 0
            a_m2 = 0
            p_buy_date_m4 = ''
            p_sim_seed_m4 = ''
            p_win_num_m4 = ''
            p_prd_num_m4 = ''
            p_buy_date_m3f = ''
            p_sim_seed_m3f = ''
            p_win_num_m3f = ''
            p_prd_num_m3f = ''
            p_buy_date_m3l = ''
            p_sim_seed_m3l = ''
            p_win_num_m3l = ''
            p_prd_num_m3l = ''
            p_buy_date_m3 = ''
            p_sim_seed_m3 = ''
            p_win_num_m3 = ''
            p_prd_num_m3 = ''
            p_buy_date_m2 = ''
            p_sim_seed_m2 = ''
            p_win_num_m2 = ''
            p_prd_num_m2 = ''
            if t_w >= 0 and t_buy_date != v_buy_date:
                sim_seed, sim_cnt = self.capture(t_w, t_n)
                t_p = self.reproduce_one(sim_seed, sim_cnt)

                pdf = pd.DataFrame(rows)
                if len(pdf) >= v_date_cnt:
                    pdf = pdf.sort_values(by=['date'], ascending=[False])
                    pdf = pdf[:v_date_cnt]
                    for pi in range(len(pdf)):
                        p_sim_cnt = pdf['sim_cnt'].iloc[pi]
                        p_date = pdf['date'].iloc[pi]
                        p_q = self.reproduce_one(sim_seed, p_sim_cnt)
                        i_m4 = pdf['m4'].iloc[pi]
                        i_m3f = pdf['m3f'].iloc[pi]
                        i_m3l = pdf['m3l'].iloc[pi]
                        i_m3 = pdf['m3'].iloc[pi]
                        i_m2 = pdf['m2'].iloc[pi]
                        pi_buy_date_m4 = pdf['p_buy_date_m4'].iloc[pi]
                        pi_sim_seed_m4 = pdf['p_sim_seed_m4'].iloc[pi]
                        pi_win_num_m4 = pdf['p_win_num_m4'].iloc[pi]
                        pi_prd_num_m4 = pdf['p_prd_num_m4'].iloc[pi]
                        pi_buy_date_m3f = pdf['p_buy_date_m3f'].iloc[pi]
                        pi_sim_seed_m3f = pdf['p_sim_seed_m3f'].iloc[pi]
                        pi_win_num_m3f = pdf['p_win_num_m3f'].iloc[pi]
                        pi_prd_num_m3f = pdf['p_prd_num_m3f'].iloc[pi]
                        pi_buy_date_m3l = pdf['p_buy_date_m3l'].iloc[pi]
                        pi_sim_seed_m3l = pdf['p_sim_seed_m3l'].iloc[pi]
                        pi_win_num_m3l = pdf['p_win_num_m3l'].iloc[pi]
                        pi_prd_num_m3l = pdf['p_prd_num_m3l'].iloc[pi]
                        pi_buy_date_m3 = pdf['p_buy_date_m3'].iloc[pi]
                        pi_sim_seed_m3 = pdf['p_sim_seed_m3'].iloc[pi]
                        pi_win_num_m3 = pdf['p_win_num_m3'].iloc[pi]
                        pi_prd_num_m3 = pdf['p_prd_num_m3'].iloc[pi]
                        pi_buy_date_m2 = pdf['p_buy_date_m2'].iloc[pi]
                        pi_sim_seed_m2 = pdf['p_sim_seed_m2'].iloc[pi]
                        pi_win_num_m2 = pdf['p_win_num_m2'].iloc[pi]
                        pi_prd_num_m2 = pdf['p_prd_num_m2'].iloc[pi]

                        if self.match(t_w, p_q, 'm4'):
                            a_m4 += 1
                            i_m4 += 1
                            if pi_buy_date_m4 != '':
                                pi_buy_date_m4 += ', ' 
                                pi_sim_seed_m4 += ', '
                                pi_win_num_m4 += ', '
                                pi_prd_num_m4 += ', '
                            pi_buy_date_m4 += str(t_buy_date)
                            pi_sim_seed_m4 += str(sim_seed)
                            pi_win_num_m4 += str(t_w)
                            pi_prd_num_m4 += str(p_q)
                                
                        if self.match(t_w, p_q, 'm3f'):
                            a_m3f += 1
                            i_m3f += 1
                            if pi_buy_date_m3f != '':
                                pi_buy_date_m3f += ', ' 
                                pi_sim_seed_m3f += ', '
                                pi_win_num_m3f += ', '
                                pi_prd_num_m3f += ', '
                            pi_buy_date_m3f += str(t_buy_date)
                            pi_sim_seed_m3f += str(sim_seed)
                            pi_win_num_m3f += str(t_w)
                            pi_prd_num_m3f += str(p_q)
                        if self.match(t_w, p_q, 'm3l'):
                            a_m3l += 1
                            i_m3l += 1
                            if pi_buy_date_m3l != '':
                                pi_buy_date_m3l += ', ' 
                                pi_sim_seed_m3l += ', '
                                pi_win_num_m3l += ', '
                                pi_prd_num_m3l += ', '
                            pi_buy_date_m3l += str(t_buy_date)
                            pi_sim_seed_m3l += str(sim_seed)
                            pi_win_num_m3l += str(t_w)
                            pi_prd_num_m3l += str(p_q)
                        if self.match(t_w, p_q, 'm3'):
                            a_m3 += 1
                            i_m3 += 1
                            if pi_buy_date_m3 != '':
                                pi_buy_date_m3 += ', ' 
                                pi_sim_seed_m3 += ', '
                                pi_win_num_m3 += ', '
                                pi_prd_num_m3 += ', '
                            pi_buy_date_m3 += str(t_buy_date)
                            pi_sim_seed_m3 += str(sim_seed)
                            pi_win_num_m3 += str(t_w)
                            pi_prd_num_m3 += str(p_q)
                        if self.match(t_w, p_q, 'm2'):
                            a_m2 += 1
                            i_m2 += 1
                            if pi_buy_date_m2 != '':
                                pi_buy_date_m2 += ', ' 
                                pi_sim_seed_m2 += ', '
                                pi_win_num_m2 += ', '
                                pi_prd_num_m2 += ', '
                            pi_buy_date_m2 += str(t_buy_date)
                            pi_sim_seed_m2 += str(sim_seed)
                            pi_win_num_m2 += str(t_w)
                            pi_prd_num_m2 += str(p_q)
                        for xi in range(len(rows)):
                            if rows[xi]['date'] == p_date:
                                rows[xi]['m4'] = i_m4
                                rows[xi]['m3f'] = i_m3f
                                rows[xi]['m3l'] = i_m3l
                                rows[xi]['m3'] = i_m3
                                rows[xi]['m2'] = i_m2

                                rows[xi]['p_buy_date_m4'] = pi_buy_date_m4
                                rows[xi]['p_sim_seed_m4'] = pi_sim_seed_m4
                                rows[xi]['p_win_num_m4'] = pi_win_num_m4
                                rows[xi]['p_prd_num_m4'] = pi_prd_num_m4

                                rows[xi]['p_buy_date_m3f'] = pi_buy_date_m3f
                                rows[xi]['p_sim_seed_m3f'] = pi_sim_seed_m3f
                                rows[xi]['p_win_num_m3f'] = pi_win_num_m3f
                                rows[xi]['p_prd_num_m3f'] = pi_prd_num_m3f

                                rows[xi]['p_buy_date_m3l'] = pi_buy_date_m3l
                                rows[xi]['p_sim_seed_m3l'] = pi_sim_seed_m3l
                                rows[xi]['p_win_num_m3l'] = pi_win_num_m3l
                                rows[xi]['p_prd_num_m3l'] = pi_prd_num_m3l

                                rows[xi]['p_buy_date_m3'] = pi_buy_date_m3
                                rows[xi]['p_sim_seed_m3'] = pi_sim_seed_m3
                                rows[xi]['p_win_num_m3'] = pi_win_num_m3
                                rows[xi]['p_prd_num_m3'] = pi_prd_num_m3

                                rows[xi]['p_buy_date_m2'] = pi_buy_date_m2
                                rows[xi]['p_sim_seed_m2'] = pi_sim_seed_m2
                                rows[xi]['p_win_num_m2'] = pi_win_num_m2
                                rows[xi]['p_prd_num_m2'] = pi_prd_num_m2

                    m4_cnt += a_m4
                    m3f_cnt += a_m3f
                    m3l_cnt += a_m3l
                    m3_cnt += a_m3
                    m2_cnt += a_m2
                    
            else:
                sim_seed = self.capture_seed(1, t_n)
            rw = {'date': t_date, 'buy_date': t_buy_date, 'next_date': t_next_date, 'w': t_w, 'n': t_n, 'p': t_p, 'sim_seed': sim_seed, 'sim_cnt': sim_cnt, 'm4': m4, 'm3f': m3f, 'm3l': m3l, 'm3': m3, 'm2': m2, 'a_m4': a_m4, 'a_m3f': a_m3f, 'a_m3l': a_m3l, 'a_m3': a_m3, 'a_m2': a_m2, 'm4_cnt': m4_cnt, 'm3f_cnt': m3f_cnt, 'm3l_cnt': m3l_cnt, 'm3_cnt': m3_cnt, 'm2_cnt': m2_cnt, 'p_buy_date_m4': p_buy_date_m4, 'p_sim_seed_m4': p_sim_seed_m4, 'p_win_num_m4': p_win_num_m4, 'p_prd_num_m4': p_prd_num_m4      , 'p_buy_date_m3f': p_buy_date_m3f, 'p_sim_seed_m3f': p_sim_seed_m3f, 'p_win_num_m3f': p_win_num_m3f, 'p_prd_num_m3f': p_prd_num_m3f        , 'p_buy_date_m3l': p_buy_date_m3l, 'p_sim_seed_m3l': p_sim_seed_m3l, 'p_win_num_m3l': p_win_num_m3l, 'p_prd_num_m3l': p_prd_num_m3l        , 'p_buy_date_m3': p_buy_date_m3, 'p_sim_seed_m3': p_sim_seed_m3, 'p_win_num_m3': p_win_num_m3, 'p_prd_num_m3': p_prd_num_m3, 'p_buy_date_m2': p_buy_date_m2, 'p_sim_seed_m2': p_sim_seed_m2, 'p_win_num_m2': p_win_num_m2, 'p_prd_num_m2': p_prd_num_m2}
            rows.append(rw)
        zdf = pd.DataFrame(rows)
        xdf = zdf[zdf['buy_date'] == v_buy_date]
        pdf = zdf[zdf['buy_date'] < v_buy_date]
        kdf = pdf.sort_values(by=['buy_date'], ascending=[False])
        json_pred = None
        m4_rsi = -1
        m4_pred = ''
        m4pc = 0
        if len(xdf) == 1 and len(pdf) >= v_date_cnt:
            s_sim_cnt = ''
            s_pred = ''
            if pso == 'A':
                pdf = pdf[(pdf['m4'] > 0)|(pdf['m3f'] > 0)|(pdf['m3l'] > 0)|(pdf['m3'] > 0)|(pdf['m2'] > 0)]
            mb_m4 = 0
            mb_m3f = 0
            mb_m3l = 0
            mb_m3 = 0
            mb_m2 = 0
            if len(pdf) > 0:
                if pso == 'A':
                    pdf = pdf.sort_values(by=['m4', 'm3f', 'm3l', 'm3', 'm2', 'date'], ascending=[False, False, False, False, False, False])
                if pso == 'B':
                    pdf = pdf.sort_values(by=['date'], ascending=[False])

                if tck_cnt > 0 and len(pdf) > tck_cnt:
                    pdf = pdf[:tck_cnt]
                l_sim_cnt = list(pdf['sim_cnt'].values)
                ls_sim_cnt = [str(x) for x in l_sim_cnt]
                s_sim_cnt = ', '.join(ls_sim_cnt)
                l_pred = []
                x_sim_seed = xdf['sim_seed'].iloc[0]
                for x_sim_cnt in l_sim_cnt:
                    x = self.reproduce_one(x_sim_seed, x_sim_cnt)
                    l_pred.append(x)
                zrsi = 0
                zrsiw = int(xdf['w'].iloc[0])
                for zp in l_pred:
                    if zp == zrsiw:
                        break
                    zrsi += 1
                if zrsi < len(l_pred):
                    m4_rsi = zrsi
                ls_pred = [str(x) for x in l_pred]
                s_pred = ', '.join(ls_pred)
                for pi in range(len(pdf)):
                    sv_sim_seed = ', ' + str(x_sim_seed) + ','
                    
                    txt = ', ' + pdf['p_sim_seed_m4'].iloc[pi] + ','
                    if sv_sim_seed in txt:
                        mb_m4 = 1
                    
                    txt = ', ' + pdf['p_sim_seed_m3f'].iloc[pi] + ','
                    if sv_sim_seed in txt:
                        mb_m3f = 1
                    
                    txt = ', ' + pdf['p_sim_seed_m3l'].iloc[pi] + ','
                    if sv_sim_seed in txt:
                        mb_m3l = 1
                    
                    txt = ', ' + pdf['p_sim_seed_m3'].iloc[pi] + ','
                    if sv_sim_seed in txt:
                        mb_m3 = 1
                    
                    txt = ', ' + pdf['p_sim_seed_m2'].iloc[pi] + ','
                    if sv_sim_seed in txt:
                        mb_m2 = 1
            else:
                pdf = None
            m4p_use = False
            if pso == 'B' and pdf is not None:
                if len(pdf) > 0:
                    if len(xdf) > 0:
                        lx_pred = self.capture_m4p(pdf, xdf['sim_seed'].iloc[0])
                        if len(lx_pred) > 0:
                            m4pc = 1
                            m4p_cnt = self.m4p_cnt
                            if m4p_cnt > 0:
                                if len(lx_pred) > m4p_cnt:
                                    lx_pred = lx_pred[:m4p_cnt]
                            lx_pred = [str(x) for x in lx_pred]
                            m4_pred = ', '.join(lx_pred)
                            m4p_use = True
                            
            if not m4p_use and len(kdf) > 0 and len(xdf) > 0:
                kdf = kdf[kdf['m4'] > 0]
                if len(kdf) > 0:
                    kdf = kdf.sort_values(by=['buy_date'], ascending=[False])
                    m4p_cnt = self.m4p_cnt
                    if m4p_cnt > 0:
                        if len(kdf) > m4p_cnt:
                            kdf = kdf[:m4p_cnt]
                    lx_pred = []
                    x_sim_seed = xdf['sim_seed'].iloc[0]
                    for xi in range(len(kdf)):
                        x = self.reproduce_one(x_sim_seed, kdf['sim_cnt'].iloc[xi])
                        lx_pred.append(str(x))
                    m4_pred = ', '.join(lx_pred)
            json_pred = {'date': xdf['date'].iloc[0], 'buy_date': xdf['buy_date'].iloc[0], 'next_date': xdf['next_date'].iloc[0], 'w': int(xdf['w'].iloc[0]), 'n': int(xdf['n'].iloc[0]), 'sim_seed': int(xdf['sim_seed'].iloc[0]), 'date_cnt': v_date_cnt, 'tck_cnt': tck_cnt, 'sim_cnt': s_sim_cnt, 'pred': s_pred, 'm4_rsi': m4_rsi, 'm4_pred': m4_pred, 'm4pc': m4pc, 'pcnt': 1, 'm4': int(xdf['a_m4'].iloc[0]), 'm3f': int(xdf['a_m3f'].iloc[0]), 'm3l': int(xdf['a_m3l'].iloc[0]), 'm3': int(xdf['a_m3'].iloc[0]), 'm2': int(xdf['a_m2'].iloc[0]), 'm4_cnt': int(xdf['m4_cnt'].iloc[0]), 'm3f_cnt': int(xdf['m3f_cnt'].iloc[0]), 'm3l_cnt': int(xdf['m3l_cnt'].iloc[0]), 'm3_cnt': int(xdf['m3_cnt'].iloc[0]), 'm2_cnt': int(xdf['m2_cnt'].iloc[0]), 'mb_m4': mb_m4, 'mb_m3f': mb_m3f, 'mb_m3l': mb_m3l, 'mb_m3': mb_m3, 'mb_m2': mb_m2}
        else:
            pdf = None
            
        if json_pred is not None:
            text = '''
  -------------------------------
           PREDICTION
  -------------------------------
        '''
            print(text)
            print(str(json_pred))

        try:
            self.save_cache()
        except Exception as e:
            msg = str(e)
            print(f'=> [E] {msg}')

        text = '''
  -------------------------------
       ANALYZE SIMULATION
====================================
        '''
        print(text)
        
        return zdf, json_pred, pdf

    def observe(self, lotte_kind, v_buy_date, o_max_tck = 2, o_date_cnt = 56, o_runtime = 60 * 60 * 11.5, date_cnt = 56, buffer_dir = '/kaggle/buffers/orottick4', data_df = None):
        self.print_heading()

        start_time = time.time()

        more = {}
        
        text = '''
====================================
             OBSERVE
  -------------------------------
        '''
        print(text)

        text = '''
  -------------------------------
           PARAMETERS
  -------------------------------
        '''
        print(text) 

        v_data_df_is_none = False
        if data_df is None:
            v_data_df_is_none = True
            
        print(f'[BUFFER_DIR] {buffer_dir}')
        print(f'[LOTTE_KIND] {lotte_kind}')
        print(f'[DATA_DF_IS_NONE] {v_data_df_is_none}')
        print(f'[BUY_DATE] {v_buy_date}')
        print(f'[DATE_CNT] {date_cnt}')
        print(f'[O_DATE_CNT] {o_date_cnt}')
        print(f'[TCK_CNT] {o_max_tck}')
        print(f'[RUNTIME] {o_runtime}')

        text = '''
  -------------------------------
        '''
        print(text) 

        d1 = datetime.strptime(v_buy_date, "%Y.%m.%d")
        g = -1
        d2 = d1 + timedelta(minutes=int(+(g*(60 * 24))))
        v_date = d2.strftime('%Y.%m.%d')

        if data_df is None:
            data_df = self.download_drawing(buffer_dir, lotte_kind, v_date)
            
        if data_df is None:
            return None, more

        ddf = data_df[data_df['buy_date'] < v_buy_date]
        ddf = ddf.sort_values(by=['buy_date'], ascending=[False])
        ddf = ddf[:o_date_cnt]
        ddf = ddf.sort_values(by=['buy_date'], ascending=[True])

        tck_cnt = 0
        pcnt = 0
        m4_cnt = 0
        m3f_cnt = 0
        m3l_cnt = 0
        m3_cnt = 0
        m2_cnt = 0
        rows = []
        for ri in range(len(ddf)):
            if o_runtime is not None:
                if time.time() - start_time > o_runtime:
                    break
                
            t_date = ddf['date'].iloc[ri]
            t_buy_date = ddf['buy_date'].iloc[ri]
            t_next_date = ddf['next_date'].iloc[ri]
            t_w = ddf['w'].iloc[ri]
            t_n = ddf['n'].iloc[ri]

            text = '''
  -------------------------------
  [O] __DATE__ : __W__
  -------------------------------
        '''
            print(text.replace('__DATE__', str(t_buy_date)).replace('__W__', str(t_w))) 

            runtime = None
            if o_runtime is not None:
                o_overtime = time.time() - start_time
                runtime = o_runtime - o_overtime
            zdf, json_prd, pdf = self.simulate(t_buy_date, buffer_dir, lotte_kind, data_df, date_cnt, o_max_tck, runtime)
            more[f'pred_{t_buy_date}'] = json_prd
            more[f'sim_{t_buy_date}'] = zdf
            more[f'pick_{t_buy_date}'] = pdf
            
            t_pred = json_prd['pred']
            vry = True
            if self.m4p_obs:
                t_pred = json_prd['m4_pred']
                if self.m4p_vry:
                    if len(t_pred.split(', ')) != 1:
                        vry = False
            t_prd_lst = t_pred.split(', ')
            if o_max_tck > 0:
                if len(t_prd_lst) > o_max_tck:
                    t_prd_lst = t_prd_lst[:o_max_tck]
            nlst = [int(x) for x in t_prd_lst]
            pcnt += 1
            prd_num = len(nlst)
            tck_cnt += prd_num
            m4 = 0
            m3f = 0
            m3l = 0
            m3 = 0
            m2 = 0
            for t_p in nlst:
                if vry and self.match(t_w, t_p, 'm4'):
                    m4 += 1
                if vry and self.match(t_w, t_p, 'm3f'):
                    m3f += 1
                if vry and self.match(t_w, t_p, 'm3l'):
                    m3l += 1
                if vry and self.match(t_w, t_p, 'm3'):
                    m3 += 1
                if vry and self.match(t_w, t_p, 'm2'):
                    m2 += 1
            m4_cnt += m4
            m3f_cnt += m3f
            m3l_cnt += m3l
            m3_cnt += m3
            m2_cnt += m2

            rw = json_prd
            rw['m4'] = m4
            rw['m3f'] = m3f
            rw['m3l'] = m3l
            rw['m3'] = m3
            rw['m2'] = m2
            rw['m4_cnt'] = m4_cnt
            rw['m3f_cnt'] = m3f_cnt
            rw['m3l_cnt'] = m3l_cnt
            rw['m3_cnt'] = m3_cnt
            rw['m2_cnt'] = m2_cnt
            rw['pcnt'] = pcnt
            
            rows.append(rw)

            print(str(rw))

        odf = pd.DataFrame(rows)
        odf = odf.sort_values(by=['buy_date'], ascending=[False])
        
        text = '''
  -------------------------------
             OBSERVE
====================================
        '''
        print(text)

        return odf, more

    def build_cache(self, v_buy_date, cache_cnt = -1, buffer_dir = '/kaggle/buffers/orottick4', lotte_kind = 'p4a', data_df = None, runtime = None):
        self.print_heading()
            
        text = '''
====================================
            BUILD CACHE
  -------------------------------
        '''
        print(text) 

        text = '''
  -------------------------------
           PARAMETERS
  -------------------------------
        '''
        print(text) 

        v_data_df_is_none = False
        if data_df is None:
            v_data_df_is_none = True
            
        print(f'[BUFFER_DIR] {buffer_dir}')
        print(f'[LOTTE_KIND] {lotte_kind}')
        print(f'[DATA_DF_IS_NONE] {v_data_df_is_none}')
        print(f'[BUY_DATE] {v_buy_date}')
        print(f'[CACHE_CNT] {cache_cnt}')
        print(f'[RUNTIME] {runtime}')

        text = '''
  -------------------------------
        '''
        print(text) 
                
        if data_df is None:
            d1 = datetime.strptime(v_buy_date, "%Y.%m.%d")
            g = -1
            d2 = d1 + timedelta(minutes=int(+(g*(60 * 24))))
            v_date = d2.strftime('%Y.%m.%d')
    
            data_df = self.download_drawing(buffer_dir, lotte_kind, v_date)
            if data_df is None:
                return None

        start_time = time.time()
        ddf = data_df[data_df['buy_date'] <= v_buy_date]
        ddf = ddf.sort_values(by=['date'], ascending=[False])
        if cache_cnt > 0:
            if len(ddf) > cache_cnt:
                ddf = ddf[:cache_cnt]
        ddf = ddf.sort_values(by=['buy_date'], ascending=[False])

        rows = []
        keycheck = {}
        sz = len(ddf)
        for ri in range(len(ddf)):
            if runtime is not None:
                if time.time() - start_time > runtime:
                    break
                    
            w = ddf['w'].iloc[ri]
            n = ddf['n'].iloc[ri]
            if w < 0:
                continue
                
            date = ddf['date'].iloc[ri]

            sim_seed, sim_cnt = self.capture(w, n)
            p = self.reproduce_one(sim_seed, sim_cnt)

            if ri % 50 == 0:
                print(f'=> [BC1] {date} : {ri} / {sz} -> {w}, {n} -> {sim_seed}, {sim_cnt} -> {p}')

            rw = {'date': date, 'w': w, 'n': n, 'sim_seed': sim_seed, 'sim_cnt': sim_cnt}
            rows.append(rw)
            
            if ri % 1000 == 0:
                try:
                    self.save_cache()
                except Exception as e:
                    msg = str(e)
                    print(f'=> [E] {msg}')
                    
        cdf = pd.DataFrame(rows)
        cdf = cdf.sort_values(by=['date'], ascending=[True])
        sz = len(cdf) * len(cdf)
        li = 0
        for ria in range(len(cdf)):
            if runtime is not None:
                if time.time() - start_time > runtime:
                    break
                    
            sim_seed = cdf['sim_seed'].iloc[ria]
            date_1 = cdf['date'].iloc[ria]
            w = cdf['w'].iloc[ria]
            n = cdf['n'].iloc[ria]
            for rib in range(len(cdf)):
                if runtime is not None:
                    if time.time() - start_time > runtime:
                        break
                        
                if rib >= ria:
                    break
                    
                date_2 = cdf['date'].iloc[rib]
                sim_cnt = cdf['sim_cnt'].iloc[rib]
                p = self.reproduce_one(sim_seed, sim_cnt)
                li += 1
                if li % 1000 == 0:
                    print(f'=> [BC2] {date_1}, {date_2} : {li} / {sz} -> {w}, {n} -> {sim_seed}, {sim_cnt} -> {p}')
                    
        try:
            self.save_cache()
        except Exception as e:
            msg = str(e)
            print(f'=> [E] {msg}')
        
        rows = []
        for key in self.cache_capture.keys():
            fds = key.split('_')
            w = int(fds[0])
            n = int(fds[1])
            fds = self.cache_capture[key]
            sim_seed = fds[0]
            sim_cnt = fds[1]
            df = ddf[(ddf['w'] == w)&(ddf['n'] == n)]
            if len(df) == 0:
                continue
            date = df['date'].iloc[0]
            buy_date = df['buy_date'].iloc[0]
            next_date = df['next_date'].iloc[0]
            rw = {'date': date, 'buy_date': buy_date, 'next_date': next_date, 'w': w, 'n': n, 'sim_seed': sim_seed, 'sim_cnt': sim_cnt}
            rows.append(rw)

        if len(rows) == 0:
            return None
            
        cdf = pd.DataFrame(rows)
        cdf = cdf.sort_values(by=['buy_date'], ascending=[False])

        text = '''
  -------------------------------
            BUILD CACHE
====================================
        '''
        print(text)

        return cdf

    def get_option(options, key, def_val):
        if key in options:
            return options[key]
        return def_val

    def run(options, github_pkg, non_github_create_fn = None):
        BUY_DATE = Orottick4Simulator.get_option(options, 'BUY_DATE', '2025.03.27')
        BUFFER_DIR = Orottick4Simulator.get_option(options, 'BUFFER_DIR', '/kaggle/buffers/orottick4')
        LOTTE_KIND = Orottick4Simulator.get_option(options, 'LOTTE_KIND', 'p4a')
        DATA_DF = Orottick4Simulator.get_option(options, 'DATA_DF', None)
        DATE_CNT = Orottick4Simulator.get_option(options, 'DATE_CNT', 56 * 5)
        O_DATE_CNT = Orottick4Simulator.get_option(options, 'O_DATE_CNT', 7)
        TCK_CNT = Orottick4Simulator.get_option(options, 'TCK_CNT', 56 * 5)
        F_TCK_CNT = Orottick4Simulator.get_option(options, 'F_TCK_CNT', 250)
        RUNTIME = Orottick4Simulator.get_option(options, 'RUNTIME', 60 * 60 * 11.5)
        PRD_SORT_ORDER = Orottick4Simulator.get_option(options, 'PRD_SORT_ORDER', 'B')
        HAS_STEP_LOG = Orottick4Simulator.get_option(options, 'HAS_STEP_LOG', True)
        RANGE_CNT = Orottick4Simulator.get_option(options, 'RANGE_CNT', 52)
        M4P_OBS = Orottick4Simulator.get_option(options, 'M4P_OBS', False)
        M4P_CNT = Orottick4Simulator.get_option(options, 'M4P_CNT', 3)
        M4P_VRY = Orottick4Simulator.get_option(options, 'M4P_VRY', False)
        M4P_ONE = Orottick4Simulator.get_option(options, 'M4P_ONE', True)
        RESULT_DIR = Orottick4Simulator.get_option(options, 'RESULT_DIR', '/kaggle/working')
        LOAD_CACHE_DIR = Orottick4Simulator.get_option(options, 'LOAD_CACHE_DIR', '/kaggle/working')
        SAVE_CACHE_DIR = Orottick4Simulator.get_option(options, 'SAVE_CACHE_DIR', '/kaggle/working')
        CACHE_CNT = Orottick4Simulator.get_option(options, 'CACHE_CNT', -1)
        USE_GITHUB = Orottick4Simulator.get_option(options, 'USE_GITHUB', False)
        METHOD = Orottick4Simulator.get_option(options, 'METHOD', 'simulate')
        M4P_COLLECT_DATA_DIRS = Orottick4RLSimulator.get_option(options, 'M4P_COLLECT_DATA_DIRS', [])
        M4P_COLLECT_SAVE_DIR = Orottick4RLSimulator.get_option(options, 'M4P_COLLECT_SAVE_DIR', '/kaggle/working')
        
        if non_github_create_fn is None:
            USE_GITHUB = True
            
        if USE_GITHUB:
            ok4s = github_pkg.Orottick4Simulator(PRD_SORT_ORDER, HAS_STEP_LOG, M4P_OBS, M4P_CNT, M4P_VRY, LOAD_CACHE_DIR, SAVE_CACHE_DIR)
        else:
            ok4s = non_github_create_fn(PRD_SORT_ORDER, HAS_STEP_LOG, M4P_OBS, M4P_CNT, M4P_VRY, LOAD_CACHE_DIR, SAVE_CACHE_DIR)
        
        if METHOD == 'build_cache':
            cdf = ok4s.build_cache(BUY_DATE, CACHE_CNT, BUFFER_DIR, LOTTE_KIND, DATA_DF, RUNTIME)
            if cdf is not None:
                try:
                    cdf.to_csv(f'{RESULT_DIR}/{LOTTE_KIND}-cache-{BUY_DATE}.csv', index=False)
                except Exception as e:
                    msg = str(e)
                    print(f'=> [E] {msg}')
        
        if METHOD == 'simulate':
            zdf, json_pred, pdf = ok4s.simulate(BUY_DATE, BUFFER_DIR, LOTTE_KIND, DATA_DF, DATE_CNT, TCK_CNT, RUNTIME)
            if zdf is not None:
                zdf.to_csv(f'{RESULT_DIR}/{LOTTE_KIND}-sim-{BUY_DATE}.csv', index=False)
            if pdf is not None:
                pdf.to_csv(f'{RESULT_DIR}/{LOTTE_KIND}-pick-{BUY_DATE}.csv', index=False)
            if json_pred is not None:
                with open(f'{RESULT_DIR}/{LOTTE_KIND}-pred-{BUY_DATE}.json', 'w') as f:
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

+ M4PC: __M4PC__

+ Predict Notebook:


  -------------------------------
              MONEY
  -------------------------------

+ Period No: 

+ Day No: 

+ Tickets: 250

+ Cost: $325

+ Total Cost: $325

+ Broker Cost: $0

+ Total Broker Cost: $0

+ Prize: $0

+ Total Prize: $0

+ Current ROI: 0.0%

+ Current ROI (w/o broker): 0.0%


  -------------------------------
            REAL BUY
  -------------------------------

+ Buy Number: __M4__

+ Confirmation Number: 

+ Cost: $1.3

+ Total Cost: $1.3

+ Broker Cost: $0.3

+ Total Broker Cost: $0.3

+ Prize: $0

+ Total Prize: $0

+ Current ROI: 0.0%

+ Current ROI (w/o broker): 0.0%
        
                '''
                if F_TCK_CNT != TCK_CNT:
                    lx_pred = str(json_pred['pred']).split(', ')
                    if len(lx_pred) > F_TCK_CNT:
                        lx_pred = lx_pred[:F_TCK_CNT]
                    json_pred['pred'] = ', '.join(lx_pred)
                m4pc = json_pred['m4pc']
                if m4pc > 0:
                    if M4P_ONE:
                        lx = str(json_pred['m4_pred']).split(', ')
                        if len(lx) != 1:
                            m4pc = 0
                json_pred['m4pc'] = m4pc
                text = text.replace('__LK__', str(LOTTE_KIND)).replace('__BD__', str(BUY_DATE)).replace('__RS__', str(json_pred['pred'])).replace('__M4__', str(json_pred['m4_pred'])).replace('__M4PC__', str(json_pred['m4pc']))
                with open(f'{RESULT_DIR}/{LOTTE_KIND}-pred-{BUY_DATE}.txt', 'w') as f:
                    f.write(text)
                print(text)
        
        if METHOD == 'observe':
            odf, more = ok4s.observe(LOTTE_KIND, BUY_DATE, TCK_CNT, O_DATE_CNT, RUNTIME, DATE_CNT, BUFFER_DIR, DATA_DF)
        
            if odf is not None and more is not None and len(odf) > 0:
                odf.to_csv(f'{RESULT_DIR}/{LOTTE_KIND}-observe-{BUY_DATE}.csv', index=False)
                qdf = odf[odf['m4'] > 0]
                if len(qdf) > 0:
                    for ri in range(len(qdf)):
                        t_buy_date = qdf['buy_date'].iloc[ri]
        
                        key = 'pred_' + t_buy_date    
                        if key in more:
                            json_pred = more[key]
                            if json_pred is not None:
                                with open(f'{RESULT_DIR}/{LOTTE_KIND}-pred-{t_buy_date}.json', 'w') as f:
                                    json.dump(json_pred, f)
        
                        key = 'sim_' + t_buy_date                
                        if key in more:
                            xdf = more[key]
                            if xdf is not None:
                                xdf.to_csv(f'{RESULT_DIR}/{LOTTE_KIND}-sim-{t_buy_date}.csv', index=False)
        
                        key = 'pick_' + t_buy_date                
                        if key in more:
                            xdf = more[key]
                            if xdf is not None:
                                xdf.to_csv(f'{RESULT_DIR}/{LOTTE_KIND}-pick-{t_buy_date}.csv', index=False)
        
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
                    odf.to_csv(f'{RESULT_DIR}/{LOTTE_KIND}-observe-{v_buy_date}.csv', index=False)
                    qdf = odf[odf['m4'] > 0]
                    if len(qdf) > 0:
                        for ri in range(len(qdf)):
                            t_buy_date = qdf['buy_date'].iloc[ri]
            
                            key = 'pred_' + t_buy_date    
                            if key in more:
                                json_pred = more[key]
                                if json_pred is not None:
                                    with open(f'{RESULT_DIR}/{LOTTE_KIND}-pred-{t_buy_date}.json', 'w') as f:
                                        json.dump(json_pred, f)
            
                            key = 'sim_' + t_buy_date                
                            if key in more:
                                xdf = more[key]
                                if xdf is not None:
                                    xdf.to_csv(f'{RESULT_DIR}/{LOTTE_KIND}-sim-{t_buy_date}.csv', index=False)
            
                            key = 'pick_' + t_buy_date                
                            if key in more:
                                xdf = more[key]
                                if xdf is not None:
                                    xdf.to_csv(f'{RESULT_DIR}/{LOTTE_KIND}-pick-{t_buy_date}.csv', index=False)
        
                range_idx += 1
        
        if METHOD == 'download':  
            d1 = datetime.strptime(BUY_DATE, "%Y.%m.%d")
            g = -1
            d2 = d1 + timedelta(minutes=int(+(g*(60 * 24))))
            v_date = d2.strftime('%Y.%m.%d')
        
            data_df = ok4s.download_drawing(BUFFER_DIR, LOTTE_KIND, v_date)
        
            if data_df is not None:
                data_df.to_csv(f'{RESULT_DIR}/{LOTTE_KIND}-{BUY_DATE}.csv', index=False)

        if METHOD == 'm4p_collect':
            ok4s.m4p_collect(LOTTE_KIND, O_DATE_CNT, DATE_CNT, M4P_COLLECT_DATA_DIRS, M4P_COLLECT_SAVE_DIR)
            
# ------------------------------------------------------------ #
