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
import lightgbm as lgb
import optuna

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

        self.m4pcm = None
        
        self.m4pm = None
        self.m4p_cnt = -1
        self.m4p_ranker_only = False
        self.m4p_max = 30
        self.m4p_ranker_max = 2 * self.m4p_max

        self.m4pl_max = 2
        self.m4pl_min = -2
        self.m4pl_step = 0.0225

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
        m4pc = 0
        lp2, vm4pc2 = self.capture_m4p_manual(pdf, x_sim_seed)
        lp, vm4pc = self.capture_m4p_ranker(pdf, x_sim_seed)
        if vm4pc2 > 0:
            m4pc = 1
        if not self.m4p_ranker_only:
            if len(lp) == 0:
                return lp2, m4pc
        return lp, m4pc
        
    def capture_m4p_ranker(self, pdf, x_sim_seed):
        m4pc = 0
        if self.m4pm is None:
            print(f'== [M4PM] Model is not found!')
            return [], m4pc
        
        pdf2 = pdf.sort_values(by=['buy_date'], ascending=[False])
        features = self.m4pm['features']
        rnkp_min = self.m4pm['rnkp_min']
        rnkp_max = self.m4pm['rnkp_max']
        pdf2['rnkp'] = self.m4pm['model'].predict(pdf2[features])
        pdf4 = pdf2[(pdf2['rnkp'] >= rnkp_min)&(pdf2['rnkp'] <= rnkp_max)]
        if len(pdf4) == 0:
            print(f'== [M4PM] Ranking is not found!')
            return [], m4pc

        pdf2a = pdf4.sort_values(by=['rnkp', 'buy_date'], ascending=[True, False])
        pdf2b = pdf4.sort_values(by=['rnkp', 'buy_date'], ascending=[False, False])

        if self.m4p_cnt > 0:
            h_cnt = int(round(self.m4p_cnt / 2.0))
            if self.m4p_cnt % 2 != 0:
                h_cnt += 1
            if len(pdf2) >= 2 * h_cnt:
                pdf2a = pdf2a[:h_cnt]
                pdf2b = pdf2b[:h_cnt]
                pdf2 = pd.concat([pdf2a, pdf2b])
                pdf2 = pdf2.sort_values(by=['rnkp', 'buy_date'], ascending=[True, False])
            else:
                pdf2 = pdf2a
        else:
            pdf2 = pdf2a
            
        l_rnkp = list(pdf2['rnkp'].values)
        print(f'==> [RNKPL] {l_rnkp}')
        
        adf = pdf2
        l_pred = []
        for ri in range(len(adf)):
            x_sim_cnt = adf['sim_cnt'].iloc[ri]
            p = self.reproduce_one(x_sim_seed, x_sim_cnt)
            l_pred.append(p)

        print(f'== [M4PM] Success: {l_pred}')

        if len(l_pred) > 0:
            if self.m4p_ranker_only:
                if len(l_pred) >= self.m4p_ranker_max:
                    m4pc = 1
            else:
                m4pc = 1

        return l_pred, m4pc
        
    def capture_m4p_manual(self, pdf, x_sim_seed):
        m4pc = 0
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
            return l_pred, m4pc

        adf = adf.sort_values(by=['m4p_no'], ascending=[True])
        for ri in range(len(adf)):
            x_sim_cnt = adf['sim_cnt'].iloc[ri]
            p = self.reproduce_one(x_sim_seed, x_sim_cnt)
            l_pred.append(p)

        if len(l_pred) > 0:
            m4pc = 1
            
        return l_pred, m4pc

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

    def m4p_prepare(self, lotte_kind, data_dir, save_dir):
        self.print_heading()

        text = '''
====================================
           M4P PREPARE
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
        print(f'DATA_DIR: {data_dir}')
        print(f'SAVE_DIR: {save_dir}')

        text = '''
  -------------------------------
        '''
        print(text) 

        apdf = None
        lg_obs = glob.glob(f'{data_dir}/{lotte_kind}-observe-*.*.*.csv')
        sz_1 = len(lg_obs)
        i_1 = 0
        for fn_obs in lg_obs:
            i_1 += 1
            odf = pd.read_csv(fn_obs)
            odf = odf[odf['m4'] > 0]
            if len(odf) == 0:
                continue
            sz_2 = len(odf)
            for xi in range(len(odf)):
                i_2 = xi + 1
                x_buy_date = odf['buy_date'].iloc[xi]
                x_sim_seed = odf['sim_seed'].iloc[xi]
                x_w = odf['w'].iloc[xi]
                pdf = pd.read_csv(f'{data_dir}/{lotte_kind}-pick-{x_buy_date}.csv')
                m_keys = ['m4', 'm3f', 'm3l', 'm3', 'm2']
                pdf['x_buy_date'] = x_buy_date
                pdf['fw'] = x_w
                pdf['fp'] = -1
                pdf['m4p_no'] = 0
                for key in m_keys:
                    pdf[f'x_{key}'] = 0
                l_fp = []
                l_dict = {}
                for key in m_keys:
                    l_dict[f'{key}'] = []
                for yi in range(len(pdf)):
                    y_sim_cnt = pdf['sim_cnt'].iloc[yi]
                    fp = self.reproduce_one(x_sim_seed, y_sim_cnt)  
                    l_fp.append(fp)
                    for key in m_keys:
                        v = 0
                        if self.match(x_w, fp, key):
                            v = 1
                        l_dict[key].append(v)
                pdf['fp'] = l_fp
                for key in m_keys:
                    pdf[f'x_{key}'] = l_dict[key]
                pdf = pdf.sort_values(by=['x_m4', 'x_m3f', 'x_m3l', 'x_m3', 'x_m2', 'buy_date'], ascending=[False, False, False, False, False, False])
                l_m4p_no = [x+1 for x in range(len(pdf))]
                pdf['m4p_no'] = l_m4p_no
                #pdf = pdf[:5]
                if apdf is None:
                    apdf = pdf
                else:
                    apdf = pd.concat([apdf, pdf])
                apdf = apdf.sort_values(by=['x_buy_date', 'm4p_no'], ascending=[False, True])
                
                print(f'== [P] {i_1} / {sz_1} -> {i_2} / {sz_2}')

        apdf.to_csv(f'{save_dir}/{lotte_kind}-all.csv', index=False)
        lx_buy_date = list(apdf['x_buy_date'].unique())
        sz = len(lx_buy_date)
        sz_valid = int(round(sz * 0.2))
        lx_valid = lx_buy_date[:sz_valid]
        lx_train = lx_buy_date[sz_valid:]
        
        valid_df = apdf[apdf['x_buy_date'].isin(lx_valid)]
        valid_df.to_csv(f'{save_dir}/{lotte_kind}-valid.csv', index=False)       

        train_df = apdf[apdf['x_buy_date'].isin(lx_train)]
        train_df.to_csv(f'{save_dir}/{lotte_kind}-train.csv', index=False)       
        
        text = '''
  -------------------------------
           M4P PREPARE
====================================
        '''
        print(text) 

    def m4p_train(self, lotte_kind, data_dir, save_dir):
        self.print_heading()

        text = '''
====================================
            M4P TRAIN
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
        print(f'DATA_DIR: {data_dir}')
        print(f'SAVE_DIR: {save_dir}')

        m4p_max = self.m4p_max
        
        all_df = pd.read_csv(f'{data_dir}/{lotte_kind}-all.csv')
        #all_df = all_df[all_df['m4p_no'] <= m4p_max]
        sz = len(all_df)
        print(f'ALL_SZ: {sz}')

        train_df = pd.read_csv(f'{data_dir}/{lotte_kind}-train.csv')
        train_df = train_df[train_df['m4p_no'] <= m4p_max]
        sz = len(train_df)
        print(f'TRAIN_SZ: {sz}')

        valid_df = pd.read_csv(f'{data_dir}/{lotte_kind}-valid.csv')
        valid_df = valid_df[valid_df['m4p_no'] <= m4p_max]
        sz = len(valid_df)
        print(f'VALID_SZ: {sz}')

        text = '''
  -------------------------------
        '''
        print(text) 

        SEED = 311
        random.seed(SEED)
        os.environ["PYTHONHASHSEED"] = str(SEED)
        np.random.seed(SEED)

        features = ['m4', 'm3f', 'm3l', 'm3', 'm2', 'a_m4', 'a_m3f', 'a_m3l', 'a_m3', 'a_m2', 'm4_cnt', 'm3f_cnt', 'm3l_cnt', 'm3_cnt', 'm2_cnt']
        target = 'm4p_no'

        # model query data
        train_query = train_df['x_buy_date'].value_counts().sort_index()
        valid_query = valid_df['x_buy_date'].value_counts().sort_index()

        def objective(trial):
            # search param
            param = {
                'reg_alpha': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
                'reg_lambda': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'num_leaves': trial.suggest_int('num_leaves', 2, 256),
                'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.1, 1), 
                #'subsample': trial.suggest_uniform('subsample', 1e-8, 1), 
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100), 
            }
             
            #train model
            model = lgb.LGBMRanker(n_estimators=1000, **param, random_state=SEED,early_stopping_rounds=50,verbose=10)
            model.fit(
                train_df[features],
                train_df[target],
                group=train_query,
                eval_set=[(valid_df[features], valid_df[target])],
                eval_group=[list(valid_query)],
                eval_at=[1, 3, 5, 10, 20], # calc validation ndcg@1,3,5,10,20
                #early_stopping_rounds=50,
                #verbose=10
            )
            
            vadf = all_df.sort_values(by=['x_buy_date', 'm4p_no'], ascending=[False, True])
            vcnt = 0
            lx_buy_date = list(vadf['x_buy_date'].unique())
            for x_buy_date in lx_buy_date:
                df = vadf[vadf['x_buy_date'] == x_buy_date]
                df['rnkp'] = model.predict(df[features])
                df = df.sort_values(by=['rnkp', 'buy_date'], ascending=[True, False])
                df['rnkn'] = [x+1 for x in range(len(df))]
                df2 = df[((df['rnkn'] >= 1)&(df['rnkn'] <= m4p_max))&(df['m4p_no'] == 1)]
                vcnt += len(df2)

            sz = len(lx_buy_date)
            print(f'== [RNK1_CNT] ==> {vcnt} / {sz}')
            
            # maximize mean ndcg
            scores = []
            for name, score in model.best_score_['valid_0'].items():
                scores.append(score)
            return vcnt + np.mean(scores)

        def count_match(model, min_p, inverve = False):
            vadf = all_df.sort_values(by=['x_buy_date', 'm4p_no'], ascending=[False, True])
            vcnt = 0
            lx_buy_date = list(vadf['x_buy_date'].unique())
            for x_buy_date in lx_buy_date:
                df = vadf[vadf['x_buy_date'] == x_buy_date]
                df['rnkp'] = model.predict(df[features])
                df = df.sort_values(by=['rnkp', 'buy_date'], ascending=[True, False])
                if inverve:
                    df = df[df['rnkp'] <= min_p]
                else:
                    df = df[df['rnkp'] >= min_p]
                if len(df) == 0:
                    continue
                df['rnkn'] = [x+1 for x in range(len(df))]
                df2 = df[((df['rnkn'] >= 1)&(df['rnkn'] <= m4p_max))&(df['m4p_no'] == 1)]
                vcnt += len(df2)

            return vcnt
            
        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=SEED) #fix random seed
                                   )
        study.optimize(objective, n_trials=100)

        print('Number of finished trials:', len(study.trials))
        print('Best trial:', study.best_trial.params)        

        # train with best params
        best_params = study.best_trial.params
        model = lgb.LGBMRanker(n_estimators=1000, **best_params, random_state=SEED,early_stopping_rounds=50,verbose=10)
        model.fit(
            train_df[features],
            train_df[target],
            group=train_query,
            eval_set=[(valid_df[features], valid_df[target])],
            eval_group=[list(valid_query)],
            eval_at=[1, 3, 5, 10, 20],
            #early_stopping_rounds=50,
            #verbose=10
        )

        lx_buy_date = list(all_df['x_buy_date'].unique())
        rnkp_sz = len(lx_buy_date)

        p_min = self.m4pl_min
        p_max = self.m4pl_max
        p_step = self.m4pl_step
        pmn = p_min
        rows = []
        while pmn <= p_max:
            cnt = count_match(model, pmn, False)
            rw = {'pmn': pmn, 'cnt': cnt}
            rows.append(rw)
            pmn += p_step

        mdf = pd.DataFrame(rows)
        mdf = mdf.sort_values(by=['cnt', 'pmn'], ascending=[False, False])
        rnkp_min = mdf['pmn'].iloc[0]
        rnkp_min_cnt = mdf['cnt'].iloc[0]
        mdf.to_csv(f'{save_dir}/{lotte_kind}-rnkp-min.csv', index=False)

        print(f'== [RNKP_MIN] ==> {rnkp_min} -> {rnkp_min_cnt} / {rnkp_sz}')

        p_min = self.m4pl_max
        p_max = self.m4pl_min
        p_step = self.m4pl_step
        pmn = p_min
        rows = []
        while pmn >= p_max:
            cnt = count_match(model, pmn, True)
            rw = {'pmn': pmn, 'cnt': cnt}
            rows.append(rw)
            pmn -= p_step

        mdf = pd.DataFrame(rows)
        mdf = mdf.sort_values(by=['cnt', 'pmn'], ascending=[False, False])
        rnkp_max = mdf['pmn'].iloc[0]
        rnkp_max_cnt = mdf['cnt'].iloc[0]
        mdf.to_csv(f'{save_dir}/{lotte_kind}-rnkp-max.csv', index=False)

        print(f'== [RNKP_MAX] ==> {rnkp_max} -> {rnkp_max_cnt} / {rnkp_sz}')

        m4pm = {'m4p_max': self.m4p_max, 'm4pl_max': self.m4pl_max, 'm4pl_min': self.m4pl_min, 'm4pl_step': self.m4pl_step, 'params': best_params, 'features': features, 'rnkp_min': rnkp_min, 'rnkp_min_cnt': rnkp_min_cnt, 'rnkp_max': rnkp_max, 'rnkp_max_cnt': rnkp_max_cnt, 'rnkp_sz': rnkp_sz, 'model': model}
        with open(f'{save_dir}/{lotte_kind}-m4pm.pkl', 'wb') as f:
            pickle.dump(m4pm, f)

        text = '''
  -------------------------------
            M4P TRAIN
====================================
        '''
        print(text) 

    def refine_m4pc_ds(self, ddf):
        dict_date = {}
        dict_year = {}
        columns = list(ddf.columns)
        ddf['ix'] = [x+1 for x in range(len(ddf))]
        list_ix = []
        col_year = []
        for ri in range(len(ddf)):
            buy_date = ddf['buy_date'].iloc[ri]
            year = int(buy_date.split('.')[0])
            col_year.append(year)
            if buy_date not in dict_date:
                dict_date[buy_date] = 1
                if year in dict_year:
                    dict_year[year] = dict_year[year] + 1
                else:
                    dict_year[year] = 1
                list_ix.append(ddf['ix'].iloc[ri])
        ddf['year'] = col_year
        list_year = []
        for year in dict_year.keys():
            cnt = dict_year[year]
            if cnt < 365:
                continue
            list_year.append(year)
        if len(list_ix) == 0:
            return None, None
        ddf = ddf[ddf['ix'].isin(list_ix)]
        if len(ddf) == 0:
            return None, None
        if len(list_year) == 0:
            return None, None
        ddf = ddf[ddf['year'].isin(list_year)]
        if len(ddf) == 0:
            return None, None
        nddf = ddf[columns]
        columns.append('year')
        oddf = ddf[columns]
        oddf = oddf.sort_values(by=['buy_date'], ascending=[False])
        nddf = nddf.sort_values(by=['buy_date'], ascending=[False])
        rows = []
        for year in list_year:
            ddf = oddf[oddf['year'] == year]
            sz = len(ddf)
            df = ddf[ddf['m4'] > 0]
            m4_1 = len(df)
            df = ddf[ddf['m4'] <= 0]
            m4_0 = len(df)
            df = ddf[ddf['m4pc'] == 1]
            m4pc_1 = len(df)
            df = ddf[ddf['m4pc'] == 0]
            m4pc_0 = len(df)
            df = ddf[(ddf['m4'] > 0)&(ddf['m4pc'] == 1)]
            m4_1__m4pc_1 = len(df)
            df = ddf[(ddf['m4'] > 0)&(ddf['m4pc'] == 0)]
            m4_1__m4pc_0 = len(df)
            df = ddf[(ddf['m4'] <= 0)&(ddf['m4pc'] == 1)]
            m4_0__m4pc_1 = len(df)
            df = ddf[(ddf['m4'] <= 0)&(ddf['m4pc'] == 0)]
            m4_0__m4pc_0 = len(df)
            ddf = ddf.sort_values(by=['buy_date'], ascending=[False])
            last_date = ddf['buy_date'].iloc[0]
            last_year = int(last_date.split('.')[0])
            rw = {'last_date': last_date, 'last_year': last_year, 'sz': sz, 'm4_1': m4_1, 'm4_0': m4_0, 'm4pc_1': m4pc_1, 'm4pc_0': m4pc_0, 'm4_1__m4pc_1': m4_1__m4pc_1, 'm4_1__m4pc_0': m4_1__m4pc_0, 'm4_0__m4pc_1': m4_0__m4pc_1, 'm4_0__m4pc_0': m4_0__m4pc_0}
            rows.append(rw)        
        return nddf, oddf, rows
        
    def m4pc_train(self, lotte_kind, data_dir, save_dir, runtime):
        global test_df, all_df, valid_df, train_df

        start_time = time.time()
        
        self.print_heading()

        text = '''
====================================
            M4PC TRAIN
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
        print(f'DATA_DIR: {data_dir}')
        print(f'SAVE_DIR: {save_dir}')

        SEED = 311
        random.seed(SEED)
        os.environ["PYTHONHASHSEED"] = str(SEED)
        np.random.seed(SEED)

        all_df = None
        rdf_glob_fn = glob.glob(f'{data_dir}/{lotte_kind}-m4pc-rdf-*.*.*.csv')
        for rdf_fn in rdf_glob_fn:
            df = pd.read_csv(rdf_fn)
            if all_df is None:
                all_df = df
            else:
                all_df = pd.concat([all_df, df])

        oddf, all_df, srows = self.refine_m4pc_ds(all_df)

        for rw in srows:
            print(str(rw))

        all_df['m4pc'] = [1 if all_df['m4'].iloc[x] > 0 else 0 for x in range(len(all_df))]

        all_df = all_df.sort_values(by=['buy_date'], ascending=[False])
        list_year = list(all_df['year'].unique())

        if len(list_year) >= 2:
            list_year_test = [list_year[0]]
            list_year_all = [list_year[x] for x in range(len(list_year)) if x > 0]
            test_df = all_df[all_df['year'].isin(list_year_test)]
            all_df = all_df[all_df['year'].isin(list_year_all)]            
        else:
            test_df = all_df.sample(frac=1)
            all_df = all_df.sample(frac=1)

        valid_df = None
        train_df = None
                
        text = '''
  -------------------------------
        '''
        print(text) 

        features = ['date_cnt_ir', 'date_cnt_or', 'year_ir', 'year_or', 'p_year_ir', 'p_year_or', 'month_ir', 'month_or', 'day_ir', 'day_or', 'month_day_ir', 'month_day_or']

        min_date_cnt = 1
        max_date_cnt = 56 * 5
        dcnt_step = 5
        dcnt_min = min_date_cnt
        dcnt_max = dcnt_min + dcnt_step - 1
        while dcnt_max <= max_date_cnt:
            features.append(f'date_cnt_{dcnt_max}_ir')
            features.append(f'date_cnt_{dcnt_max}_or')
            dcnt_min += dcnt_step
            dcnt_max = dcnt_min + dcnt_step - 1

        target = 'm4pc'

        try_no = 1

        def objective(trial):
            # search param
            param = {
                'reg_alpha': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
                'reg_lambda': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
                'max_depth': trial.suggest_int('max_depth', 2, 8),
                'num_leaves': trial.suggest_int('num_leaves', 2, 256),
                'colsample_bytree': 1,
                #'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.1, 1), 
                #'subsample': trial.suggest_uniform('subsample', 1e-8, 1), 
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100), 
            }
             
            #train model
            model = lgb.LGBMClassifier(n_estimators=1000, **param, random_state=SEED,early_stopping_rounds=50,verbose=-1)
            model.fit(
                train_df[features],
                train_df[target],
                eval_set=[(valid_df[features], valid_df[target])],
                #eval_at=[1, 3, 5, 10, 20], # calc validation ndcg@1,3,5,10,20
                #early_stopping_rounds=50,
                #verbose=10
            )
            
            vadf = all_df.sort_values(by=['buy_date'], ascending=[False])
            df = vadf[vadf['m4pc'] == 1]
            vcnt_sz = len(df)
            vadf['nm4pc'] = model.predict(vadf[features])
            df = vadf[(vadf['nm4pc'] == 1)&(vadf['m4pc'] == 1)]
            vcnt2 = len(df)
            df = vadf[(vadf['nm4pc'] == 1)&(vadf['m4pc'] == 0)]
            vcnt3 = len(df)
            vcnt = vcnt_sz - (vcnt2 - vcnt3)

            df = vadf[vadf['m4pc'] == 1]
            sz = len(df)
            tn = trial.number
            print(f'== [M4PC_CNT_{try_no}_{tn}] ==> {vcnt}, {vcnt2}, {vcnt3} / {sz}')
            
            # maximize mean ndcg
            scores = []
            for name, score in model.best_score_['valid_0'].items():
                scores.append(score)
            return np.mean(scores) + vcnt

        def do_try():
            global test_df, all_df, valid_df, train_df
            
            #test_df = test_df.sample(frac=1)
            all_df = all_df.sample(frac=1)
            
            adf1 = all_df[all_df['m4pc'] == 1]
            adf0 = all_df[all_df['m4pc'] == 0]
    
            sz = len(adf1)
            sz = int(round(sz * 0.2))
            valid1_df = adf1[:sz]
            train1_df = adf1[sz:]
    
            sz = len(adf0)
            sz = int(round(sz * 0.2))
            valid0_df = adf0[:sz]
            train0_df = adf0[sz:]
    
            sz = len(valid1_df) * 5
            valid0_df = valid0_df[:sz]
            sz = len(train1_df) * 5
            train0_df = train0_df[:sz]
            
            valid_df = pd.concat([valid1_df, valid0_df])
            train_df = pd.concat([train1_df, train0_df])
        
            valid_df = valid_df.sample(frac=1)
            train_df = train_df.sample(frac=1)

            study = optuna.create_study(direction='minimize',
                                        sampler=optuna.samplers.TPESampler(seed=SEED) #fix random seed
                                       )
            study.optimize(objective, n_trials=100)
    
            print('Number of finished trials:', len(study.trials))
            print('Best trial:', study.best_trial.params)        
    
            # train with best params
            best_params = study.best_trial.params
            model = lgb.LGBMClassifier(n_estimators=1000, **best_params, random_state=SEED,early_stopping_rounds=50,verbose=-1)
            model.fit(
                train_df[features],
                train_df[target],
                eval_set=[(valid_df[features], valid_df[target])],
                #eval_at=[1, 3, 5, 10, 20],
                #early_stopping_rounds=50,
                #verbose=10
            )

            rw = {'try_no': try_no, 'score': 0}
            
            ddf = all_df.sort_values(by=['buy_date'], ascending=[False])
            ddf['nm4pc'] = model.predict(ddf[features])

            sz = len(ddf)
            df = ddf[ddf['m4'] > 0]
            m4_1 = len(df)
            df = ddf[ddf['m4'] <= 0]
            m4_0 = len(df)
            df = ddf[ddf['nm4pc'] == 1]
            m4pc_1 = len(df)
            df = ddf[ddf['nm4pc'] == 0]
            m4pc_0 = len(df)
            df = ddf[(ddf['m4'] > 0)&(ddf['nm4pc'] == 1)]
            m4_1__m4pc_1 = len(df)
            df = ddf[(ddf['m4'] > 0)&(ddf['nm4pc'] == 0)]
            m4_1__m4pc_0 = len(df)
            df = ddf[(ddf['m4'] <= 0)&(ddf['nm4pc'] == 1)]
            m4_0__m4pc_1 = len(df)
            df = ddf[(ddf['m4'] <= 0)&(ddf['nm4pc'] == 0)]
            m4_0__m4pc_0 = len(df)
            nrw = {'a_score': 0, 'a_sz': sz, 'a_m4_1': m4_1, 'a_m4_0': m4_0, 'a_m4pc_1': m4pc_1, 'a_m4pc_0': m4pc_0, 'a_m4_1__m4pc_1': m4_1__m4pc_1, 'a_m4_1__m4pc_0': m4_1__m4pc_0, 'a_m4_0__m4pc_1': m4_0__m4pc_1, 'a_m4_0__m4pc_0': m4_0__m4pc_0}
            for key in nrw.keys():
                rw[key] = nrw[key]
                
            vadf = ddf
            df = vadf[vadf['m4'] > 0]
            vcnt_sz = len(df)
            df = vadf[(vadf['nm4pc'] == 1)&(vadf['m4'] > 0)]
            vcnt2 = len(df)
            df = vadf[(vadf['nm4pc'] == 1)&(vadf['m4'] <= 0)]
            vcnt3 = len(df)
            score = vcnt_sz - (vcnt2 - vcnt3)
            
            rw['a_score'] = score

            ddf = test_df.sort_values(by=['buy_date'], ascending=[False])
            ddf['nm4pc'] = model.predict(ddf[features])

            sz = len(ddf)
            df = ddf[ddf['m4'] > 0]
            m4_1 = len(df)
            df = ddf[ddf['m4'] <= 0]
            m4_0 = len(df)
            df = ddf[ddf['nm4pc'] == 1]
            m4pc_1 = len(df)
            df = ddf[ddf['nm4pc'] == 0]
            m4pc_0 = len(df)
            df = ddf[(ddf['m4'] > 0)&(ddf['nm4pc'] == 1)]
            m4_1__m4pc_1 = len(df)
            df = ddf[(ddf['m4'] > 0)&(ddf['nm4pc'] == 0)]
            m4_1__m4pc_0 = len(df)
            df = ddf[(ddf['m4'] <= 0)&(ddf['nm4pc'] == 1)]
            m4_0__m4pc_1 = len(df)
            df = ddf[(ddf['m4'] <= 0)&(ddf['nm4pc'] == 0)]
            m4_0__m4pc_0 = len(df)
            nrw = {'t_score': 0, 't_sz': sz, 't_m4_1': m4_1, 't_m4_0': m4_0, 't_m4pc_1': m4pc_1, 't_m4pc_0': m4pc_0, 't_m4_1__m4pc_1': m4_1__m4pc_1, 't_m4_1__m4pc_0': m4_1__m4pc_0, 't_m4_0__m4pc_1': m4_0__m4pc_1, 't_m4_0__m4pc_0': m4_0__m4pc_0}
            for key in nrw.keys():
                rw[key] = nrw[key]

            vadf = ddf
            df = vadf[vadf['m4'] > 0]
            vcnt_sz = len(df)
            df = vadf[(vadf['nm4pc'] == 1)&(vadf['m4'] > 0)]
            vcnt2 = len(df)
            df = vadf[(vadf['nm4pc'] == 1)&(vadf['m4'] <= 0)]
            vcnt3 = len(df)
            score = vcnt_sz - (vcnt2 - vcnt3)

            rw['t_score'] = score

            rw['score'] = rw['a_score'] + rw['t_score']
            
            print(f'== [M4PC_CNT_{try_no}_FINAL] ==> ' + str(rw))

            vm4pcm = {'params': best_params, 'features': features, 'scores': rw, 'model': model}

            return rw, vm4pcm

        rows = []
        try_no = 1
        dict_m4pcm = {}
        while try_no <= 50000:
            if runtime is not None:
                if time.time() - start_time > runtime:
                    break
            srw, vm4pcm = do_try()
            if srw['a_m4_1__m4pc_1'] > 0 and srw['t_m4_1__m4pc_1'] > 0:
                rows.append(srw)
                try:
                    sdf = pd.DataFrame(rows)
                    sdf = sdf.sort_values(by=['score', 't_score', 'a_score'], ascending=[True, True, True])
                    sdf.to_csv(f'{save_dir}/summary.csv', index=False)
                    with open(f'{save_dir}/{lotte_kind}-m4pcm-{try_no}.pkl', 'wb') as f:
                        pickle.dump(m4pcm, f)
                    dict_m4pcm[try_no] = m4pcm
                except Exception as e:
                    print(f'== [E:{try_no}] ==> ' + str(e))        
            try_no += 1

        if len(rows) > 0:
            sdf = pd.DataFrame(rows)
            sdf = sdf.sort_values(by=['score', 't_score', 'a_score'], ascending=[True, True, True])
            try_no = sdf['try_no'].iloc[0]
            sdf.to_csv(f'{save_dir}/summary.csv', index=False)
            m4pcm = dict_m4pcm[try_no]
            
            with open(f'{save_dir}/{lotte_kind}-m4pcm.pkl', 'wb') as f:
                pickle.dump(m4pcm, f)

        text = '''
  -------------------------------
            M4PC TRAIN
====================================
        '''
        print(text) 

    def research_a(self, v_buy_date, buffer_dir = '/kaggle/buffers/orottick4', lotte_kind = 'p4a', data_df = None, v_date_cnt = 365 * 5, has_log_step = False, runtime = None, catch_kind = 'same_month', silent = False):
        self.print_heading()

        text = '''
====================================
          RESEARCH A
  -------------------------------
        '''
        if not silent:
            print(text) 

        text = '''
  -------------------------------
           PARAMETERS
  -------------------------------
        '''
        if not silent:
            print(text) 

        v_data_df_is_none = False
        if data_df is None:
            v_data_df_is_none = True

        if not silent:
            print(f'[BUFFER_DIR] {buffer_dir}')
            print(f'[LOTTE_KIND] {lotte_kind}')
            print(f'[DATA_DF_IS_NONE] {v_data_df_is_none}')
            print(f'[BUY_DATE] {v_buy_date}')
            print(f'[DATE_CNT] {v_date_cnt}')
            print(f'[RUNTIME] {runtime}')

        text = '''
  -------------------------------
        '''
        if not silent:
            print(text) 

        start_time = time.time()

        rdf = None
        cdf = None
        
        if data_df is None:
            d1 = datetime.strptime(v_buy_date, "%Y.%m.%d")
            g = -1
            d2 = d1 + timedelta(minutes=int(+(g*(60 * 24))))
            v_date = d2.strftime('%Y.%m.%d')
    
            data_df = self.download_drawing(buffer_dir, lotte_kind, v_date)
            if data_df is None:
                return rdf, cdf

        ddf = data_df[data_df['buy_date'] < v_buy_date]
        if len(ddf) == 0:
            return rdf, cdf
            
        ddf = ddf[(ddf['w'] >= 0)&(ddf['n'] >= 0)]
        if len(ddf) == 0:
            return rdf, cdf

        ddf = ddf.sort_values(by=['buy_date'], ascending=[False])
        if len(ddf) > v_date_cnt:
            ddf = ddf[:v_date_cnt]
        if len(ddf) == 0:
            return rdf, cdf

        ddf = ddf.sort_values(by=['buy_date'], ascending=[True])

        bdfd = v_buy_date.split('.')
        x_txt_year = bdfd[0]
        x_txt_month = bdfd[1]
        x_txt_day = bdfd[2]
        
        dict_year = {}
        dict_month = {}
        dict_year_month = {}
        dict_day = {}

        dict_year_m4 = {}
        dict_month_m4 = {}
        dict_year_month_m4 = {}
        dict_day_m4 = {}

        for ri in range(len(ddf)):
            t_buy_date = ddf['buy_date'].iloc[ri]
            bdfd = t_buy_date.split('.')

            year = bdfd[0]
            if year in dict_year:
                dict_year[year] = dict_year[year] + 1
            else:
                dict_year[year] = 1

            month = bdfd[1]
            if month in dict_month:
                dict_month[month] = dict_month[month] + 1
            else:
                dict_month[month] = 1
            
            year_month = bdfd[0] + '.' + bdfd[1]
            if year_month in dict_year_month:
                dict_year_month[year_month] = dict_year_month[year_month] + 1
            else:
                dict_year_month[year_month] = 1

            day = bdfd[2]
            if day in dict_day:
                dict_day[day] = dict_day[day] + 1
            else:
                dict_day[day] = 1
            
        rows = []
        dsz = len(ddf) * len(ddf)
        dix = 0
        dix_m4 = 0
        dcnt = 1000
        dcnt_m4 = 100
        for ria in range(len(ddf)):
            if runtime is not None:
                if time.time() - start_time > runtime:
                    break

            a_date = ddf['date'].iloc[ria]
            a_buy_date = ddf['buy_date'].iloc[ria]
            a_next_date = ddf['next_date'].iloc[ria]
            a_w = ddf['w'].iloc[ria]
            a_n = ddf['n'].iloc[ria]
            a_sim_seed, a_sim_cnt = self.capture(a_w, a_n)

            bdfd = a_buy_date.split('.')
            a_txt_year = bdfd[0]
            a_year = int(a_txt_year)
            a_year_cnt = dict_year[a_txt_year]

            a_txt_month = bdfd[1]
            a_month = int(a_txt_month)
            a_month_cnt = dict_month[a_txt_month]
            
            a_txt_year_month = bdfd[0] + '.' + bdfd[1]
            a_year_month_cnt = dict_year_month[a_txt_year_month]

            a_txt_day = bdfd[2]
            a_day = int(a_txt_day)
            a_day_cnt = dict_day[a_txt_day]

            for rib in range(len(ddf)):
                if runtime is not None:
                    if time.time() - start_time > runtime:
                        break

                if rib >= ria:
                    break

                b_date = ddf['date'].iloc[rib]
                b_buy_date = ddf['buy_date'].iloc[rib]
                b_next_date = ddf['next_date'].iloc[rib]
                b_w = ddf['w'].iloc[rib]
                b_n = ddf['n'].iloc[rib]
                b_sim_seed, b_sim_cnt = self.capture(b_w, b_n)

                a_p = self.reproduce_one(a_sim_seed, b_sim_cnt)

                a_m4 = 0
                if self.match(a_w, a_p, 'm4'):
                    a_m4 = 1
                    dix_m4 += 1

                dix += 1
                if dix % dcnt == 0:
                    if has_log_step:
                        print(f'== [R] {dix}, {dix_m4} / {dsz}')
                    
                if a_m4 <= 0:
                    continue

                a_date_no = ria + 1
                a_date_cnt = ria - rib
                b_date_no = rib + 1

                rw = {'a_date': a_date, 'a_buy_date': a_buy_date, 'a_next_date': a_next_date, 'a_txt_year': a_txt_year, 'a_year': a_year, 'a_year_cnt': a_year_cnt, 'a_year_cnt_m4': 0, 'a_txt_month': a_txt_month, 'a_month': a_month, 'a_month_cnt': a_month_cnt, 'a_month_cnt_m4': 0, 'a_txt_year_month': a_txt_year_month, 'a_year_month_cnt': a_year_month_cnt, 'a_year_month_cnt_m4': 0, 'a_txt_day': a_txt_day, 'a_day': a_day, 'a_day_cnt': a_day_cnt, 'a_day_cnt_m4': 0, 'a_w': a_w, 'a_n': a_n, 'a_sim_seed': a_sim_seed, 'a_sim_cnt': a_sim_cnt, 'a_p': a_p, 'a_m4': a_m4, 'a_date_no': a_date_no, 'a_date_cnt': a_date_cnt, 'a_date_cnt_same': 0, 'b_date': b_date, 'b_buy_date': b_buy_date, 'b_next_date': b_next_date, 'b_w': b_w, 'b_n': b_n, 'b_sim_seed': b_sim_seed, 'b_sim_cnt': b_sim_cnt, 'b_date_no': b_date_no}
                rows.append(rw)
                
                if dix_m4 % dcnt_m4 == 0:
                    if has_log_step:
                        print(str(rw))

                if a_txt_year in dict_year_m4:
                    dict_year_m4[a_txt_year] = dict_year_m4[a_txt_year] + 1
                else:
                    dict_year_m4[a_txt_year] = 1

                if a_txt_month in dict_month_m4:
                    dict_month_m4[a_txt_month] = dict_month_m4[a_txt_month] + 1
                else:
                    dict_month_m4[a_txt_month] = 1
            
                if a_txt_year_month in dict_year_month_m4:
                    dict_year_month_m4[a_txt_year_month] = dict_year_month_m4[a_txt_year_month] + 1
                else:
                    dict_year_month_m4[a_txt_year_month] = 1

                if a_txt_day in dict_day_m4:
                    dict_day_m4[a_txt_day] = dict_day_m4[a_txt_day] + 1
                else:
                    dict_day_m4[a_txt_day] = 1

        if len(rows) > 0:
            rdf = pd.DataFrame(rows)
            ordf = rdf.sort_values(by=['a_buy_date', 'b_buy_date'], ascending=[False, False])
            rdf = rdf.sort_values(by=['a_buy_date', 'b_buy_date'], ascending=[False, False])

            try:
                l_txt_year = list(rdf['a_txt_year'].unique())
                nrdf = None
                for t_txt_year in l_txt_year:
                    df = rdf[rdf['a_txt_year'] == t_txt_year]
                    v = 0
                    if t_txt_year in dict_year_m4:
                        v = dict_year_m4[t_txt_year]
                    df['a_year_cnt_m4'] = v
                    if nrdf is None:
                        nrdf = df
                    else:
                        nrdf = pd.concat([nrdf, df])
                rdf = nrdf
    
                l_txt_month = list(rdf['a_txt_month'].unique())
                nrdf = None
                for t_txt_month in l_txt_month:
                    df = rdf[rdf['a_txt_month'] == t_txt_month]
                    v = 0
                    if t_txt_month in dict_month_m4:
                        v = dict_month_m4[t_txt_month]
                    df['a_month_cnt_m4'] = v
                    if nrdf is None:
                        nrdf = df
                    else:
                        nrdf = pd.concat([nrdf, df])
                rdf = nrdf
    
                l_txt_year_month = list(rdf['a_txt_year_month'].unique())
                nrdf = None
                for t_txt_year_month in l_txt_year_month:
                    df = rdf[rdf['a_txt_year_month'] == t_txt_year_month]
                    v = 0
                    if t_txt_year_month in dict_year_month_m4:
                        v = dict_year_month_m4[t_txt_year_month]
                    df['a_year_month_cnt_m4'] = v
                    if nrdf is None:
                        nrdf = df
                    else:
                        nrdf = pd.concat([nrdf, df])
                rdf = nrdf
    
                l_txt_day = list(rdf['a_txt_day'].unique())
                nrdf = None
                for t_txt_day in l_txt_day:
                    df = rdf[rdf['a_txt_day'] == t_txt_day]
                    v = 0
                    if t_txt_day in dict_day_m4:
                        v = dict_day_m4[t_txt_day]
                    df['a_day_cnt_m4'] = v
                    if nrdf is None:
                        nrdf = df
                    else:
                        nrdf = pd.concat([nrdf, df])
                rdf = nrdf

                l_date_cnt = list(rdf['a_date_cnt'].unique())
                nrdf = None
                for t_date_cnt in l_date_cnt:
                    df = rdf[rdf['a_date_cnt'] == t_date_cnt]
                    df['a_date_cnt_same'] = len(df)
                    if nrdf is None:
                        nrdf = df
                    else:
                        nrdf = pd.concat([nrdf, df])
                rdf = nrdf

                rdf = rdf.sort_values(by=['a_date_cnt_same', 'a_date_cnt', 'a_buy_date', 'b_buy_date'], ascending=[False, True, False, False])

                cdf = rdf[rdf['a_date_cnt_same'] > 1]
                if len(cdf) == 0:
                    cdf = None
                else:
                    if catch_kind == 'previous_year':
                        xt_year = int(x_txt_year) - 1
                        cdf = cdf[(cdf['a_year'] == xt_year)]
                    elif catch_kind == 'same_date':
                        cdf = cdf[(cdf['a_txt_day'] == x_txt_day)&(cdf['a_txt_month'] == x_txt_month)]                        
                    else:
                        cdf = cdf[(cdf['a_txt_month'] == x_txt_month)]
                    if len(cdf) == 0:
                        cdf = None
            except Exception as e:
                msg = str(e)
                print(f'=> [E] {msg}')
                rdf = ordf
                
        try:
            self.save_cache()
        except Exception as e:
            msg = str(e)
            print(f'=> [E] {msg}')

        text = '''
  -------------------------------
          RESEARCH A
====================================
        '''
        if not silent:
            print(text) 
        
        return rdf, cdf

    def capture_m4pc(self, v_buy_date, data_df, runtime):
        if self.m4pcm is None:
            return None

        SEED = 311
        random.seed(SEED)
        os.environ["PYTHONHASHSEED"] = str(SEED)
        np.random.seed(SEED)
        
        rw, rdf, cdf = self.m4pc_data(v_buy_date, data_df, runtime)
        if rdf is None:
            return None

        p = self.m4pcm['model'].predict(rdf[self.m4pcm['features']])

        rdf['p'] = p
        m4pc2 = str(p[0])
        m4pc3 = int(p[0])
        m4pc = int(rdf['p'].iloc[0])
        
        print(f'== [M4PC] ==> {v_buy_date} -> {m4pc}, {m4pc2}, {m4pc3}')
        
        return m4pc
        
    def m4pc_data(self, v_buy_date, data_df, runtime):
        rdf = None
        cdf = None
        rw = None
        catch_kind = 'same_date'
        start_time = time.time()
        xdf = data_df[data_df['buy_date'] == v_buy_date]
        if len(xdf) == 0:
            return rw, rdf, cdf
        ddf = data_df[data_df['buy_date'] < v_buy_date]
        if len(ddf) == 0:
            return rw, rdf, cdf
        ddf = data_df[data_df['buy_date'] < v_buy_date]
        ardf, acdf = self.research_a(v_buy_date, None, None, data_df, 365 * 5, False, runtime, catch_kind, True)
        if ardf is None:
            return rw, rdf, cdf

        rw = {}

        min_date_cnt = 1
        max_date_cnt = 56 * 5
        df1 = ardf[(ardf['a_date_cnt'] >= min_date_cnt)&(ardf['a_date_cnt'] <= max_date_cnt)]
        rw['date_cnt_ir'] = len(df1)
        rw['date_cnt_or'] = len(ardf) - len(df1)

        rw['year_ir'] = 0
        rw['year_or'] = 0
        rw['p_year_ir'] = 0
        rw['p_year_or'] = 0
        rw['month_ir'] = 0
        rw['month_or'] = 0
        rw['day_ir'] = 0
        rw['day_or'] = 0
        rw['month_day_ir'] = 0
        rw['month_day_or'] = 0

        dcnt_step = 5
        dcnt_min = min_date_cnt
        dcnt_max = dcnt_min + dcnt_step - 1
        while dcnt_max <= max_date_cnt:
            rw[f'date_cnt_{dcnt_max}_ir'] = 0
            rw[f'date_cnt_{dcnt_max}_or'] = 0
            dcnt_min += dcnt_step
            dcnt_max = dcnt_min + dcnt_step - 1

        bdfd = v_buy_date.split('.')
        a_year = int(bdfd[0])
        a_p_year = a_year - 1
        a_month = int(bdfd[1])
        a_day = int(bdfd[2])

        if len(df1) > 0:
            df = df1[df1['a_year'] == a_year]
            rw['year_ir'] = len(df)
            rw['year_or'] = len(df1) - len(df)

            df = df1[df1['a_year'] == a_p_year]
            rw['p_year_ir'] = len(df)
            rw['p_year_or'] = len(df1) - len(df)

            df = df1[df1['a_month'] == a_month]
            rw['month_ir'] = len(df)
            rw['month_or'] = len(df1) - len(df)

            df = df1[df1['a_day'] == a_day]
            rw['day_ir'] = len(df)
            rw['day_or'] = len(df1) - len(df)

            df = df1[(df1['a_month'] == a_month)&(df1['a_day'] == a_day)]
            rw['month_day_ir'] = len(df)
            rw['month_day_or'] = len(df1) - len(df)
            
            dcnt_step = 5
            dcnt_min = min_date_cnt
            dcnt_max = dcnt_min + dcnt_step - 1
            while dcnt_max <= max_date_cnt:
                df = df1[(df1['a_date_cnt'] >= dcnt_min)&(df1['a_date_cnt'] <= dcnt_max)]
                rw[f'date_cnt_{dcnt_max}_ir'] = len(df)
                rw[f'date_cnt_{dcnt_max}_or'] = len(df1) - len(df)
                dcnt_min += dcnt_step
                dcnt_max = dcnt_min + dcnt_step - 1
        
        rdf = pd.DataFrame([rw])
        cdf = ardf
        
        return rw, rdf, cdf

    def m4pc_prepare(self, v_buy_date, buffer_dir = '/kaggle/buffers/orottick4', lotte_kind = 'p4a', data_df = None, v_date_cnt = 365 * 5, has_log_step = False, runtime = None):
        self.print_heading()

        more = {}
        
        text = '''
====================================
           M4PC PREPARE
  -------------------------------
        '''
        print(text) 

        text = '''
  -------------------------------
           PARAMETERS
  -------------------------------
        '''
        print(text) 

        catch_kind = 'previous_year'
        v_data_df_is_none = False
        if data_df is None:
            v_data_df_is_none = True
            
        print(f'[BUFFER_DIR] {buffer_dir}')
        print(f'[LOTTE_KIND] {lotte_kind}')
        print(f'[DATA_DF_IS_NONE] {v_data_df_is_none}')
        print(f'[BUY_DATE] {v_buy_date}')
        print(f'[DATE_CNT] {v_date_cnt}')
        print(f'[RUNTIME] {runtime}')
        print(f'[CATCH_KIND] {catch_kind}')

        text = '''
  -------------------------------
        '''
        print(text) 

        start_time = time.time()

        rdf = None
        cdf = None
        
        if data_df is None:
            d1 = datetime.strptime(v_buy_date, "%Y.%m.%d")
            g = -1
            d2 = d1 + timedelta(minutes=int(+(g*(60 * 24))))
            v_date = d2.strftime('%Y.%m.%d')
    
            data_df = self.download_drawing(buffer_dir, lotte_kind, v_date)
            if data_df is None:
                return rdf, cdf, more

        ddf = data_df[data_df['buy_date'] <= v_buy_date]
        if len(ddf) == 0:
            return rdf, cdf, more
            
        ddf = ddf[(ddf['w'] >= 0)&(ddf['n'] >= 0)]
        if len(ddf) == 0:
            return rdf, cdf, more

        xrdf, xcdf = self.research_a(v_buy_date, None, None, data_df, 365 * 5, False, runtime, 'same_date', True)
        xrdf = xrdf[xrdf['a_date_cnt'] <= 56*5]
        
        oddf = ddf.sort_values(by=['buy_date'], ascending=[False])
        ddf = ddf.sort_values(by=['buy_date'], ascending=[False])
        if len(ddf) > v_date_cnt:
            ddf = ddf[:v_date_cnt]
        if len(ddf) == 0:
            return rdf, cdf, more

        ddf = ddf.sort_values(by=['buy_date'], ascending=[False])

        rows = []
        dix = 0
        dcnt = 10
        dix_m4 = 0
        dcnt_m4 = 10
        dsz = len(ddf)
        if dsz < 365:
            dcnt_m4 = 1
        for ri in range(len(ddf)):
            if runtime is not None:
                if time.time() - start_time > runtime:
                    break

            a_date = ddf['date'].iloc[ri]
            a_buy_date = ddf['buy_date'].iloc[ri]
            a_next_date = ddf['next_date'].iloc[ri]
            a_w = ddf['w'].iloc[ri]
            a_n = ddf['n'].iloc[ri]
            a_runtime = None
            if runtime is not None:
                o_runtime = time.time() - start_time
                a_runtime = runtime - o_runtime

            a_m4 = 0
            a_m4pc = 0
            df = xrdf[xrdf['a_buy_date'] == a_buy_date]
            if len(df) > 0:
                a_m4 = len(df)
                a_m4pc = 1
            a_rw, a_rdf, a_cdf = self.m4pc_data(a_buy_date, oddf, a_runtime)
            if a_rdf is not None:
                more[f'rdf_{a_buy_date}'] = a_rdf
            if a_cdf is not None:
                more[f'cdf_{a_buy_date}'] = a_cdf
                
            dix += 1
            if a_m4 > 0:
                dix_m4 += 1

            if dix > 0 and dix % dcnt == 0:
                if has_log_step:
                    print(f'== [R] ==> {dix}, {dix_m4} / {dsz}')

            rw = {'date': a_date, 'buy_date': a_buy_date, 'next_date': a_next_date, 'w': a_w, 'n': a_n, 'm4pc': a_m4pc, 'm4': a_m4}
            if a_rw is not None:
                for key in a_rw.keys():
                    rw[key] = a_rw[key]
                rows.append(rw)

            if dix_m4 <= 1:
                if dix > 0 and dix <= 50:
                    if has_log_step:
                        print(str(rw))                    
                elif dix % dcnt == 0:
                    if has_log_step:
                        print(str(rw))

            if dix_m4 > 1 and dix_m4 % dcnt_m4 == 0:
                if has_log_step:
                    print(str(rw))

        if len(rows) > 0:
            rdf = pd.DataFrame(rows)
            cdf = xrdf
                
        try:
            self.save_cache()
        except Exception as e:
            msg = str(e)
            print(f'=> [E] {msg}')

        text = '''
  -------------------------------
           M4PC PREPARE
====================================
        '''
        print(text)
        
        return rdf, cdf, more
        
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
        g_sim_seed = -1
        g_sim_cnt = -1
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
                        p_w = pdf['w'].iloc[pi]
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
                for xli in range(len(l_sim_cnt)):
                    x_sim_cnt = l_sim_cnt[xli]
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
                        v_runtime = None
                        if runtime is not None:
                            vo_runtime = time.time() - start_time 
                            v_runtime = runtime - vo_runtime 
                        vm4pc2 = self.capture_m4pc(v_buy_date, data_df, v_runtime)
                        
                        lx_pred, vm4pc = self.capture_m4p(pdf, xdf['sim_seed'].iloc[0])
                        if len(lx_pred) > 0:
                            m4pc = vm4pc
                            m4p_cnt = self.m4p_cnt
                            if m4p_cnt > 0:
                                if len(lx_pred) > m4p_cnt:
                                    lx_pred = lx_pred[:m4p_cnt]
                            lx_pred = [str(x) for x in lx_pred]
                            m4_pred = ', '.join(lx_pred)
                            m4p_use = True 

                        if vm4pc2 is not None:
                            m4pc = vm4pc2
                             
                            
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
        M4P_COLLECT_DATA_DIRS = Orottick4Simulator.get_option(options, 'M4P_COLLECT_DATA_DIRS', [])
        M4P_COLLECT_SAVE_DIR = Orottick4Simulator.get_option(options, 'M4P_COLLECT_SAVE_DIR', '/kaggle/working')

        M4P_PREPARE_DATA_DIR = Orottick4Simulator.get_option(options, 'M4P_PREPARE_DATA_DIR', '/kaggle/working')
        M4P_PREPARE_SAVE_DIR = Orottick4Simulator.get_option(options, 'M4P_PREPARE_SAVE_DIR', '/kaggle/working')

        M4P_TRAIN_DATA_DIR = Orottick4Simulator.get_option(options, 'M4P_TRAIN_DATA_DIR', '/kaggle/working')
        M4P_TRAIN_SAVE_DIR = Orottick4Simulator.get_option(options, 'M4P_TRAIN_SAVE_DIR', '/kaggle/working')

        M4P_MODEL_DIR = Orottick4Simulator.get_option(options, 'M4P_MODEL_DIR', '/kaggle/working')

        M4P_RANKER_ONLY = Orottick4Simulator.get_option(options, 'M4P_RANKER_ONLY', True)

        M4P_MAX = Orottick4Simulator.get_option(options, 'M4P_MAX', 30)

        M4PL_MAX = Orottick4Simulator.get_option(options, 'M4PL_MAX', 2)
        M4PL_MIN = Orottick4Simulator.get_option(options, 'M4PL_MIN', -2)
        M4PL_STEP = Orottick4Simulator.get_option(options, 'M4PL_STEP', 0.0225)

        M4PC_TRAIN_DATA_DIR = Orottick4Simulator.get_option(options, 'M4PC_TRAIN_DATA_DIR', '/kaggle/working')
        M4PC_TRAIN_SAVE_DIR = Orottick4Simulator.get_option(options, 'M4PC_TRAIN_SAVE_DIR', '/kaggle/working')

        M4PC_MODEL_DIR = Orottick4Simulator.get_option(options, 'M4PC_MODEL_DIR', '/kaggle/working')

        TCK_PRIZE = Orottick4Simulator.get_option(options, 'TCK_PRIZE', 5000)
        BRK_COST = Orottick4Simulator.get_option(options, 'BRK_COST', 0.3)
        PREDICT_NOTEBOOK = Orottick4Simulator.get_option(options, 'PREDICT_NOTEBOOK', 'https://www.kaggle.com/code/dinhttrandrise/orottick4-predict-rsp-a-o-2025-04-06')
        PERIOD_NO = Orottick4Simulator.get_option(options, 'PERIOD_NO', 1)
        DAY_NO = Orottick4Simulator.get_option(options, 'DAY_NO', 1)
        REAL_BUY_TIMES = Orottick4Simulator.get_option(options, 'REAL_BUY_TIMES', 1)

        if non_github_create_fn is None:
            USE_GITHUB = True
            
        if USE_GITHUB:
            ok4s = github_pkg.Orottick4Simulator(PRD_SORT_ORDER, HAS_STEP_LOG, M4P_OBS, M4P_CNT, M4P_VRY, LOAD_CACHE_DIR, SAVE_CACHE_DIR)
        else:
            ok4s = non_github_create_fn(PRD_SORT_ORDER, HAS_STEP_LOG, M4P_OBS, M4P_CNT, M4P_VRY, LOAD_CACHE_DIR, SAVE_CACHE_DIR)

        ok4s.m4p_max = M4P_MAX
        ok4s.m4p_ranker_max = M4P_MAX * 2

        ok4s.m4pl_max = M4PL_MAX
        ok4s.m4pl_min = M4PL_MIN
        ok4s.m4pl_step = M4PL_STEP

        m4pm_fn = f'{M4P_MODEL_DIR}/{LOTTE_KIND}-m4pm.pkl'
        if os.path.exists(m4pm_fn):
            with open(m4pm_fn, 'rb') as f:
                ok4s.m4pm = pickle.load(f)
                if M4P_CNT >= ok4s.m4p_ranker_max:
                    ok4s.m4p_cnt = M4P_CNT
                else:
                    ok4s.m4p_cnt = ok4s.m4p_ranker_max
                ok4s.m4p_ranker_only = M4P_RANKER_ONLY
                if 'm4p_max' in ok4s.m4pm:
                    ok4s.m4p_max = ok4s.m4pm['m4p_max']
                    ok4s.m4p_ranker_max = ok4s.m4p_max * 2
                if 'm4pl_max' in ok4s.m4pm:
                    ok4s.m4pl_max = ok4s.m4pm['m4pl_max']
                if 'm4pl_min' in ok4s.m4pm:
                    ok4s.m4pl_min = ok4s.m4pm['m4pl_min']
                if 'm4pl_step' in ok4s.m4pm:
                    ok4s.m4pl_step = ok4s.m4pm['m4pl_step']


        m4pcm_fn = f'{M4PC_MODEL_DIR}/{LOTTE_KIND}-m4pcm.pkl'
        if os.path.exists(m4pcm_fn):
            with open(m4pcm_fn, 'rb') as f:
                ok4s.m4pcm = pickle.load(f)
                print('== [M4PC_MODEL] ==> Loaded!')
        
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

            rs_rb = str(json_pred['pred'])
            tck_cnt = F_TCK_CNT
            if M4P_OBS:
                tck_cnt = M4P_CNT
                rs_rb = str(json_pred['m4_pred'])
            cost = (1 + BRK_COST) * tck_cnt
            brk_cost = BRK_COST * tck_cnt
            cost_rb = REAL_BUY_TIMES * cost            
            brk_cost_rb = REAL_BUY_TIMES * brk_cost  
            m4pc = json_pred['m4pc']
            m4pc_txt = ''
            if m4pc == 0:
                cost = 0
                brk_cost = 0
                cost_rb = 0
                brk_cost_rb = 0
                m4pc_txt = '  (Not bought <- m4pc = 0)'
                
                
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

+ Predict Notebook: __PREDICT_NOTEBOOK__


  -------------------------------
              MONEY
  -------------------------------

+ Period No: __PERIOD_NO__

+ Day No: __DAY_NO__

+ Tickets: __TCK_CNT__

+ Cost: $__COST__  __M4PC_TXT__

+ Total Cost: $__COST__

+ Broker Cost: $__BRK_COST__

+ Total Broker Cost: $__BRK_COST__

+ Prize: $0

+ Total Prize: $0

+ Current ROI: 0.0%

+ Current ROI (w/o broker): 0.0%


  -------------------------------
            REAL BUY
  -------------------------------

+ Buy Number: __RS_RB__

+ Confirmation Number: 

+ Cost: $__COST_RB__  __M4PC_TXT__

+ Total Cost: $__COST_RB__

+ Broker Cost: $__BRK_COST_RB__

+ Total Broker Cost: $__BRK_COST_RB__

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
                text = text.replace('__LK__', str(LOTTE_KIND)).replace('__BD__', str(BUY_DATE)).replace('__RS__', str(json_pred['pred'])).replace('__M4__', str(json_pred['m4_pred'])).replace('__M4PC__', str(json_pred['m4pc'])).replace('__PREDICT_NOTEBOOK__', PREDICT_NOTEBOOK).replace('__PERIOD_NO__', str(PERIOD_NO)).replace('__DAY_NO__', str(DAY_NO)).replace('__TCK_CNT__', str(tck_cnt)).replace('__COST__', str(cost)).replace('__BRK_COST__', str(brk_cost)).replace('__COST_RB__', str(brk_cost_rb)).replace('__BRK_COST_RB__', str(brk_cost_rb)).replace('  __M4PC_TXT__', m4pc_txt).replace('__RS_RB__', rs_rb)
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

        if METHOD == 'm4p_prepare':
            ok4s.m4p_prepare(LOTTE_KIND, M4P_PREPARE_DATA_DIR, M4P_PREPARE_SAVE_DIR)

        if METHOD == 'm4p_train':
            ok4s.m4p_train(LOTTE_KIND, M4P_TRAIN_DATA_DIR, M4P_TRAIN_SAVE_DIR)

        if METHOD == 'research_a':
            rdf, cdf = ok4s.research_a(BUY_DATE, BUFFER_DIR, LOTTE_KIND, DATA_DF, DATE_CNT, HAS_STEP_LOG, RUNTIME)

            if rdf is not None:
                rdf.to_csv(f'{RESULT_DIR}/{LOTTE_KIND}-research-a-{BUY_DATE}.csv', index=False)

            if cdf is not None:
                cdf.to_csv(f'{RESULT_DIR}/{LOTTE_KIND}-research-a-date_cnt-{BUY_DATE}.csv', index=False)

        if METHOD == 'm4pc_prepare':
            ardf, acdf, more = ok4s.m4pc_prepare(BUY_DATE, BUFFER_DIR, LOTTE_KIND, DATA_DF, DATE_CNT, HAS_STEP_LOG, RUNTIME)

            if ardf is not None:
                ardf.to_csv(f'{RESULT_DIR}/{LOTTE_KIND}-m4pc-rdf-{BUY_DATE}.csv', index=False)

            if acdf is not None:
                acdf.to_csv(f'{RESULT_DIR}/{LOTTE_KIND}-m4pc-cdf-{BUY_DATE}.csv', index=False)

            for key in more.keys():
                adf = more[key]
                if adf is not None:
                    adf.to_csv(f'{RESULT_DIR}/{LOTTE_KIND}-m4pc-{key}-{BUY_DATE}.csv', index=False)

        if METHOD == 'm4pc_train':
            ok4s.m4pc_train(LOTTE_KIND, M4PC_TRAIN_DATA_DIR, M4PC_TRAIN_SAVE_DIR, RUNTIME)

# ------------------------------------------------------------ #
