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

# ------------------------------------------------------------ #

class Orottick4Simulator:
    def __init__(self, prd_sort_order = 'A', has_step_log = True, heading_printed = False):
        self.min_num = 0
        self.max_num = 9999

        self.baseset = {0: 1000, 1: 100, 2: 10, 3: 1}

        self.heading_printed = heading_printed

        if prd_sort_order not in ['A', 'B', 'C']:
            prd_sort_order = 'A'
        self.prd_sort_order = prd_sort_order

        self.has_step_log = has_step_log
        
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
        sim_seed = 0
        p = self.reproduce_one(sim_seed, sim_cnt)
        while p != n:
            sim_seed += 1
            p = self.reproduce_one(sim_seed, sim_cnt)
        return sim_seed

    def capture(self, w, n):
        sim_seed = self.capture_seed(1, n)
        random.seed(sim_seed)
        
        sim_cnt = 0
        p = self.gen_num()
        sim_cnt += 1
        while p != w:
            p = self.gen_num()
            sim_cnt += 1

        pn = self.reproduce_one(sim_seed, 1)
        pw = self.reproduce_one(sim_seed, sim_cnt)

        if pn == n and pw == w:
            return sim_seed, sim_cnt
        else:
            return -1, -1
            
    def reproduce_one(self, sim_seed, sim_cnt):
        random.seed(sim_seed)
        n = -1
        for si in range(sim_cnt):
            n = self.gen_num()
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
        ddf = data_df[data_df['date'] <= v_buy_date]
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
        json_pred = None
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
            json_pred = {'date': xdf['date'].iloc[0], 'buy_date': xdf['buy_date'].iloc[0], 'next_date': xdf['next_date'].iloc[0], 'w': int(xdf['w'].iloc[0]), 'n': int(xdf['n'].iloc[0]), 'sim_seed': int(xdf['sim_seed'].iloc[0]), 'date_cnt': v_date_cnt, 'tck_cnt': tck_cnt, 'sim_cnt': s_sim_cnt, 'pred': s_pred, 'pcnt': 1, 'm4': int(xdf['a_m4'].iloc[0]), 'm3f': int(xdf['a_m3f'].iloc[0]), 'm3l': int(xdf['a_m3l'].iloc[0]), 'm3': int(xdf['a_m3'].iloc[0]), 'm2': int(xdf['a_m2'].iloc[0]), 'm4_cnt': int(xdf['m4_cnt'].iloc[0]), 'm3f_cnt': int(xdf['m3f_cnt'].iloc[0]), 'm3l_cnt': int(xdf['m3l_cnt'].iloc[0]), 'm3_cnt': int(xdf['m3_cnt'].iloc[0]), 'm2_cnt': int(xdf['m2_cnt'].iloc[0]), 'mb_m4': mb_m4, 'mb_m3f': mb_m3f, 'mb_m3l': mb_m3l, 'mb_m3': mb_m3, 'mb_m2': mb_m2}
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
                if self.match(t_w, t_p, 'm4'):
                    m4 += 1
                if self.match(t_w, t_p, 'm3f'):
                    m3f += 1
                if self.match(t_w, t_p, 'm3l'):
                    m3l += 1
                if self.match(t_w, t_p, 'm3'):
                    m3 += 1
                if self.match(t_w, t_p, 'm2'):
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

# ------------------------------------------------------------ #
