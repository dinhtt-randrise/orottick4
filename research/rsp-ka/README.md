```
              _   _   _    _   _ _  
  ___ _ _ ___| |_| |_(_)__| |_| | | 
 / _ \ '_/ _ \  _|  _| / _| / /_  _|
 \___/_| \___/\__|\__|_\__|_\_\ |_| 
------------------------------------
RSP-KA: Random simulating with single attribute
------------------------------------

====================================
            ABSTRACT
  -------------------------------


====================================
              GOAL
  -------------------------------

Our goal is:
+ estimating if we can predict future (w/ single attribute)
using random simulating.
+ estimating if we can gain profits by predicting future (w/ single attribute)

  -------------------------------
     Future w/ single attribue
  -------------------------------

Single attribute is expressed by one integer in specified range.

Study examples of future with single attribute:

+ Oregon Lottery - Pick 4 drawing [ https://www.oregonlottery.org/jackpot/pick-4/ ]
  o 1 PM  [ https://www.kaggle.com/code/dinhttrandrise/orottick4-data-rsp-ka-p4a-2025-01-01 ]
  o 4 PM  [ https://www.kaggle.com/code/dinhttrandrise/orottick4-data-rsp-ka-p4b-2025-01-01 ]
  o 7 PM  [ https://www.kaggle.com/code/dinhttrandrise/orottick4-data-rsp-ka-p4c-2025-01-01 ]
  o 10 PM  [ https://www.kaggle.com/code/dinhttrandrise/orottick4-data-rsp-ka-p4d-2025-01-01 ]


  -------------------------------
              Data
  -------------------------------

| date | buy_date | next_date | w | n |

+ date: the date of last drawing
+ buy_date: the date for buying lottery
+ next_date: the date of next drawing
+ w: the next drawing number (0 - 9999)
+ n: the last drawing number (0 - 9999)

  -------------------------------
    Method of random simulating
  -------------------------------

=====>] Generate random number [<=====

def gen_num():
    return random.randint(0, 9999)

=====>] Reproduce random number from sim_seed & sim_cnt [<=====

def reproduce_one(sim_seed, sim_cnt):
    global cache_capture_seed, cache_reproduce_one, cache_capture

    key = f'{sim_seed}_{sim_cnt}'
    if key in cache_reproduce_one:
        return cache_reproduce_one[key]

    random.seed(sim_seed)
    n = -1
    for si in range(sim_cnt):
        n = gen_num()  

    cache_reproduce_one[key] = n
    return n

=====>] Capture sim_seed from n [<=====

def capture_seed(sim_cnt, n):
    global cache_capture_seed, cache_reproduce_one, cache_capture
    
    key = f'{sim_cnt}_{n}'
    if key in cache_capture_seed:
        return cache_capture_seed[key]

    sim_seed = 0
    p = reproduce_one(sim_seed, sim_cnt)
    while p != n:
        sim_seed += 1
        p = reproduce_one(sim_seed, sim_cnt)   

    cache_capture_seed[key] = sim_seed
    return sim_seed

=====>] Capture sim_seed, sim_cnt from w, n [<=====

def capture(w, n):
    global cache_capture_seed, cache_reproduce_one, cache_capture

    key = f'{w}_{n}'
    if key in cache_capture:
        return cache_capture[key][0], cache_capture[key][1]

    sim_seed = capture_seed(1, n)
    random.seed(sim_seed)
    sim_cnt = 0
    p = gen_num()
    sim_cnt += 1
    while w != p:
        p = gen_num()
        sim_cnt += 1
    pn = reproduce_one(sim_seed, 1)
    pw = reproduce_one(sim_seed, sim_cnt)    
    if pn == n and pw == w:
        cache_capture[key] = [sim_seed, sim_cnt]
        return sim_seed, sim_cnt
    else:
        return -1, -1

=====>] Predict next drawing number from past drawing date [<=====

def predict(buy_date, past_buy_date, data_df):
    adf = data_df[data_df['buy_date'] == buy_date]
    bdf = data_df[data_df['buy_date'] == past_buy_date]
    if len(adf) == 0 or len(bdf) == 0:
        return -1
    a_n = adf['n'].iloc[0]
    b_w = bdf['w'].iloc[0]
    b_n = bdf['n'].iloc[0]
    a_sim_seed = capture_seed(1, a_n)
    b_sim_seed, b_sim_cnt = capture(b_w, b_n)
    p = reproduce_one(a_sim_seed, b_sim_cnt)
    return p

  -------------------------------
Method of estimating if we can predict future
  -------------------------------

=====>] Pairing [<=====

def pairing(data_df):
    ddf = data_df.sort_values(by=['buy_date'], ascending=[True])
    dsz = len(ddf) * len(ddf)
    dcnt = 1000
    dix = 0
    rows = []
    for ria in range(len(ddf)):
        if time.time() - START_TIME > RUNTIME:
            break
            
        a_buy_date = ddf['buy_date'].iloc[ria]
        a_year = int(a_buy_date.split('.')[0])
        a_w = ddf['w'].iloc[ria]
        a_n = ddf['n'].iloc[ria]
        for rib in range(len(ddf)):
            if time.time() - START_TIME > RUNTIME:
                break
            
            if rib >= ria:
                break
                
            b_buy_date = ddf['buy_date'].iloc[rib]
            a_p = predict(a_buy_date, b_buy_date, data_df)
            b_w = ddf['w'].iloc[rib]
            b_n = ddf['n'].iloc[rib]
            a_m = 0
            if a_p == a_w:
                a_m = 1
            rw = {'a_buy_date': a_buy_date, 'a_year': a_year, 'a_w': a_w, 'a_n': a_n, 'a_p': a_p, 'a_m': a_m, 'b_buy_date': b_buy_date, 'b_w': b_w, 'b_n': b_n}
            rows.append(rw)

            dix += 1
            if dix > 0 and dix % dcnt == 0:
                print(f'== [P] {a_buy_date}, {b_buy_date} : {dix} / {dsz}')

    mdf = pd.DataFrame(rows)
    mdf = mdf.sort_values(by=['a_buy_date', 'b_buy_date'], ascending=[False, False])
    return mdf

=====>] Prepare past matches data [<=====

def pm_prepare(ddf):
    ddf = ddf[(ddf['w'] >= 0)&(ddf['n'] >= 0)]
    if len(ddf) == 0:
        return
    ddf = ddf[ddf['buy_date'] < LAST_BUY_DATE]
    if len(ddf) == 0:
        return

    ddf['year'] = -1
    ddf['month'] = -1
    ddf['day'] = -1
    ddf['week_num'] = -1
    ddf['weekday'] = -1
    ddf['month_day'] = ''
    ddf['sim_seed'] = -1
    ddf['sim_cnt'] = -1
    l_year = []
    l_month = []
    l_day = []
    l_week_num = []
    l_weekday = []
    l_month_day = []
    l_sim_seed = []
    l_sim_cnt = []
    for ri in range(len(ddf)):
        buy_date = ddf['buy_date'].iloc[ri]
        bdt = datetime.strptime(buy_date, "%Y.%m.%d")
        week_num, weekday = bdt.isocalendar()[1], bdt.isocalendar()[2]
        fds = buy_date.split('.')
        year = int(fds[0])
        month = int(fds[1])
        day = int(fds[2])
        month_day = '_' + str(month) + '_' + str(day) + '_' 
        sim_seed, sim_cnt = capture(ddf['w'].iloc[ri], ddf['n'].iloc[ri])
        l_year.append(year)
        l_month.append(month)
        l_day.append(day)
        l_week_num.append(week_num)
        l_weekday.append(weekday)
        l_month_day.append(month_day)
        l_sim_seed.append(sim_seed)
        l_sim_cnt.append(sim_cnt)
    ddf['sim_seed'] = l_sim_seed
    ddf['sim_cnt'] = l_sim_cnt
    ddf['year'] = l_year
    ddf['month'] = l_month
    ddf['day'] = l_day
    ddf['month_day'] = l_month_day
    ddf['week_num'] = l_week_num
    ddf['weekday'] = l_weekday
    
    ddf = ddf[(ddf['sim_seed'] >= 0)&(ddf['sim_cnt'] >= 0)]
    if len(ddf) == 0:
        return

    pm_rows = []
    sddf = ddf.sort_values(by=['buy_date'], ascending=[True])
    for ri in range(len(sddf)):
        if time.time() - START_TIME > RUNTIME:
            break
            
        buy_date = sddf['buy_date'].iloc[ri]
        bdt = datetime.strptime(buy_date, "%Y.%m.%d")
        week_num, weekday = bdt.isocalendar()[1], bdt.isocalendar()[2]
        week_info = '_' + str(week_num) + '_' + str(weekday) + '_'
        fds = buy_date.split('.')
        year = int(fds[0])
        month = int(fds[1])
        day = int(fds[2])
        month_day = '_' + str(month) + '_' + str(day) + '_'
        xdf = sddf[sddf['buy_date'] == buy_date]
        x_sim_seed = xdf['sim_seed'].iloc[0]
        x_w = xdf['w'].iloc[0]
        
        pdf = sddf[sddf['buy_date'] < buy_date]
        pdf = pdf.sort_values(by=['buy_date'], ascending=[False])

        for pi in range(len(pdf)):
            p = reproduce_one(x_sim_seed, pdf['sim_cnt'].iloc[pi])
            fds = pdf['buy_date'].iloc[pi].split('.')
            b_year = int(fds[0])
            b_month = int(fds[1])
            b_day = int(fds[2])
            a_month_day = '_' + str(month) + '_' + str(day) + '_'
            b_month_day = '_' + str(b_month) + '_' + str(b_day) + '_'
            bdt = datetime.strptime(pdf['buy_date'].iloc[pi], "%Y.%m.%d")
            b_week_num, b_weekday = bdt.isocalendar()[1], bdt.isocalendar()[2]
            b_week_info = '_' + str(b_week_num) + '_' + str(b_weekday) + '_'
            if p == x_w:
                pm_rw = {'no': len(pm_rows)+1, 'mb': 1, 'm_pos': pi + 1, 'a_buy_date': buy_date, 'a_year': year, 'a_month': month, 'a_day': day, 'a_month_day': a_month_day, 'a_week_num': week_num, 'a_weekday': weekday, 'a_week_info': week_info, 'b_buy_date': pdf['buy_date'].iloc[pi], 'b_year': b_year, 'b_month': b_month, 'b_day': b_day, 'b_month_day': b_month_day, 'b_week_num': b_week_num, 'b_weekday': b_weekday,'b_week_info': b_week_info}
                pm_rows.append(pm_rw)
            else:
                pm_rw = {'no': len(pm_rows)+1, 'mb': 0, 'm_pos': -1, 'a_buy_date': buy_date, 'a_year': year, 'a_month': month, 'a_day': day, 'a_month_day': a_month_day, 'a_week_num': week_num, 'a_weekday': weekday, 'a_week_info': week_info, 'b_buy_date': pdf['buy_date'].iloc[pi], 'b_year': b_year, 'b_month': b_month, 'b_day': b_day, 'b_month_day': b_month_day, 'b_week_num': b_week_num, 'b_weekday': b_weekday,'b_week_info': b_week_info}
                pm_rows.append(pm_rw)

    mddf = pd.DataFrame(pm_rows)
    mddf = mddf.sort_values(by=['mb', 'a_buy_date', 'b_buy_date'], ascending=[False, False, False])
    mddf.to_csv(f'{LOTTE_KIND}-match-data-{LAST_BUY_DATE}.csv', index=False)


====================================
             PROCESS
  -------------------------------


  -------------------------------
             Pairing
  -------------------------------

=====>] Oregon Lottery - Pick 4 drawing [<=====

+ 1 PM: https://www.kaggle.com/code/dinhttrandrise/orottick4-pairing-rsp-ka-p4a-2-2025-01-01

+ 4 PM: https://www.kaggle.com/code/dinhttrandrise/orottick4-pairing-rsp-ka-p4b-2-2025-01-01

+ 7 PM: https://www.kaggle.com/code/dinhttrandrise/orottick4-pairing-rsp-ka-p4c-2-2025-01-01

+ 10 PM: https://www.kaggle.com/code/dinhttrandrise/orottick4-pairing-rsp-ka-p4d-2-2025-01-01


  -------------------------------
         Past Matches Data
  -------------------------------

=====>] Oregon Lottery - Pick 4 drawing [<=====

+ 1 PM: https://www.kaggle.com/code/dinhttrandrise/orottick4-pmd-rsp-ka-p4a-2025-01-01

+ 4 PM: https://www.kaggle.com/code/dinhttrandrise/orottick4-pmd-rsp-ka-p4b-2025-01-01

+ 7 PM: 

+ 10 PM: 

```
