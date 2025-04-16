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

=====>] Generate random drawing data from specified drawing data [<=====

def gen_rand_ds(bddf):
    bddf = bddf.sort_values(by=['buy_date'], ascending=[True])
    rows = []
    for ri in range(len(bddf)):
        b = int(SEED_ADD + int(bddf['n'].iloc[ri]))
        random.seed(b)
        n = random.randint(0, 9999)
        rw = {'date': bddf['date'].iloc[ri], 'w': -1, 'n': n}
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
    df.to_csv(f'{LOTTE_KIND}-{BUY_DATE}.csv', index=False)

    sz = len(df)
    print(f'== [Success] ==> Random drawing data is generated. It contains {sz} rows.')


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

=====>] Collect R4 data from pairing dataset [<=====

def r4_collect(ddf):
    ddf = ddf[ddf['a_m'] > 0]
    rows = []
    dsz = len(ddf)
    dcnt = 1000
    dix = 0
    for ri in range(len(ddf)):
        if time.time() - START_TIME > RUNTIME:
            break

        dix += 1
        if dix % dcnt == 0:
            print(f'== [R4] ==> {dix} / {dsz}')

        if R4_DIX_START is not None:
            if dix <= R4_DIX_START:
                continue
                
        no = R4_NO_START + len(rows) + 1
        w1 = ddf['a_w'].iloc[ri]
        n1 = ddf['a_n'].iloc[ri]
        w2 = ddf['b_w'].iloc[ri]
        n2 = ddf['b_n'].iloc[ri] 
        max_buy_date = ddf['b_buy_date'].iloc[ri]
        max_year = int(max_buy_date.split('.')[0]) 
        a_max_buy_date = ddf['a_buy_date'].iloc[ri]
        fds = a_max_buy_date.split('.')
        a_max_year = int(fds[0])
        a_max_year_month = '_' + fds[0] + '.' + fds[1] + '_'
        
        key = str(w1) + '_' + str(n1) + '_' + str(w2) + '_' + str(n2)
        rw = {'no': no, 'w1': w1, 'n1': n1, 'w2': w2, 'n2': n2, 'key': key, 'max_buy_date': max_buy_date, 'max_year': max_year, 'a_max_year': a_max_year, 'a_max_year_month': a_max_year_month, 'a_max_buy_date': a_max_buy_date}
        rows.append(rw)

    sdf = pd.DataFrame([{'dix': dix, 'dsz': dsz}])
    sdf.to_csv('r4-sum.csv', index=False)
    
    if len(rows) > 0:
        or4df = pd.DataFrame(rows)
        r4df = None
        key_list = list(or4df['key'].unique())
        for key in key_list:
            df = or4df[or4df['key'] == key]
            df = df.sort_values(by=['a_max_year_month', 'max_buy_date'], ascending=[True, True])
            df = df[:1]
            if r4df is None:
                r4df = df
            else:
                r4df = pd.concat([r4df, df])
        r4df['no'] = [R4_NO_START+x+1 for x in range(len(r4df))]
        r4df.to_csv('r4-data.csv', index=False)

=====>] Merge multiple R4 data files [<=====

def r4_merge(data_files):
    ddf = None
    for fn in data_files:
        if os.path.exists(fn):
            df = pd.read_csv(fn)
            if ddf is None:
                ddf = df
            else:
                ddf = pd.concat([ddf, df])
    if ddf is not None:
        ddf['no'] = [x+1 for x in range(len(ddf))]
        or4df = ddf
        r4df = None
        key_list = list(or4df['key'].unique())
        for key in key_list:
            df = or4df[or4df['key'] == key] 
            df = df.sort_values(by=['a_max_year_month', 'max_buy_date'], ascending=[True, True])
            df = df[:1]
            if r4df is None:
                r4df = df
            else:
                r4df = pd.concat([r4df, df])
        r4df = r4df.sort_values(by=['max_buy_date', 'a_max_buy_date'], ascending=[False, True])
        r4df['no'] = [x+1 for x in range(len(r4df))]
        r4df.to_csv('r4-data.csv', index=False)

        r4df = r4df.sort_values(by=['a_max_buy_date', 'max_buy_date'], ascending=[False, False])
        r4df.to_csv('r4-data-a.csv', index=False)

        lu_n = list(r4df['n1'].unique())
        u_n = len(lu_n)

        lu_w = list(r4df['w1'].unique())
        u_w = len(lu_w)

        sdf = pd.DataFrame([{'u_w': u_w, 'u_n': u_n, 'u_max': 10000}])
        sdf.to_csv('r4-unique.csv', index=False)

=====>] Analyze prediction by using R4 data [<=====

def r4_analyze(r4df, ddf):
    ddf = ddf[(ddf['w'] >= 0)&(ddf['n'] >= 0)]
    ddf['year'] = 0
    ddf['pcnt'] = 0
    ddf['pls'] = ' '
    ddf['m'] = 0
    ddf['a_max_buy_date'] = ''
    l_year = []
    l_pcnt = []
    l_pls = []
    l_m = []
    l_a_max_buy_date = []
    for ri in range(len(ddf)):
        buy_date = ddf['buy_date'].iloc[ri]
        year = int(buy_date.split('.')[0])
        n = ddf['n'].iloc[ri]
        w = ddf['w'].iloc[ri]
        a_max_buy_date = ddf['date'].iloc[ri]
        pdf = r4df[(r4df['n1'] == n)&(r4df['max_buy_date'] < buy_date)]
        if CHK_MAX_YEAR:
            if len(pdf) > 0:
                pdf = pdf[pdf['max_year'] < year] 
        if CHK_A_MAX_YEAR:
            if len(pdf) > 0:
                pdf = pdf[pdf['a_max_year'] <= year]
        if CHK_A_MAX_YEAR_MONTH:
            fds = buy_date.split('.')
            year_month = '_' + fds[0] + '.' + fds[1] + '_'
            if len(pdf) > 0:
                pdf = pdf[pdf['a_max_year_month'] <= year_month]
        pls = ' '
        m = 0
        pcnt = len(pdf)
        if len(pdf) == 0:
            m = -1
        elif len(pdf) > 0:
            ln = list(pdf['w1'].unique())
            pls = ', '.join([str(x) for x in ln])
            for pi in range(len(pdf)):
                w1 = pdf['w1'].iloc[pi]
                if w1 == w:
                    m = 1
                    a_max_buy_date = pdf['a_max_buy_date'].iloc[pi]
        l_pls.append(pls)
        l_m.append(m)
        l_pcnt.append(pcnt)
        l_year.append(year)
        l_a_max_buy_date.append(a_max_buy_date)
    ddf['pls'] = l_pls
    ddf['m'] = l_m
    ddf['pcnt'] = l_pcnt
    ddf['year'] = l_year
    ddf['a_max_buy_date'] = l_a_max_buy_date

    ddf.to_csv('r4-data.csv', index=False)

    df1 = ddf[ddf['m'] == 1]
    m_1_cnt = len(df1)
    m_1a_cnt = m_1_cnt
    if len(df1) > 0:
        df1.to_csv('r4-m-1.csv', index=False)
        df1a = df1[df1['buy_date'] > df1['a_max_buy_date']]
        m_1a_cnt = len(df1a)
        if len(df1a) > 0:
            df1a.to_csv('r4-m-1a.csv', index=False)

    df0 = ddf[ddf['m'] == 0]
    m_0_cnt = len(df0)
    if len(df0) > 0:
        df0.to_csv('r4-m-0.csv', index=False)

    dfz = ddf[ddf['m'] == -1]
    m_z_cnt = len(dfz)
    if len(dfz) > 0:
        dfz.to_csv('r4-m-z.csv', index=False)

    pcnt_max = int(ddf['pcnt'].max())
    
    all_cnt = len(ddf)

    sdf = pd.DataFrame([{'pcnt_max': pcnt_max, 'm_z_cnt': m_z_cnt, 'm_1_cnt': m_1_cnt, 'm_1a_cnt': m_1a_cnt, 'm_0_cnt': m_0_cnt, 'all_cnt': all_cnt}])
    sdf.to_csv('r4-sum.csv', index=False)

    rows = []
    year_list = list(ddf['year'].unique())
    for year in year_list:
        dfk = ddf[ddf['year'] == year]

        df1 = dfk[dfk['m'] == 1]
        m_1_cnt = len(df1)
        m_1a_cnt = m_1_cnt
        if len(df1) > 0:
            df1.to_csv(f'r4-m-1-{year}.csv', index=False) 
            df1a = df1[df1['buy_date'] > df1['a_max_buy_date']]
            m_1a_cnt = len(df1a)
            if len(df1a) > 0:
                df1a.to_csv(f'r4-m-1a-{year}.csv', index=False)

        df0 = dfk[dfk['m'] == 0]
        m_0_cnt = len(df0)
        if len(df0) > 0:
            df0.to_csv(f'r4-m-0-{year}.csv', index=False) 

        dfz = dfk[dfk['m'] == -1]
        m_z_cnt = len(dfz)
        if len(dfz) > 0:
            dfz.to_csv(f'r4-m-z-{year}.csv', index=False) 

        all_cnt = len(dfk)

        b_cnt = m_1_cnt + m_0_cnt
        cost = MP_COST * b_cnt
        prize = MP_PRIZE * m_1_cnt
        profit = prize - cost

        rw = {'year': year, 'pcnt_max': pcnt_max, 'm_z_cnt': m_z_cnt, 'm_1_cnt': m_1_cnt, 'm_1a_cnt': m_1a_cnt, 'm_0_cnt': m_0_cnt, 'all_cnt': all_cnt, 'buy_cnt': b_cnt, 'cost': cost, 'prize': prize, 'profit': profit} 
        rows.append(rw)
        sdf = pd.DataFrame([rw])
        sdf.to_csv(f'r4-m-{year}.csv', index=False)
            
    sdf = pd.DataFrame(rows)
    sdf.to_csv('r4-m.csv', index=False)


====================================
             PROCESS
  -------------------------------

  -------------------------------
   Generate random drawing data
  -------------------------------

=====>] Oregon Lottery - Pick 4 drawing [<=====

+ 1 PM:
  o ra1 (SEED_ADD = 0): https://www.kaggle.com/code/dinhttrandrise/orottick4-data-rsp-ka-ra1-2025-01-01
  o ra2 (SEED_ADD = 1): https://www.kaggle.com/code/dinhttrandrise/orottick4-data-rsp-ka-ra2-2025-01-01
  o ra3 (SEED_ADD = 2): https://www.kaggle.com/code/dinhttrandrise/orottick4-data-rsp-ka-ra3-2025-01-01
  o ra4 (SEED_ADD = 3): https://www.kaggle.com/code/dinhttrandrise/orottick4-data-rsp-ka-ra4-2025-01-01
  o ra5 (SEED_ADD = 4): https://www.kaggle.com/code/dinhttrandrise/orottick4-data-rsp-ka-ra5-2025-01-01
  o ra6 (SEED_ADD = 5): https://www.kaggle.com/code/dinhttrandrise/orottick4-data-rsp-ka-ra6-2025-01-01
  o ra7 (SEED_ADD = 6): https://www.kaggle.com/code/dinhttrandrise/orottick4-data-rsp-ka-ra7-2025-01-01
  o ra8 (SEED_ADD = 7): https://www.kaggle.com/code/dinhttrandrise/orottick4-data-rsp-ka-ra8-2025-01-01

+ 4 PM: 

+ 7 PM: 

+ 10 PM: 


  -------------------------------
             Pairing
  -------------------------------

=====>] Oregon Lottery - Pick 4 drawing [<=====

+ 1 PM: https://www.kaggle.com/code/dinhttrandrise/orottick4-pairing-rsp-ka-p4a-2-2025-01-01

+ 4 PM: https://www.kaggle.com/code/dinhttrandrise/orottick4-pairing-rsp-ka-p4b-2-2025-01-01

+ 7 PM: https://www.kaggle.com/code/dinhttrandrise/orottick4-pairing-rsp-ka-p4c-2-2025-01-01

+ 10 PM: https://www.kaggle.com/code/dinhttrandrise/orottick4-pairing-rsp-ka-p4d-2-2025-01-01


  -------------------------------
             R4 Data
  -------------------------------

=====>] Oregon Lottery - Pick 4 drawing [<=====

+ 1 PM: https://www.kaggle.com/code/dinhttrandrise/orottick4-r4-data-rsp-ka-p4a-2025-01-01

+ 4 PM: https://www.kaggle.com/code/dinhttrandrise/orottick4-r4-data-rsp-ka-p4b-2025-01-01

+ 7 PM: https://www.kaggle.com/code/dinhttrandrise/orottick4-r4-data-rsp-ka-p4c-2025-01-01

+ 10 PM: https://www.kaggle.com/code/dinhttrandrise/orottick4-r4-data-rsp-ka-p4d-2025-01-01


  -------------------------------
          Merged R4 Data
  -------------------------------

=====>] Oregon Lottery - Pick 4 drawing [<=====

+ All kinds (1PM + 4PM + 7PM + 10PM): https://www.kaggle.com/code/dinhttrandrise/orottick4-r4-data-rsp-ka-p4-2025-01-01
 

  -------------------------------
Analyze prediction by using R4 data
  -------------------------------

=====>] Oregon Lottery - Pick 4 drawing (R4 Data: All kinds [1PM + 4PM + 7PM + 10PM]) [<=====

+ 1 PM: https://www.kaggle.com/code/dinhttrandrise/orottick4-r4-analyze-rsp-ka-p4a-2025-01-01

+ 4 PM: https://www.kaggle.com/code/dinhttrandrise/orottick4-r4-analyze-rsp-ka-p4b-2025-01-01

+ 7 PM: https://www.kaggle.com/code/dinhttrandrise/orottick4-r4-analyze-rsp-ka-p4c-2025-01-01

+ 10 PM: https://www.kaggle.com/code/dinhttrandrise/orottick4-r4-analyze-rsp-ka-p4d-2025-01-01


====================================
             RESULTS
  -------------------------------

  -------------------------------
Prediction by using R4 data (All kinds [1PM + 4PM + 7PM + 10PM])
  -------------------------------

=====>] Oregon Lottery - Pick 4 drawing [<=====

+ 1 PM:
  o Possible: Yes
  o Profitable: Yes

```
![](https://github.com/dinhtt-randrise/orottick4/blob/412d3d8ffcb8fcaf5bec34bd6d4c05b9eeff3a51/research/rsp-ka/r4-p4-p4a.png)

```
+ 4 PM: 
  o Possible: Yes
  o Profitable: Yes

```
![](https://github.com/dinhtt-randrise/orottick4/blob/9c532ed503f892e73d78542e215cf542e77f8668/research/rsp-ka/r4-p4-p4b.png)

```
+ 7 PM: 
  o Possible: Yes
  o Profitable: Yes

```
![](https://github.com/dinhtt-randrise/orottick4/blob/7709ac5c240f04275dd07a3581de8e5c42eef80f/research/rsp-ka/r4-p4-p4c.png)

```
+ 10 PM: 
  o Possible: Yes
  o Profitable: Yes

```
![](https://github.com/dinhtt-randrise/orottick4/blob/92e6ec8b0174d92e3aa7fca00d44dac65a70f7c3/research/rsp-ka/r4-p4-p4d.png)

```



```
