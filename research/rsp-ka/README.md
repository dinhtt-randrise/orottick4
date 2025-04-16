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
+ Nasdaq - Amazon.com stock price [ https://www.nasdaq.com/market-activity/stocks/amzn/historical ]
  o Open price  [ https://www.kaggle.com/code/dinhttrandrise/orottick4-data-rsp-ka-s1o-2025-01-01 ]
  o Close price  [ https://www.kaggle.com/code/dinhttrandrise/orottick4-data-rsp-ka-s1c-2025-01-01 ]
  o High price  [ https://www.kaggle.com/code/dinhttrandrise/orottick4-data-rsp-ka-s1h-2025-01-01 ]
  o Low price  [ https://www.kaggle.com/code/dinhttrandrise/orottick4-data-rsp-ka-s1l-2025-01-01 ]
+ Nasdaq - Microsoft stock price [ https://www.nasdaq.com/market-activity/stocks/msft/historical ]
  o Open price  [ https://www.kaggle.com/code/dinhttrandrise/orottick4-data-rsp-ka-s2o-2025-01-01 ]
  o Close price  [ https://www.kaggle.com/code/dinhttrandrise/orottick4-data-rsp-ka-s2c-2025-01-01 ]
  o High price  [ https://www.kaggle.com/code/dinhttrandrise/orottick4-data-rsp-ka-s2h-2025-01-01 ]
  o Low price  [ https://www.kaggle.com/code/dinhttrandrise/orottick4-data-rsp-ka-s2l-2025-01-01 ]

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

=====>] Analyze pairing dataset in one year [<=====

def load_cm_model():
    global CM_MODEL

    CM_MODEL = None
    if CM_MODEL_FILE is not None:
        if os.path.exists(CM_MODEL_FILE):
            with open(CM_MODEL_FILE, 'rb') as f:
                CM_MODEL = pickle.load(f)
                print(f'== [CM_MODEL] ==> Loaded!')

def calc_cv_tax_rate(cv):
    rate = 0.15

    if cv < 4400:
        return rate + 0.0475
    elif cv < 11050:
        return rate + 0.0675
    elif cv < 125000:
        return rate + 0.0875
    else:
        return rate + 0.0990
        
def calc_roi(a_buy_date, a_date_cnt, a_pos, dfb, ddf):
    if INVEST_KIND == 'MP':
        df1 = dfb[dfb['a_m'] > 0]
        a_m_cnt = len(df1)
        a_cost = MP_COST * a_date_cnt
        a_prize = a_m_cnt * MP_PRIZE
        a_return = a_prize - a_cost
        if a_pos == 0:
            a_m_cnt = 0
            a_cost = 0
            a_prize = 0
            a_return = 0
    
        return a_cost, a_prize, a_return
    elif INVEST_KIND == 'CV':
        if a_pos == 0:
            return 0, 0, 0            
        tddf = ddf.sort_values(by=['a_buy_date', 'b_buy_date'], ascending=[False, False])
        tadf = tddf[tddf['a_buy_date'] == a_buy_date]
        tbdf = tddf[tddf['a_buy_date'] < a_buy_date]
        if len(tadf) == 0 or len(tbdf) == 0:
            return 0, 0, 0
        b_date_list = list(tbdf['a_buy_date'].unique())
        if len(b_date_list) < CV_CYCLE_SIZE:
            return 0, 0, 0
        b_date_list = b_date_list[CV_CYCLE_SIZE:]
        if len(b_date_list) == 0:
            return 0, 0, 0            
        b_buy_date = b_date_list[0]
        tbdf = tddf[tddf['a_buy_date'] == b_buy_date]
        if len(tbdf) == 0:
            return 0, 0, 0
        b_w = tbdf['a_w'].iloc[0] * CV_ADJUST_RATE * CV_STOCK_CNT
        aa_cost = 0
        aa_prize = 0
        aa_return = 0
        for ria in range(len(dfb)):
            a_w = dfb['a_p'].iloc[ria] * CV_ADJUST_RATE * CV_STOCK_CNT
            a_cost = b_w
            a_prize = a_w - b_w
            if a_prize > 0:
                a_tax = calc_cv_tax_rate(a_prize) * a_prize
                a_prize = 0
                a_return = a_prize
            else:
                a_return = a_prize
                a_prize = 0
            aa_cost += a_cost
            aa_prize += a_prize
            aa_return += a_return
        return aa_cost, aa_prize, aa_return            
    else:
        return 0, 0, 0
        
def check_match_possible(a_buy_date, vddf):
    possible = 1
    pdf = None
    if CM_MODEL is None:
        return possible, pdf
    else:
        return possible, pdf
        
def analyze_year(year, vddf):
    ddf = vddf[vddf['a_year'] == year]
    if len(ddf) == 0:
        return None, None
    a_buy_date_list = list(ddf['a_buy_date'].unique())
    if len(a_buy_date_list) < PAIR_DATE_CNT_MIN:
        return None, None
    rows = []
    for a_buy_date in a_buy_date_list:
        a_m_cnt = 0
        dfa = ddf[ddf['a_buy_date'] == a_buy_date]
        if len(dfa) > 0:
            df1 = dfa[dfa['a_m'] > 0]
            a_m_cnt = len(df1)
        rw = {'a_buy_date': a_buy_date, 'a_m_cnt': a_m_cnt, 'a_pos': 0}
        if len(dfa) > 0:
            dfa = dfa.sort_values(by=['b_buy_date'], ascending=[False])
            possible, dfc = check_match_possible(a_buy_date, ddf)
            if dfc is not None:
                if len(dfc) > 0:
                    dfa = dfc
            a_pos = possible
            if a_pos > 0:
                if 'a_pos' in dfa.columns:
                    dfp = dfa[dfa['a_pos'] > 0]
                    if len(dfp) == 0:
                        a_pos = 0
            rw['a_pos'] = a_pos
            aa_pos = a_pos
            a_date_cnt = DATE_CNT_MIN
            while a_date_cnt <= DATE_CNT_MAX:
                df0 = dfa
                if len(df0) >= a_date_cnt:
                    df0 = df0[:a_date_cnt]
                a_pos = aa_pos
                if a_pos > 0:
                    if 'a_pos' in df0.columns:
                        dfp = df0[df0['a_pos'] > 0]
                        if len(dfp) == 0:
                            a_pos = 0
                df1 = df0[df0['a_m'] > 0]
                a_m_cnt = len(df1)
                a_cost, a_prize, a_return = calc_roi(a_buy_date, a_date_cnt, a_pos, df0, ddf)
                rw[f'a_pos_{a_date_cnt}'] = a_pos
                rw[f'a_m_cnt_{a_date_cnt}'] = a_m_cnt
                rw[f'a_cost_{a_date_cnt}'] = a_cost
                rw[f'a_prize_{a_date_cnt}'] = a_prize
                rw[f'a_return_{a_date_cnt}'] = a_return
                a_date_cnt += DATE_CNT_STEP
        rows.append(rw)
    cdf = pd.DataFrame(rows)
    cdf.to_csv(f'match-count-{year}.csv', index=False)
    df1 = cdf[cdf['a_m_cnt'] > 0]
    a_m_1_cnt = len(df1)
    df0 = cdf[cdf['a_m_cnt'] == 0]
    a_m_0_cnt = len(df0)
    dfz = cdf[cdf[f'a_pos'] > 0]
    a_pos_cnt = len(dfz)
    rw = {'a_year': year, 'a_m_1_cnt': a_m_1_cnt, 'a_m_0_cnt': a_m_0_cnt, 'a_pos_cnt': a_pos_cnt}
    a_date_cnt = DATE_CNT_MIN
    while a_date_cnt <= DATE_CNT_MAX:
        df1 = cdf[cdf[f'a_m_cnt_{a_date_cnt}'] > 0]
        rw[f'a_m_1_cnt_{a_date_cnt}'] = len(df1)
        df0 = cdf[cdf[f'a_m_cnt_{a_date_cnt}'] == 0]
        rw[f'a_m_0_cnt_{a_date_cnt}'] = len(df0)
        a_cost = cdf[f'a_cost_{a_date_cnt}'].sum()
        a_prize = cdf[f'a_prize_{a_date_cnt}'].sum()
        a_return = cdf[f'a_return_{a_date_cnt}'].sum()
        a_pos_cnt = cdf[f'a_pos_{a_date_cnt}'].sum()
        rw[f'a_pos_cnt_{a_date_cnt}'] = a_pos_cnt
        rw[f'a_cost_{a_date_cnt}'] = a_cost
        rw[f'a_prize_{a_date_cnt}'] = a_prize
        rw[f'a_return_{a_date_cnt}'] = a_return
        a_date_cnt += DATE_CNT_STEP
    sdf = pd.DataFrame([rw])
    sdf.to_csv(f'match-in-year-{year}.csv', index=False)
    return cdf, sdf

=====>] Analyze pairing dataset in year range [<=====

def analyze_year_range(ddf):
    a_year_list = []
    for ri in range(len(ddf)):
        a_year = int(str(ddf['a_buy_date'].iloc[ri]).split('.')[0])
        a_year_list.append(a_year)
    ddf['a_year'] = a_year_list
    year_start = YEAR_MAX
    year_end = YEAR_MIN
    year_step = -YEAR_STEP
    year = year_start
    asdf = None
    while year >= year_end:
        cdf, sdf = analyze_year(year, ddf)
        if sdf is not None:
            if asdf is None:
                asdf = sdf
            else:
                asdf = pd.concat([asdf, sdf])
        year += year_step
    if asdf is not None:
        asdf = asdf.sort_values(by=['a_year'], ascending=[False])
        asdf.to_csv(f'match-in-year.csv', index=False)
        rows = []
        a_date_cnt = DATE_CNT_MIN
        while a_date_cnt <= DATE_CNT_MAX:
            a_cost = asdf[f'a_cost_{a_date_cnt}'].sum()
            a_prize = asdf[f'a_prize_{a_date_cnt}'].sum()
            a_return = asdf[f'a_return_{a_date_cnt}'].sum()
            a_m_1_cnt = asdf[f'a_m_1_cnt_{a_date_cnt}'].sum()
            a_m_0_cnt = asdf[f'a_m_0_cnt_{a_date_cnt}'].sum()
            a_pos_cnt = asdf[f'a_pos_cnt_{a_date_cnt}'].sum()
            df1 = asdf[asdf[f'a_return_{a_date_cnt}'] >= RETURN_IN_YEAR_MIN]
            a_riy_1_cnt = len(df1)
            df0 = asdf[asdf[f'a_return_{a_date_cnt}'] < RETURN_IN_YEAR_MIN]
            a_riy_0_cnt = len(df0)
            rw = {'a_date_cnt': a_date_cnt, 'a_m_1_cnt': a_m_1_cnt, 'a_m_0_cnt': a_m_0_cnt, 'a_pos_cnt': a_pos_cnt, 'a_cost': a_cost, 'a_prize': a_prize, 'a_return': a_return, 'a_riy_1_cnt': a_riy_1_cnt, 'a_riy_0_cnt': a_riy_0_cnt}
            rows.append(rw)
            a_date_cnt += DATE_CNT_STEP
        amdf = pd.DataFrame(rows)
        amdf = amdf.sort_values(by=['a_riy_1_cnt', 'a_return', 'a_cost', 'a_date_cnt'], ascending=[False, False, True, True])
        amdf.to_csv(f'matches.csv', index=False)

        possible = 'No'
        df1 = amdf[amdf['a_m_1_cnt'] >= amdf['a_m_0_cnt'] * M_1_CNT_RATE]
        if len(df1) > 0:
            possible = 'Yes'
        profitable = 'No'
        df2 = amdf[amdf['a_return'] >= RETURN_ALL_MIN]
        if len(df2) > 0:
            profitable = 'Yes'
        arsdf = pd.DataFrame({'key': ['Possible', 'Profitable'], 'value': [possible, profitable]})
        arsdf.to_csv('results.csv', index=False)


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

=====>] Nasdaq - Amazon.com stock price [<=====

+ Open price: https://www.kaggle.com/code/dinhttrandrise/orottick4-pairing-rsp-ka-s1o-1-2025-01-01

+ Close price: https://www.kaggle.com/code/dinhttrandrise/orottick4-pairing-rsp-ka-s1c-1-2025-01-01

+ High price: https://www.kaggle.com/code/dinhttrandrise/orottick4-pairing-rsp-ka-s1h-1-2025-01-01

+ Low price: https://www.kaggle.com/code/dinhttrandrise/orottick4-pairing-rsp-ka-s1l-1-2025-01-01


  -------------------------------
   Analyze (Not check matchable)
  -------------------------------

=====>] Oregon Lottery - Pick 4 drawing [<=====

+ 1 PM: https://www.kaggle.com/code/dinhttrandrise/orottick4-analyze-rsp-ka-p4a-nc-2025-01-01

+ 4 PM: https://www.kaggle.com/code/dinhttrandrise/orottick4-analyze-rsp-ka-p4b-nc-2025-01-01

+ 7 PM: https://www.kaggle.com/code/dinhttrandrise/orottick4-analyze-rsp-ka-p4c-nc-2025-01-01

+ 10 PM: https://www.kaggle.com/code/dinhttrandrise/orottick4-analyze-rsp-ka-p4d-nc-2025-01-01


====================================
             RESULTS
  -------------------------------

  -------------------------------
       Not check matchable
  -------------------------------

=====>] Oregon Lottery - Pick 4 drawing [<=====

+ 1 PM:
  o Possible: Yes
  o Profitable: No

+ 4 PM: 

+ 7 PM: 

+ 10 PM: 


```
