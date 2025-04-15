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

Our goal is estimating if we can predict future (w/ single attribute)
using random simulating.

  -------------------------------
     Future w/ single attribue
  -------------------------------

Single attribute is expressed by one integer in specified range.

Study example of future w/ single attribute:

+ Oregon Lottery - Pick 4 drawing [ https://www.oregonlottery.org/jackpot/pick-4/ ]
  o 1 PM
  o 4 PM
  o 7 PM
  o 10 PM

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
    random.seed(sim_seed)
    n = -1
    for si in range(sim_cnt):
        n = gen_num()    
    return n

=====>] Capture sim_seed from n [<=====

def capture_seed(sim_cnt, n):
    sim_seed = 0
    p = reproduce_one(sim_seed, sim_cnt)
    while p != n:
        sim_seed += 1
        p = reproduce_one(sim_seed, sim_cnt)    
    return sim_seed

=====>] Capture sim_seed, sim_cnt from w, n [<=====

def capture(w, n):
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
    rows = []
    for ria in range(len(ddf)):
        a_buy_date = ddf['buy_date'].iloc[ria]
        a_year = int(a_buy_date.split('.')[0])
        a_w = ddf['w'].iloc[ria]
        a_n = ddf['n'].iloc[ria]
        for rib in range(len(ddf)):
            if rib <= ria:
                continue
            b_buy_date = ddf['buy_date'].iloc[rib]
            a_p = predict(a_buy_date, b_buy_date, data_df)
            b_w = ddf['w'].iloc[rib]
            b_n = ddf['n'].iloc[rib]
            a_m = 0
            if a_p == a_w:
                a_m = 1
            rw = {'a_buy_date': a_buy_date, 'a_w': a_w, 'a_n': a_n, 'a_p': a_p, 'a_m': a_m, 'b_buy_date': b_buy_date, 'b_w': b_w, 'b_n': b_n}
            rows.append(rw)
    mdf = pd.DataFrame(rows)
    mdf = mdf.sort_values(by=['a_buy_date', 'b_buy_date'], ascending=[False, False])
    return mdf


```
