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

```
