import numpy as np
import pandas as pd
import seaborn as sns

from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score

import random
import matplotlib.pyplot as plt

import datetime
import pytz

import statsmodels.api as sm
#import statsmodels.formula.api as smf




### stock price breath ###
russle = local_csv('russle1000.csv')
russle = russle.rename(columns={'As of 06/27/2014 Russell Indexes.': 'symbol'})
russle.symbol = russle.symbol[1:].apply(lambda x: x.split(' ')[-1])
russle = russle[~russle.symbol.isin(['Ticker', 'Indexes.'])]
russle = russle.dropna()
russle.reset_index(inplace=True, drop=True)
#This is a bit of a hack. You can call get_pricing on a big list of stocks. However, some of stocks
#in this list are listed under different tickers in Quantopian's price data. For time sake, I will use 
#this loop to drop those stocks out and avoid an exception. 
print len(russle)
russle_price = pd.DataFrame()
for sym in russle.symbol: 
    try:
        price = get_pricing(sym, fields='price', frequency='daily', start_date='2008-01-01', end_date='2015-03-31')
        russle_price[sym] = price
    except:
        print sym
        
        
        



russle_delta = russle_price.pct_change()
russle_breadth = russle_delta.apply(lambda x: x.apply(lambda y: 1 if y > 0 else -1))
total_breadth = russle_breadth.sum(axis=1)
total_breadth.plot()
plt.title('# of Advancing Stocks - # of Declining Stocks in the Russle 1000')







#I use an exponential decay moving average to smooth the data
month_rolling_breadth = pd.stats.moments.ewma(total_breadth, 10, min_periods=10)
month_rolling_breadth.plot()
plt.title('# of Advancing Stocks - # of Declining Stocks in the Russle 1000 (10 Day EWMA)')



### stock price strength ###

#We are scraping this site:
# http://online.wsj.com/mdc/public/page/2_3021-newhinyse-newhighs-20150319.html?mod=mdc_pastcalendar
# I use the Put/Call ratio csv used later in this notebook to get the dates for our scrape.

# datesold = pd.read_csv('/Users/acampbell/Downloads/equitypc.csv', skiprows=[0,1])
# datesold = datesold['DATE']

# dates = datesold.apply(lambda x: pd.to_datetime(x))
# dates = dates.apply(lambda x: x.strftime("%Y%m%d"))

# highs = []
# lows = []
# newdates = []
# for date in dates[797:]:
#     url = 'http://online.wsj.com/mdc/public/page/2_3021-newhinyse-newhighs-' + date + '.html?mod=mdc_pastcalendar'
#     x = urllib2.urlopen(url) # Opens URLS
#     htmlSource = x.read()
#     x.close()
#     soup = bs4.BeautifulSoup(htmlSource)
#     table = soup.find_all('table')[2]
#     rows = table.find_all(attrs={'colspan':'6'})
#     high = int(rows[0].string.split(' ')[3])
#     low = int(rows[1].string.split(' ')[3])
#     print high
#     print low
    
#     highs.append(high)
#     lows.append(low)
#     newdates.append(date)
    
# df = pd.DataFrame({'dates': newdates, 'highs': highs, 'lows': lows})

# df['date'] = datesold[797:].reset_index().DATE
# df[['date', 'highs', 'lows']].to_csv('52_day_highs_lows.csv')
highlows = local_csv('52_day_highs_lows.csv', date_column='date')
highlows = highlows.highs - highlows.lows
highlows.plot()
plt.title('Number 52 Week Highs Minus 52 Week Lows on the NYSE')




### junk bond demand ###

hy = local_csv('high_yield_BofA.csv', date_column='observation_date', skiprows=10)
ig = local_csv('AA_yield_BofA.csv',  skiprows=10, date_column='observation_date')
yield_spread = pd.DataFrame({'igyield': ig.BAMLC0A2CAAEY, 'highyield': hy.BAMLH0A0HYM2EY}) 
print yield_spread.highyield.isnull().sum()
print yield_spread.igyield.isnull().sum()


yield_spread['spread'] = yield_spread.highyield - yield_spread.igyield
yield_spread[['highyield', 'igyield']].plot()
plt.title('High Yield and Investment Grade Bond Yields')
plt.ylabel('%')


yield_spread.spread.plot()
plt.title('High Yeild - Investment Grade Spread')
plt.ylabel('Yield Spread %')


### market volatility ###

vix = local_csv('YAHOO-INDEX_VIX .csv', date_column='Date')
vix = vix['Adjusted Close']
vix.plot()
plt.title('VIX Index')


### put/call options ###

putcall = local_csv('equitypc.csv', skiprows=2, date_column='DATE')
putcall = putcall['P/C Ratio']
putcall = pd.stats.moments.ewma(putcall, 5, min_periods=5)
putcall.plot()
plt.title('Equity-only Put/Call Ratio')



### safe haven demand ###

spy = get_pricing('SPY', frequency='daily', fields='price', start_date='2009-01-01', end_date='2015-03-10')
treas = get_pricing('TLT', frequency='daily', fields='price', start_date='2009-01-01', end_date='2015-03-10')
returns = pd.DataFrame() 
returns['spyret'] = pd.Series(data=((spy[30:].values - spy[:-30].values)/spy[:-30].values), 
                              index=spy[30:].index)
returns['treasret'] = pd.Series(data=((treas[30:].values - treas[:-30].values)/treas[:-30].values),
                                index=treas[30:].index)
spy_treasury_spread = returns['spyret'] - returns['treasret']
spy_treasury_spread.head()

returns.plot()
plt.title('30 Day Return from SPY vs. TLT')
plt.ylabel('Return')





spy_treasury_spread.plot()
plt.title('SPY minus Treasury Bond 30 Day Return Spread')



### stock market momentum ###

spy_roll = pd.stats.moments.rolling_mean(spy, 125, min_periods=125)
spy_momentum = (spy/spy_roll).dropna()
pd.DataFrame({'Spy 125 Day Rolling': spy_roll, 'Spy Spot': spy}).plot()
plt.title('SPY')
plt.ylabel('Price')



spy_momentum.plot()
plt.title('SPY Spot/ SPY 125 Day Mean')




### combining our metrics ###

# Need to give these series a timezone so they can be combined with other TZ aware series 
yield_spread = yield_spread.tz_localize('UTC')
highlows = highlows.tz_localize('UTC')
putcall = putcall.tz_localize('UTC')
vix = vix.tz_localize('UTC')
feargreedraw = pd.DataFrame({'stock_momentum': spy_momentum, 'stock_strength': highlows, 'stock_breadth': total_breadth,
                          'putcall_ratio': putcall, 'junk_bond_demand': yield_spread.spread, 'market_volatility': vix,
                          'safe_haven_demand': spy_treasury_spread})
feargreedraw.isnull().sum()

feargreed = feargreedraw.dropna()


#We need to flip the sign of our VIX and yield spread data so that an increase in each will lower our Fear and Greed Index (an increase in the Fear and Greed index indicates greater investor confidence)
feargreed.market_volatility = feargreed.market_volatility * -1
feargreed.junk_bond_demand = feargreed.junk_bond_demand * -1 


print len(feargreed)
feargreed.head()





#I am unsure of how to best scale my factors so they are equally weighted and produce a final score
#that is comparable to the CNN index's range. 

#This is a hacky first attempt
feargreed_scaled = pd.DataFrame()
for col in feargreed.columns:
    feargreed_scaled[col] = (feargreed[col]/feargreed[col][feargreed.index[0]])*12
#Here I take the sum of the parts to get my total index value.
feargreed_scaled['mine'] = feargreed_scaled.sum(axis=1)
feargreed_scaled.head()


feargreed_scaled[feargreed_scaled.columns[:7]].plot(linewidth=.8)
plt.title('My Fear and Greed Index Components with an a Lame Attempt at Equal Weighting')






### comparing to cnn's index ###

FG_index = local_csv('Fear_and_Greed_index.csv', date_column='date')
FG_index = FG_index.value
feargreed_scaled['CNN'] = FG_index
feargreed_scaled[['CNN', 'mine']].dropna().plot(linewidth=1.2)
plt.title('CNN Fear and Greed Index vs. My Homebrew Fear and Greed Index')



feargreed_scaled['mine_smooth'] = pd.stats.moments.ewma(feargreed_scaled.mine, 4, min_periods=4)
feargreed_scaled[['CNN', 'mine_smooth']].dropna().plot(linewidth=1.5)
plt.title('CNN Fear and Greed Index vs. My Homebrew Fear and Greed Index (4 Day EWMA)')


feargreed_scaled[['CNN', 'safe_haven_demand']].dropna().plot(alpha=.8)
plt.title('CNN Fear and Greed Index vs. Spread Between Stock and Treasury Returns')



#Partially adapted from http://vincent.is/finding-trending-things/
def rolling_zscore(data, decay=0.9):
    #Lower decay = more importance of recent points in mean calculation
    avg = float(data[0])
    squared_average = float(data[0] ** 2)

    def add_to_history(point, average, sq_average):
        average = average * decay + point * (1 - decay)
        sq_average = sq_average * decay + (point ** 2) * (1 - decay)
        return average, sq_average

    def calculate_zscore(average, sq_average, value):
        std = round(np.sqrt(sq_average - avg ** 2))
        if std == 0:
            return value - average

        return (value - average) / std
    
    zscores = []
    for point in data[1:]:
        zscores.append(calculate_zscore(avg, squared_average, point))
        avg, squared_average = add_to_history(point, avg, squared_average)

    return zscores
feargreed_rolling_z = pd.DataFrame()
for col in feargreed.columns[:7]:
    feargreed_rolling_z[col] = rolling_zscore(feargreed[col], decay=.95)
    
feargreed_rolling_z = feargreed_rolling_z.set_index(feargreed.index[1:])

feargreed_rolling_z['mine'] = feargreed_rolling_z.sum(axis=1)
feargreed_rolling_z['CNN'] = FG_index
feargreed_rolling_z['mine'] = feargreed_rolling_z.mine*12 + 50
feargreed_rolling_z[feargreed_rolling_z.columns[:7]].plot(linewidth=.8)
plt.ylim(-4,4)



feargreed_rolling_z[['CNN', 'mine']].dropna().plot()
plt.title('CNN Fear and Greed Index vs. My Homebrewed Fear and Greed Index')




feargreed_rolling_z['mine_smooth'] = pd.stats.moments.ewma(feargreed_rolling_z.mine, 4, min_periods=4)
feargreed_rolling_z[['CNN', 'mine_smooth']].dropna().plot()
plt.title('CNN Fear and Greed Index vs. My Homebrew Fear and Greed Index (4 day EWMA of rolling Z-score)')
plt.ylabel('Index Value')



### preliminary assesment of the fear and greed index as a trading signal ###

feargreed_rolling_z['spy'] = spy.pct_change()
feargreed_rolling_z = feargreed_rolling_z.dropna()
predict = pd.DataFrame({'fear_greed': feargreed_rolling_z.mine_smooth.values[:-1],
                       'pricedelta': feargreed_rolling_z.spy.values[1:]})
predict.head()





plt.scatter(predict.fear_greed, predict.pricedelta)
z = np.polyfit(predict.fear_greed, predict.pricedelta, 1)
p = np.poly1d(z)
plt.plot(predict.fear_greed, p(predict.fear_greed), 'r-')
plt.ylabel('SPY Percent Price Change')
plt.xlabel('Previous Day Fear and Greed Index')








