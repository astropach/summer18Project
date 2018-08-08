# import quandl
# import datetime
 
# quandl.ApiConfig.api_key = 'RuPd2Lqns2kvB4qR_2hS'
 
# def quandl_stocks(symbol, start_date=(2000, 1, 1), end_date=None):
#     """
#     symbol is a string representing a stock symbol, e.g. 'AAPL'
 
#     start_date and end_date are tuples of integers representing the year, month,
#     and day
 
#     end_date defaults to the current date when None
#     """
 
#     query_list = ['WIKI' + '/' + symbol + '.' + str(k) for k in range(1, 13)]
 
#     start_date = datetime.date(*start_date)
 
#     if end_date:
#         end_date = datetime.date(*end_date)
#     else:
#         end_date = datetime.date.today()
 
#     return quandl.get(query_list, 
#             returns='pandas', 
#             start_date=start_date,
#             end_date=end_date,
#             collapse='daily',
#             order='asc'
#             )
 
 
# if __name__ == '__main__':
 
#     apple_data = quandl_stocks('AAPL')
#     print(apple_data)



# import pandas as pd
# import io
# import requests
# import time
 
# def google_stocks(symbol, startdate = (1, 1, 2005), enddate = None):
 
#     startdate = str(startdate[0]) + '+' + str(startdate[1]) + '+' + str(startdate[2])
 
#     if not enddate:
#         enddate = time.strftime("%m+%d+%Y")
#     else:
#         enddate = str(enddate[0]) + '+' + str(enddate[1]) + '+' + str(enddate[2])
 
#     stock_url = "http://www.google.com/finance/historical?q=" + symbol + \
#                 "&startdate=" + startdate + "&enddate=" + enddate + "&output=csv"
 
#     raw_response = requests.get(stock_url).content
 
#     stock_data = pd.read_csv(io.StringIO(raw_response.decode('utf-8')))
 
#     return stock_data
 
 
# if __name__ == '__main__':
#     apple_data = google_stocks('AAPL')
#     print(apple_data)
 
#     apple_truncated = google_stocks('AAPL', enddate = (1, 1, 2006))
#     print(apple_truncated)

import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import data as pdr

import fix_yahoo_finance as yf
yf.pdr_override() # <== that's all it takes :-)

# download dataframe
data = pdr.get_data_yahoo("RELIANCE.NS", start="2001-01-01", end="2018-06-30")

# download Panel
#data = pdr.get_data_yahoo(["SPY", "IWM"], start="2017-01-01", end="2017-04-30")

data.to_csv('rilDown.csv', sep=',')
print(data)