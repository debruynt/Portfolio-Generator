import pandas as pd
import yfinance as yf
from datetime import datetime as dt


def get_market_cap(tickers):
    yf_tickers = [yf.Ticker(ticker) for ticker in tickers]
    market_cap = [ticker.info['marketCap'] for ticker in yf_tickers]

    return market_cap


def get_closings(tickers, start_date='01/01/2021', end_date='12/31/2021'):
    start = dt.strptime(start_date, '%m/%d/%Y')
    end = dt.strptime(end_date, '%m/%d/%Y')
    closings = yf.download(tickers, start, end)['Adj Close']

    return closings


def main():
    # Load in the data
    data = pd.read_html('https://en.wikipedia.org/wiki/List_of_S&P_500_companies')
    stock_data = data[0]
    stock_data = stock_data[['Symbol', 'GICS Sector', 'GICS Sub-Industry']]

    # Market Cap
    stock_data['Market Cap'] = get_market_cap(stock_data['Symbol'])

    # Filter to top 100 market cap
    stock_data = stock_data.sort_values(by=['Market Cap'], ascending=False)[:100]

    # Get closings for 2021 and 2022
    tickers = stock_data['Symbol'].tolist()
    closing_21 = get_closings(tickers)
    closing_22 = get_closings(tickers, start_date='01/01/2022', end_date='12/31/2022')

    # Get closings for SPY (S&P 500 Index ETF)
    spy_21 = get_closings(['SPY'])
    spy_22 = get_closings(['SPY'], start_date='01/01/2022', end_date='12/31/2022')

    # Save as CSVs
    stock_data.to_csv('ticker_data.csv', encoding='utf-8', index=False)
    closing_21.to_csv('closing21.csv', encoding='utf-8', index=False)
    closing_22.to_csv('closing22.csv', encoding='utf-8', index=True)
    spy_21.to_csv('spy_21.csv', encoding='utf-8', index=False)
    spy_22.to_csv('spy_22.csv', encoding='utf-8', index=True)



if __name__ == '__main__':
    main()