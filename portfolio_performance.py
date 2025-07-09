import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

CLOSING22 = pd.read_csv('closing22.csv').set_index('Date')
SPY22 = pd.read_csv('spy_22.csv').set_index('Date')


def div_by_first_row(df):
    """ Divides whole dataframe by the first row """
    return df.div(df.iloc[0])


def graph_top_performance(dfs, funds=1000):
    pct_return = {}

    spy = div_by_first_row(SPY22.copy())
    spy = spy.mul(funds).reset_index()
    spy['Date'] = pd.to_datetime(spy['Date'], infer_datetime_format=True)

    for obj, sols in dfs.items():
        df = sols[1]

        df['Allocation'] = (df['Pct Allocation'] / 100) * funds

        init_invest = np.transpose(df['Allocation'].to_numpy())

        # Copy to mult
        portfolio = div_by_first_row(CLOSING22.copy())

        portfolio = portfolio.mul(init_invest).apply(sum, axis=1).reset_index().rename(columns={0: 'Adj Close'})
        portfolio['Date'] = pd.to_datetime(portfolio['Date'], infer_datetime_format=True)

        pct_return[obj] = \
            (portfolio['Adj Close'].iloc[-1] - portfolio['Adj Close'].iloc[0]) / portfolio['Adj Close'].iloc[0]

        plt.plot(portfolio['Date'], portfolio['Adj Close'], label='Best ' + obj + ': ' + str(round(abs(sols[0]), 2)))
    plt.plot(spy['Date'], spy['Adj Close'], label='SPY Index')
    plt.xlabel('Date')
    plt.ylabel('Investment (USD)')
    plt.title('Performance of $' + str(funds) + ' Invested in Chosen Portfolio vs SPY Index Fund Over 2022')
    plt.legend()
    plt.show()

    pct_return['SPY'] = \
        (spy['Adj Close'].iloc[-1] - spy['Adj Close'].iloc[0]) / spy['Adj Close'].iloc[0]

    print(pct_return)


def graph_performance(sols, funds, **kwargs):
    spy = div_by_first_row(SPY22.copy())
    spy = spy.mul(funds).reset_index()
    spy['Date'] = pd.to_datetime(spy['Date'], infer_datetime_format=True)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=spy['Date'],
        y=spy['Adj Close'],
        name='SPY Index',
        mode="lines", ))

    legend = kwargs.get('legend', None)
    i = 0
    for obj, sol in sols.items():
        sol['Allocation'] = (sol['Pct Allocation'] / 100) * funds

        init_invest = np.transpose(sol['Allocation'].to_numpy())

        # Copy to mult
        portfolio = div_by_first_row(CLOSING22.copy())

        portfolio = portfolio.mul(init_invest).apply(sum, axis=1).reset_index().rename(columns={0: 'Adj Close'})
        portfolio['Date'] = pd.to_datetime(portfolio['Date'], infer_datetime_format=True)

        name = str(obj)
        if legend != None:
            name = legend[i]

        fig.add_trace(go.Scatter(
            x=portfolio['Date'],
            y=portfolio['Adj Close'],
            name=name,
            mode="lines", ))
        i += 1

    fig.update_layout(title='What Investing $' + str(funds) + ' at the Beginning of 2022 Would Have Looked Like:',
                      xaxis_title='Date',
                      yaxis_title='Investment (USD)',
                      showlegend=True)

    return fig
