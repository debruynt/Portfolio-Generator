"""
file: portfolio_generator

description:
"""

import evo
import numpy as np
import pandas as pd
import statistics as stats
import random as rnd
import portfolio_performance as pp
import pickle
import warnings
# warnings.simplefilter(action='ignore', category=[FutureWarning, DeprecationWarning])

STOCK_DATA = pd.read_csv('ticker_data.csv')
CLOSING21 = pd.read_csv('closing21.csv')
SPY21 = pd.read_csv('spy_21.csv')

# Calculate variance and return of SPY fund
spy_var = stats.variance(SPY21['Adj Close'])
spy_return = (SPY21['Adj Close'].iloc[-1] - SPY21['Adj Close'].iloc[0]) / SPY21['Adj Close'].iloc[0]
risk_free_rate = 0.03  # Assuming rfr does not change from https://www.kroll.com/en/insights/publications/cost-of-capital/kroll-us-normalized-risk-free-rate-increased-april-2022
seed = STOCK_DATA.copy()
seed['Pct Allocation'] = 1


def stock_covariances(df):
    """ Calculates the covariance of stock returns """
    # Get stock pct changes day to day
    pct_change = df.copy().pct_change()

    # Remove first row
    pct_change = pct_change.iloc[1:, :]

    return pct_change.cov()


# Make stock_cov a global var
stock_cov = stock_covariances(CLOSING21)


def calculate_beta(ticker):
    """ Calculates a stock's volatility relative to the market """
    spy_cov = SPY21['Adj Close'].cov(CLOSING21[ticker])

    return spy_var / spy_cov


def calculate_return(ticker):
    """ Calculates the % return of a stock during 2021 """
    return (CLOSING21[ticker].iloc[-1] - CLOSING21[ticker].iloc[0]) / CLOSING21[ticker].iloc[0]


def calculate_alpha(ticker, ticker_return=False, beta=False):
    """ Calculates returns of a stock relative to market performance """

    if ticker_return is False:
        ticker_return = calculate_return(ticker)

    if beta is False:
        beta = calculate_beta(ticker)

    return ticker_return - risk_free_rate - (beta * (spy_return - risk_free_rate))


def calculate_risk(ticker):
    """ Calculates risk of a stock as a form of standard deviation """
    pct_change = CLOSING21[ticker].pct_change()
    return stats.stdev(pct_change.iloc[1:]) * 250


def load_finance_data(df):
    stock_data = df.copy()

    stock_data['beta'] = stock_data.apply(lambda x: calculate_beta(x['Symbol']), axis=1)
    stock_data['return'] = stock_data.apply(lambda x: calculate_return(x['Symbol']), axis=1)
    stock_data['alpha'] = stock_data.apply(lambda x: calculate_alpha(x['Symbol'], x['return'], x['beta']), axis=1)
    stock_data['risk'] = stock_data.apply(lambda x: calculate_risk(x['Symbol']), axis=1)

    return stock_data


def portfolio_risk(sol):
    # Convert weights and covariances to numpy arrays
    weights = (sol['Pct Allocation'].copy() / 100).to_numpy()
    covariances = stock_cov.to_numpy()
    # Multiply covariances by number of days
    covariances = covariances * 250

    return (np.matmul(np.transpose(weights), np.matmul(covariances, weights))) ** 0.5


def diversity_wide(sol):
    """ calculates a diversity score for a portfolio by checking the number of stocks with greater than 5% allocation"""
    sol1 = sol.copy()
    return - len(sol1[(sol1['Pct Allocation']) > 5])


# supposed to calculate a diversity score for a portfolio by checking the number of stocks (ordered from large to small)
# that it takes to equal 20% of the Pct Allocation
def diversity_tall(sol):
    """ calculate a diversity score for a portfolio by checking the number of stocks (ordered from large to small)
        that it takes to equal 20% of the Pct Allocation """
    sol1 = sol.copy()
    rng = list(range(len(sol1) + 1))

    for x in rng:
        sol2 = sol1.nlargest(x+1, 'Pct Allocation')
        if sol2['Pct Allocation'].sum() > 20:
            return -(x+1)

# other version that I tried of diversity_tall
    """if sol2['Pct Allocation'].sum() > 20:
        return -i
    else:
        diversity_tall(sol1, i + 1)"""


def sharpe_ratio(sol):
    expected_return = sum((sol['Pct Allocation'] / 100) * sol['return'])
    portfolio_stdev = portfolio_risk(sol)

    return - ((expected_return - risk_free_rate) / portfolio_stdev)


def portfolio_alpha(sol):
    """ calculates weighted alpha of portfolio, returns the negative alpha
    (to maximize the actual alpha while minimizing the returned value) """
    sol = sol.copy()
    return - (np.sum(sol['alpha'] * (sol['Pct Allocation'] / 100)))


def treynor_ratio(sol):
    """ Calculates a solution's Treynor Ratio """
    return - ((np.sum(sol['return'] * sol['Pct Allocation']) - (risk_free_rate * 100)) / \
               np.sum(sol['beta'] * sol['Pct Allocation'] / 100))


def industry_diversification(sol):
    ind_alloc = sol.groupby('GICS Sub-Industry').sum(numeric_only=False)
    return - len(ind_alloc[ind_alloc['Pct Allocation'].between(1, 10)])


def mutate(sol):
    """ Reallocates a percent allocation from one random stock to another """
    N = sol[0].copy()
    mutes = [rnd.randrange(0, len(N.index)) for _ in range(2)]

    if N['Pct Allocation'][mutes[0]] == 0:
        return N

    else:
        N.loc[mutes[0]:mutes[0], 'Pct Allocation'] -= 1
        N.loc[mutes[1]:mutes[1], 'Pct Allocation'] += 1
        return N


def swapper(sol):
    """ Swaps two stock's allocations """
    N = sol[0].copy()
    swaps = [rnd.randrange(0, len(N.index)) for _ in range(2)]
    N.loc[swaps[0]:swaps[0], 'Pct Allocation'] = sol[0]['Pct Allocation'][swaps[1]]
    N.loc[swaps[1]:swaps[1], 'Pct Allocation'] = sol[0]['Pct Allocation'][swaps[0]]

    return N


def dump(sol):
    """ Dumps total allocation of one random stock into another """
    N = sol[0].copy()
    dumps = [rnd.randrange(0, len(N.index)) for _ in range(2)]
    N.loc[dumps[0]:dumps[0], 'Pct Allocation'] = N['Pct Allocation'][dumps[0]] + sol[0]['Pct Allocation'][dumps[1]]
    N.loc[dumps[1]:dumps[1], 'Pct Allocation'] = 0

    return N


def avg(sol):
    """ Returns the average percent allocation of each stock across two random solutions as a new solution """
    a = rnd.randrange(0, len(sol))
    b = rnd.randrange(0, len(sol))
    if a == b:
        return sol[0]
    else:
        sol1 = sol.copy(sol[a])
        sol2 = sol.copy(sol[b])
        sol1['Pct Allocation'] = (sol1['Pct Allocation'] + sol2['Pct Allocation']) / 2
        return sol1
    

def lower_risk(sol):
    """ locates stock with risks higher/lower than mean, allocates all pct from lowest to highest """
    df = sol[0]

    min_risk = df[df['risk'] < df['risk'].mean()].sample()  # Symbol
    max_risk = df[df['risk'] > df['risk'].mean()].sample()

    df.loc[df['Symbol'] == min_risk.iloc[0, 0], 'Pct Allocation'] += max_risk.iloc[0, -1]
    df.loc[df['Symbol'] == max_risk.iloc[0, 0], 'Pct Allocation'] = 0

    return df


def higher_alpha(sol):
    """ locates stock with alpha higher/lower than mean, allocates all pct from highest to lowest """
    df = sol[0]

    min_alpha = df[df['alpha'] < df['alpha'].mean()].sample()  # Symbol
    max_alpha = df[df['alpha'] > df['alpha'].mean()].sample()

    df.loc[df['Symbol'] == max_alpha.iloc[0, 0], 'Pct Allocation'] += min_alpha.iloc[0, -1]
    df.loc[df['Symbol'] == min_alpha.iloc[0, 0], 'Pct Allocation'] = 0

    return df


def main():
    # Load in data
    stock_data = load_finance_data(STOCK_DATA)

    # create population
    E = evo.Environment()

    # register the fitness criteria (objects)
    E.add_fitness_criteria('portfolio risk', portfolio_risk)
    E.add_fitness_criteria('sharpe ratio', sharpe_ratio)
    E.add_fitness_criteria('portfolio alpha', portfolio_alpha)
    E.add_fitness_criteria('treynor ratio', treynor_ratio)
    E.add_fitness_criteria('industry diversification', industry_diversification)
    E.add_fitness_criteria('diversity wide', diversity_wide)
    E.add_fitness_criteria('diversity tall', diversity_tall)

    # register all agents
    E.add_agent('mutate', mutate)
    E.add_agent('swapper', swapper)
    E.add_agent('dump', dump)
    E.add_agent('avg', avg)    
    E.add_agent('alpha', higher_alpha)
    E.add_agent('risk', lower_risk)


    # seed the population with an initial solutions
    seed_sol = stock_data.copy()
    seed_sol['Pct Allocation'] = 1

    E.add_solution(seed_sol)

    # run the evolver
    E.evolve(time_limit=10, n=1000000, dom=1000, sync=10000, status=2000)

    best_sols = E.get_best_of_each_crit()
    pp.graph_top_performance(best_sols, 6000)
    E.plot_tradeoffs('portfolio alpha', 'portfolio risk')


    # get solution evaluations
    #E.summarize(with_details=True)

    with open('evo.dat', 'wb') as file:
        pickle.dump(E, file)



if __name__ == '__main__':
    main()
