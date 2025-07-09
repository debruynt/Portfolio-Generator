import pandas as pd
import pickle
from dash import Dash, dcc, html, Input, Output, State, ctx
from dash.exceptions import PreventUpdate
from collections import defaultdict
import plotly.express as px
import portfolio_generator as pg
import portfolio_performance as pp
import plotly.graph_objects as go
import plotly.express as px
import json
import dash_bootstrap_components as dbc
from functools import reduce

OBJECTIVES = ['portfolio risk', 'sharpe ratio', 'portfolio alpha', 'treynor ratio', 'industry diversification',
              'diversity wide', 'diversity tall']

max_objectives = ['sharpe ratio', 'portfolio alpha', 'treynor ratio', 'industry diversification',
                  'diversity wide', 'diversity tall']
OBJ_FUNC = [pg.portfolio_risk, pg.sharpe_ratio, pg.portfolio_alpha, pg.treynor_ratio, pg.industry_diversification,
            pg.diversity_wide, pg.diversity_tall]
fitness = zip(OBJECTIVES, OBJ_FUNC)


# load and format solutions
def convert_evals(sols):
    """ Converts evals into their proper format and makes them more readable """
    new_sols = {}
    for eval, sol in sols.items():
        new_eval = []
        for obj, score in eval:
            if obj in max_objectives:
                score = -score
            new_eval.append((obj, round(score, 3)))
        new_sols[tuple(new_eval)] = sol
    return new_sols


def open_solutions():
    """ Opens the pickled solutions.dat file """
    with open('solutions.dat', 'rb') as file:
        loaded = pickle.load(file)
        loaded = convert_evals(loaded)

    with open('refactored_solutions.dat', 'wb') as f:
        pickle.dump(loaded, f)

    return loaded


def get_solution_evals(sols):
    """ Returns a dictionary of all solution objectives and their scores """
    objectives = defaultdict(list)

    # Get eval for each objective into a dictionary
    for eval in sols.keys():
        for obj, score in eval:
            objectives[obj].append(score)

    return objectives


# control the dashboard
def get_slider_ranges():
    """ Get max and min for each objective score """
    ranges = {}

    for obj, scores in eval_dict.items():
        step = round((max(scores) - min(scores)) / 5, 4)
        ranges[obj] = (min(scores), max(scores), step)

    return ranges


def filter_options(df, filter):
    for filt in filter:
        if filter[filt] != None:
            low = filter[filt][0]
            up = filter[filt][1]

            df = df[(df[filt] <= up) & (df[filt] >= low)]

    return df


# all solutions with evaluations
solutions = open_solutions()
eval_dict = get_solution_evals(solutions)

# load all solutions to df
fin_sol = defaultdict(list)
for el in solutions.keys():
    list(map(lambda x: fin_sol[el[x][0]].append(el[x][1]), range(len(el))))
global all_solutions
all_solutions = pd.DataFrame.from_dict(fin_sol)
all_solutions = all_solutions.reset_index().rename(columns={'index': 'solution #'})
all_solutions['solution #'] = 'Solution ' + (all_solutions['solution #'] + 1).astype(str)

# sliders for dashboard
min_max = get_slider_ranges()

# https://towardsdatascience.com/python-for-data-science-bootstrap-for-plotly-dash-interactive-visualizations-c294464e3e0e
# style with bootstrap
external_stylesheets = [dbc.themes.BOOTSTRAP]

# create the dashboard
app = Dash(__name__, external_stylesheets=external_stylesheets)

# describe dashboard components
evolver_tab = html.Div(children=[
    html.H3(''),
    html.Div(children=[
        html.P('Please Enter Desired Values'),
        html.P('Industries to Exclude')
    ], style={'width': '20%', 'display': 'inline-block'})
])

performance_tab = html.Div(children=[
    html.Div(children=[
        html.P('Industries to Exclude'),
        dcc.Checklist(id='industry-checklist', options=pg.STOCK_DATA['GICS Sub-Industry'].unique(),
                      labelStyle={'display': 'block'})
    ], style={'width': '20%', 'display': 'inline-block', 'position': 'fixed', 'top': '100px', 'left': 0, 'bottom': 0,
              'overflow': 'scroll'}),
    html.Div(children=[
        html.Button('Update Solutions', id='sol-update-button', n_clicks=1),
        html.Br(),
        html.Br(),
        html.Br(),
        html.P('Funds:'),
        dcc.Input(id='fund-input', value=600, type='number', placeholder='Input Fund Amt'),
        dcc.Graph(id='performance-graph'),
        html.H5('Select Solutions'),
        dcc.Checklist(id='solution-checklist', inline=False, inputStyle={"margin-left": "20px"}),
        html.Br(),
        html.Br(),
        html.H5('Filter Solutions'),
        html.P('Portfolio Risk'),
        dcc.RangeSlider(min_max['portfolio risk'][0], min_max['portfolio risk'][1], id='pf-slider'),
        html.P('Portfolio Alpha'),
        dcc.RangeSlider(min_max['portfolio alpha'][0], min_max['portfolio alpha'][1], id='sharpe-slider'),
        html.P('Sharpe Ratio'),
        dcc.RangeSlider(min_max['sharpe ratio'][0], min_max['sharpe ratio'][1], id='alpha-slider'),
        html.P('Treynor Ratio'),
        dcc.RangeSlider(min_max['treynor ratio'][0], min_max['treynor ratio'][1], id='tr-slider'),
        html.P('Industry Diversification'),
        dcc.RangeSlider(min_max['industry diversification'][0], min_max['industry diversification'][1],
                        id='id-slider'),
        html.P('Diversity Wide'),
        dcc.RangeSlider(min_max['diversity wide'][0], min_max['diversity wide'][1], id='dw-slider', step=1),
        html.P('Diversity Tall'),
        dcc.RangeSlider(min_max['diversity tall'][0], min_max['diversity tall'][1], id='dt-slider', step=1),
        html.Br(),
        html.Br(),
        html.H5('Examine Tradeoffs'),
        dcc.Dropdown(OBJECTIVES, OBJECTIVES[0], id='obj1-dropdown', clearable=False, ),
        dcc.Dropdown(OBJECTIVES, OBJECTIVES[2], id='obj2-dropdown', clearable=False, ),
        html.Br(),
        html.P('Filter to Only Selected Portfolios?'),
        dcc.RadioItems(id='selected-tick', options=['Yes', 'No'], value='No', style={'display': 'inline-block'}),
        dcc.Graph(id='tradeoff-graph')
    ], style={'width': '80%', 'float': 'right', 'display': 'inline-block'}),
    dcc.Store(id='eval-json')
])

show_solution_tab = html.Div(children=[
    html.H3('Your Chosen Solutions'),
    dcc.Dropdown(all_solutions['solution #'], all_solutions['solution #'][:2], id='compare-dropdown',
                 multi=True),
    html.Br(),
    html.Br(),
    html.Br(),

    html.H4('Allocation Details'),
])

# describe tabs
tab1_content = dbc.Card(
    dbc.CardBody(
        [
            performance_tab
        ]
    ),
    className="mt-3",
)

tab2_content = dbc.Card(
    dbc.CardBody(
        [
            evolver_tab
        ]
    ),
    className="mt-3",
)

tab3_content = dbc.Card(
    dbc.CardBody(
        [
            show_solution_tab,
            html.Div(id='compare-solution')
        ]),
    className="mt-3",
)

tab4_content = dbc.Card(
    dbc.CardBody(id='solution-df'),
    className="mt-3",
)

tabs = dbc.Tabs(
    [
        dbc.Tab(tab1_content, label="Performance"),
        dbc.Tab(tab4_content, label="Solution Details"),
        # dbc.Tab(tab2_content, label="Evolver"),
        dbc.Tab(tab3_content, label="Solution"),
    ]
)

# add dashboard components to the layout
app.layout = html.Div(
    children=[
        html.H3('Stock Portfolio Generator with Top 100 S&P500 Tickers'),
        tabs,
    ]
)

# callback functions
@app.callback(
    Output('sol-update-button', 'n_clicks'),
    Input('sol-update-button', 'n_clicks'),
    State('industry-checklist', 'value'),
)
def exclude_industries(click, industries):
    """ Creates and saves new solutions after reallocating funds from excluded industries """

    if 'sol-update-button' == ctx.triggered_id and industries is not None:
        refactored_sol = {}
        for eval, sol in solutions.items():
            excluded_sum = sol[sol['GICS Sub-Industry'].isin(industries)]['Pct Allocation'].sum()
            refactor = 1 + (excluded_sum / (100 - excluded_sum))

            sol['Pct Allocation'] = sol.apply(lambda x: 0 if x['GICS Sub-Industry'] in industries \
                else x['Pct Allocation'] * refactor, axis=1)

            eval = tuple([(name, f(sol)) for name, f in fitness])

        refactored_sol[eval] = sol

        refactored_sol = convert_evals(refactored_sol)

        # Save new data
        open('refactored_solutions.dat', 'wb').close()
        with open('refactored_solutions.dat', 'wb') as f:
            pickle.dump(refactored_sol, f)

        for el in solutions.keys():
            list(map(lambda x: fin_sol[el[x][0]].append(el[x][1]), range(len(el))))
        global all_solutions
        all_solutions = pd.DataFrame.from_dict(fin_sol)
        all_solutions = all_solutions.reset_index().rename(columns={'index': 'solution #'})
        all_solutions['solution #'] = 'Solution ' + (all_solutions['solution #'] + 1).astype(str)

    click = 0

    return click


@app.callback(
    Output('solution-checklist', 'options'),
    Output('solution-checklist', 'value'),
    Output('solution-df', 'children'),
    Input('pf-slider', 'value'),
    Input('sharpe-slider', 'value'),
    Input('alpha-slider', 'value'),
    Input('tr-slider', 'value'),
    Input('id-slider', 'value'),
    Input('dw-slider', 'value'),
    Input('dt-slider', 'value'),
)
def get_solution_options(pf, sharpe, alpha, tr, id, dw, dt):
    # Set filter
    filter = {'portfolio risk': pf,
              'sharpe ratio': sharpe,
              'portfolio alpha': alpha,
              'treynor ratio': tr,
              'industry diversification': id,
              'diversity wide': dw,
              'diversity tall': dt}

    solutions_df = filter_options(all_solutions, filter)

    tdf = dbc.Table.from_dataframe(solutions_df, striped=True, bordered=True, hover=True)

    options = solutions_df['solution #']

    return options, options[:2], tdf


@app.callback(
    Output('performance-graph', 'figure'),
    Input('solution-checklist', 'value'),
    Input('fund-input', 'value'),

)
def plot_sol(sol_num, funds):
    # dict of all solutions and details
    str_sols = {str(obj): sol for obj, sol in solutions.items()}

    # find all solutions
    all_sols = all_solutions[all_solutions['solution #'].isin(sol_num)].drop(['solution #'], axis=1).to_dict(
        orient='records')
    sols = []
    list(map(lambda x: sols.append(str(tuple(x.items()))), all_sols))
    sols = {obj: str_sols[obj] for obj in sols}

    return pp.graph_performance(sols, funds, legend=sol_num)


@app.callback(
    Output('tradeoff-graph', 'figure'),
    Input('obj1-dropdown', 'value'),
    Input('obj2-dropdown', 'value'),
    Input('selected-tick', 'value'),
    Input('solution-checklist', 'value')
)
def plot_tradeoffs(obj1, obj2, selected, sols):
    if selected == 'No':
        solutions = all_solutions
    else:
        solutions = all_solutions[all_solutions['solution #'].isin(sols)]

    fig = px.scatter(solutions, x=obj1, y=obj2)
    return fig


@app.callback(
    Output('compare-solution', 'children'),
    Input('compare-dropdown', 'value')
)
def compare_solutions(sol_num):
    # dict of all solutions and details
    str_sols = {str(obj): sol for obj, sol in solutions.items()}

    # find all solutions
    all_sols = all_solutions[all_solutions['solution #'].isin(sol_num)].drop(['solution #'], axis=1).to_dict(
        orient='records')
    sols = []
    list(map(lambda x: sols.append(str(tuple(x.items()))), all_sols))
    sols_dfs = [str_sols[obj][['Symbol', 'Pct Allocation']] for obj in sols]

    compare_df = reduce(lambda left, right: pd.merge(left, right, on='Symbol'), sols_dfs)
    col = ['Ticker Symbol'] + sol_num
    compare_df.columns = col

    tdf = dbc.Table.from_dataframe(compare_df, striped=True, bordered=True, hover=True)

    return tdf


# run the dashboard
app.run_server(debug=True)
