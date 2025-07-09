"""
Created on Sun Nov  1 19:48:48 2020
@author: John Rachlin
@file: evo_v4.py: An evolutionary computing framework (version 4)
Assumes no Solutions class.
"""

import random as rnd
import copy
from functools import reduce
import pickle
import time
from profiler import profile
import os
import json
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt


class Environment:

    def __init__(self):
        """ Population constructor """
        self.pop = {} # The solution population eval -> solution
        self.fitness = {} # Registered fitness functions: name -> objective function
        self.agents = {}  # Registered agents:  name -> (operator, num_solutions_input)

    def size(self):
        """ The size of the current population """
        return len(self.pop)

    def add_fitness_criteria(self, name, f):
        """ Register a fitness criterion (objective) with the
        environment. Any solution added to the environment is scored 
        according to this objective """
        self.fitness[name] = f
        
    def add_agent(self, name, op, k=1):
        """ Register a named agent with the population.
        The operator (op) function defines what the agent does.
        k defines the number of solutions the agent operates on. """
        self.agents[name] = (op, k)

    def add_solution(self, sol):
        """ Add a solution to the population """
        eval = tuple([(name, f(sol)) for name, f in self.fitness.items()])
        self.pop[eval] = sol


    @profile
    def run_agent(self, name):
        """ Invoke an agent against the population """
        op, k = self.agents[name]
        picks = self.get_random_solutions(k)
        new_solution = op(picks)
        self.add_solution(new_solution)




    def get_random_solutions(self, k=1):
        """ Pick k random solutions from the population """
        if self.size() == 0: # No solutions in population
            return []
        else:
            popvals = tuple(self.pop.values())
            return [copy.deepcopy(rnd.choice(popvals)) for _ in range(k)]


    @staticmethod
    def _dominates(p, q):
        """ p = evaluation of solution: ((obj1, score1), (obj2, score2), ... )"""
        pscores = [score for _,score in p]
        qscores = [score for _,score in q]
        score_diffs = list(map(lambda x,y: y-x, pscores, qscores))
        min_diff = min(score_diffs)
        max_diff = max(score_diffs)
        return min_diff >= 0.0 and max_diff > 0.0


    @staticmethod
    def _reduce_nds(S, p):
        return S - {q for q in S if Environment._dominates(p,q)}


    def remove_dominated(self):
        """ Remove dominated solutions """
        nds = reduce(Environment._reduce_nds, self.pop.keys(), self.pop.keys())
        self.pop = {k:self.pop[k] for k in nds}

    @staticmethod
    def _reduce_viol(S, T):
        objective, max_value = T
        return S - {q for q in S if dict(q)[objective]>max_value}

    @profile
    def remove_constraint_violators(self):
        """ Remove solutions whose objective values exceed one or
        more user-defined constraints as listed in constraints.dat """

        # Read the latest constraints file into a dictionary
        with open('constraints.json', 'r') as f:
            limits = json.load(f)

        # Determine non-violators and update population
        nonviol = reduce(Environment._reduce_viol, limits.items(), self.pop.keys())
        self.pop = {k:self.pop[k] for k in nonviol}


    def summarize(self, with_details=False, source=""):
        header = ",".join(self.fitness.keys())
        if source != "":
            header = "groupname,"+header
        print(header)

        for eval in self.pop.keys():
            vals = ",  ".join([str(score) for _, score in eval])
            if source != "":
                vals = source + ", " + vals
            print(vals)

        if with_details:
            counter = 0
            for eval, sol in self.pop.items():
                counter += 1
                print(f"\n\nSOLUTION {counter}")
                for objective, score in eval:
                    print(f"{objective:15}: {score}")
                print(str(sol))
                #print(pd.DataFrame(sol))

    def __str__(self):
        """ Output the solutions in the population """
        rslt = ""
        for eval,sol in self.pop.items():
            rslt += str(dict(eval))+"\n" # +str(sol)+"\n"
        return rslt

    @profile
    def evolve(self, n=1, dom=100, viol=100, status=1000, sync=1000, time_limit=None, reset=False):
        """ Run n random agents (default=1)
        dom defines how often we remove dominated (unfit) solutions
        status defines how often we display the current population

        n = # of agent invocations
        dom = interval for removing dominated solutions
        viol = interval for removing solutions that violate user-defined upper limits
        status = interval for display the current population
        sync = interval for merging results with solutions.dat (for parallel invocation)
        time_limit = the evolution time limit (seconds).  Evolve function stops when limit reached

        """

        # Initialize solutions file
        if reset and os.path.exists('solutions.dat'):
            os.remove('solutions.dat')

        # Initialize user constraints
        if reset or not os.path.exists('constraints.json'):
            with open('constraints.json', 'w') as f:
                json.dump({name: 99999 for name in self.fitness},
                          f, indent=4)

        start = time.time_ns()
        elapsed = (time.time_ns() - start) / 10 ** 9
        agent_names = list(self.agents.keys())

        i = 0
        while i < n and self.size() > 0 and (time_limit is None or elapsed < time_limit):

            pick = rnd.choice(agent_names)
            self.run_agent(pick)

            if i % sync == 0:
                try:
                    # Merge saved solutions into population
                    with open('solutions.dat', 'rb') as file:
                        loaded = pickle.load(file)
                        for eval, sol in loaded.items():
                            self.pop[eval] = sol
                except Exception as e:
                    print(e)

                # Remove the dominated solutions
                self.remove_dominated()

                # Resave the non-dominated solutions
                with open('solutions.dat', 'wb') as file:
                    pickle.dump(self.pop, file)

            if i % dom == 0:
                self.remove_dominated()

            if i % viol == 0:
                self.remove_constraint_violators()

            if i % status == 0:
                self.remove_dominated()

                #print(self)
                print("Iteration          :", i)
                print("Population size    :", self.size())
                print("Elapsed Time (Sec) :", elapsed, "\n\n\n")
                self.summarize()

            i += 1
            elapsed = (time.time_ns() - start) / 10 ** 9

        # Clean up the population
        print("Total elapsed time (sec): ", round(elapsed, 4))
        print(i)
        self.remove_dominated()

    def get_best_of_each_crit(self):
        best = {obj:(100, 0) for obj in self.fitness.keys()}

        for eval, sol in self.pop.items():
            for obj, score in eval:
                if score < best[obj][0]:
                    best[obj] = (score, sol)

        return best

    def get_solution_evals(self):
        """ Returns a dictionary of all solution objectives and their scores """
        objectives = defaultdict(list)

        # Get eval for each objective into a dictionary
        for eval in self.pop.keys():
            for obj, score in eval:
                objectives[obj].append(score)

        return objectives

    def export(self, groupname='groupname'):
        """ Exports all the solution objective scores into a csv file """
        objectives = self.get_solution_evals()

        # Convert to DataFrame and add a groupname column
        df = pd.DataFrame.from_dict(objectives)
        df.insert(0, 'groupname', groupname)

        # Export to csv
        df.to_csv(groupname+'_results.csv', encoding='utf-8', index=False)


    def plot_tradeoffs(self, obj1, obj2):
        """ Print a tradeoff graph of two objectives and their solutions """
        evals = self.get_solution_evals()

        plt.scatter(x=evals[obj1], y=evals[obj2])
        plt.xlabel(obj1)
        plt.ylabel(obj2)
        plt.title('Solution tradeoff between ' + obj1 + ' and ' + obj2)
        plt.show()



