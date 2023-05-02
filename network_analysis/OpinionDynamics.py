#!/usr/bin/env python
# coding: utf-8

# In[5]:


from scipy.optimize import curve_fit
from sklearn import preprocessing
import networkx as nx
import pandas as pd
import numpy as np
import re

import nltk
from nltk.probability import FreqDist
import itertools
import collections
import powerlaw
from nltk import bigrams
from nltk.util import ngrams
import os
import glob
import pquality
from nltk.corpus import stopwords
import string
import math

from scipy.optimize import curve_fit
from sklearn import preprocessing
from cdlib import algorithms
import networkx as nx
import pandas as pd
import numpy as np

from pattern.web import download, plaintext
stopwords_file = download("https://raw.githubusercontent.com/stopwords-iso/stopwords-it/master/stopwords-it.txt").decode("utf-8")
stopwords_text = plaintext(stopwords_file)
italian_stopwords = set(stopwords_text.split())
stop_words = set(stopwords.words('italian'))

import matplotlib.pyplot as plt


# In[2]:


def remove_useless(text):
    tweets = [word for word in text if not word in stop_words ]
    tweets = [word for word in tweets if not word in italian_stopwords]
    tweets = [word for word in tweets if len(word) >3]

    return tweets

def expfunc(x, a, b):
        return a*pow(x,-b)
    
    
def istogramma(G, inizio = 1):
    print(G.number_of_nodes(),G.number_of_edges(),nx.number_connected_components(G))

    k = nx.degree_histogram(G)

    
    hist = nx.degree_histogram(G)
    bins = np.arange(len(k))
    x = bins[inizio:]
    y = np.log(hist)

    popt, pcov = curve_fit(expfunc, x, k[inizio:])

    plt.plot(range(0, len(hist)), hist,'.',label='degree distribution')
    plt.plot(x, expfunc(x, *popt), 'r--', label='fitted distribution')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(10**np.log10(0.5), 1e4)
    plt.show()

    print(f"b={popt[1]}")
    
    
from random import choice
import future.utils
from collections import defaultdict
import tqdm
import abc
import past.builtins
import networkx as nx
from ndlib.models.DiffusionModel import DiffusionModel
import six
import warnings
import numpy as np
import future.utils

class AlgorithmicB2(DiffusionModel):#leggera modifica al codice sorgente per poter settare i valori iniziali
    """
    Model Parameters to be specified via ModelConfig
    :param epsilon: bounded confidence threshold from the Deffuant model, in [0,1]
    :param gamma: strength of the algorithmic bias, positive, real
    Node states are continuous values in [0,1].
    The initial state is generated randomly uniformly from the domain [0,1].
    """

    def __init__(self, graph, seed=None):
        """
             Model Constructor
             :param graph: A networkx graph object
         """
        super(self.__class__, self).__init__(graph, seed)

        self.discrete_state = False

        self.available_statuses = {
            "Infected": 0
        }

        self.parameters = {
            "model": {
                "epsilon": {
                    "descr": "Bounded confidence threshold",
                    "range": [0, 1],
                    "optional": False
                },
                "gamma": {
                    "descr": "Algorithmic bias",
                    "range": [0, 100],
                    "optional": False
                }
            },
            "nodes": {},
            "edges": {}
        }

        self.name = "Agorithmic Bias"

        self.node_data = {}
        self.ids = None
        self.sts = None

    def set_initial_status(self, configuration=None):
        """
        Override behaviour of methods in class DiffusionModel.
        Overwrites initial status using random real values.
        """
        super(AlgorithmicB2, self).set_initial_status(configuration)

        # set node status
        for node in self.status:
                if node in v:
                    self.status[node]= np.random.rand()*0.33+0.66 #np.random.normal(1, sigma)
                else:
                    if node in ss:
                        self.status[node]= np.random.rand()*0.33
                    else:
                        self.status[node]= np.random.rand()*0.33+0.33
        self.initial_status = self.status.copy()

        ### Initialization numpy representation

        max_edgees = (self.graph.number_of_nodes() * (self.graph.number_of_nodes() - 1)) / 2
        nids = np.array(list(self.status.items()))
        self.ids = nids[:, 0]

        if max_edgees == self.graph.number_of_edges():
            self.sts = nids[:, 1]

        else:
            for i in self.graph.nodes:
                i_neigh = list(self.graph.neighbors(i))
                i_ids = nids[:, 0][i_neigh]
                i_sts = nids[:, 1][i_neigh]
                # non uso mai node_data[:,1]
                # per tenere aggiornato node_data() all'interno del for dovrei ciclare ogni item=nodo
                # e se uno dei suoi vicini è n1 o n2 aggiornare l'array sts
                # la complessità dovrebbe essere O(N)
                # se invece uso solo actual_status, considerando che per ogni nodo ho la lista dei neighbors in memoria
                # a ogni ciclo devo soltanto tirarmi fuori la lista degli stati degli avg_k vicini e prendere i
                # loro stati da actual_status
                # quindi la complessità dovrebbe essere O(N*p) < O(N)
                # sto pensando ad un modo per farlo in O(1) ma non mi è ancora venuto in mente

                self.node_data[i] = (i_ids, i_sts)

    # def clean_initial_status(self, valid_status=None):
    #     for n, s in future.utils.iteritems(self.status):
    #         if s > 1 or s < 0:
    #             self.status[n] = 0

    @staticmethod
    def prob(distance, gamma, min_dist):
        if distance < min_dist:
            distance = min_dist
        return np.power(distance, -gamma)

    def pb1(self, statuses, i_status):
        dist = np.abs(statuses - i_status)
        null = np.full(statuses.shape[0], 0.00001)
        max_base = np.maximum(dist, null)
        dists = max_base ** -self.params['model']['gamma']
        return dists

    def iteration(self, node_status=True):
        """
        Execute a single model iteration
        :return: Iteration_id, Incremental node status (dictionary node->status)
        """
        # One iteration changes the opinion of N agent pairs using the following procedure:
        # - first one agent is selected
        # - then a second agent is selected based on a probability that decreases with the distance to the first agent
        # - if the two agents have a distance smaller than epsilon, then they change their status to the average of
        # their previous statuses

        actual_status = self.status.copy()

        if self.actual_iteration == 0:
            self.actual_iteration += 1
            delta, node_count, status_delta = self.status_delta(self.status)
            if node_status:
                return {"iteration": 0, "status": actual_status,
                        "node_count": node_count.copy(), "status_delta": status_delta.copy()}
            else:
                return {"iteration": 0, "status": {},
                        "node_count": node_count.copy(), "status_delta": status_delta.copy()}

        n = self.graph.number_of_nodes()

        # interact with peers
        for i in range(0, n):

            # ho rimesso la selezione del nodo a random
            # n1 = list(self.graph.nodes)[np.random.randint(0, n)]
            n1 = int(choice(self.ids))

            if len(self.node_data) == 0:
                sts = self.sts
                ids = self.ids
                # toglie se stesso dalla lista degli id e degli status perché mi sembra rimanesse
                # e quindi con gamma alto a volte sceglieva se stesso per interagire
                neigh_sts = np.delete(sts, n1)
                neigh_ids = np.delete(ids, n1)
            else:
                neigh_ids = self.node_data[n1][0]
                neigh_sts = np.array([actual_status[id] for id in neigh_ids])

            # ho cambiato come crea l'array degli stati
            # niegh_sts = self.node_data[n1][1]

            # uso neigh_sts e actual_status[n1] come argomenti della funzione
            # perché altrimenti self.status[n1] è quello che viene dalla precedente
            # iterazione ma non viene aggiornato in corso di interazioni all'interno di questo for
            # e potrebbe essere cambiato in precedenza
            # e nel codice vecchio su usava invece lo stato sempre aggiornato

            # selection_prob = self.pb1(sts, self.status[n1])
            selection_prob = self.pb1(neigh_sts, actual_status[n1])

            # compute probabilities to select a second node among the neighbours
            total = np.sum(selection_prob)
            selection_prob = selection_prob / total
            cumulative_selection_probability = np.cumsum(selection_prob)

            r = np.random.random_sample()
            # n2 = np.argmax(cumulative_selection_probability >= r) -1
            n2 = np.argmax(
                cumulative_selection_probability >= r)
            # seleziono n2 dagli id dei neighbors di n1
            n2 = int(neigh_ids[n2])

            # update status of n1 and n2
            diff = np.abs(actual_status[n1] - actual_status[n2])

            if diff < self.params['model']['epsilon']:
                avg = (actual_status[n1] + actual_status[n2]) / 2.0
                actual_status[n1] = avg
                actual_status[n2] = avg
                # se la rete è completa aggiorno all'interno del ciclo
                # self.sts, così lo riprendo sempre aggiornato
                if len(self.node_data) == 0:
                    self.sts[n1] = avg
                    self.sts[n2] = avg

        # delta, node_count, status_delta = self.status_delta(actual_status)
        delta = actual_status
        node_count = {}
        status_delta = {}

        self.status = actual_status
        self.actual_iteration += 1

        if node_status:
            return {"iteration": self.actual_iteration - 1, "status": delta,
                    "node_count": node_count.copy(), "status_delta": status_delta.copy()}
        else:
            return {"iteration": self.actual_iteration - 1, "status": {},
                    "node_count": node_count.copy(), "status_delta": status_delta.copy()}

    def steady_state(self, max_iterations=100000, nsteady=1000, sensibility=0.00001, node_status=True,
                     progress_bar=False):
        """
        Execute a bunch of model iterations
        :param max_iterations: the maximum number of iterations to execute
        :param nsteady: number of required stable states
        :param sensibility: sensibility check for a steady state
        :param node_status: if the incremental node status has to be returned.
        :param progress_bar: whether to display a progress bar, default False
        :return: a list containing for each iteration a dictionary {"iteration": iteration_id, "status": dictionary_node_to_status}
        """
        system_status = []
        steady_it = 0
        for it in tqdm.tqdm(range(0, max_iterations), disable=not progress_bar):
            its = self.iteration(node_status)

            if it > 0:
                old = np.array(list(system_status[-1]['status'].values()))
                actual = np.array(list(its['status'].values()))
                res = np.abs(old - actual)
                if np.all((res < sensibility)):
                    steady_it += 1
                else:
                    steady_it = 0

            system_status.append(its)
            if steady_it == nsteady:
                return system_status[:-nsteady]

        return system_status



class CognitiveOpDynModel2(DiffusionModel):
    """
    Model Parameters to be specified via ModelConfig
    :param I: external information value in [0,1]
    :param T_range_min: the minimum of the range of initial values for  T. Range [0,1].
    :param T_range_max: the maximum of the range of initial values for  T. Range [0,1].
    :param B_range_min: the minimum of the range of initial values for  B. Range [0,1]
    :param B_range_max: the maximum of the range of initial values for  B. Range [0,1].
    :param R_fraction_negative: fraction of individuals having the node parameter R=-1.
    :param R_fraction_positive: fraction of individuals having the node parameter R=1
    :param R_fraction_neutral: fraction of individuals having the node parameter R=0
    The following relation should hold: R_fraction_negative+R_fraction_neutral+R_fraction_positive=1.
    To achieve this, the fractions selected will be normalised to sum 1.
    Node states are continuous values in [0,1].
    The initial state is generated randomly uniformly from the domain defined by model parameters.
    """

    def __init__(self, graph, seed=None):
        """
             Model Constructor
             :param graph: A networkx graph object
         """
        super(self.__class__, self).__init__(graph, seed)

        self.discrete_state = False

        self.available_statuses = {
            "Infected": 0
        }

        self.parameters = {
            "model": {
                "I": {
                    "descr": "External information",
                    "range": [0, 1],
                    "optional": False
                },
                "T_range_min": {
                    "descr": "Minimum of the range of initial values for T",
                    "range": [0, 1],
                    "optional": False
                },
                "T_range_max": {
                    "descr": "Maximum of the range of initial values for T",
                    "range": [0, 1],
                    "optional": False
                },
                "B_range_min": {
                    "descr": "Minimum of the range of initial values for B",
                    "range": [0, 1],
                    "optional": False
                },
                "B_range_max": {
                    "descr": "Maximum of the range of initial values for B",
                    "range": [0, 1],
                    "optional": False
                },
                "R_fraction_negative": {
                    "descr": "Fraction of nodes having R=-1",
                    "range": [0, 1],
                    "optional": False
                },
                "R_fraction_neutral": {
                    "descr": "Fraction of nodes having R=0",
                    "range": [0, 1],
                    "optional": False
                },
                "R_fraction_positive": {
                    "descr": "Fraction of nodes having R=1",
                    "range": [0, 1],
                    "optional": False
                }
            },
            "nodes": {},
            "edges": {}
        }

        self.name = "Cognitive Opinion Dynamics"

    def set_initial_status(self, configuration=None):
        """
        Override behaviour of methods in class DiffusionModel.
        Overwrites initial status using random real values.
        Generates random node profiles.
        """
        super(CognitiveOpDynModel2, self).set_initial_status(configuration)

        # set node status
        for node in self.status:
                if node in v:
                    self.status[node]= np.random.rand()*0.33+0.66 #np.random.normal(1, sigma)
                else:
                    if node in ss:
                        self.status[node]= np.random.rand()*0.33
                    else:
                        self.status[node]= np.random.rand()*0.33+0.33
        self.initial_status = self.status.copy()

        # set new node parameters
        self.params['nodes']['cognitive'] = {}

        # first correct the input model parameters and retreive T_range, B_range and R_distribution
        T_range = (self.params['model']['T_range_min'], self.params['model']['T_range_max'])
        if self.params['model']['T_range_min'] > self.params['model']['T_range_max']:
            T_range = (self.params['model']['T_range_max'], self.params['model']['T_range_min'])

        B_range = (self.params['model']['B_range_min'], self.params['model']['B_range_max'])
        if self.params['model']['B_range_min'] > self.params['model']['B_range_max']:
            B_range = (self.params['model']['B_range_max'], self.params['model']['B_range_min'])
        s = float(self.params['model']['R_fraction_negative'] + self.params['model']['R_fraction_neutral'] +
                  self.params['model']['R_fraction_positive'])
        R_distribution = (self.params['model']['R_fraction_negative']/s, self.params['model']['R_fraction_neutral']/s,
                          self.params['model']['R_fraction_positive']/s)

        # then sample parameters from the ranges and distribution
        for node in self.graph.nodes:
            R_prob = np.random.random_sample()
            if R_prob < R_distribution[0]:
                R = -1
            elif R_prob < (R_distribution[0] + R_distribution[1]):
                R = 0
            else:
                R = 1
            # R, B and T parameters in a tuple
            self.params['nodes']['cognitive'][node] = (R,
                                                       B_range[0] + (B_range[1] - B_range[0])*np.random.random_sample(),
                                                       T_range[0] + (T_range[1] - T_range[0])*np.random.random_sample())

    def clean_initial_status(self, valid_status=None):
        for n, s in future.utils.iteritems(self.status):
            if s > 1 or s < 0:
                self.status[n] = 0

    def iteration(self, node_status=True):
        """
        Execute a single model iteration
        :return: Iteration_id, Incremental node status (dictionary node->status)
        """
        # One iteration changes the opinion of all agents using the following procedure:
        # - first all agents communicate with institutional information I using a deffuant like rule
        # - then random pairs of agents are selected to interact  (N pairs)
        # - interaction depends on state of agents but also internal cognitive structure

        self.clean_initial_status(None)

        actual_status = {node: nstatus for node, nstatus in future.utils.iteritems(self.status)}

        if self.actual_iteration == 0:
            self.actual_iteration += 1
            delta, node_count, status_delta = self.status_delta(self.status)
            if node_status:
                return {"iteration": 0, "status": self.status.copy(),
                        "node_count": node_count.copy(), "status_delta": status_delta.copy()}
            else:
                return {"iteration": 0, "status": {},
                        "node_count": node_count.copy(), "status_delta": status_delta.copy()}

        # first interact with I
        I = self.params['model']['I']
        for node in self.graph.nodes:
            T = self.params['nodes']['cognitive'][node][2]
            R = self.params['nodes']['cognitive'][node][0]
            actual_status[node] = actual_status[node] + T * (I - actual_status[node])
            if R == 1:
                actual_status[node] = 0.5 * (1 + actual_status[node])
            if R == -1:
                actual_status[node] *= 0.5

        # then interact with peers
        for i in range(0, self.graph.number_of_nodes()):
            # select a random node
            n1 = list(self.graph.nodes)[np.random.randint(0, self.graph.number_of_nodes())]

            # select all of the nodes neighbours (no digraph possible)
            neighbours = list(self.graph.neighbors(n1))
            if len(neighbours) == 0:
                continue

            # select second node - a random neighbour
            n2 = neighbours[np.random.randint(0, len(neighbours))]

            # update status of n1 and n2
            p1 = pow(actual_status[n1], 1.0 / self.params['nodes']['cognitive'][n1][1])
            p2 = pow(actual_status[n2], 1.0 / self.params['nodes']['cognitive'][n2][1])

            oldn1 = self.status[n1]
            if np.random.random_sample() < p2:  # if node 2 talks, node 1 gets changed
                T1 = self.params['nodes']['cognitive'][n1][2]
                R1 = self.params['nodes']['cognitive'][n1][0]
                actual_status[n1] += (1 - T1) * (actual_status[n2] - actual_status[n1])
                if R1 == 1:
                    actual_status[n1] = 0.5 * (1 + actual_status[n1])
                if R1 == -1:
                    actual_status[n1] *= 0.5
            if np.random.random_sample() < p1:  # if node 1 talks, node 2 gets changed
                T2 = self.params['nodes']['cognitive'][n2][2]
                R2 = self.params['nodes']['cognitive'][n2][0]
                actual_status[n2] += (1 - T2) * (oldn1 - actual_status[n2])
                if R2 == 1:
                    actual_status[n2] = 0.5 * (1 + actual_status[n2])
                if R2 == -1:
                    actual_status[n2] *= 0.5

        delta, node_count, status_delta = self.status_delta(actual_status)
        self.status = actual_status
        self.actual_iteration += 1

        if node_status:
            return {"iteration": self.actual_iteration - 1, "status": delta.copy(),
                    "node_count": node_count.copy(), "status_delta": status_delta.copy()}
        else:
            return {"iteration": self.actual_iteration - 1, "status": {},
                    "node_count": node_count.copy(), "status_delta": status_delta.copy()}
        
 
class OpinionEvo2(object):

    def __init__(self, model, trends):
        """
        :param model: The model object
        :param trends: The computed simulation trends
        """
        self.model = model
        self.srev = trends
        self.ylabel = "Opinion"

    def plot(self, filename=None):
        """
        Generates the plot
        :param filename: Output filename
        :param percentile: The percentile for the trend variance area
        """

        descr = ""
        infos = self.model.get_info()

        for k, v in future.utils.iteritems(infos):
            descr += "%s: %s, " % (k, v)
        descr = descr[:-2].replace("_", " ")

        nodes2opinions = {}
        node2col = {}

        last_it = self.srev[-1]['iteration'] + 1
        last_seen = {}

        for it in self.srev:
            sts = it['status']
            its = it['iteration']
            for n, v in sts.items():
                if n in nodes2opinions:
                    last_id = last_seen[n]
                    last_value = nodes2opinions[n][last_id]

                    for i in range(last_id, its):
                        nodes2opinions[n][i] = last_value

                    nodes2opinions[n][its] = v
                    last_seen[n] = its
                else:
                    nodes2opinions[n] = [0]*last_it
                    nodes2opinions[n][its] = v
                    last_seen[n] = 0
                    if v < 0.33:
                        node2col[n] = '#ff0000'
                    elif 0.33 <= v <= 0.66:
                        node2col[n] = '#00ff00'
                    else:
                        node2col[n] = '#0000ff'

        mx = 0
        for k, l in future.utils.iteritems(nodes2opinions):
            if mx < last_seen[k]:
                mx = last_seen[k]
            x = list(range(0, last_seen[k]))
            y = l[0:last_seen[k]]
            plt.plot(x, y, lw=1, alpha=0.5, color=node2col[k])

        plt.title(descr)
        plt.xlabel("Iterations", fontsize=24)
        plt.ylabel(self.ylabel, fontsize=24)
        plt.legend(loc="best", fontsize=18)

        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename)
            plt.clf()
        else:
            plt.show()


# In[6]:


df = pd.read_csv('df_tutti_1.csv')
user=[]
for i in range(0, len(df['user'])):
    c=df['user'][i].split("'")
    user.append(c[3])
use = np.array(user)
unique_user = np.unique(use)
men_user=[[] for _ in range(0,len(df['mentionedUsers']))]

for i in range(0,len(df['mentionedUsers'])):
    if isinstance(df['mentionedUsers'][i],float):
        men_user[i].append('nan')
    else:
        v=df['mentionedUsers'][i].split("{")
        v.remove(v[0])
        if len(v)>1:
            for j in range(0, len(v)-1):
                d=v[j].split("'")
                men_user[i].append(d[3])
        elif len(v)==1:
            d=v[0].split("'")
            men_user[i].append(d[3])
df.drop(df.columns.difference(['renderedContent']), axis=1, inplace=True)
df['renderedContent'] = df['renderedContent'].apply(lambda x: x.split())
df['user']=user
df['men_user']=men_user

df['renderedContent'] = df['renderedContent'].apply(lambda x: remove_useless(x))


G = nx.Graph()

for k ,v ,m in zip(df['user'].items(),df['men_user'].items(),df['renderedContent'].items()):
    for i in range(0, len(v[1])):
        if v[1][i]=='nan':pass
        else:  
            G.add_edge(k[1], v[1][i], tweet=m[1])
G.remove_edges_from(nx.selfloop_edges(G))

components = nx.connected_components(G)

for component in components:
    print(len(component))
    real_G = G.subgraph(component)
    real_G = nx.Graph(real_G)
    break
istogramma(real_G,10)


# In[7]:



leiden_coms = algorithms.leiden(real_G)

#Inter-Communities words
for j in range(0, len(leiden_coms.communities)):
    df=pd.DataFrame(leiden_coms.communities[j], columns=['User'])
    comb=itertools.combinations(leiden_coms.communities[j],2)     
    d=[]
    if df.shape[0]>10:
        for i in comb:
            if G.has_edge(i[0],i[1]):
                d.extend(G.edges[i[0],i[1]]['tweet'])
            else:pass
    dt=pd.DataFrame()
    dt['word']=d
    print(f"\nCommunity {j}: {len(leiden_coms.communities[j])}")
    print(dt['word'].value_counts().nlargest(5))


# In[8]:


#v1 communities con Meloni Salvini etc come best words
#s1 communities con Conte Calanda etc

s1=leiden_coms.communities[0]+leiden_coms.communities[2]+leiden_coms.communities[6]+leiden_coms.communities[7]+leiden_coms.communities[9]+leiden_coms.communities[11]
v1=leiden_coms.communities[1]+leiden_coms.communities[4]+leiden_coms.communities[5]+leiden_coms.communities[14]+leiden_coms.communities[15]+leiden_coms.communities[19]


# In[9]:


from scipy.optimize import curve_fit


import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep



import ndlib.models.ModelConfig as mc
import ndlib.models.opinions as op

from ndlib.viz.mpl.DiffusionTrend import DiffusionTrend
from ndlib.viz.mpl.DiffusionPrevalence import DiffusionPrevalence

from ndlib.viz.mpl.OpinionEvolution import OpinionEvolution

from ndlib.viz.mpl.TrendComparison import DiffusionTrendComparison
from ndlib.viz.mpl.PrevalenceComparison import DiffusionPrevalenceComparison



def modello(model ,nome="graph", infected=None, q=0, ODM=None, ABM=None):
    
    config = mc.Configuration()
    
    if type(infected)== type([1,2]) :
        
        config.add_model_initial_configuration("Infected", infected)
    else:
        if type(infected) == type(0.1):
            config.add_model_parameter("fraction_infected", infected)
    
    
    if q!=0: config.add_model_parameter("q", q)
        
    
    if ODM!=None:
        config.add_model_parameter("I", ODM)
        config.add_model_parameter("B_range_min", 0)
        config.add_model_parameter("B_range_max", 1)
        config.add_model_parameter("T_range_min", 0)
        config.add_model_parameter("T_range_max", 1)
        config.add_model_parameter("R_fraction_negative", 1.0 / 3)
        config.add_model_parameter("R_fraction_neutral", 1.0 / 3)
        config.add_model_parameter("R_fraction_positive", 1.0 / 3)
        model.set_initial_status(config)
        iterations = model.iteration_bunch(20)
        viz = OpinionEvo2(model, iterations)
        viz.plot(nome+".pdf")
        return model, iterations
    if ABM!=None:
        config.add_model_parameter("epsilon", ABM[0])
        config.add_model_parameter("gamma", ABM[1])
        model.set_initial_status(config)

        # Simulation execution
        iterations = model.iteration_bunch(20)
        viz = OpinionEvo2(model, iterations)
        viz.plot(nome+".pdf")
        return model, iterations
    
    model.set_initial_status(config)

    iterations = model.iteration_bunch(200)
    trends = model.build_trends(iterations)
    
    viz = DiffusionTrend(model, trends)
    viz.plot(percentile=90)
    
    viz = DiffusionPrevalence(model, trends)
    viz.plot(percentile=90)
    return model, trends

def new_graph():#serve per creare il lo stesso grafico ma i nodi sono numInteri (serve per AlgoritBias and CognitiOp)
    H = nx.Graph()
    v =[]
    s =[]
    for i, node_name in enumerate(real_G.nodes()):
        H.add_node(i, name=node_name)
        if node_name in v1:
            v.append(i)
        else:
            if node_name in s1:
                s.append(i)
    for edge in real_G.edges():
        u = list(real_G.nodes()).index(edge[0])
        c = list(real_G.nodes()).index(edge[1])
        H.add_edge(u, c)
    return H,v,s

H,v,ss=new_graph()

c=[]
for node in H.nodes():
    if node not in v and node not in ss:#settiamo i nodi che non sono nelle communities principali come "centro"
        c.append(node)


# # Communities

# In[12]:


#Voter and SznajdModel


# In[10]:


print("\n\n VoterModel")
Vot_M, t_V_M = modello(op.VoterModel(real_G),"VoterModel", v1)
print("\n\n SznajdModel")
S_M, t_S_M = modello(op.SznajdModel(real_G),"SznajdModel", v1)


# In[11]:



viz = DiffusionTrendComparison([Vot_M, S_M], [t_V_M, t_S_M], statuses=['Infected'])
viz.plot()


# In[12]:



viz = DiffusionPrevalenceComparison([Vot_M, S_M], [t_V_M, t_S_M], statuses=['Infected'])
viz.plot()


# In[ ]:


#QVoter and MajorityRule


# In[13]:


#Qparameter tuning
ts=[]
h=[]
for i in range(1,300,10):
    h.append(i)
    MV_M, t_MV_M = modello(op.MajorityRuleModel(real_G),"MajorityRuleModel", v1, i)
    ts.append(t_MV_M[-1]['trends']['node_count'][0][-1])
t=[]
for i in range(1,300,10):
    QVot_M, t_QV_M = modello(op.QVoterModel(real_G),"QVoterModel", v1,i)
    t.append(t_QV_M[-1]['trends']['node_count'][0][-1])


# In[14]:


plt.plot(h,t, color='red', label='QVoterModel', marker='.')
plt.plot(h,ts, color='blue', label='MajorityRuleModel', marker='.')
plt.legend(loc='best')
plt.xlabel("Q parameter")
plt.ylabel("Susceptible count")
plt.show()


# In[15]:


QVot_M, t_QV_M = modello(op.QVoterModel(real_G),"QVoterModel", v1,15)
MV_M, t_MV_M = modello(op.MajorityRuleModel(real_G),"MajorityRuleModel", v1, 15)


# In[16]:


viz = DiffusionTrendComparison([QVot_M, MV_M], [t_QV_M, t_MV_M], statuses=['Infected'])
viz.plot()


# In[17]:


viz = DiffusionPrevalenceComparison([QVot_M, MV_M], [t_QV_M, t_MV_M], statuses=['Infected'])
viz.plot()


# In[ ]:


#Cognitve


# In[18]:


m,iterations= modello(CognitiveOpDynModel2(H) ,nome="CognitiveOpDynModel2(H)", infected=None, q=0, ODM=0.9, ABM=None)
mediaS=[]
mediaV=[]
mediaC=[]
errS=[]
errV=[]
errC=[]
l=[]
a=0
sss={}
vv={}
cc={}
for i in iterations:#there might be a bug on the program or the 1 status keep the node away from the trend list
    for key in ss:
        try:
            sss[key] = i['status'][key]
        except:
            pass
    for key in v:
        try:
            vv[key]= i['status'][key]
        except:
            pass
    for key in c:
        try:
            cc[key]= i['status'][key]
        except:
            pass
    mediaS.append(np.mean(list(sss.values())))
    errS.append(np.var(list(sss.values())))
    mediaV.append(np.mean(list(vv.values())))
    errV.append(np.var(list(vv.values())))
    mediaC.append(np.mean(list(cc.values())))
    errC.append(np.var(list(cc.values())))
    l.append(a)
    a+=1
plt.errorbar(l, mediaS,yerr=errS, color='blue',label='Left', marker='.')
plt.errorbar(l, mediaV,yerr=errV, color='red',label='Right', marker='.')
plt.errorbar(l, mediaC,yerr=errC, color='green',label='Center', marker='.')
plt.legend(loc='best')
plt.xlabel("Iteration")
plt.ylabel("Status mean value")
plt.show()


# In[19]:


mS=[]
mV=[]
mC=[]
vS=[]
vV=[]
vC=[]
param=[]
for q in range(1,100,5):
    q=q*0.01
    
    m,iterations= modello(CognitiveOpDynModel2(H) ,nome="CognitiveOpDynModel2(H)", infected=None, q=0, ODM=q, ABM=None)
    
    
    mediaS=[]
    mediaV=[]
    mediaC=[]
    errS=[]
    errV=[]
    errC=[]
    l=[]
    a=0
    sss={}
    vv={}
    cc={}
    for i in [iterations[0],iterations[-1]]:#there might be a bug on the program or the 1 status keep the node away from the trend list
        for key in ss:
            try:
                sss[key] = i['status'][key]
            except:
                pass
        for key in v:
            try:
                vv[key]= i['status'][key]
            except:
                pass
        for key in c:
            try:
                cc[key]= i['status'][key]
            except:
                pass
        mediaS.append(np.mean(list(sss.values())))
        errS.append(np.var(list(sss.values())))
        mediaV.append(np.mean(list(vv.values())))
        errV.append(np.var(list(vv.values())))
        mediaC.append(np.mean(list(cc.values())))
        errC.append(np.var(list(cc.values())))
    mS.append(mediaS[-1]-mediaS[0])
    vS.append(errS[-1]+errS[0])
    mV.append(mediaV[-1]-mediaV[0])
    vV.append(errV[-1]+errV[0])
    mC.append(mediaC[-1]-mediaC[0])
    vC.append(errC[-1]+errC[0])
    param.append(q)
plt.errorbar(param, mS,vS, color='blue',label='Left', marker='.')
plt.errorbar(param, mV,vV, color='red',label='Right', marker='.')
plt.errorbar(param, mC,vC, color='green',label='Center', marker='.')
plt.legend(loc='best')
plt.xlabel("External Information Parameter")
plt.ylabel("Status mean value")
plt.show()


# In[ ]:


#AlgB


# In[20]:


m,iterations= modello(AlgorithmicB2(H) ,nome="AlgorithmicB2(H)", infected=None, q=0, ODM=None, ABM=(0.2,80))

mediaS=[]
mediaV=[]
mediaC=[]
errS=[]
errV=[]
errC=[]
l=[]
a=0
for i in iterations:
    sss = {key: i['status'][key] for key in ss}
    vv = {key: i['status'][key] for key in v}
    cc = {key: i['status'][key] for key in c}
    
    mediaS.append(np.mean(list(sss.values())))
    errS.append(np.var(list(sss.values())))
    mediaV.append(np.mean(list(vv.values())))
    errV.append(np.var(list(vv.values())))
    mediaC.append(np.mean(list(cc.values())))
    errC.append(np.var(list(cc.values())))
    l.append(a)
    a+=1
plt.errorbar(l, mediaS,yerr=errS, color='blue',label='Left', marker='.')
plt.errorbar(l, mediaV,yerr=errV, color='red',label='Right', marker='.')
plt.errorbar(l, mediaC,yerr=errC, color='green',label='Center', marker='.')
plt.legend(loc='best')
plt.xlabel("Iteration")
plt.ylabel("Status mean value")
plt.show()


# In[21]:


mS=[]
mV=[]
mC=[]
vS=[]
vV=[]
vC=[]
param=[]
for q in range(1,100,5):
    q=q*0.01
    m,iterations= modello(AlgorithmicB2(H) ,nome="AlgorithmicB2(H)", infected=None, q=0, ODM=None, ABM=(q,20))

    

    mediaS=[]
    mediaV=[]
    mediaC=[]
    errS=[]
    errV=[]
    errC=[]
    for i in [iterations[0],iterations[-1]]:
        sss = {key: i['status'][key] for key in ss}
        vv = {key: i['status'][key] for key in v}
        cc = {key: i['status'][key] for key in c}
        mediaS.append(np.mean(list(sss.values())))
        errS.append(np.var(list(sss.values())))
        mediaV.append(np.mean(list(vv.values())))
        errV.append(np.var(list(vv.values())))
        mediaC.append(np.mean(list(cc.values())))
        errC.append(np.var(list(cc.values())))
    mS.append(mediaS[-1]-mediaS[0])
    vS.append(errS[-1]+errS[0])
    mV.append(mediaV[-1]-mediaV[0])
    vV.append(errV[-1]+errV[0])
    mC.append(mediaC[-1]-mediaC[0])
    vC.append(errC[-1]+errC[0])
    param.append(q)
plt.errorbar(param, mS,vS, color='blue',label='Left', marker='.')
plt.errorbar(param, mV,vV, color='red',label='Right', marker='.')
plt.errorbar(param, mC,vC, color='green',label='Center', marker='.')
plt.legend(loc='best')
plt.xlabel("Epsilon (gamma=20)")
plt.ylabel("Status mean value")
plt.show()


# In[22]:


mS=[]
mV=[]
mC=[]
vS=[]
vV=[]
vC=[]
param=[]
for q in range(1,100,5):
    m,iterations= modello(AlgorithmicB2(H) ,nome="AlgorithmicB2(H)", infected=None, q=0, ODM=None, ABM=(0.2,q))

    

    mediaS=[]
    mediaV=[]
    mediaC=[]
    errS=[]
    errV=[]
    errC=[]
    for i in [iterations[0],iterations[-1]]:
        sss = {key: i['status'][key] for key in ss}
        vv = {key: i['status'][key] for key in v}
        cc = {key: i['status'][key] for key in c}
        mediaS.append(np.mean(list(sss.values())))
        errS.append(np.var(list(sss.values())))
        mediaV.append(np.mean(list(vv.values())))
        errV.append(np.var(list(vv.values())))
        mediaC.append(np.mean(list(cc.values())))
        errC.append(np.var(list(cc.values())))
    mS.append(mediaS[-1]-mediaS[0])
    vS.append(errS[-1]+errS[0])
    mV.append(mediaV[-1]-mediaV[0])
    vV.append(errV[-1]+errV[0])
    mC.append(mediaC[-1]-mediaC[0])
    vC.append(errC[-1]+errC[0])
    param.append(q)
plt.errorbar(param, mS,vS, color='blue',label='Left', marker='.')
plt.errorbar(param, mV,vV, color='red',label='Right', marker='.')
plt.errorbar(param, mC,vC, color='green',label='Center', marker='.')
plt.legend(loc='best')
plt.xlabel("Gamma (epsilon=0.2)")
plt.ylabel("Status mean value")
plt.show()


# # RANDOM and HUBS

# In[23]:


for i in range(1,real_G.number_of_nodes()):

    ba_model = nx.barabasi_albert_graph(real_G.number_of_nodes(), i)
    if ba_model.number_of_edges()>real_G.number_of_edges():
        break

ba_model = nx.barabasi_albert_graph(real_G.number_of_nodes(), i-1)

for i in range(1,10):
    i=i*0.1
    er_model = nx.erdos_renyi_graph(real_G.number_of_nodes(), i)
    if er_model.number_of_edges()>real_G.number_of_edges():
        break
er_model = nx.erdos_renyi_graph(real_G.number_of_nodes(), i)


# In[24]:


#VOTER
y=[]
yerr=[]
yBA=[]
yerrBA=[]
yHub=[]
yerrHub=[]
yRN=[]
yerrRN=[]
x=[]
degrees = dict(real_G.degree())
sorted_deg=dict(sorted(degrees.items(), key=lambda x:x[1], reverse=True))
nnode=real_G.number_of_nodes()
for i in range (1,10):
    i = i*0.1
    Vot_M, t_V_M = modello(op.VoterModel(real_G),"VoterModel", i)
    Vot_MBA, t_V_MBA = modello(op.VoterModel(ba_model),"VoterModel", i)
    Vot_MHub, t_V_MHub = modello(op.VoterModel(real_G),"VoterModel", list(sorted_deg.keys())[:int(nnode*i)])
    Vot_MRN, t_V_MRN = modello(op.VoterModel(er_model),"VoterModel", i)
    x.append(i)
    y.append(np.mean(t_V_M[-1]['trends']['node_count'][1][190:])/nnode)
    yerr.append(np.var(t_V_M[-1]['trends']['node_count'][1][190:])/nnode)
    yBA.append(np.mean(t_V_MBA[-1]['trends']['node_count'][1][190:])/nnode)
    yerrBA.append(np.var(t_V_MBA[-1]['trends']['node_count'][1][190:])/nnode)
    yHub.append(np.mean(t_V_MHub[-1]['trends']['node_count'][1][190:])/nnode)
    yerrHub.append(np.var(t_V_MHub[-1]['trends']['node_count'][1][190:])/nnode)
    yRN.append(np.mean(t_V_MRN[-1]['trends']['node_count'][1][190:])/nnode)
    yerrRN.append(np.var(t_V_MRN[-1]['trends']['node_count'][1][190:])/nnode)
plt.errorbar(x, y,yerr, color='blue',label='Voter', marker='.')
plt.errorbar(x, yBA,yerrBA, color='red',label='Voter BAgraph', marker='.')
plt.errorbar(x, yHub,yerrHub, color='green',label='Voter Hub', marker='.')
plt.errorbar(x, yRN,yerrRN, color='black',label='Voter ER graph', marker='.')
plt.legend(loc='best')
plt.xlabel("Starting percentage of infected")
plt.ylabel("Final percantage (over last 10 iterations)")
plt.show()


# In[25]:


#SZNAJD
y=[]
yerr=[]
yBA=[]
yerrBA=[]
yHub=[]
yerrHub=[]
yRN=[]
yerrRN=[]
x=[]
degrees = dict(real_G.degree())
sorted_deg=dict(sorted(degrees.items(), key=lambda x:x[1], reverse=True))
nnode=real_G.number_of_nodes()
for i in range (1,10):
    i = i*0.1
    Vot_M, t_V_M = modello(op.SznajdModel(real_G),"SznajdModel", i)
    Vot_MBA, t_V_MBA = modello(op.SznajdModel(ba_model),"SznajdModel", i)
    Vot_MHub, t_V_MHub = modello(op.SznajdModel(real_G),"SznajdModel", list(sorted_deg.keys())[:int(nnode*i)])
    Vot_MRN, t_V_MRN = modello(op.SznajdModel(er_model),"SznajdModel", i)
    x.append(i)
    y.append(np.mean(t_V_M[-1]['trends']['node_count'][1][190:])/nnode)
    yerr.append(np.var(t_V_M[-1]['trends']['node_count'][1][190:])/nnode)
    yBA.append(np.mean(t_V_MBA[-1]['trends']['node_count'][1][190:])/nnode)
    yerrBA.append(np.var(t_V_MBA[-1]['trends']['node_count'][1][190:])/nnode)
    yHub.append(np.mean(t_V_MHub[-1]['trends']['node_count'][1][195:])/nnode)
    yerrHub.append(np.var(t_V_MHub[-1]['trends']['node_count'][1][195:])/nnode)
    yRN.append(np.mean(t_V_MRN[-1]['trends']['node_count'][1][190:])/nnode)
    yerrRN.append(np.var(t_V_MRN[-1]['trends']['node_count'][1][190:])/nnode)
plt.errorbar(x, y,yerr, color='blue',label='SznajdModel', marker='.')
plt.errorbar(x, yBA,yerrBA, color='red',label='SznajdModel BAgraph', marker='.')
plt.errorbar(x, yHub,yerrHub, color='green',label='SznajdModel Hub', marker='.')
plt.errorbar(x, yRN,yerrRN, color='black',label='SznajdModel ER graph', marker='.')
plt.legend(loc='best')
plt.xlabel("Starting percentage of infected")
plt.ylabel("Final percantage (over last 10 iterations)")
plt.show()


# In[ ]:


#QVOTER


# In[26]:


y=[]
yerr=[]
yBA=[]
yerrBA=[]
yHub=[]
yerrHub=[]
yRN=[]
yerrRN=[]
x=[]
degrees = dict(real_G.degree())
sorted_deg=dict(sorted(degrees.items(), key=lambda x:x[1], reverse=True))
nnode=real_G.number_of_nodes()
for i in range (1,10):
    i = i*0.1
    Vot_M, t_V_M = modello(op.QVoterModel(real_G),"QVoterModel", i,10)
    Vot_MBA, t_V_MBA = modello(op.QVoterModel(ba_model),"QVoterModel", i,10)
    Vot_MHub, t_V_MHub = modello(op.QVoterModel(real_G),"QVoterModel", list(sorted_deg.keys())[:int(nnode*i)],10)
    Vot_MRN, t_V_MRN = modello(op.QVoterModel(er_model),"QVoterModel", i,10)
    x.append(i)
    y.append(np.mean(t_V_M[-1]['trends']['node_count'][1][190:])/nnode)
    yerr.append(np.var(t_V_M[-1]['trends']['node_count'][1][190:])/nnode)
    yBA.append(np.mean(t_V_MBA[-1]['trends']['node_count'][1][190:])/nnode)
    yerrBA.append(np.var(t_V_MBA[-1]['trends']['node_count'][1][190:])/nnode)
    yHub.append(np.mean(t_V_MHub[-1]['trends']['node_count'][1][195:])/nnode)
    yerrHub.append(np.var(t_V_MHub[-1]['trends']['node_count'][1][195:])/nnode)
    yRN.append(np.mean(t_V_MRN[-1]['trends']['node_count'][1][190:])/nnode)
    yerrRN.append(np.var(t_V_MRN[-1]['trends']['node_count'][1][190:])/nnode)
plt.errorbar(x, y,yerr, color='blue',label='QVoter', marker='.')
plt.errorbar(x, yBA,yerrBA, color='red',label='QVoter BAgraph', marker='.')
plt.errorbar(x, yHub,yerrHub, color='green',label='QVoter Hub', marker='.')
plt.errorbar(x, yRN,yerrRN, color='black',label='QVoter ER graph', marker='.')
plt.legend(loc='best')
plt.xlabel("Starting percentage of infected")
plt.ylabel("Final percantage (over last 50 iterations)")
plt.show()


# In[27]:


y=[]
yerr=[]
yBA=[]
yerrBA=[]
yHub=[]
yerrHub=[]
yRN=[]
yerrRN=[]
x=[]
degrees = dict(real_G.degree())
sorted_deg=dict(sorted(degrees.items(), key=lambda x:x[1], reverse=True))
nnode=real_G.number_of_nodes()
for i in range (1,100,4):
    Vot_M, t_V_M = modello(op.QVoterModel(real_G),"QVoterModel", 0.3,i)
    Vot_MBA, t_V_MBA = modello(op.QVoterModel(ba_model),"QVoterModel", 0.3,i)
    Vot_MHub, t_V_MHub = modello(op.QVoterModel(real_G),"QVoterModel", list(sorted_deg.keys())[:int(nnode*0.3)],i)
    Vot_MRN, t_V_MRN = modello(op.QVoterModel(er_model),"QVoterModel", 0.3,i)
    x.append(i)
    y.append(np.mean(t_V_M[-1]['trends']['node_count'][1][190:])/nnode)
    yerr.append(np.var(t_V_M[-1]['trends']['node_count'][1][190:])/nnode)
    yBA.append(np.mean(t_V_MBA[-1]['trends']['node_count'][1][190:])/nnode)
    yerrBA.append(np.var(t_V_MBA[-1]['trends']['node_count'][1][190:])/nnode)
    yHub.append(np.mean(t_V_MHub[-1]['trends']['node_count'][1][190:])/nnode)
    yerrHub.append(np.var(t_V_MHub[-1]['trends']['node_count'][1][190:])/nnode)
    yRN.append(np.mean(t_V_MRN[-1]['trends']['node_count'][1][190:])/nnode)
    yerrRN.append(np.var(t_V_MRN[-1]['trends']['node_count'][1][190:])/nnode)
plt.errorbar(x, y,yerr, color='blue',label='QVoter', marker='.')
plt.errorbar(x, yBA,yerrBA, color='red',label='QVoter BAgraph', marker='.')
plt.errorbar(x, yHub,yerrHub, color='green',label='QVoter Hub', marker='.')
plt.errorbar(x, yRN,yerrRN, color='black',label='QVoter ER graph', marker='.')
plt.legend(loc='best')
plt.xlabel("Q value")
plt.ylabel("Final percantage (over last 10 iterations starting at 0.3)")
plt.show()


# In[ ]:


#majority rule


# In[28]:


y=[]
yerr=[]
yBA=[]
yerrBA=[]
yHub=[]
yerrHub=[]
yRN=[]
yerrRN=[]
x=[]
degrees = dict(real_G.degree())
sorted_deg=dict(sorted(degrees.items(), key=lambda x:x[1], reverse=True))
nnode=real_G.number_of_nodes()
for i in range (1,10):
    i = i*0.1
    Vot_M, t_V_M = modello(op.MajorityRuleModel(real_G),"VoterModel", i,10)
    Vot_MBA, t_V_MBA = modello(op.MajorityRuleModel(ba_model),"VoterModel", i,10)
    Vot_MHub, t_V_MHub = modello(op.MajorityRuleModel(real_G),"VoterModel", list(sorted_deg.keys())[:int(nnode*i)],10)
    Vot_MRN, t_V_MRN = modello(op.MajorityRuleModel(er_model),"VoterModel", i,10)
    x.append(i)
    y.append(np.mean(t_V_M[-1]['trends']['node_count'][1][190:])/nnode)
    yerr.append(np.var(t_V_M[-1]['trends']['node_count'][1][190:])/nnode)
    yBA.append(np.mean(t_V_MBA[-1]['trends']['node_count'][1][190:])/nnode)
    yerrBA.append(np.var(t_V_MBA[-1]['trends']['node_count'][1][190:])/nnode)
    yHub.append(np.mean(t_V_MHub[-1]['trends']['node_count'][1][190:])/nnode)
    yerrHub.append(np.var(t_V_MHub[-1]['trends']['node_count'][1][190:])/nnode)
    yRN.append(np.mean(t_V_MRN[-1]['trends']['node_count'][1][190:])/nnode)
    yerrRN.append(np.var(t_V_MRN[-1]['trends']['node_count'][1][190:])/nnode)
plt.errorbar(x, y,yerr, color='blue',label='MajorityRuleModel', marker='.')
plt.errorbar(x, yBA,yerrBA, color='red',label='MajorityRuleModel BAgraph', marker='.')
plt.errorbar(x, yHub,yerrHub, color='green',label='MajorityRuleModel Hub', marker='.')
plt.errorbar(x, yRN,yerrRN, color='black',label='MajorityRuleModel ER graph', marker='.')
plt.legend(loc='best')
plt.xlabel("Starting percentage of infected")
plt.ylabel("Final percantage (over last 50 iterations)")
plt.show()


# In[29]:


y=[]
yerr=[]
yBA=[]
yerrBA=[]
yHub=[]
yerrHub=[]
yRN=[]
yerrRN=[]
x=[]
degrees = dict(real_G.degree())
sorted_deg=dict(sorted(degrees.items(), key=lambda x:x[1], reverse=True))
nnode=real_G.number_of_nodes()
for i in range (1,100,4):
    Vot_M, t_V_M = modello(op.MajorityRuleModel(real_G),"QVoterModel", 0.3,i)
    Vot_MBA, t_V_MBA = modello(op.MajorityRuleModel(ba_model),"QVoterModel", 0.3,i)
    Vot_MHub, t_V_MHub = modello(op.MajorityRuleModel(real_G),"QVoterModel", list(sorted_deg.keys())[:int(nnode*0.3)],i)
    Vot_MRN, t_V_MRN = modello(op.MajorityRuleModel(er_model),"VoterModel", 0.3,i)
    x.append(i)
    y.append(np.mean(t_V_M[-1]['trends']['node_count'][1][190:])/nnode)
    yerr.append(np.var(t_V_M[-1]['trends']['node_count'][1][190:])/nnode)
    yBA.append(np.mean(t_V_MBA[-1]['trends']['node_count'][1][190:])/nnode)
    yerrBA.append(np.var(t_V_MBA[-1]['trends']['node_count'][1][190:])/nnode)
    yHub.append(np.mean(t_V_MHub[-1]['trends']['node_count'][1][190:])/nnode)
    yerrHub.append(np.var(t_V_MHub[-1]['trends']['node_count'][1][190:])/nnode)
    yRN.append(np.mean(t_V_MRN[-1]['trends']['node_count'][1][190:])/nnode)
    yerrRN.append(np.var(t_V_MRN[-1]['trends']['node_count'][1][190:])/nnode)
plt.errorbar(x, y,yerr, color='blue',label='MajorityRuleModel', marker='.')
plt.errorbar(x, yBA,yerrBA, color='red',label='MajorityRuleModel BAgraph', marker='.')
plt.errorbar(x, yHub,yerrHub, color='green',label='MajorityRuleModel Hub', marker='.')
plt.errorbar(x, yRN,yerrRN, color='black',label='MajorityRuleModel ER graph', marker='.')
plt.legend(loc='best')
plt.xlabel("Q value")
plt.ylabel("Final percantage (over last 10 iterations starting at 0.3)")
plt.show()


# In[ ]:


#AlgBias


# In[32]:


infect_percentage=0.2
m,iterations= modello(op.AlgorithmicBiasModel(H) ,nome="CognitiveOpDynModel2(H)", infected=None, q=0,ODM=None, ABM=(0.3,0.2))
media=[]
err=[]
l=[]
a=0
for i in iterations:#there might be a bug on the program or the 1 status keep the node away from the trend list
    media.append(np.mean(list(i['status'].values())))
    err.append(np.var(list(i['status'].values())))
    l.append(a)
    a+=1
    
    
m,iterationsER= modello(op.AlgorithmicBiasModel(er_model) ,nome="CognitiveOpDynModel2(H)", infected=None, q=0,ODM=None, ABM=(0.3,0.2))
mediaER=[]
errER=[]
for i in iterationsER:#there might be a bug on the program or the 1 status keep the node away from the trend list
    mediaER.append(np.mean(list(i['status'].values())))
    errER.append(np.var(list(i['status'].values())))

m,iterationsBA= modello(op.AlgorithmicBiasModel(ba_model) ,nome="CognitiveOpDynModel2(H)", infected=None, q=0,ODM=None, ABM=(0.3,0.2))
mediaBA=[]
errBA=[]
for i in iterationsBA:#there might be a bug on the program or the 1 status keep the node away from the trend list
    mediaBA.append(np.mean(list(i['status'].values())))
    errBA.append(np.var(list(i['status'].values())))    
    

m,iterationsHUB= modello(AlgorithmicB3(H) ,nome="CognitiveOpDynModel2(H)", infected=None, q=0, ODM=None, ABM=(0.3,0.2))
mediaHUB=[]
errHUB=[]
for i in iterationsHUB:#there might be a bug on the program or the 1 status keep the node away from the trend list
    mediaHUB.append(np.mean(list(i['status'].values())))
    errHUB.append(np.var(list(i['status'].values())))      
    
plt.errorbar(l, media,yerr=err, color='blue', marker='.', label='Normal')
plt.errorbar(l, mediaER,yerr=errER, color='red', marker='.', label='ER model')
plt.errorbar(l, mediaBA,yerr=errBA, color='green', marker='.', label='BA model')
plt.errorbar(l, mediaHUB,yerr=errHUB, color='black', marker='.', label='Hubs(0.2)')
plt.legend(loc='best')
plt.xlabel("Iteration")
plt.ylabel("Status mean value")
plt.show()


# In[33]:


y=[]
yer=[]
for i in range (0,10):
    infect_percentage=i*0.1
    m,iterationsHUB= modello(AlgorithmicB3(H) ,nome="CognitiveOpDynModel2(H)", infected=None, q=0, ODM=None, ABM=(0.3,20))
    mediaHUB=[]
    errHUB=[]
    l=[]
    a=0
    
    for i in iterationsHUB:#there might be a bug on the program or the 1 status keep the node away from the trend list
        mediaHUB.append(np.mean(list(i['status'].values())))
        errHUB.append(np.var(list(i['status'].values())))
        l.append(a)
        a+=1
    y.append(mediaHUB)
    yer.append(errHUB)
i=0
for yy, yerr in zip(y,yer):
    plt.errorbar(l, yy,yerr=yerr, marker='.', label=str(i*0.1))
    i+=1
plt.legend(loc='best')
plt.xlabel("Iteration")
plt.ylabel("Status mean value")
plt.show()


# In[ ]:


deltaHub=[]
errHub=[]
delta=[]
err=[]
deltaBA=[]
errBA=[]
deltaER=[]
errER=[]
param=[]
infect_percentage=0.2
for q in range(1,100,10):
    param.append(q)
    m,iterations= modello(AlgorithmicB3(H) ,nome="CognitiveOpDynModel3(H)", infected=None, q=0, ODM=None, ABM=(0.8,q))
    deltaHub.append(np.mean(list(iterations[-1]['status'].values()))-np.mean(list(iterations[0]['status'].values())))
    errHub.append(np.var(list(iterations[-1]['status'].values()))+np.var(list(iterations[0]['status'].values())))
    
    
    m,iterations= modello(op.AlgorithmicBiasModel(H) ,nome="CognitiveOpDynModel3(H)", infected=None, q=0, ODM=None, ABM=(0.8,q))
    delta.append(np.mean(list(iterations[-1]['status'].values()))-np.mean(list(iterations[0]['status'].values())))
    err.append(np.var(list(iterations[-1]['status'].values()))+np.var(list(iterations[0]['status'].values())))

    
    m,iterations= modello(op.AlgorithmicBiasModel(ba_model) ,nome="CognitiveOpDynModel3(H)", infected=None, q=0, ODM=None, ABM=(0.8,q))
    deltaBA.append(np.mean(list(iterations[-1]['status'].values()))-np.mean(list(iterations[0]['status'].values())))
    errBA.append(np.var(list(iterations[-1]['status'].values()))+np.var(list(iterations[0]['status'].values())))
    
    m,iterations= modello(op.AlgorithmicBiasModel(er_model) ,nome="CognitiveOpDynModel3(H)", infected=None, q=0, ODM=None, ABM=(0.8,q))
    deltaER.append(np.mean(list(iterations[-1]['status'].values()))-np.mean(list(iterations[0]['status'].values())))
    errER.append(np.var(list(iterations[-1]['status'].values()))+np.var(list(iterations[0]['status'].values())))

plt.errorbar(param, delta,err, color='blue',label='Normal', marker='.')
plt.errorbar(param, deltaBA,errBA, color='red',label='BA model', marker='.')
plt.errorbar(param, deltaER,errER, color='green',label='ER model', marker='.')
plt.errorbar(param, deltaHub,errHub, color='purple',label='Hubs(0.2)', marker='.')
plt.legend(loc='best')
plt.xlabel("Gamma (epsilone=0.2)")
plt.ylabel("Delta value")
plt.show()


# In[ ]:


deltaHub=[]
errHub=[]
delta=[]
err=[]
deltaBA=[]
errBA=[]
deltaER=[]
errER=[]
param=[]
infect_percentage=0.2
for q in range(1,10):
    q=q*0.1
    param.append(q)
    m,iterations= modello(AlgorithmicB3(H) ,nome="CognitiveOpDynModel3(H)", infected=None, q=0, ODM=None, ABM=(q,50))
    deltaHub.append(np.mean(list(iterations[-1]['status'].values()))-np.mean(list(iterations[0]['status'].values())))
    errHub.append(np.var(list(iterations[-1]['status'].values()))+np.var(list(iterations[0]['status'].values())))
    
    
    m,iterations= modello(op.AlgorithmicBiasModel(H) ,nome="CognitiveOpDynModel3(H)", infected=None, q=0, ODM=None, ABM=(q,50))
    delta.append(np.mean(list(iterations[-1]['status'].values()))-np.mean(list(iterations[0]['status'].values())))
    err.append(np.var(list(iterations[-1]['status'].values()))+np.var(list(iterations[0]['status'].values())))

    
    m,iterations= modello(op.AlgorithmicBiasModel(ba_model) ,nome="CognitiveOpDynModel3(H)", infected=None, q=0, ODM=None, ABM=(q,50))
    deltaBA.append(np.mean(list(iterations[-1]['status'].values()))-np.mean(list(iterations[0]['status'].values())))
    errBA.append(np.var(list(iterations[-1]['status'].values()))+np.var(list(iterations[0]['status'].values())))
    
    m,iterations= modello(op.AlgorithmicBiasModel(er_model) ,nome="CognitiveOpDynModel3(H)", infected=None, q=0, ODM=None, ABM=(q,50))
    deltaER.append(np.mean(list(iterations[-1]['status'].values()))-np.mean(list(iterations[0]['status'].values())))
    errER.append(np.var(list(iterations[-1]['status'].values()))+np.var(list(iterations[0]['status'].values())))

plt.errorbar(param, delta,err, color='blue',label='Normal', marker='.')
plt.errorbar(param, deltaBA,errBA, color='red',label='BA model', marker='.')
plt.errorbar(param, deltaER,errER, color='green',label='ER model', marker='.')
plt.errorbar(param, deltaHub,errHub, color='purple',label='Hubs(0.2)', marker='.')
plt.legend(loc='best')
plt.xlabel("Epsilon (gamma =50)")
plt.ylabel("Delta value")
plt.show()


# In[ ]:


#cognitive opinion


# In[34]:


infect_percentage=0.2

m,iterations= modello(op.CognitiveOpDynModel(H) ,nome="CognitiveOpDynModel2(H)", infected=None, q=0, ODM=0.3, ABM=None)
media=[]
err=[]
l=[]
a=0
for i in iterations:#there might be a bug on the program or the 1 status keep the node away from the trend list
    media.append(np.mean(list(i['status'].values())))
    err.append(np.var(list(i['status'].values())))
    l.append(a)
    a+=1
    
    
m,iterationsER= modello(op.CognitiveOpDynModel(er_model) ,nome="CognitiveOpDynModel2(H)", infected=None, q=0, ODM=0.3, ABM=None)
mediaER=[]
errER=[]
for i in iterationsER:#there might be a bug on the program or the 1 status keep the node away from the trend list
    mediaER.append(np.mean(list(i['status'].values())))
    errER.append(np.var(list(i['status'].values())))

m,iterationsBA= modello(op.CognitiveOpDynModel(ba_model) ,nome="CognitiveOpDynModel2(H)", infected=None, q=0, ODM=0.3, ABM=None)
mediaBA=[]
errBA=[]
for i in iterationsBA:#there might be a bug on the program or the 1 status keep the node away from the trend list
    mediaBA.append(np.mean(list(i['status'].values())))
    errBA.append(np.var(list(i['status'].values())))    
    

m,iterationsHUB= modello(CognitiveOpDynModel3(H)  ,nome="CognitiveOpDynModel2(H)", infected=None, q=0, ODM=0.3, ABM=None)
mediaHUB=[]
errHUB=[]
for i in iterationsHUB:#there might be a bug on the program or the 1 status keep the node away from the trend list
    mediaHUB.append(np.mean(list(i['status'].values())))
    errHUB.append(np.var(list(i['status'].values())))      
    
plt.errorbar(l, media,yerr=err, color='blue', marker='.', label='Normal')
plt.errorbar(l, mediaER,yerr=errER, color='red', marker='.', label='ER model')
plt.errorbar(l, mediaBA,yerr=errBA, color='green', marker='.', label='BA model')
plt.errorbar(l, mediaHUB,yerr=errHUB, color='black', marker='.', label='Hubs(0.2)')
plt.legend(loc='best')
plt.xlabel("Iteration")
plt.ylabel("Status mean value")
plt.show()


# In[35]:


y=[]
yer=[]
for i in range (0,10):
    infect_percentage=i*0.1
    m,iterationsHUB= modello(CognitiveOpDynModel3(H)  ,nome="CognitiveOpDynModel2(H)", infected=None, q=0, ODM=0.3, ABM=None)
    mediaHUB=[]
    errHUB=[]
    l=[]
    a=0
    
    for i in iterationsHUB:#there might be a bug on the program or the 1 status keep the node away from the trend list
        mediaHUB.append(np.mean(list(i['status'].values())))
        errHUB.append(np.var(list(i['status'].values())))
        l.append(a)
        a+=1
    y.append(mediaHUB)
    yer.append(errHUB)
i=0
for yy, yerr in zip(y,yer):
    plt.errorbar(l, yy,yerr=yerr, marker='.', label=str(i*0.1))
    i+=1
plt.legend(loc='best')
plt.xlabel("Iteration")
plt.ylabel("Status mean value")
plt.show()


# In[ ]:


deltaHub=[]
errHub=[]
delta=[]
err=[]
deltaBA=[]
errBA=[]
deltaER=[]
errER=[]
param=[]
infect_percentage=0.2
for q in range(1,100,10):
    q=q*0.01
    param.append(q)
    m,iterations= modello(CognitiveOpDynModel3(H) ,nome="CognitiveOpDynModel3(H)", infected=None, q=0, ODM=q, ABM=None)
    deltaHub.append(np.mean(list(iterations[-1]['status'].values()))-np.mean(list(iterations[0]['status'].values())))
    errHub.append(np.var(list(iterations[-1]['status'].values()))+np.var(list(iterations[0]['status'].values())))
    
    
    m,iterations= modello(op.CognitiveOpDynModel(H) ,nome="CognitiveOpDynModel3(H)", infected=None, q=0, ODM=q, ABM=None)
    delta.append(np.mean(list(iterations[-1]['status'].values()))-np.mean(list(iterations[0]['status'].values())))
    err.append(np.var(list(iterations[-1]['status'].values()))+np.var(list(iterations[0]['status'].values())))

    
    m,iterations= modello(op.CognitiveOpDynModel(ba_model) ,nome="CognitiveOpDynModel3(H)", infected=None, q=0, ODM=q, ABM=None)
    deltaBA.append(np.mean(list(iterations[-1]['status'].values()))-np.mean(list(iterations[0]['status'].values())))
    errBA.append(np.var(list(iterations[-1]['status'].values()))+np.var(list(iterations[0]['status'].values())))
    
    m,iterations= modello(op.CognitiveOpDynModel(er_model) ,nome="CognitiveOpDynModel3(H)", infected=None, q=0, ODM=q, ABM=None)
    deltaER.append(np.mean(list(iterations[-1]['status'].values()))-np.mean(list(iterations[0]['status'].values())))
    errER.append(np.var(list(iterations[-1]['status'].values()))+np.var(list(iterations[0]['status'].values())))

plt.errorbar(param, delta,err, color='blue',label='Normal', marker='.')
plt.errorbar(param, deltaBA,errBA, color='red',label='BA model', marker='.')
plt.errorbar(param, deltaER,errER, color='green',label='ER model', marker='.')
plt.errorbar(param, deltaHub,errHub, color='purple',label='Hubs(0.2)', marker='.')
plt.legend(loc='best')
plt.xlabel("External Information parameter")
plt.ylabel("Status mean value")
plt.show()


# In[ ]:





# In[ ]:





# In[31]:


#algortbias and cognitive per hubs



degrees = dict(H.degree())
sorted_deg=dict(sorted(degrees.items(), key=lambda x:x[1], reverse=True))

class AlgorithmicB3(DiffusionModel):
    """
    Model Parameters to be specified via ModelConfig
    :param epsilon: bounded confidence threshold from the Deffuant model, in [0,1]
    :param gamma: strength of the algorithmic bias, positive, real
    Node states are continuous values in [0,1].
    The initial state is generated randomly uniformly from the domain [0,1].
    """

    def __init__(self, graph, seed=None):
        """
             Model Constructor
             :param graph: A networkx graph object
         """
        super(self.__class__, self).__init__(graph, seed)

        self.discrete_state = False

        self.available_statuses = {
            "Infected": 0
        }

        self.parameters = {
            "model": {
                "epsilon": {
                    "descr": "Bounded confidence threshold",
                    "range": [0, 1],
                    "optional": False
                },
                "gamma": {
                    "descr": "Algorithmic bias",
                    "range": [0, 100],
                    "optional": False
                }
            },
            "nodes": {},
            "edges": {}
        }

        self.name = "Agorithmic Bias"

        self.node_data = {}
        self.ids = None
        self.sts = None

    def set_initial_status(self, configuration=None):
        """
        Override behaviour of methods in class DiffusionModel.
        Overwrites initial status using random real values.
        """
        super(AlgorithmicB3, self).set_initial_status(configuration)

        # set node status
        for node in self.status:
                if node in list(sorted_deg.keys())[:int(nnode*infect_percentage)]:
                    self.status[node]= 0.99
                else:
                    self.status[node]=np.random.rand()
                    
        self.initial_status = self.status.copy()

        ### Initialization numpy representation

        max_edgees = (self.graph.number_of_nodes() * (self.graph.number_of_nodes() - 1)) / 2
        nids = np.array(list(self.status.items()))
        self.ids = nids[:, 0]

        if max_edgees == self.graph.number_of_edges():
            self.sts = nids[:, 1]

        else:
            for i in self.graph.nodes:
                i_neigh = list(self.graph.neighbors(i))
                i_ids = nids[:, 0][i_neigh]
                i_sts = nids[:, 1][i_neigh]
                # non uso mai node_data[:,1]
                # per tenere aggiornato node_data() all'interno del for dovrei ciclare ogni item=nodo
                # e se uno dei suoi vicini è n1 o n2 aggiornare l'array sts
                # la complessità dovrebbe essere O(N)
                # se invece uso solo actual_status, considerando che per ogni nodo ho la lista dei neighbors in memoria
                # a ogni ciclo devo soltanto tirarmi fuori la lista degli stati degli avg_k vicini e prendere i
                # loro stati da actual_status
                # quindi la complessità dovrebbe essere O(N*p) < O(N)
                # sto pensando ad un modo per farlo in O(1) ma non mi è ancora venuto in mente

                self.node_data[i] = (i_ids, i_sts)

    # def clean_initial_status(self, valid_status=None):
    #     for n, s in future.utils.iteritems(self.status):
    #         if s > 1 or s < 0:
    #             self.status[n] = 0

    @staticmethod
    def prob(distance, gamma, min_dist):
        if distance < min_dist:
            distance = min_dist
        return np.power(distance, -gamma)

    def pb1(self, statuses, i_status):
        dist = np.abs(statuses - i_status)
        null = np.full(statuses.shape[0], 0.00001)
        max_base = np.maximum(dist, null)
        dists = max_base ** -self.params['model']['gamma']
        return dists

    def iteration(self, node_status=True):
        """
        Execute a single model iteration
        :return: Iteration_id, Incremental node status (dictionary node->status)
        """
        # One iteration changes the opinion of N agent pairs using the following procedure:
        # - first one agent is selected
        # - then a second agent is selected based on a probability that decreases with the distance to the first agent
        # - if the two agents have a distance smaller than epsilon, then they change their status to the average of
        # their previous statuses

        actual_status = self.status.copy()

        if self.actual_iteration == 0:
            self.actual_iteration += 1
            delta, node_count, status_delta = self.status_delta(self.status)
            if node_status:
                return {"iteration": 0, "status": actual_status,
                        "node_count": node_count.copy(), "status_delta": status_delta.copy()}
            else:
                return {"iteration": 0, "status": {},
                        "node_count": node_count.copy(), "status_delta": status_delta.copy()}

        n = self.graph.number_of_nodes()

        # interact with peers
        for i in range(0, n):

            # ho rimesso la selezione del nodo a random
            # n1 = list(self.graph.nodes)[np.random.randint(0, n)]
            n1 = int(choice(self.ids))

            if len(self.node_data) == 0:
                sts = self.sts
                ids = self.ids
                # toglie se stesso dalla lista degli id e degli status perché mi sembra rimanesse
                # e quindi con gamma alto a volte sceglieva se stesso per interagire
                neigh_sts = np.delete(sts, n1)
                neigh_ids = np.delete(ids, n1)
            else:
                neigh_ids = self.node_data[n1][0]
                neigh_sts = np.array([actual_status[id] for id in neigh_ids])

            # ho cambiato come crea l'array degli stati
            # niegh_sts = self.node_data[n1][1]

            # uso neigh_sts e actual_status[n1] come argomenti della funzione
            # perché altrimenti self.status[n1] è quello che viene dalla precedente
            # iterazione ma non viene aggiornato in corso di interazioni all'interno di questo for
            # e potrebbe essere cambiato in precedenza
            # e nel codice vecchio su usava invece lo stato sempre aggiornato

            # selection_prob = self.pb1(sts, self.status[n1])
            selection_prob = self.pb1(neigh_sts, actual_status[n1])

            # compute probabilities to select a second node among the neighbours
            total = np.sum(selection_prob)
            selection_prob = selection_prob / total
            cumulative_selection_probability = np.cumsum(selection_prob)

            r = np.random.random_sample()
            # n2 = np.argmax(cumulative_selection_probability >= r) -1
            n2 = np.argmax(
                cumulative_selection_probability >= r)
            # seleziono n2 dagli id dei neighbors di n1
            n2 = int(neigh_ids[n2])

            # update status of n1 and n2
            diff = np.abs(actual_status[n1] - actual_status[n2])

            if diff < self.params['model']['epsilon']:
                avg = (actual_status[n1] + actual_status[n2]) / 2.0
                actual_status[n1] = avg
                actual_status[n2] = avg
                # se la rete è completa aggiorno all'interno del ciclo
                # self.sts, così lo riprendo sempre aggiornato
                if len(self.node_data) == 0:
                    self.sts[n1] = avg
                    self.sts[n2] = avg

        # delta, node_count, status_delta = self.status_delta(actual_status)
        delta = actual_status
        node_count = {}
        status_delta = {}

        self.status = actual_status
        self.actual_iteration += 1

        if node_status:
            return {"iteration": self.actual_iteration - 1, "status": delta,
                    "node_count": node_count.copy(), "status_delta": status_delta.copy()}
        else:
            return {"iteration": self.actual_iteration - 1, "status": {},
                    "node_count": node_count.copy(), "status_delta": status_delta.copy()}

    def steady_state(self, max_iterations=100000, nsteady=1000, sensibility=0.00001, node_status=True,
                     progress_bar=False):
        """
        Execute a bunch of model iterations
        :param max_iterations: the maximum number of iterations to execute
        :param nsteady: number of required stable states
        :param sensibility: sensibility check for a steady state
        :param node_status: if the incremental node status has to be returned.
        :param progress_bar: whether to display a progress bar, default False
        :return: a list containing for each iteration a dictionary {"iteration": iteration_id, "status": dictionary_node_to_status}
        """
        system_status = []
        steady_it = 0
        for it in tqdm.tqdm(range(0, max_iterations), disable=not progress_bar):
            its = self.iteration(node_status)

            if it > 0:
                old = np.array(list(system_status[-1]['status'].values()))
                actual = np.array(list(its['status'].values()))
                res = np.abs(old - actual)
                if np.all((res < sensibility)):
                    steady_it += 1
                else:
                    steady_it = 0

            system_status.append(its)
            if steady_it == nsteady:
                return system_status[:-nsteady]

        return system_status

class CognitiveOpDynModel3(DiffusionModel):
    """
    Model Parameters to be specified via ModelConfig
    :param I: external information value in [0,1]
    :param T_range_min: the minimum of the range of initial values for  T. Range [0,1].
    :param T_range_max: the maximum of the range of initial values for  T. Range [0,1].
    :param B_range_min: the minimum of the range of initial values for  B. Range [0,1]
    :param B_range_max: the maximum of the range of initial values for  B. Range [0,1].
    :param R_fraction_negative: fraction of individuals having the node parameter R=-1.
    :param R_fraction_positive: fraction of individuals having the node parameter R=1
    :param R_fraction_neutral: fraction of individuals having the node parameter R=0
    The following relation should hold: R_fraction_negative+R_fraction_neutral+R_fraction_positive=1.
    To achieve this, the fractions selected will be normalised to sum 1.
    Node states are continuous values in [0,1].
    The initial state is generated randomly uniformly from the domain defined by model parameters.
    """

    def __init__(self, graph, seed=None):
        """
             Model Constructor
             :param graph: A networkx graph object
         """
        super(self.__class__, self).__init__(graph, seed)

        self.discrete_state = False

        self.available_statuses = {
            "Infected": 0
        }

        self.parameters = {
            "model": {
                "I": {
                    "descr": "External information",
                    "range": [0, 1],
                    "optional": False
                },
                "T_range_min": {
                    "descr": "Minimum of the range of initial values for T",
                    "range": [0, 1],
                    "optional": False
                },
                "T_range_max": {
                    "descr": "Maximum of the range of initial values for T",
                    "range": [0, 1],
                    "optional": False
                },
                "B_range_min": {
                    "descr": "Minimum of the range of initial values for B",
                    "range": [0, 1],
                    "optional": False
                },
                "B_range_max": {
                    "descr": "Maximum of the range of initial values for B",
                    "range": [0, 1],
                    "optional": False
                },
                "R_fraction_negative": {
                    "descr": "Fraction of nodes having R=-1",
                    "range": [0, 1],
                    "optional": False
                },
                "R_fraction_neutral": {
                    "descr": "Fraction of nodes having R=0",
                    "range": [0, 1],
                    "optional": False
                },
                "R_fraction_positive": {
                    "descr": "Fraction of nodes having R=1",
                    "range": [0, 1],
                    "optional": False
                }
            },
            "nodes": {},
            "edges": {}
        }

        self.name = "Cognitive Opinion Dynamics"

    def set_initial_status(self, configuration=None):
        """
        Override behaviour of methods in class DiffusionModel.
        Overwrites initial status using random real values.
        Generates random node profiles.
        """
        super(CognitiveOpDynModel3, self).set_initial_status(configuration)

        # set node status
        for node in self.status:
                if node in list(sorted_deg.keys())[:int(nnode*infect_percentage)]:
                    self.status[node]= 0.99
                else:
                    self.status[node]=np.random.rand()
        self.initial_status = self.status.copy()

        # set new node parameters
        self.params['nodes']['cognitive'] = {}

        # first correct the input model parameters and retreive T_range, B_range and R_distribution
        T_range = (self.params['model']['T_range_min'], self.params['model']['T_range_max'])
        if self.params['model']['T_range_min'] > self.params['model']['T_range_max']:
            T_range = (self.params['model']['T_range_max'], self.params['model']['T_range_min'])

        B_range = (self.params['model']['B_range_min'], self.params['model']['B_range_max'])
        if self.params['model']['B_range_min'] > self.params['model']['B_range_max']:
            B_range = (self.params['model']['B_range_max'], self.params['model']['B_range_min'])
        s = float(self.params['model']['R_fraction_negative'] + self.params['model']['R_fraction_neutral'] +
                  self.params['model']['R_fraction_positive'])
        R_distribution = (self.params['model']['R_fraction_negative']/s, self.params['model']['R_fraction_neutral']/s,
                          self.params['model']['R_fraction_positive']/s)

        # then sample parameters from the ranges and distribution
        for node in self.graph.nodes:
            R_prob = np.random.random_sample()
            if R_prob < R_distribution[0]:
                R = -1
            elif R_prob < (R_distribution[0] + R_distribution[1]):
                R = 0
            else:
                R = 1
            # R, B and T parameters in a tuple
            self.params['nodes']['cognitive'][node] = (R,
                                                       B_range[0] + (B_range[1] - B_range[0])*np.random.random_sample(),
                                                       T_range[0] + (T_range[1] - T_range[0])*np.random.random_sample())

    def clean_initial_status(self, valid_status=None):
        for n, s in future.utils.iteritems(self.status):
            if s > 1 or s < 0:
                self.status[n] = 0

    def iteration(self, node_status=True):
        """
        Execute a single model iteration
        :return: Iteration_id, Incremental node status (dictionary node->status)
        """
        # One iteration changes the opinion of all agents using the following procedure:
        # - first all agents communicate with institutional information I using a deffuant like rule
        # - then random pairs of agents are selected to interact  (N pairs)
        # - interaction depends on state of agents but also internal cognitive structure

        self.clean_initial_status(None)

        actual_status = {node: nstatus for node, nstatus in future.utils.iteritems(self.status)}

        if self.actual_iteration == 0:
            self.actual_iteration += 1
            delta, node_count, status_delta = self.status_delta(self.status)
            if node_status:
                return {"iteration": 0, "status": self.status.copy(),
                        "node_count": node_count.copy(), "status_delta": status_delta.copy()}
            else:
                return {"iteration": 0, "status": {},
                        "node_count": node_count.copy(), "status_delta": status_delta.copy()}

        # first interact with I
        I = self.params['model']['I']
        for node in self.graph.nodes:
            T = self.params['nodes']['cognitive'][node][2]
            R = self.params['nodes']['cognitive'][node][0]
            actual_status[node] = actual_status[node] + T * (I - actual_status[node])
            if R == 1:
                actual_status[node] = 0.5 * (1 + actual_status[node])
            if R == -1:
                actual_status[node] *= 0.5

        # then interact with peers
        for i in range(0, self.graph.number_of_nodes()):
            # select a random node
            n1 = list(self.graph.nodes)[np.random.randint(0, self.graph.number_of_nodes())]

            # select all of the nodes neighbours (no digraph possible)
            neighbours = list(self.graph.neighbors(n1))
            if len(neighbours) == 0:
                continue

            # select second node - a random neighbour
            n2 = neighbours[np.random.randint(0, len(neighbours))]

            # update status of n1 and n2
            p1 = pow(actual_status[n1], 1.0 / self.params['nodes']['cognitive'][n1][1])
            p2 = pow(actual_status[n2], 1.0 / self.params['nodes']['cognitive'][n2][1])

            oldn1 = self.status[n1]
            if np.random.random_sample() < p2:  # if node 2 talks, node 1 gets changed
                T1 = self.params['nodes']['cognitive'][n1][2]
                R1 = self.params['nodes']['cognitive'][n1][0]
                actual_status[n1] += (1 - T1) * (actual_status[n2] - actual_status[n1])
                if R1 == 1:
                    actual_status[n1] = 0.5 * (1 + actual_status[n1])
                if R1 == -1:
                    actual_status[n1] *= 0.5
            if np.random.random_sample() < p1:  # if node 1 talks, node 2 gets changed
                T2 = self.params['nodes']['cognitive'][n2][2]
                R2 = self.params['nodes']['cognitive'][n2][0]
                actual_status[n2] += (1 - T2) * (oldn1 - actual_status[n2])
                if R2 == 1:
                    actual_status[n2] = 0.5 * (1 + actual_status[n2])
                if R2 == -1:
                    actual_status[n2] *= 0.5

        delta, node_count, status_delta = self.status_delta(actual_status)
        self.status = actual_status
        self.actual_iteration += 1

        if node_status:
            return {"iteration": self.actual_iteration - 1, "status": delta.copy(),
                    "node_count": node_count.copy(), "status_delta": status_delta.copy()}
        else:
            return {"iteration": self.actual_iteration - 1, "status": {},
                    "node_count": node_count.copy(), "status_delta": status_delta.copy()}

