# -*- coding: utf-8 -*-
"""Plot results read from a result set
"""
from __future__ import division
import os
import argparse
import collections
import logging

import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from icarus.util import Settings, Tree, config_logging, step_cdf
from icarus.tools import means_confidence_interval
from icarus.results import plot_lines, plot_bar_chart
from icarus.registry import RESULTS_READER


# Logger object
logger = logging.getLogger('plot')

# These lines prevent insertion of Type 3 fonts in figures
# Publishers don't want them
plt.rcParams['ps.useafm'] = True
plt.rcParams['pdf.use14corefonts'] = True

# If True text is interpreted as LaTeX, e.g. underscore are interpreted as 
# subscript. If False, text is interpreted literally
plt.rcParams['text.usetex'] = False

# Aspect ratio of the output figures
plt.rcParams['figure.figsize'] = 8, 5

# Size of font in legends
LEGEND_SIZE = 14

# Line width in pixels
LINE_WIDTH = 1.5

# Plot
PLOT_EMPTY_GRAPHS = True

# This dict maps strategy names to the style of the line to be used in the plots
# Off-path strategies: solid lines
# On-path strategies: dashed lines
# No-cache: dotted line
STRATEGY_STYLE = {
         'HR_SYMM':         'b-o',
         'HR_ASYMM':        'g-D',
         'HR_MULTICAST':    'm-^',         
         'HR_HYBRID_AM':    'c-s',
         'HR_HYBRID_SM':    'r-v',
         'LCE':             'b--p',
         'LCD':             'g-->',
         'CL4M':            'g-->',
         'PROB_CACHE':      'c--<',
         'RAND_CHOICE':     'r--<',
         'RAND_BERNOULLI':  'g--*',
         'NO_CACHE':        'k:o',
         'OPTIMAL':         'k-o'
                }

# This dict maps name of strategies to names to be displayed in the legend
STRATEGY_LEGEND = {
         'LCE':             'LCE',
         'LCD':             'LCD',
         'HR_SYMM':         'HR Symm',
         'HR_ASYMM':        'HR Asymm',
         'HR_MULTICAST':    'HR Multicast',         
         'HR_HYBRID_AM':    'HR Hybrid AM',
         'HR_HYBRID_SM':    'HR Hybrid SM',
         'CL4M':            'CL4M',
         'PROB_CACHE':      'ProbCache',
         'RAND_CHOICE':     'Random (choice)',
         'RAND_BERNOULLI':  'Random (Bernoulli)',
         'NO_CACHE':        'No caching',
         'OPTIMAL':         'Optimal'
                    }

# Color and hatch styles for bar charts of cache hit ratio and link load vs topology
STRATEGY_BAR_COLOR = {
    'LCE':          'k',
    'LCD':          '0.4',
    'NO_CACHE':     '0.5',
    'HR_ASYMM':     '0.6',
    'HR_SYMM':      '0.7'
    }

STRATEGY_BAR_HATCH = {
    'LCE':          None,
    'LCD':          '//',
    'NO_CACHE':     'x',
    'HR_ASYMM':     '+',
    'HR_SYMM':      '\\'
    }


def plot_cache_hits_vs_alpha(resultset, topology, cache_size, alpha_range, strategies, plotdir):
    if 'NO_CACHE' in strategies:
        strategies.remove('NO_CACHE')
    desc = {}
    desc['title'] = 'Cache hit ratio: T=%s C=%s' % (topology, cache_size)
    desc['ylabel'] = 'Cache hit ratio'
    desc['xlabel'] = u'Content distribution \u03b1'
    desc['xparam'] = ('workload', 'alpha')
    desc['xvals'] = alpha_range
    desc['filter'] = {'topology': {'name': topology},
                      'cache_placement': {'network_cache': cache_size}}
    desc['ymetrics'] = [('CACHE_HIT_RATIO', 'MEAN')]*len(strategies)
    desc['ycondnames'] = [('strategy', 'name')]*len(strategies)
    desc['ycondvals'] = strategies
    desc['errorbar'] = True
    desc['legend_loc'] = 'upper left'
    desc['line_style'] = STRATEGY_STYLE
    desc['legend'] = STRATEGY_LEGEND
    desc['plotempty'] = PLOT_EMPTY_GRAPHS
    plot_lines(resultset, desc, 'CACHE_HIT_RATIO_T=%s@C=%s.pdf'
               % (topology, cache_size), plotdir)


def plot_cache_hits_vs_cache_size(resultset, topology, alpha, cache_size_range, strategies, plotdir):
    desc = {}
    if 'NO_CACHE' in strategies:
        strategies.remove('NO_CACHE')
    desc['title'] = 'Cache hit ratio: T=%s A=%s' % (topology, alpha)
    desc['xlabel'] = u'Cache to population ratio'
    desc['ylabel'] = 'Cache hit ratio'
    desc['xscale'] = 'log'
    desc['xparam'] = ('cache_placement','network_cache')
    desc['xvals'] = cache_size_range
    desc['filter'] = {'topology': {'name': topology},
                      'workload': {'name': 'STATIONARY', 'alpha': alpha}}
    desc['ymetrics'] = [('CACHE_HIT_RATIO', 'MEAN')]*len(strategies)
    desc['ycondnames'] = [('strategy', 'name')]*len(strategies)
    desc['ycondvals'] = strategies
    desc['errorbar'] = True
    desc['legend_loc'] = 'upper left'
    desc['line_style'] = STRATEGY_STYLE
    desc['legend'] = STRATEGY_LEGEND
    desc['plotempty'] = PLOT_EMPTY_GRAPHS
    plot_lines(resultset, desc,'CACHE_HIT_RATIO_T=%s@A=%s.pdf'
               % (topology, alpha), plotdir)
    

def plot_link_load_vs_alpha(resultset, topology, cache_size, alpha_range, strategies, plotdir):
    desc = {}
    desc['title'] = 'Internal link load: T=%s C=%s' % (topology, cache_size)
    desc['xlabel'] = u'Content distribution \u03b1'
    desc['ylabel'] = 'Internal link load'
    desc['xparam'] = ('workload', 'alpha')
    desc['xvals'] = alpha_range
    desc['filter'] = {'topology': {'name': topology},
                      'cache_placement': {'network_cache': cache_size}}
    desc['ymetrics'] = [('LINK_LOAD', 'MEAN_INTERNAL')]*len(strategies)
    desc['ycondnames'] = [('strategy', 'name')]*len(strategies)
    desc['ycondvals'] = strategies
    desc['errorbar'] = True
    desc['legend_loc'] = 'upper right'
    desc['line_style'] = STRATEGY_STYLE
    desc['legend'] = STRATEGY_LEGEND
    desc['plotempty'] = PLOT_EMPTY_GRAPHS
    plot_lines(resultset, desc, 'LINK_LOAD_INTERNAL_T=%s@C=%s.pdf'
               % (topology, cache_size), plotdir)


def plot_link_load_vs_cache_size(resultset, topology, alpha, cache_size_range, strategies, plotdir):
    desc = {}
    desc['title'] = 'Internal link load: T=%s A=%s' % (topology, alpha)
    desc['xlabel'] = 'Cache to population ratio'
    desc['ylabel'] = 'Internal link load'
    desc['xscale'] = 'log'
    desc['xparam'] = ('cache_placement','network_cache')
    desc['xvals'] = cache_size_range
    desc['filter'] = {'topology': {'name': topology},
                      'workload': {'name': 'stationary', 'alpha': alpha}}
    desc['ymetrics'] = [('LINK_LOAD', 'MEAN_INTERNAL')]*len(strategies)
    desc['ycondnames'] = [('strategy', 'name')]*len(strategies)
    desc['ycondvals'] = strategies
    desc['errorbar'] = True
    desc['legend_loc'] = 'upper right'
    desc['line_style'] = STRATEGY_STYLE
    desc['legend'] = STRATEGY_LEGEND
    desc['plotempty'] = PLOT_EMPTY_GRAPHS
    plot_lines(resultset, desc, 'LINK_LOAD_INTERNAL_T=%s@A=%s.pdf'
               % (topology, alpha), plotdir)
    

def plot_latency_vs_alpha(resultset, topology, cache_size, alpha_range, strategies, plotdir):
    desc = {}
    desc['title'] = 'Latency: T=%s C=%s' % (topology, cache_size)
    desc['xlabel'] = u'Content distribution \u03b1'
    desc['ylabel'] = 'Latency (ms)'
    desc['xparam'] = ('workload', 'alpha')
    desc['xvals'] = alpha_range
    desc['filter'] = {'topology': {'name': topology},
                      'cache_placement': {'network_cache': cache_size}}
    desc['ymetrics'] = [('LATENCY', 'MEAN')]*len(strategies)
    desc['ycondnames'] = [('strategy', 'name')]*len(strategies)
    desc['ycondvals'] = strategies
    desc['errorbar'] = True
    desc['legend_loc'] = 'upper right'
    desc['line_style'] = STRATEGY_STYLE
    desc['legend'] = STRATEGY_LEGEND
    desc['plotempty'] = PLOT_EMPTY_GRAPHS
    plot_lines(resultset, desc, 'LATENCY_T=%s@C=%s.pdf'
               % (topology, cache_size), plotdir)


def plot_latency_vs_cache_size(resultset, topology, alpha, cache_size_range, strategies, plotdir):
    desc = {}
    desc['title'] = 'Latency: T=%s A=%s' % (topology, alpha)
    desc['xlabel'] = 'Cache to population ratio'
    desc['ylabel'] = 'Latency'
    desc['xscale'] = 'log'
    desc['xparam'] = ('cache_placement','network_cache')
    desc['xvals'] = cache_size_range
    desc['filter'] = {'topology': {'name': topology},
                      'workload': {'name': 'STATIONARY', 'alpha': alpha}}
    desc['ymetrics'] = [('LATENCY', 'MEAN')]*len(strategies)
    desc['ycondnames'] = [('strategy', 'name')]*len(strategies)
    desc['ycondvals'] = strategies
    desc['metric'] = ('LATENCY', 'MEAN')
    desc['errorbar'] = True
    desc['legend_loc'] = 'upper right'
    desc['line_style'] = STRATEGY_STYLE
    desc['legend'] = STRATEGY_LEGEND
    desc['plotempty'] = PLOT_EMPTY_GRAPHS
    plot_lines(resultset, desc, 'LATENCY_T=%s@A=%s.pdf'
               % (topology, alpha), plotdir)
    

def plot_cache_hits_vs_topology(resultset, alpha, cache_size, topology_range, strategies, plotdir):
    """
    Plot bar graphs of cache hit ratio for specific values of alpha and cache
    size for various topologies.
    
    The objective here is to show that our algorithms works well on all
    topologies considered
    """
    if 'NO_CACHE' in strategies:
        strategies.remove('NO_CACHE')
    desc = {}
    desc['title'] = 'Cache hit ratio: A=%s C=%s' % (alpha, cache_size)
    desc['ylabel'] = 'Cache hit ratio'
    desc['xparam'] = ('topology', 'name')
    desc['xvals'] = topology_range
    desc['filter'] = {'cache_placement': {'network_cache': cache_size},
                      'workload': {'name': 'STATIONARY', 'alpha': alpha}}
    desc['ymetrics'] = [('CACHE_HIT_RATIO', 'MEAN')]*len(strategies)
    desc['ycondnames'] = [('strategy', 'name')]*len(strategies)
    desc['ycondvals'] = strategies
    desc['errorbar'] = True
    desc['legend_loc'] = 'lower right'
    desc['bar_color'] = STRATEGY_BAR_COLOR
    desc['bar_hatch'] = STRATEGY_BAR_HATCH
    desc['legend'] = STRATEGY_LEGEND
    desc['plotempty'] = PLOT_EMPTY_GRAPHS
    plot_bar_chart(resultset, desc, 'CACHE_HIT_RATIO_A=%s_C=%s.pdf'
                   % (alpha, cache_size), plotdir)
    

def plot_link_load_vs_topology(resultset, alpha, cache_size, topology_range, strategies, plotdir):
    """
    Plot bar graphs of link load for specific values of alpha and cache
    size for various topologies.
    
    The objective here is to show that our algorithms works well on all
    topologies considered
    """
    desc = {}
    desc['title'] = 'Internal link load: A=%s C=%s' % (alpha, cache_size)
    desc['ylabel'] = 'Internal link load'
    desc['xparam'] = ('topology', 'name')
    desc['xvals'] = topology_range
    desc['filter'] = {'cache_placement': {'network_cache': cache_size},
                      'workload': {'name': 'STATIONARY', 'alpha': alpha}}
    desc['ymetrics'] = [('LINK_LOAD', 'MEAN_INTERNAL')]*len(strategies)
    desc['ycondnames'] = [('strategy', 'name')]*len(strategies)
    desc['ycondvals'] = strategies
    desc['errorbar'] = True
    desc['legend_loc'] = 'lower right'
    desc['bar_color'] = STRATEGY_BAR_COLOR
    desc['bar_hatch'] = STRATEGY_BAR_HATCH
    desc['legend'] = STRATEGY_LEGEND
    desc['plotempty'] = PLOT_EMPTY_GRAPHS
    plot_bar_chart(resultset, desc, 'LINK_LOAD_INTERNAL_A=%s_C=%s.pdf'
                   % (alpha, cache_size), plotdir)


def searchDictMultipleCat(lst, category_list, attr_value_pairs, num_pairs, collector, subtype):
    """
    Search the resultset list for a particular [category, attribute, value] parameter such as ['strategy', 'extra_quota', 3]. attr_value_pairs include the key-value pairs.
    and once such a key is found, extract the result for a collector, subtype such as ['CACHE_HIT_RATIO', 'MEAN']

    Returns the result if found in the dictionary lst; otherwise returns None

    """
    result = None
    for l in lst:
        num_match = 0
        for key, val in l[0].items():
            #print key + '-and-' + category + '-\n'
            if key in category_list:
                if (isinstance(val, dict)):
                    for key1, val1 in val.items():
                        for key2, val2 in attr_value_pairs.items():
                            if key1 == key2 and val1 == val2:
                                num_match = num_match + 1
                    if num_match == num_pairs:
                        result = l[1]
                        break
                else:
                    print 'Something is wrong with the search for attr-value pairs\n'
                    return None

        if result is not None:
            break
    
    if result is None:
        print 'Error searched attribute, value pairs:\n' 
        for k, v in attr_value_pairs.items():
            print '[ ' + repr(k) + ' , ' + repr(v) + ' ]  '
        print 'is not found, returning none\n'
        return None
    
    found = None
    for key, val in result.items():
        if key == collector:
            for key1, val1 in val.items():
                if key1 == subtype:
                    found = val1
                    break
            if found is not None:
                break

    if found is None:
        print 'Error searched collector, subtype ' + repr(collector) + ',' + repr(subtype) + 'is not found\n'

    return found

def searchDictMultipleCat1(lst, category_list, attr_value_list, num_pairs, collector, subtype):
    """
    Search the resultset list for a particular [category, attribute, value] parameter such as ['strategy', 'extra_quota', 3]. attr_value_pairs include the key-value pairs.
    and once such a key is found, extract the result for a collector, subtype such as ['CACHE_HIT_RATIO', 'MEAN']

    Returns the result if found in the dictionary lst; otherwise returns None

    """
    result = None
    for l in lst:
        num_match = 0
        for key, val in l[0].items():
            #print key + '-and-' + category + '-\n'
            if key in category_list:
                if (isinstance(val, dict)):
                    for key1, val1 in val.items():
                        for arr in attr_value_list:
                            key2 = arr[0]
                            val2 = arr[1]
                            if key1 == key2 and val1 == val2:
                                num_match = num_match + 1
                    if num_match == num_pairs:
                        result = l[1]
                        break
                else:
                    print 'Something is wrong with the search for attr-value pairs\n'
                    return None

        if result is not None:
            break
    
    if result is None:
        print 'Error searched attribute, value pairs:\n' 
        for arr in attr_value_list:
            k = arr[0]
            v = arr[1]
            print '[ ' + repr(k) + ' , ' + repr(v) + ' ]  '
        print 'is not found, returning none\n'
        return None
    
    found = None
    for key, val in result.items():
        if key == collector:
            for key1, val1 in val.items():
                if key1 == subtype:
                    found = val1
                    break
            if found is not None:
                break

    if found is None:
        print 'Error searched collector, subtype ' + repr(collector) + ',' + repr(subtype) + 'is not found\n'

    return found

def searchDict(lst, category, attr_value_pairs, num_pairs, collector, subtype):
    """
    Search the resultset list for a particular [category, attribute, value] parameter such as ['strategy', 'extra_quota', 3]. attr_value_pairs include the key-value pairs.
    and once such a key is found, extract the result for a collector, subtype such as ['CACHE_HIT_RATIO', 'MEAN']

    Returns the result if found in the dictionary lst; otherwise returns None

    """
    result = None
    for l in lst:
        for key, val in l[0].items():
            #print key + '-and-' + category + '-\n'
            if key == category:
                if (isinstance(val, dict)):
                    num_match = 0
                    for key1, val1 in val.items():
                        for key2, val2 in attr_value_pairs.items():
                            if key1 == key2 and val1 == val2:
                                num_match = num_match + 1
                    if num_match == num_pairs:
                        result = l[1]
                        break
                else:
                    print 'Something is wrong with the search for attr-value pairs\n'
                    return None
        if result is not None:
            break
    
    if result is None:
        print 'Error searched attribute, value pairs:\n' 
        for k, v in attr_value_pairs.items():
            print '[ ' + repr(k) + ' , ' + repr(v) + ' ]  '
        print 'is not found, returning none\n'
        return None
    
    found = None
    for key, val in result.items():
        if key == collector:
            for key1, val1 in val.items():
                if key1 == subtype:
                    found = val1
                    break
            if found is not None:
                break

    if found is None:
        print 'Error searched collector, subtype ' + repr(collector) + ',' + repr(subtype) + 'is not found\n'

    return found

def print_lru_probability_results(lst):

    probs = [0.1, 0.25, 0.50, 0.75, 1.0]
    strategies = ['LRU']

    for strategy in strategies:
        for p in probs:
            filename = 'sat_' + str(strategy) + '_' + str(p)
            f = open(filename, 'w')
            f.write('# Sat. rate for LRU over time\n')
            f.write('#\n')
            f.write('# Time     Sat. Rate\n')
            sat_times = searchDict(lst, 'strategy', {'name':  strategy, 'p' : p}, 2, 'LATENCY', 'SAT_TIMES')
            for k in sorted(sat_times):
                s = str(k[0][0]) + "\t" + str(k[1]) + "\n"
                f.write(s)
            f.close()
    
    for strategy in strategies:
        for p in probs:
            filename = 'idle_' + str(strategy) + '_' + str(p)
            f = open(filename, 'w')
            f.write('# Idle time of strategies over time\n')
            f.write('#\n')
            f.write('# Time     Idle percentage\n')
            idle_times = searchDict(lst, 'strategy', {'name':  strategy, 'p' : p}, 2, 'LATENCY', 'IDLE_TIMES')
            for k in sorted(idle_times):
                s = str(k[0][0]) + "\t" + str(k[1]) + "\n"
                f.write(s)
            f.close()

def print_strategies_performance(lst):

    strategies = ['SDF', 'HYBRID', 'MFU'] 
    service_budget = 500
    alpha = 0.75
    replacement_interval = 30.0
    n_services = 1000

    # Print Sat. rates:
    for strategy in strategies:
        filename = 'sat_' + str(strategy)
        f = open(filename, 'w')
        f.write('# Sat. rate over time\n')
        f.write('#\n')
        f.write('# Time     Sat. Rate\n')
        sat_times = searchDictMultipleCat(lst, ['strategy', 'computation_placement', 'workload'], {'name' : strategy, 'service_budget' : service_budget, 'alpha' : alpha}, 3, 'LATENCY', 'SAT_TIMES')
        for k in sorted(sat_times):
            s = str(k[0][0]) + "\t" + str(k[1]) + "\n"
            f.write(s)
        f.close()
    
    # Print Idle times:
    for strategy in strategies:
        filename = 'idle_' + str(strategy)
        f = open(filename, 'w')
        f.write('# Idle time of strategies over time\n')
        f.write('#\n')
        f.write('# Time     Idle percentage\n')
        idle_times = searchDictMultipleCat(lst, ['strategy', 'computation_placement', 'workload'], {'name' : strategy, 'service_budget' : service_budget, 'alpha' : alpha}, 3, 'LATENCY', 'IDLE_TIMES')
        for k in sorted(idle_times):
            s = str(k[0][0]) + "\t" + str(k[1]) + "\n"
            f.write(s)
        f.close()
    
    # Print per-service Sat. rates:
    for strategy in strategies:
        filename = 'sat_service_' + str(strategy)
        f = open(filename, 'w')
        f.write('# Per-service Sat. rate over time\n')
        f.write('#\n')
        f.write('# Time     Sat. Rate\n')
        sat_services = searchDictMultipleCat(lst, ['strategy', 'computation_placement', 'workload'], {'name' : strategy, 'service_budget' : service_budget, 'alpha' : alpha}, 3, 'LATENCY', 'PER_SERVICE_SATISFACTION')
        #f.write(str(sat_services))
        for indx in range(1, n_services):
            s = str(indx) + "\t" + str(sat_services[indx]) + "\n"
            f.write(s)
        f.close()

def print_scheduling_experiments(lst):
    strategies = ['SDF', 'HYBRID', 'MFU'] 
    schedule_policies = ['EDF', 'FIFO']
    service_budget = 500
    alpha = 0.75
    replacement_interval = 30.0

    # Print Sat. rates:
    for strategy in strategies:
        for policy in schedule_policies:
            filename = 'sat_' + str(strategy) + '_' + str(policy)
            f = open(filename, 'w')
            f.write('# Sat. rate over time\n')
            f.write('#\n')
            f.write('# Time     Sat. Rate\n')
            sat_times = searchDictMultipleCat1(lst, ['strategy', 'computation_placement', 'workload', 'sched_policy'], [['name', strategy], ['service_budget', service_budget], ['alpha', alpha], ['name', policy]], 4, 'LATENCY', 'SAT_TIMES')
            for k in sorted(sat_times):
                s = str(k[0][0]) + "\t" + str(k[1]) + "\n"
                f.write(s)
            f.close()
    
    # Print idle times:
    for strategy in strategies:
        for policy in schedule_policies:
            filename = 'idle_' + str(strategy) + '_' + str(policy)
            f = open(filename, 'w')
            f.write('# Idle times over time\n')
            f.write('#\n')
            f.write('# Time     Idle percentage\n')
            idle_times = searchDictMultipleCat1(lst, ['strategy', 'computation_placement', 'workload', 'sched_policy'], [['name', strategy], ['service_budget', service_budget], ['alpha', alpha], ['name', policy]], 4, 'LATENCY', 'IDLE_TIMES')
            for k in sorted(idle_times):
                s = str(k[0][0]) + "\t" + str((1.0*k[1])) + "\n"
                f.write(s)
            f.close()

def print_zipf_experiment(lst):
    
    strategies = ['SDF', 'HYBRID', 'MFU'] 
    alphas = [0.1, 0.25, 0.50, 0.75, 1.0]
    replacement_interval = 30.0
    service_budget = 500

    # Print Sat. rates:
    for strategy in strategies:
        for alpha in alphas:
            filename = 'sat_' + str(strategy) + '_' + str(alpha)
            f = open(filename, 'w')
            f.write('# Sat. rate over time\n')
            f.write('#\n')
            f.write('# Time     Sat. Rate\n')
            sat_times = searchDictMultipleCat(lst, ['strategy', 'computation_placement', 'workload'], {'name' : strategy, 'service_budget' : service_budget, 'alpha' : alpha}, 3, 'LATENCY', 'SAT_TIMES')
            for k in sorted(sat_times):
                s = str(k[0][0]) + "\t" + str(k[1]) + "\n"
                f.write(s)
            f.close()
    
    # Print Idle times:
    for strategy in strategies:
        for alpha in alphas:
            filename = 'idle_' + str(strategy) + '_' + str(alpha)
            f = open(filename, 'w')
            f.write('# Idle times over time\n')
            f.write('#\n')
            f.write('# Time     Idle percentage\n')
            idle_times = searchDictMultipleCat(lst, ['strategy', 'computation_placement', 'workload'], {'name' : strategy, 'service_budget' : service_budget, 'alpha' : alpha}, 3, 'LATENCY', 'IDLE_TIMES')
            for k in sorted(sat_times):
                s = str(k[0][0]) + "\t" + str((1.0*k[1])) + "\n"
                f.write(s)
            f.close()

def print_engagement_time_results(lst):
    """
    Print results for 2 services with varying engagement times. Second service is much more sensitive to changes in QoS compared to the first one.
    """
    
    strategies = ['DOUBLE_AUCTION']
    #service_times = [[120.0, 120.0], [135.0, 105.0], [150.0, 90.0], [165.0, 75.0], [180.0, 60.0], [105.0, 135.0], [90.0, 150.0], [75.0, 165.0], [60.0, 180.0]]
    service_times_service0_constant = [[60.0, 15.0], [60.0, 30.0], [60.0, 45.0], [60.0, 60.0], [60.0, 75.0], [60.0, 90.0], [60.0, 105.0], [60.0, 120.0]] 
    service_times_service1_constant = [[15.0, 60.0],[30.0, 60.0],[45.0, 60.0], [60,60], [75.0, 60.0], [90.0, 60.0], [105.0, 60.0], [120.0, 60.0]] 

    n_services = 2

    for strategy in ['DOUBLE_AUCTION']:
        filename = 'engagement_results_service0_constant'
        f = open(filename, 'w')
        f.write('# Engagement time results for two services\n')
        f.write('# Service 0 has u_min 50 (less demanding) and Service 1 has u_min 0\n')
        s = "EngagementTimes"
        s += "\tidle_times" 
        s += "\trevenue" 
        for serv in range(n_services):
            s += "\tvm_prices" + str(serv)
            s += "\tqos_service" + str(serv)
            s += "\tsat_rate" + str(serv)
        s += "\n"
        f.write(s)
        qos_service = None
        service_revenue = None
        vm_prices = None
        sat_rate = None 
        idle_times = None
        for service_time in service_times_service0_constant:
            s = str(int(service_time[1]))# + "-" + str(service_time[1])
            service_prices = searchDictMultipleCat(lst, ['strategy', 'netconf'], {'name' : strategy, 'service_times' : service_time}, 2, 'LATENCY', 'NODE_VM_PRICES')
            serv_utils = searchDictMultipleCat(lst, ['strategy', 'netconf'], {'name' : strategy, 'service_times' : service_time}, 2, 'LATENCY', 'IDLE_TIMES_AVG')
            service_prices = dict(service_prices)
            service_revenue = searchDictMultipleCat(lst, ['strategy', 'netconf'], {'name' : strategy, 'service_times' : service_time}, 2, 'LATENCY', 'REVENUE_TIMES_AVG')
            s += "\t" + str(serv_utils)
            s += "\t" + str(service_revenue)
            for serv in range(n_services):
                qos_service = searchDictMultipleCat(lst, ['strategy', 'netconf'], {'name' : strategy, 'service_times' : service_time}, 2, 'LATENCY', 'QoS_SERVICE')
                sat_rate = searchDictMultipleCat(lst, ['strategy', 'netconf'], {'name' : strategy, 'service_times' : service_time}, 2, 'LATENCY', 'SERVICE_SAT_RATE')
                s += "\t"  + str(service_prices[serv]) 
                s += "\t" + str(qos_service[serv]) + "\t" + str(sat_rate[serv]) 
            s += '\n'
            f.write(s)
        f.close()
            
    for strategy in ['DOUBLE_AUCTION']:
        filename = 'engagement_results_service1_constant'
        f = open(filename, 'w')
        f.write('# Engagement time results for two services\n')
        f.write('# Service 0 has u_min 50 (less demanding) and Service 1 has u_min 0\n')
        s = "EngagementTimes"
        s += "\tidle_times" 
        s += "\trevenue" 
        for serv in range(n_services):
            s += "\tvm_prices" + str(serv)
            s += "\tqos_service" + str(serv)
            s += "\tsat_rate" + str(serv)
        s += "\n"
        f.write(s)
        qos_service = None
        service_revenue = None
        vm_prices = None
        sat_rate = None 
        idle_times = None
        for service_time in service_times_service1_constant:
            s = str(int(service_time[0])) # + "-" + str(service_time[1])
            service_prices = searchDictMultipleCat(lst, ['strategy', 'netconf'], {'name' : strategy, 'service_times' : service_time}, 2, 'LATENCY', 'NODE_VM_PRICES')
            serv_utils = searchDictMultipleCat(lst, ['strategy', 'netconf'], {'name' : strategy, 'service_times' : service_time}, 2, 'LATENCY', 'IDLE_TIMES_AVG')
            service_prices = dict(service_prices)
            service_revenue = searchDictMultipleCat(lst, ['strategy', 'netconf'], {'name' : strategy, 'service_times' : service_time}, 2, 'LATENCY', 'REVENUE_TIMES_AVG')
            s += "\t" + str(serv_utils)
            s += "\t" + str(service_revenue)
            for serv in range(n_services):
                qos_service = searchDictMultipleCat(lst, ['strategy', 'netconf'], {'name' : strategy, 'service_times' : service_time}, 2, 'LATENCY', 'QoS_SERVICE')
                sat_rate = searchDictMultipleCat(lst, ['strategy', 'netconf'], {'name' : strategy, 'service_times' : service_time}, 2, 'LATENCY', 'SERVICE_SAT_RATE')
                s += "\t"  + str(service_prices[serv]) 
            
                s += "\t" + str(qos_service[serv]) + "\t" + str(sat_rate[serv]) 
            s += '\n'
            f.write(s)
        f.close()
    
def print_vm_results_tree(lst):
    """
    Print results for varying number of VMs for 1 node 2 services 10 classes
    """
    strategies = ['LFU_TRACE', 'DOUBLE_AUCTION', 'SELF_TUNING_TRACE', 'STATIC'] 
    strategies = ['DOUBLE_AUCTION'] 
    #num_cloudlets = 7
    num_cloudlets = 160
    #num_of_vms = [7*num_cloudlets, 14*num_cloudlets, 21*num_cloudlets, 28*num_cloudlets, 35*num_cloudlets, 42*num_cloudlets, 49*num_cloudlets, 56*num_cloudlets, 63*num_cloudlets, 70*num_cloudlets]
    #num_of_vms = [2*num_cloudlets, 7*num_cloudlets, 14*num_cloudlets, 21*num_cloudlets, 28*num_cloudlets, 35*num_cloudlets, 42*num_cloudlets, 49*num_cloudlets, 56*num_cloudlets, 63*num_cloudlets, 70*num_cloudlets]
    num_of_vms = [10*num_cloudlets, 20*num_cloudlets, 30*num_cloudlets, 40*num_cloudlets, 50*num_cloudlets, 60*num_cloudlets, 70*num_cloudlets, 80*num_cloudlets]
    num_classes = 10
    num_services = 1
    
    for strategy in strategies:
        filename = 'vm_results_' + str(strategy) 
        f = open(filename, 'w')
        f.write('# QoS for different number of VMs\n')
        f.write('#\n')
        s = "num_of_vms"
        #for serv in range(num_services):
        s += "\tqos"
        s += "\trevenue"
        s += "\tprice"
        s += "\tidle\n"
        f.write(s)
        qos_service = None
        service_revenue = None
        vm_prices = None
        sat_rate = None 
        idle_times = None
        for vms in num_of_vms:
            qos_avg = searchDictMultipleCat(lst, ['strategy', 'computation_placement'], {'name' : strategy, 'service_budget' : vms}, 2, 'LATENCY', 'QOS_TIMES_AVG')
            revenue_avg = searchDictMultipleCat(lst, ['strategy', 'computation_placement'], {'name' : strategy, 'service_budget' : vms}, 2, 'LATENCY', 'REVENUE_TIMES_AVG')
            price_avg = searchDictMultipleCat(lst, ['strategy', 'computation_placement'], {'name' : strategy, 'service_budget' : vms}, 2, 'LATENCY', 'PRICE_TIMES_AVG')
            idle_avg = searchDictMultipleCat(lst, ['strategy', 'computation_placement'], {'name' : strategy, 'service_budget' : vms}, 2, 'LATENCY', 'IDLE_TIMES_AVG')
            #node_rates = searchDictMultipleCat(lst, ['strategy', 'computation_placement'], {'name' : strategy, 'service_budget' : vms}, 2, 'LATENCY', 'NODE_RATE_TIMES')
            #node_eff_rates = searchDictMultipleCat(lst, ['strategy', 'computation_placement'], {'name' : strategy, 'service_budget' : vms}, 2, 'LATENCY', 'NODE_EFF_RATE_TIMES')
            #node_rates = dict(node_rates)
            #node_eff_rates = dict(node_eff_rates)
                
            s = str(int(vms/num_cloudlets))
            s += "\t" + str(qos_avg) 
            s += "\t" + str(revenue_avg) 
            s += "\t" + str(price_avg) 
            s += "\t" + str(idle_avg) 
            s += '\n'
            f.write(s)
        f.close()

def print_per_node_vm_results_tree(lst):
    """
    Print results for varying number of VMs for 1 node 2 services 10 classes
    """
    strategies = ['LFU_TRACE', 'DOUBLE_AUCTION', 'SELF_TUNING_TRACE', 'STATIC'] 
    num_cloudlets = 7
    #num_of_vms = [7*num_cloudlets, 14*num_cloudlets, 21*num_cloudlets, 28*num_cloudlets, 35*num_cloudlets, 42*num_cloudlets, 49*num_cloudlets]
    #num_of_vms = [7*num_cloudlets, 14*num_cloudlets, 21*num_cloudlets, 28*num_cloudlets, 35*num_cloudlets, 42*num_cloudlets, 49*num_cloudlets, 56*num_cloudlets, 63*num_cloudlets, 70*num_cloudlets]
    num_of_vms = [2*num_cloudlets, 7*num_cloudlets, 14*num_cloudlets, 21*num_cloudlets, 28*num_cloudlets, 35*num_cloudlets, 42*num_cloudlets, 49*num_cloudlets, 56*num_cloudlets, 63*num_cloudlets, 70*num_cloudlets]
    
    for strategy in strategies:
        filename = 'vm_per_node_results_' + str(strategy) 
        f = open(filename, 'w')
        f.write('# QoS for different number of VMs\n')
        f.write('#\n')
        s = "num_of_vms"
        #for serv in range(num_services):
        s += "\tqos_level_0"
        s += "\tprice_level_0"
        s += "\trevenue_level_0"
        s += "\tidle_level_0"
        s += "\texec_reqs_level_0"
        s += "\tqos_level_1"
        s += "\tprice_level_1"
        s += "\trevenue_level_1"
        s += "\tidle_level_1"
        s += "\texec_reqs_level_1"
        s += "\tqos_level_2"
        s += "\tprice_level_2"
        s += "\trevenue_level_2"
        s += "\tidle_level_2"
        s += "\texec_reqs_level_2\n"
        f.write(s)
        qos_service = None
        service_revenue = None
        vm_prices = None
        sat_rate = None 
        idle_times = None
        for vms in num_of_vms:
            per_node_qos_avg = searchDictMultipleCat(lst, ['strategy', 'computation_placement'], {'name' : strategy, 'service_budget' : vms}, 2, 'LATENCY', 'PER_NODE_QOS')
            per_node_qos_avg = dict(per_node_qos_avg)
            per_node_price = searchDictMultipleCat(lst, ['strategy', 'computation_placement'], {'name' : strategy, 'service_budget' : vms}, 2, 'LATENCY', 'PER_NODE_PRICE_TIMES')
            per_node_price = dict(per_node_price)
            per_node_revenue_avg = searchDictMultipleCat(lst, ['strategy', 'computation_placement'], {'name' : strategy, 'service_budget' : vms}, 2, 'LATENCY', 'PER_NODE_REV')
            per_node_revenue_avg = dict(per_node_revenue_avg)
            per_node_idle_avg = searchDictMultipleCat(lst, ['strategy', 'computation_placement'], {'name' : strategy, 'service_budget' : vms}, 2, 'LATENCY', 'PER_NODE_IDLE_TIMES_AVG')
            per_node_idle_avg = dict(per_node_idle_avg)
            per_node_exec_reqs = searchDictMultipleCat(lst, ['strategy', 'computation_placement'], {'name' : strategy, 'service_budget' : vms}, 2, 'LATENCY', 'PER_NODE_EXEC_REQS')
            per_node_exec_reqs = dict(per_node_exec_reqs)
            #node_rates = searchDictMultipleCat(lst, ['strategy', 'computation_placement'], {'name' : strategy, 'service_budget' : vms}, 2, 'LATENCY', 'NODE_RATE_TIMES')
            #node_eff_rates = searchDictMultipleCat(lst, ['strategy', 'computation_placement'], {'name' : strategy, 'service_budget' : vms}, 2, 'LATENCY', 'NODE_EFF_RATE_TIMES')
            #node_rates = dict(node_rates)
            #node_eff_rates = dict(node_eff_rates)
                
            s = str(int(vms/num_cloudlets))
            s += "\t" + str(per_node_qos_avg[0]) 
            s += "\t" + str(per_node_price[0]) 
            s += "\t" + str(per_node_revenue_avg[0]) 
            s += "\t" + str(per_node_idle_avg[0]) 
            s += "\t" + str(per_node_exec_reqs[0]) 
            s += "\t" + str((1.0*(per_node_qos_avg[1] + per_node_qos_avg[2]))/2) 
            s += "\t" + str(1.0*(per_node_price[1])) # + per_node_qos_avg[2]))/2) 
            s += "\t" + str((1.0*(per_node_revenue_avg[1] + per_node_revenue_avg[2]))/2) 
            s += "\t" + str((1.0*(per_node_idle_avg[1] + per_node_idle_avg[2]))/2) 
            s += "\t" + str((1.0*(per_node_exec_reqs[1] + per_node_exec_reqs[2]))/2) 
            s += "\t" + str((1.0*(per_node_qos_avg[3] + per_node_qos_avg[4] +per_node_qos_avg[5] + per_node_qos_avg[6]))/4) 
            s += "\t" + str(1.0*(per_node_price[3])) # + per_node_qos_avg[2]))/2) 
            s += "\t" + str((1.0*(per_node_revenue_avg[3] + per_node_revenue_avg[4] +per_node_revenue_avg[5] + per_node_revenue_avg[6]))/4) 
            s += "\t" + str((1.0*(per_node_idle_avg[3] + per_node_idle_avg[4] +per_node_idle_avg[5] + per_node_idle_avg[6]))/4) 
            s += "\t" + str((1.0*(per_node_exec_reqs[3] + per_node_exec_reqs[4] +per_node_exec_reqs[5] + per_node_exec_reqs[6]))/4) 
            s += '\n'
            f.write(s)
        f.close()

def print_vm_results(lst):
    """
    Print results for varying number of VMs for 1 node 2 services 10 classes
    """
    strategies = ['DOUBLE_AUCTION'] 
    num_of_vms = [1, 5, 10, 20, 30, 40, 50, 60]
    num_classes = 10
    num_services = 1
    
    for strategy in strategies:
        filename = 'vm_results_' + str(strategy) 
        f = open(filename, 'w')
        f.write('# QoS for different number of VMs\n')
        f.write('#\n')
        s = "num_of_vms"
        #for serv in range(num_services):
        s += "\tqos_service"
        s += "\trevenue"
        s += "\tvm_prices"
        s += "\tsat_rate"
        s += "\tidle_times"
        s += "\tnodeRate"
        s += "\tnodeEffRate\n"
        f.write(s)
        qos_service = None
        service_revenue = None
        vm_prices = None
        sat_rate = None 
        idle_times = None
        for vms in num_of_vms:
            qos_service = searchDictMultipleCat(lst, ['strategy', 'computation_placement'], {'name' : strategy, 'service_budget' : vms}, 2, 'LATENCY', 'QoS_SERVICE')
            qos_service = list(qos_service)
            service_revenue = searchDictMultipleCat(lst, ['strategy', 'computation_placement'], {'name' : strategy, 'service_budget' : vms}, 2, 'LATENCY', 'SERVICE_REVENUE')
            service_revenue = list(service_revenue)
            service_prices = searchDictMultipleCat(lst, ['strategy', 'computation_placement'], {'name' : strategy, 'service_budget' : vms}, 2, 'LATENCY', 'PRICE_TIMES_AVG')
            sat_rate = searchDictMultipleCat(lst, ['strategy', 'computation_placement'], {'name' : strategy, 'service_budget' : vms}, 2, 'LATENCY', 'SERVICE_SAT_RATE')
            sat_rate = list(sat_rate)
            serv_utils = searchDictMultipleCat(lst, ['strategy', 'computation_placement'], {'name' : strategy, 'service_budget' : vms}, 2, 'LATENCY', 'IDLE_TIMES_AVG')
            node_rates = searchDictMultipleCat(lst, ['strategy', 'computation_placement'], {'name' : strategy, 'service_budget' : vms}, 2, 'LATENCY', 'NODE_RATE_TIMES')
            node_eff_rates = searchDictMultipleCat(lst, ['strategy', 'computation_placement'], {'name' : strategy, 'service_budget' : vms}, 2, 'LATENCY', 'NODE_EFF_RATE_TIMES')
            node_rates = dict(node_rates)
            node_eff_rates = dict(node_eff_rates)

            node_rates_at_time0 = node_rates[0.0]
            for node_rate_arr in node_rates_at_time0:
                n = node_rate_arr[0] #node
                r = node_rate_arr[1] #rate
                if n == 0:
                    node_0_service_0_rate = r[0]
                    break
            node_eff_rates_at_time0 = node_eff_rates[0.0]
            for node_rate_arr in node_eff_rates_at_time0:
                n = node_rate_arr[0] #node
                r = node_rate_arr[1] #rate
                if n == 0:
                    node_0_service_0_eff_rate = r[0]
                    break
                
            s = str(vms)
            s += "\t" + str(qos_service[0]) 
            s += "\t" + str(service_revenue[0]) 
            s += "\t" + str(service_prices) 
            s += "\t" + str(sat_rate[0]) 
            s +=  "\t" + str(serv_utils)
            s += "\t" + str(node_0_service_0_rate)
            s += "\t" + str(node_0_service_0_eff_rate)
            s += '\n'
            f.write(s)
        f.close()

def print_trace_results(lst):
    """
    Print results for Google traces with varying observation periods (i.e., price recomputation at each period)
    """

    num_of_nodes = 3

    periods = [60]
    strategies = ['LFU_TRACE', 'DOUBLE_AUCTION_TRACE', 'SELF_TUNING_TRACE', 'STATIC']
    #strategies = ['SELF_TUNING_TRACE']
    for strategy in strategies:
        filename = "trace_performance_" + str(strategy) + ".txt"
        f = open(filename, 'w')
        f.write("# Price, Sat, Idle, QoS times for strategy: " + str(strategy) + "\n")
        f.write('#\n')
        #for node in range(num_of_nodes):
        s = "Time    Price    SatRate    PercentIdleTime   QoS    Revenue\n"
        f.write(s)

        qos_times = searchDictMultipleCat(lst, ['strategy'], {'name' : strategy}, 1, 'LATENCY', 'QOS_TIMES')
        qos_times = dict(qos_times)
        sat_times = searchDictMultipleCat(lst, ['strategy'], {'name' : strategy}, 1, 'LATENCY', 'SAT_TIMES')
        sat_times = dict(sat_times)
        idle_times = searchDictMultipleCat(lst, ['strategy'], {'name' : strategy}, 1, 'LATENCY', 'NODE_IDLE_TIMES')
        idle_times = dict(idle_times)
        rev_times = searchDictMultipleCat(lst, ['strategy'], {'name' : strategy}, 1, 'LATENCY', 'REVENUE_TIMES')
        rev_times = dict(rev_times)
        price_times = searchDictMultipleCat(lst, ['strategy'], {'name' : strategy}, 1, 'LATENCY', 'PRICE_TIMES')
        price_times = dict(price_times)

        print "\n\n"
        print "Price_times: " + repr(dict(price_times))
        print "\n\n"
        print "Sat_times: " + repr(sat_times)
        print "\n\n"
        print "idle_times: " + repr(idle_times)
        print "\n\n"
        print "rev_times: " + repr(idle_times)
        
        for t in sorted(price_times):
            if t == 0:
                continue
            s = str(t)
            s += "   " + str(1.0*sum(price_times[t])/len(price_times[t]))
            s += "   " + str(1.0*sum(sat_times[t])/len(sat_times[t]))
            s += "   " + str(1.0*sum(idle_times[t])/len(idle_times[t]))
            s += "   " + str(qos_times[t])
            s += "   " + str(rev_times[t])
            s += "\n"
            f.write(s)
        f.close()

#def print_vm_results_one_service_one_node_k_classes(lst):
def print_rate_dist_results(lst):
    """
    Print results for varying number of VMs for 1 node 1 service 10 classes
    """

    strategies = ['DOUBLE_AUCTION', 'FIFO'] 
    alpha = 0.75
    num_of_vms = 20
    num_classes = 10
    num_services = 1
    #rate_dists = [[0.33, 0.67], [0.5, 0.5], [0.67, 0.33]]
    #rate_dists = [[0.01, 0.99], [0.5, 0.5], [0.99, 0.01]]
    rate_dists = [[0.30, 0.25, 0.10, 0.065, 0.058, 0.053, 0.05, 0.045, 0.04, 0.039], [0.039, 0.04, 0.045, 0.05, 0.053, 0.058, 0.065, 0.1, 0.25, 0.3], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]
    
    for strategy in strategies:
        for rate_dist in rate_dists:
            suffix = '' 
            if rate_dist[0] > rate_dist[1]:
                suffix = 'low_util_popular'
            elif rate_dist[0] < rate_dist[1]:
                suffix = 'high_util_popular'
            else:
                suffix = 'equal'

            filename = 'qos_' + str(strategy) + '_classes_' + suffix
            f = open(filename, 'w')
            f.write('# ' + suffix + '\n')
            f.write('#\n')
            s = "Class\tQoS\tRevenue\tSatisfaction\tUtility\tPrice\n"
            #for c in range(num_classes):
            #    s += '  qos_class_' + str(c+1) + '  revenue_class_' + str(c+1) + '  class_sat_rate_' + str(c+1) + '  class_util_' + str(c+1) + '   class_price_' + str(c+1)
            #s += "\n"
            f.write(s)
            
            qos_class = searchDictMultipleCat(lst, ['strategy', 'netconf'], {'name' : strategy, 'rate_dist' : rate_dist}, 2, 'LATENCY', 'QoS_CLASS')
            class_revenue = searchDictMultipleCat(lst, ['strategy', 'netconf'], {'name' : strategy, 'rate_dist' : rate_dist}, 2, 'LATENCY', 'CLASS_REVENUE')
            class_sat_rate = searchDictMultipleCat(lst, ['strategy', 'netconf'], {'name' : strategy, 'rate_dist' : rate_dist}, 2, 'LATENCY', 'CLASS_SAT_RATE')
            class_utils = searchDictMultipleCat(lst, ['strategy', 'netconf'], {'name' : strategy, 'rate_dist' : rate_dist}, 2, 'LATENCY', 'NODE_UTILITIES')
            class_utils = dict(class_utils)
            print "class utils:" + repr(class_utils)
            class_prices = searchDictMultipleCat(lst, ['strategy', 'netconf'], {'name' : strategy, 'rate_dist' : rate_dist}, 2, 'LATENCY', 'PRICE_TIMES')
            class_prices = dict(class_prices)
            print  "class prices" + repr(class_prices)
            for c in range(num_classes):
                s = str(c+1)
                s += "\t" + str(qos_class[c]) + "\t" + str(class_revenue[c]) + "\t" + str(class_sat_rate[c]) + "\t" + str(class_utils[0][0][c]) + "\t" + str(class_prices[0.0][0])
                s += "\n"
                f.write(s)
            f.close()
    
    for strategy in strategies:
        filename = 'price_' + str(strategy) 
        f = open(filename, 'w')
        f.write('# ' + 'Prices for different correlation of QoS and popularity' + '\n')
        f.write('#\n')
        s = "Correlation\tPrice\n"
        f.write(s)
        for rate_dist in rate_dists:
            suffix = '' 
            if rate_dist[0] > rate_dist[1]:
                suffix = 'Negative'
            elif rate_dist[0] < rate_dist[1]:
                suffix = 'Positive'
            else:
                suffix = 'Uncorrelated'

            class_prices = searchDictMultipleCat(lst, ['strategy', 'netconf'], {'name' : strategy, 'rate_dist' : rate_dist}, 2, 'LATENCY', 'PRICE_TIMES')
            class_prices = dict(class_prices)
            s = suffix + "\t" + str(class_prices[0.0][0])
            s += "\n"
            f.write(s)
        f.close()
    
    for strategy in strategies:
        filename = 'idle_' + str(strategy) 
        f = open(filename, 'w')
        f.write('# ' + 'Idle times for different correlation of QoS and popularity' + '\n')
        f.write('#\n')
        s = "Correlation\tIdleTime\n"
        f.write(s)
        for rate_dist in rate_dists:
            suffix = '' 
            if rate_dist[0] > rate_dist[1]:
                suffix = 'Negative'
            elif rate_dist[0] < rate_dist[1]:
                suffix = 'Positive'
            else:
                suffix = 'Uncorrelated'

            idle_times = searchDictMultipleCat(lst, ['strategy', 'netconf'], {'name' : strategy, 'rate_dist' : rate_dist}, 2, 'LATENCY', 'IDLE_TIMES_AVG')
            s = suffix + "\t" + str(idle_times)
            s += "\n"
            f.write(s)
        f.close()

def print_budget_experiment(lst):
    
    strategies = ['SDF', 'HYBRID', 'MFU'] 
    alpha = 0.75
    replacement_interval = 30.0 
    N_SERVICES = 1000
    #budgets = [N_SERVICES, 2*N_SERVICES, 3*N_SERVICES, 4*N_SERVICES, 5*N_SERVICES]
    budgets = [N_SERVICES/8, N_SERVICES/4, N_SERVICES/2, 0.75*N_SERVICES, N_SERVICES, 2*N_SERVICES]


    # Print Sat. rates:
    for strategy in strategies:
        for budget in budgets:
            filename = 'sat_' + str(strategy) + '_' + str(budget)
            f = open(filename, 'w')
            f.write('# Sat. rate over time\n')
            f.write('#\n')
            f.write('# Time     Sat. Rate\n')
            sat_times = searchDictMultipleCat(lst, ['strategy', 'computation_placement', 'workload'], {'name' : strategy, 'service_budget' : budget, 'alpha' : alpha}, 3, 'LATENCY', 'SAT_TIMES')
            for k in sorted(sat_times):
                s = str(k[0][0]) + "\t" + str(k[1]) + "\n"
                f.write(s)
            f.close()
    
    # Print Idle times:
    for strategy in strategies:
        for budget in budgets:
            filename = 'idle_' + str(strategy) + '_' + str(budget)
            f = open(filename, 'w')
            f.write('# Idle times over time\n')
            f.write('#\n')
            f.write('# Time     Idle percentage\n')
            idle_times = searchDictMultipleCat(lst, ['strategy', 'computation_placement', 'workload'], {'name' : strategy, 'service_budget' : budget, 'alpha' : alpha}, 3, 'LATENCY', 'IDLE_TIMES')
            for k in sorted(idle_times):
                s = str(k[0][0]) + "\t" + str((1.0*k[1])) + "\n"
                f.write(s)
            f.close()

def printTree(tree, d = 0):
    if (tree == None or len(tree) == 0):
        print "\t" * d, "-"
    else:
        for key, val in tree.items():
            if (isinstance(val, dict)):
                print "\t" * d, key
                printTree(val, d+1)
            else:
                print "\t" * d, key, str(val)

def run(config, results, plotdir):
    """Run the plot script
    
    Parameters
    ----------
    config : str
        The path of the configuration file
    results : str
        The file storing the experiment results
    plotdir : str
        The directory into which graphs will be saved
    """
    resultset = RESULTS_READER['PICKLE'](results)
    #Onur: added this BEGIN
    lst = resultset.dump()
    """
    for l in lst:
        print 'PARAMETERS:\n'
        printTree(l[0])
        print 'RESULTS:\n'
        printTree(l[1])
    """
    #print_lru_probability_results(lst) 
    #print_rate_dist_results(lst) # toy example
    #print_trace_results(lst)
    #print_vm_results(lst) # toy experiment
    print_vm_results_tree(lst) # real experiment 
    #print_per_node_vm_results_tree(lst)
    
    #print_engagement_time_results(lst) # toy example

    #print_strategies_performance(lst)
    #print_budget_experiment(lst)
    #print_scheduling_experiments(lst)
    #print_zipf_experiment(lst)

    # /home/uceeoas/.local/bin/python ./plotresults.py --results results.pickle --output ./ config.py
    """
    settings = Settings()
    settings.read_from(config)
    config_logging(settings.LOG_LEVEL)
    resultset = RESULTS_READER[settings.RESULTS_FORMAT](results)
    # Create dir if not existsing
    if not os.path.exists(plotdir):
        os.makedirs(plotdir)
    # Parse params from settings
    topologies = settings.TOPOLOGIES
    cache_sizes = settings.NETWORK_CACHE
    alphas = settings.ALPHA
    strategies = settings.STRATEGIES
    # Plot graphs
    for topology in topologies:
        for cache_size in cache_sizes:
            logger.info('Plotting cache hit ratio for topology %s and cache size %s vs alpha' % (topology, str(cache_size)))
            plot_cache_hits_vs_alpha(resultset, topology, cache_size, alphas, strategies, plotdir)
            logger.info('Plotting link load for topology %s vs cache size %s' % (topology, str(cache_size)))
            plot_link_load_vs_alpha(resultset, topology, cache_size, alphas, strategies, plotdir)
            logger.info('Plotting latency for topology %s vs cache size %s' % (topology, str(cache_size)))
            plot_latency_vs_alpha(resultset, topology, cache_size, alphas, strategies, plotdir)
    for topology in topologies:
        for alpha in alphas:
            logger.info('Plotting cache hit ratio for topology %s and alpha %s vs cache size' % (topology, str(alpha)))
            plot_cache_hits_vs_cache_size(resultset, topology, alpha, cache_sizes, strategies, plotdir)
            logger.info('Plotting link load for topology %s and alpha %s vs cache size' % (topology, str(alpha)))
            plot_link_load_vs_cache_size(resultset, topology, alpha, cache_sizes, strategies, plotdir)
            logger.info('Plotting latency for topology %s and alpha %s vs cache size' % (topology, str(alpha)))
            plot_latency_vs_cache_size(resultset, topology, alpha, cache_sizes, strategies, plotdir)
    for cache_size in cache_sizes:
        for alpha in alphas:
            logger.info('Plotting cache hit ratio for cache size %s vs alpha %s against topologies' % (str(cache_size), str(alpha)))
            plot_cache_hits_vs_topology(resultset, alpha, cache_size, topologies, strategies, plotdir)
            logger.info('Plotting link load for cache size %s vs alpha %s against topologies' % (str(cache_size), str(alpha)))
            plot_link_load_vs_topology(resultset, alpha, cache_size, topologies, strategies, plotdir)
    logger.info('Exit. Plots were saved in directory %s' % os.path.abspath(plotdir))
    """

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("-r", "--results", dest="results",
                        help='the results file',
                        required=True)
    parser.add_argument("-o", "--output", dest="output",
                        help='the output directory where plots will be saved',
                        required=True)
    parser.add_argument("config",
                        help="the configuration file")
    args = parser.parse_args()
    run(args.config, args.results, args.output)

if __name__ == '__main__':
    main()
