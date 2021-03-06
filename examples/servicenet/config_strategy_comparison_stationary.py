# -*- coding: utf-8 -*-
"""This module contains all configuration information used to run simulations
"""
from multiprocessing import cpu_count
from collections import deque
import copy
import random
from icarus.util import Tree

# GENERAL SETTINGS

# Level of logging output
# Available options: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL = 'INFO'

# If True, executes simulations in parallel using multiple processes
# to take advantage of multicore CPUs
PARALLEL_EXECUTION = True

# Number of processes used to run simulations in parallel.
# This option is ignored if PARALLEL_EXECUTION = False
N_PROCESSES = cpu_count()

# Granularity of caching.
# Currently, only OBJECT is supported
CACHING_GRANULARITY = 'OBJECT'

# Warm-up strategy
#WARMUP_STRATEGY = 'MFU' #'HYBRID'
WARMUP_STRATEGY = 'DOUBLE_AUCTION_TRACE' #'HYBRID'
WARMUP_STRATEGY = 'SELF_TUNING_TRACE'
WARMUP_STRATEGY = 'LFU_TRACE'

# Format in which results are saved.
# Result readers and writers are located in module ./icarus/results/readwrite.py
# Currently only PICKLE is supported 
RESULTS_FORMAT = 'PICKLE'

# Number of times each experiment is replicated
# This is necessary for extracting confidence interval of selected metrics
N_REPLICATIONS = 1

# List of metrics to be measured in the experiments
# The implementation of data collectors are located in ./icaurs/execution/collectors.py
DATA_COLLECTORS = ['LATENCY']

# Range of alpha values of the Zipf distribution using to generate content requests
# alpha values must be positive. The greater the value the more skewed is the 
# content popularity distribution
# Range of alpha values of the Zipf distribution using to generate content requests
# alpha values must be positive. The greater the value the more skewed is the 
# content popularity distribution
# Note: to generate these alpha values, numpy.arange could also be used, but it
# is not recommended because generated numbers may be not those desired. 
# E.g. arange may return 0.799999999999 instead of 0.8. 
# This would give problems while trying to plot the results because if for
# example I wanted to filter experiment with alpha=0.8, experiments with
# alpha = 0.799999999999 would not be recognized 
ZIPF_EXP = 0.75
#ALPHA = [0.00001]
#ALPHAS = [0.7, 1.0]

# Total size of network cache as a fraction of content population
NETWORK_CACHE = 0.05

# Number of content objects
N_CONTENTS = 10 #9218 #10

# SERVICE POPULATION
N_SERVICES = N_CONTENTS

# Number of requests per second (over the whole network)
#NETWORK_REQUEST_RATE = 100.0 # this rate does not mean anything anymore see per-service rates below

# Number of content requests generated to prepopulate the caches
# These requests are not logged
N_WARMUP_REQUESTS = 300 #30000


# List of all implemented topologies
# Topology implementations are located in ./icarus/scenarios/topology.py
TOPOLOGIES =  ['TREE']
N_CLASSES = 10
RATE = 10 #(per edge node) was 100
RATE_DIST = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1] # how service rates are distributed among classes
TREE_DEPTH = 2 #was 1
BRANCH_FACTOR = 2

ALPHAS = [random.random() for x in range(N_CONTENTS)]

if len(ALPHAS) != N_CONTENTS:
    raise RuntimeError("Incorrect size for ALPHAS.\n") 
if len(RATE_DIST) != N_CLASSES:
    raise RuntimeError("Incorrect size for RATE_DIST.\n") 

# Number of content requests generated after the warmup and logged
# to generate results. 
SECS = 60 #do not change
MINS = 240
N_MEASURED_REQUESTS = RATE*SECS*MINS*(TREE_DEPTH*BRANCH_FACTOR)

# Replacement Interval in seconds
REPLACEMENT_INTERVAL = 240.0 #60.0
NUM_REPLACEMENTS = 10000

# List of caching and routing strategies
# The code is located in ./icarus/models/strategy.py
#STRATEGIES = ['SDF', 'HYBRID', 'MFU']  # service-based routing
STRATEGIES = ['SELF_TUNING_TRACE'] 
#STRATEGIES = ['SDF']  
#STRATEGIES = ['HYBRID'] 
#STRATEGIES = ['LRU']  

# Cache replacement policy used by the network caches.
# Supported policies are: 'LRU', 'LFU', 'FIFO', 'RAND' and 'NULL'
# Cache policy implmentations are located in ./icarus/models/cache.py
CACHE_POLICY = 'LRU'

# Task scheduling policy used by the cloudlets.
# Supported policies are: 'EDF' (Earliest Deadline First), 'FIFO'
SCHED_POLICY = 'EDF'

# Queue of experiments
EXPERIMENT_QUEUE = deque()
default = Tree()

default['workload'] = {'name':       'STATIONARY',
                       'n_contents': N_CONTENTS,
                       'n_warmup':   0,
                       'n_measured': N_MEASURED_REQUESTS,
                       'rates':       RATE,
                       'rate_dist' :     RATE_DIST,
                       'seed':  0,
                       'n_services': N_SERVICES,
                       'alpha' : ZIPF_EXP
                      }
default['cache_placement']['name'] = 'UNIFORM'
#default['computation_placement']['name'] = 'CENTRALITY'
default['computation_placement']['name'] = 'UNIFORM'
#default['computation_placement']['name'] = 'CENTRALITY'
default['computation_placement']['service_budget'] = N_SERVICES*5 # number of VMs in the memory
default['cache_placement']['network_cache'] = default['computation_placement']['service_budget']
default['computation_placement']['computation_budget'] = N_SERVICES*5 # one core per each VM
default['content_placement']['name'] = 'UNIFORM'
default['cache_policy']['name'] = CACHE_POLICY
default['sched_policy']['name'] = SCHED_POLICY
default['strategy']['replacement_interval'] = REPLACEMENT_INTERVAL
default['strategy']['n_replacements'] = NUM_REPLACEMENTS
default['topology']['name'] = 'PATH'
# TREE topology
default['topology']['name'] = 'TREE'
default['topology']['k'] = BRANCH_FACTOR
default['topology']['h'] = TREE_DEPTH
# PATH topology
#default['topology']['n'] = TREE_DEPTH + 1
default['topology']['n_classes'] = N_CLASSES
default['topology']['min_delay'] = 0.004
default['topology']['max_delay'] = 0.034
default['warmup_strategy']['name'] = WARMUP_STRATEGY
default['netconf']['alphas'] = ALPHAS # Sensitivity of the services to changes in QoS
default['netconf']['rate_dist'] = RATE_DIST # determines how service rate is distributed among classes

# Create experiments multiplexing all desired parameters

# 1. Experiments with 1 cloudlet 1 service and k classes
#for strategy in ['LFU_TRACE', 'DOUBLE_AUCTION_TRACE', 'SELF_TUNING_TRACE', 'STATIC_TRACE']:
for strategy in ['SELF_TUNING_TRACE']:
    for num_of_vms in range(140, 141): #was 100, 101
        experiment = copy.deepcopy(default)
        experiment['strategy']['name'] = strategy
        experiment['warmup_strategy']['name'] = strategy
        if strategy is 'STATIC_TRACE':
            experiment['strategy']['trace_file'] = 'top_n_trace.txt'
            experiment['strategy']['n_measured_requests'] = N_MEASURED_REQUESTS
            experiment['warmup_strategy']['name'] = 'LFU_TRACE' #strategy
        experiment['computation_placement']['service_budget'] = num_of_vms 
        experiment['computation_placement']['computation_budget'] = num_of_vms 
        experiment['desc'] = "strategy: %s, num_of_vms: %s" \
                             % (strategy, str(num_of_vms))
        EXPERIMENT_QUEUE.append(experiment)

# Experiments with different replacement intervals
"""
num_of_vms = 1000
for strategy in ['LFU_TRACE', 'DOUBLE_AUCTION_TRACE', 'SELF_TUNING_TRACE', 'STATIC_TRACE']:
    for replacement_period in [10, 20, 30, 40, 50, 60, 120]:
        experiment = copy.deepcopy(default)
        experiment['strategy']['name'] = strategy
        experiment['warmup_strategy']['name'] = strategy
        experiment['strategy']['replacement_interval'] = replacement_period
        if strategy is 'STATIC_TRACE':
            experiment['strategy']['trace_file'] = 'top_n_trace.txt'
            experiment['strategy']['n_measured_requests'] = N_MEASURED_REQUESTS
            experiment['warmup_strategy']['name'] = 'LFU_TRACE' #strategy
        experiment['computation_placement']['service_budget'] = num_of_vms 
        experiment['computation_placement']['computation_budget'] = num_of_vms 
        experiment['desc'] = "strategy: %s, num_of_vms: %s" \
                             % (strategy, str(num_of_vms))
        EXPERIMENT_QUEUE.append(experiment)
"""
