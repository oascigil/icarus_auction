# -*- coding: utf-8 -*-
"""This module contains all configuration information used to run simulations
"""
from multiprocessing import cpu_count
from collections import deque
import copy
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
WARMUP_STRATEGY = 'DOUBLE_AUCTION' #'HYBRID'

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

#ALPHA: This is obselete, use U_MIN variable to adjust QoS sensitivity of services
ALPHAS = [0.7, 0.7]

# User engagement Times
ENGAGEMENT_TIMES = [60.0]

# Total size of network cache as a fraction of content population
NETWORK_CACHE = 0.05

# Number of content objects
N_CONTENTS = 1

# SERVICE POPULATION
N_SERVICES = N_CONTENTS

# Price computation mode for the auction-based service placement (True: maximizes utilisation of the cloudlets; False: maximizes revenue of the cloudlets)
MONETARYFOCUS = False

# Number of requests per second (over the whole network)
#NETWORK_REQUEST_RATE = 100.0 # this rate does not mean anything anymore see per-service rates below

# Number of content requests generated to prepopulate the caches
# These requests are not logged
N_WARMUP_REQUESTS = 100 #30000

# Number of content requests generated after the warmup and logged
# to generate results. 
#N_MEASURED_REQUESTS = 1000 #60*30000 #100000

# List of all implemented topologies
# Topology implementations are located in ./icarus/scenarios/topology.py
TOPOLOGIES =  ['PATH']
N_CLASSES = 10
#RATES = [5, 5] # A rate per service
RATES = [1] # A rate per service

U_MINS = [0.0]

RATE_DIST = [0.30, 0.25, 0.10, 0.065, 0.058, 0.053, 0.05, 0.045, 0.04, 0.039] #negative correlation between QoS and popularity
#RATE_DIST.reverse() #positive corr demand and QoS
#RATE_DIST = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1] # how service rates are distributed among classes
#RATE_DIST = [0.5, 0.5] 
TREE_DEPTH = 1
BRANCH_FACTOR = 2

# Parameter check
if len(RATES) != N_SERVICES:
    raise RuntimeError("Incorrect size for RATES.\n") 
if len(RATE_DIST) != N_CLASSES:
    raise RuntimeError("Incorrect size for RATE_DIST.\n") 
if len(ENGAGEMENT_TIMES) != N_SERVICES:
    raise RuntimeError("Incorrect size for ENGAGEMENT_TIMES.\n") 

SECS = 60 #do not change
MINS = 60*24
NETWORK_REQUEST_RATE = sum(RATES)
N_MEASURED_REQUESTS = NETWORK_REQUEST_RATE*SECS*MINS

# Replacement Interval in seconds
REPLACEMENT_INTERVAL = 300.0
NUM_REPLACEMENTS = 10000

# List of caching and routing strategies
# The code is located in ./icarus/models/strategy.py
#STRATEGIES = ['SDF', 'HYBRID', 'MFU']  # service-based routing
STRATEGIES = ['DOUBLE_AUCTION'] 
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
                       'n_warmup':   N_WARMUP_REQUESTS,
                       'n_measured': N_MEASURED_REQUESTS,
                       'rates':       RATES,
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
#default['topology']['name'] = 'TREE'
#default['topology']['k'] = BRANCH_FACTOR
#default['topology']['h'] = TREE_DEPTH
default['topology']['n'] = TREE_DEPTH + 1
default['topology']['n_classes'] = N_CLASSES
default['topology']['min_delay'] = 0.004
default['topology']['max_delay'] = 0.034
default['warmup_strategy']['name'] = WARMUP_STRATEGY
default['netconf']['alphas'] = ALPHAS # Sensitivity of the services to changes in QoS
default['netconf']['umins'] = U_MINS
default['netconf']['service_times'] = ENGAGEMENT_TIMES
default['netconf']['rate_dist'] = RATE_DIST # determines how service rate is distributed among classes
default['netconf']['monetaryFocus'] = MONETARYFOCUS

# Create experiments multiplexing all desired parameters

# 1. Experiments with 1 cloudlet n service and k classes

default['topology']['n'] = 1
num_of_vms = 20

#rate_dists = [[0.33, 0.67], [0.5, 0.5], [0.67, 0.33]]
rate_dists = [[0.30, 0.25, 0.10, 0.065, 0.058, 0.053, 0.05, 0.045, 0.04, 0.039], [0.039, 0.04, 0.045, 0.05, 0.053, 0.058, 0.065, 0.1, 0.25, 0.3], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]
#rate_dists = [[0.01, 0.99], [0.5, 0.5], [0.99, 0.01]]

#for strategy in ['DOUBLE_AUCTION', 'FIFO']:
for strategy in ['DOUBLE_AUCTION', 'FIFO']:
    for rate_dist in rate_dists:
        experiment = copy.deepcopy(default)
        experiment['netconf']['rate_dist'] = rate_dist # determines how service rate is distributed among classes
        experiment['workload']['rate_dist'] = rate_dist #FIXME: netconf rate_dist seems to have no affect on the workload rate_dist; therefore, needs to be set seprately
        experiment['strategy']['name'] = strategy
        experiment['warmup_strategy']['name'] = strategy
        experiment['computation_placement']['service_budget'] = num_of_vms #N_SERVICES*5 # number of VMs in the memory
        experiment['computation_placement']['computation_budget'] = num_of_vms #N_SERVICES*5 # one core per each VM
        experiment['desc'] = "strategy: %s, rate_dist: %s" % (strategy, str(rate_dist))
        EXPERIMENT_QUEUE.append(experiment)
"""
default['topology']['n'] = 2

for strategy in ['DOUBLE_AUCTION']:
    for num_of_vms in range(1, 11):
        experiment = copy.deepcopy(default)
        experiment['strategy']['name'] = strategy
        experiment['warmup_strategy']['name'] = strategy
        experiment['computation_placement']['service_budget'] = num_of_vms #N_SERVICES*5 # number of VMs in the memory
        experiment['computation_placement']['computation_budget'] = num_of_vms #N_SERVICES*5 # one core per each VM
        experiment['desc'] = "strategy: %s, num_of_vms: %s" \
                             % (strategy, str(num_of_vms))
        EXPERIMENT_QUEUE.append(experiment)
"""
# Compare SDF, LFU, Hybrid for default values
"""
for strategy in STRATEGIES:
    experiment = copy.deepcopy(default)
    experiment['strategy']['name'] = strategy
    experiment['desc'] = "strategy: %s, prob: %s" \
                         % (strategy, str(p))
    EXPERIMENT_QUEUE.append(experiment)
"""
"""
budgets = [N_SERVICES/8, N_SERVICES/4, N_SERVICES/2, 0.75*N_SERVICES, N_SERVICES, 2*N_SERVICES]
# Experiment with different budgets
for strategy in STRATEGIES:
    for budget in budgets:
        experiment = copy.deepcopy(default)
        experiment['strategy']['name'] = strategy
        experiment['warmup_strategy']['name'] = strategy
        experiment['computation_placement']['service_budget'] = budget
        experiment['strategy']['replacement_interval'] = REPLACEMENT_INTERVAL
        experiment['strategy']['n_replacements'] = NUM_REPLACEMENTS
        experiment['desc'] = "strategy: %s, budget: %s" \
                             % (strategy, str(budget))
        EXPERIMENT_QUEUE.append(experiment)
"""
# Experiment comparing FIFO with EDF 
"""
for schedule_policy in ['EDF', 'FIFO']:
    for strategy in STRATEGIES:
        experiment = copy.deepcopy(default)
        experiment['strategy']['name'] = strategy
        experiment['warmup_strategy']['name'] = strategy
        experiment['sched_policy']['name'] = schedule_policy
        experiment['desc'] = "strategy: %s, schedule policy: %s" \
                             % (strategy, str(schedule_policy))
        EXPERIMENT_QUEUE.append(experiment)
"""
# Experiment with various zipf values
"""
for alpha in [0.1, 0.25, 0.50, 0.75, 1.0]:
    for strategy in STRATEGIES:
        experiment = copy.deepcopy(default)
        experiment['workload']['alpha'] = alpha
        experiment['strategy']['name'] = strategy
        experiment['desc'] = "strategy: %s, zipf: %s" \
                         % (strategy, str(alpha))
        EXPERIMENT_QUEUE.append(experiment)
"""
# Experiment with various request rates (for sanity checking)
