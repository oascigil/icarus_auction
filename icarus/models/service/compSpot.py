# -*- coding: utf-8 -*-
"""Computational Spot implementation
This module contains the implementation of a set of VMs residing at a node. Each VM is abstracted as a FIFO queue. 
todo: - apply social and maximisation policy
      - clean the code
"""
from __future__ import division
from collections import deque
import random
import abc
import copy

from cvxpy import *
import numpy
import math
import optparse
import sys

from icarus.util import inheritdoc

__all__ = [
        'ComputationalSpot',
        'Task'
           ]

# Status codes
REQUEST = 0
RESPONSE = 1
TASK_COMPLETE = 2
# Admission results:
DEADLINE_MISSED = 0
CONGESTION = 1
SUCCESS = 2
CLOUD = 3
NO_INSTANCES = 4

class Task(object):
    """
    A request to execute a service at a node
    """
    def __init__(self, time, deadline, rtt_delay, node, service, service_time, flow_id, receiver, finishTime=None, traffic_class = None):
        
        self.service = service
        self.time = time
        self.expiry = deadline
        self.rtt_delay = rtt_delay
        self.exec_time = service_time
        self.node = node
        self.flow_id = flow_id
        self.receiver = receiver
        self.finishTime = finishTime
        self.traffic_class = traffic_class

    def print_task(self):
        print ("Task.service: " + repr(self.service))
        print ("Task.time: " + repr(self.time))
        print ("Task.deadline: " + repr(self.expiry))
        print ("Task.rtt_delay: " + repr(self.rtt_delay))
        print ("Task.exec_time: " + repr(self.exec_time))
        print ("Task.node: " + repr(self.node))
        print ("Task.flow_id: " + repr(self.flow_id))
        print ("Task.receiver: " + repr(self.receiver))
        print ("Task.finishTime: " + repr(self.finishTime))

class CpuInfo(object):
    """
    Information on running tasks, their finish times, etc. on each core of the CPU
    """
    
    def __init__(self, numOfCores):

        self.numOfCores = numOfCores
        # Hypothetical finish time of the last request (i.e., tail of the queue)
        self.coreFinishTime = [0]*self.numOfCores 
        # Currently running service instance at each cpu core (scheduled earlier)
        self.coreService = [None]*self.numOfCores
        # Currently running service instance at each cpu core (scheduled earlier)
        self.coreServiceClass = [[None, None]]*self.numOfCores
        # Idle time of the server
        self.idleTime = 0.0

    def get_idleTime(self, time):
        
        # update the idle times
        for indx in range(0, self.numOfCores):
            if self.coreFinishTime[indx] < time:
                self.idleTime += time - self.coreFinishTime[indx]
                self.coreFinishTime[indx] = time
                self.coreService[indx] = None
                self.coreServiceClass[indx] = [None, None]
        
        return self.idleTime

    def get_available_core(self, time):
        """
        Retrieve the core which is/will be available the soonest
        """
        
        # update the running services 
        num_free_cores = 0
        for indx in range(0, self.numOfCores):
            if self.coreFinishTime[indx] <= time:
                self.idleTime += time - self.coreFinishTime[indx]
                self.coreFinishTime[indx] = time
                self.coreService[indx] = None
                self.coreServiceClass[indx] = [None, None]
                num_free_cores += 1
            
        indx = self.coreFinishTime.index(min(self.coreFinishTime))
        if self.coreFinishTime[indx] <= time:
            return indx, num_free_cores
        
        return None, 0

    def get_free_core(self, time):
        """
        Retrieve a currently available core, if there is any. 
        If there isn't, return None
        """
        core_indx = self.get_available_core(time)
        if self.coreFinishTime[core_indx] <= time:
            self.coreService[indx] = None
            self.coreServiceClass[indx] = [None, None]
            return core_indx
        else:
            return None
    
    def get_next_available_core(self):
        indx = self.coreFinishTime.index(min(self.coreFinishTime)) 
        
        return indx

    def assign_task_to_core(self, core_indx, fin_time, service, traffic_class=None):
        
        if self.coreFinishTime[core_indx] > fin_time:
            raise ValueError("Error in assign_task_to_core: there is a running task")
        
        self.coreService[core_indx] = service
        self.coreFinishTime[core_indx] = fin_time
        if traffic_class is not None:
            self.coreServiceClass[core_indx] = [service, traffic_class]

    def update_core_status(self, time):
        
        for indx in range(0, self.numOfCores):
            if self.coreFinishTime[indx] <= time:
                self.idleTime += time - self.coreFinishTime[indx]
                self.coreFinishTime[indx] = time
                self.coreService[indx] = None
                self.coreServiceClass[indx] = [None, None]

    def count_running_service(self, service):
        """Count the currently running VM instances for service s
        """

        return self.coreService.count(service)

    def count_running_service_type(self, service, traffic_class):
        """Count the currently running VM instances for service type (service+class)
        """
        return self.coreServiceClass.count([service, traffic_class])

    def print_core_status(self):
        for indx in range(0, len(self.coreService)):
            print ("Core: " + repr(indx) + " finish time: " + repr(self.coreFinishTime[indx]) + " service: " + repr(self.coreService[indx]))

class ComputationalSpot(object):
    """ 
    A set of computational resources, where the basic unit of computational resource 
    is a VM. Each VM is bound to run a specific service instance and abstracted as a 
    Queue. The service time of the Queue is extracted from the service properties. 
    """

    def __init__(self, model, numOfCores, n_services, services, node, num_classes, sched_policy = "EDF", dist=None, monetaryFocus=False, debugMode=False):
        """Constructor

        Parameters
        ----------
        numOfCores: total number of VMs available at the computational spot
        n_services : number of services in the memory
        services : list of all the services (service population) with their attributes
        """

        #if numOfCores == -1:
        #    self.numOfCores = 100000 # this should really be infinite
        #    self.is_cloud = True
        #else:
        self.numOfCores = n_services#numOfCores
        self.is_cloud = False
        self.debugMode = debugMode

        self.service_population = len(services)
        self.model = model
        self.num_classes = num_classes
        self.monetaryFocus = monetaryFocus
        if self.monetaryFocus:
            print "Monetary Focus set to True"
        else:
            print "Monetary Focus set to False"

        print ("Number of VMs @node: " + repr(node) + " " + repr(n_services))
        print ("Number of cores @node: " + repr(node) + " " + repr(numOfCores))
        print ("Number of classes @node: " + repr(node) + " " + repr(num_classes))

        #if n_services < numOfCores:
        #    n_services = numOfCores*2
        
        #if n_services > self.service_population:
        #    n_services = self.service_population

        self.sched_policy = sched_policy

        # CPU info
        self.cpuInfo = CpuInfo(self.numOfCores)
        
        # num. of vms (memory capacity) #TODO rename this
        self.n_services = n_services
        
        # Task queue of the comp. spot
        self.taskQueue = []

        # Rate and effective rate at the selected price (time-series)
        self.rate_times = {}
        self.eff_rate_times = {}
        
        # num. of instances of each service in the memory
        self.numberOfInstances = [[0 for x in range(self.num_classes)] for y in range(self.service_population)]
        
        # price for each service and class (service type) ued by self-tuning apprach
        self.service_class_price = [[0.0 for x in range(self.num_classes)] for y in range(self.service_population)]

        # server missed requests (due to congestion)
        self.missed_requests = [0] * self.service_population

        # service request count (per service)
        self.running_requests = [0 for x in range(0, self.service_population)] #correct!
        
        # delegated service request counts (per service)
        self.delegated_requests = [0 for x in range(0, self.service_population)]
        
        self.services = services
        self.view = None
        self.node = node
        self.delay_to_cloud = self.model.topology.node[self.node]['delay_to_cloud'] # Note this value may not be accurate for Rocketfuel

        # Price of each VM
        self.vm_prices = None
        self.service_class_rate = [[0.0 for x in range(self.num_classes)] for y in range(self.service_population)]
        self.service_class_count = [[0 for x in range(self.num_classes)] for y in range(self.service_population)]
        self.utilities = [[0.0 for x in range(self.num_classes)] for y in range(self.service_population)] # This is the QoS gain or the bid (different from the actual QoS)
        self.qos = [[0.0 for x in range(self.num_classes)] for y in range(self.service_population)]  # This is the actual QoS (not the QS gain)
        self.compute_utilities() # compute the utilities of each service and class
        print ("Utility @ node: " + repr(self.node) + ": " + repr(self.utilities))
        # Outputs from the get_prices() call:
        self.admitted_service_rate = [0.0]*self.service_population
        self.admitted_service_class_rate = [[0.0 for x in range(self.num_classes)] for y in range(self.service_population)]

        # Setup all the variables: numberOfInstances, cpuInfo, etc. ...
        num_services = 0
        #if dist is None and self.is_cloud == False:
        if self.is_cloud == False:
            # setup a random set of services to run in the memory
            while num_services < self.n_services:
                service_index = random.choice(range(0, self.service_population))
                class_index = random.choice(range(0, self.num_classes))
                self.numberOfInstances[service_index][class_index] += 1
                num_services += 1
        #            evicted = self.model.cache[node].put(service_index) # HACK: should use controller here   
        #            print ("Evicted: " + repr(evicted))

    def schedule(self, time):
        """
        Return the next task to be executed, if there is any.
        """

        # TODO: This method can simply fetch the task with the smallest finish time
        # No need to repeat the same computation already carried out by simulate()

        core_indx, num_free_cores = self.cpuInfo.get_available_core(time)
        
        if (len(self.taskQueue) > 0) and (core_indx is not None):
            coreService = self.cpuInfo.coreService
            for task_indx in range(0, len(self.taskQueue)):
                aTask = self.taskQueue[task_indx]
                serv_count = coreService.count(aTask.service)
                if self.numberOfInstances[aTask.service] > 0: 
                    available_vms = self.numberOfInstances[aTask.service] - serv_count
                    if available_vms > 0:
                        self.taskQueue.pop(task_indx)
                        self.cpuInfo.assign_task_to_core(core_indx, time + aTask.exec_time, aTask.service)
                        aTask.finishTime = time + aTask.exec_time
                        return aTask
                else: # This can happen during service replacement transitions
                    self.taskQueue.pop(task_indx)
                    self.cpuInfo.assign_task_to_core(core_indx, time + aTask.exec_time, aTask.service)
                    aTask.finishTime = time + aTask.exec_time
                    return aTask

        return None

    def simulate_execution(self, aTask, time, debug):
        """
        Simulate the execution of tasks in the taskQueue and compute each
        task's finish time. 

        Parameters:
        -----------
        taskQueue: a queue (List) of tasks (Task objects) waiting to be executed
        cpuFinishTime: Current Finish times of CPUs
        """
        # Add the task to the taskQueue
        taskQueueCopy = self.taskQueue[:] #shallow copy

        cpuInfoCopy = copy.deepcopy(self.cpuInfo)

        # Check if the queue has any task that misses its deadline (after adding this)
        #coreFinishTimeCopy = self.cpuInfo.coreFinishTime[:]
        #cpuServiceCopy = self.cpuService[:]
        
        if debug:
            for task_indx in range(0, len(taskQueueCopy)):
                taskQueueCopy[task_indx].print_task()
            cpuInfoCopy.print_core_status()

        sched_failed = False
        aTask = None
        while len(taskQueueCopy) > 0:
            if not sched_failed:
                core_indx = cpuInfoCopy.get_next_available_core()
            time = cpuInfoCopy.coreFinishTime[core_indx]
            cpuInfoCopy.update_core_status(time)
            sched_failed = False

            for task_indx in range(0, len(taskQueueCopy)):
                aTask = taskQueueCopy[task_indx]
                if self.numberOfInstances[aTask.service] > 0: 
                    serv_count = cpuInfoCopy.coreService.count(aTask.service)
                    available_cores = self.numberOfInstances[aTask.service] - serv_count
                    if available_cores > 0:
                        taskQueueCopy.pop(task_indx)
                        cpuInfoCopy.assign_task_to_core(core_indx, time + aTask.exec_time, aTask.service)
                        aTask.finishTime = time + aTask.exec_time
                        break
                else: # This can happen during service replacement transitions
                    taskQueueCopy.pop(task_indx)
                    cpuInfoCopy.assign_task_to_core(core_indx, time + aTask.exec_time, aTask.service)
                    aTask.finishTime = time + aTask.exec_time
                    break
            else: # for loop concluded without a break
                sched_failed = True
                core_indx = cpuInfoCopy.coreService.index(aTask.service)

    def admit_task_auction_queuing(self, service, time, flow_id, traffic_class, receiver, rtt_delay, controller, debug, rtt_to_nextnode):
        """
        Admit a task if there is an idle VM or can be queued
        """
        serviceTime = self.services[service].service_time
        self.cpuInfo.update_core_status(time) #need to call before simulate
        core_indx, num_free_cores = self.cpuInfo.get_available_core(time)
        if core_indx == None:
            utility = self.utilities[service][traffic_class]
            price = self.vm_prices[0]
            if utility > price:
                deadline = time + serviceTime #+ rtt_to_nextnode
                aTask = Task(time, deadline, rtt_delay, self.node, service, serviceTime, flow_id, receiver, 0.0, traffic_class)
                self.taskQueue.append(aTask)
                self.simulate_execution(aTask, time, debug)
                for task in self.taskQueue:
                    if debug:
                        print("After simulate:")
                        task.print_task()
                    if (task.expiry - task.rtt_delay) < task.finishTime:
                        self.missed_requests[service] += 1
                        if debug:
                            print("Refusing TASK: Congestion")
                        self.taskQueue.remove(aTask)
                        #print "Time: " + repr(time) + " core indx: " + repr(core_indx) + " num_free_cores: " + repr(num_free_cores) + " Idle time: " + repr(self.cpuInfo.idleTime) + " Traffic class: " + repr(traffic_class) + " REJECTED CONG " + "queue len: " + repr(len(self.taskQueue))
                        controller.reject_service(time, flow_id, service, self.is_cloud, traffic_class, self.node, price)
                        return [False, CONGESTION]
                #print "Time: " + repr(time) + " core indx: " + repr(core_indx) + " num_free_cores: " + repr(num_free_cores) + " Idle time: " + repr(self.cpuInfo.idleTime) + " Traffic class: " + repr(traffic_class) + " ACCEPTED QUEUED " + "queue len: " + repr(len(self.taskQueue))
                # New task can be admitted, add to service Queue
                self.running_requests[service] += 1
                # Run the next task (if there is any)
                newTask = self.schedule(time) 
                if newTask is not None:
                    controller.add_event(newTask.finishTime, newTask.receiver, newTask.service, self.node, newTask.flow_id, newTask.traffic_class, newTask.rtt_delay, TASK_COMPLETE) 
                    controller.execute_service(newTask.finishTime, flow_id, newTask.service, self.is_cloud, newTask.traffic_class, self.node, price) 
                return [True, SUCCESS]
            else:
                controller.reject_service(time, flow_id, service, self.is_cloud, traffic_class, self.node, price)
                return [False, CONGESTION]
        else:
            utility = self.utilities[service][traffic_class]
            price = self.vm_prices[num_free_cores-1]
            if utility < price: # reject the request
                #print "Time: " + repr(time) + " core indx: " + repr(core_indx) + " num_free_cores: " + repr(num_free_cores) + " Idle time: " + repr(self.cpuInfo.idleTime) + " Traffic class: " + repr(traffic_class) + " REJECTED UTIL " + "queue len: " + repr(len(self.taskQueue))
                controller.reject_service(time, flow_id, service, self.is_cloud, traffic_class, self.node, price)
                return [False, CONGESTION]
            finishTime = time + serviceTime
            self.cpuInfo.assign_task_to_core(core_indx, finishTime, service)
            controller.add_event(finishTime, receiver, service, self.node, flow_id, traffic_class, rtt_delay, TASK_COMPLETE) 
            controller.execute_service(time, flow_id, service, self.is_cloud, traffic_class, self.node, price)
            self.service_class_count[service][traffic_class] += 1
            #print "Time: " + repr(time) + " core indx: " + repr(core_indx) + " num_free_cores: " + repr(num_free_cores) + " Idle time: " + repr(self.cpuInfo.idleTime) + "Traffic class: " + repr(traffic_class) + "ACCEPTED " + "queue len: " + repr(len(self.taskQueue))
            return [True, SUCCESS]

    def admit_task_auction(self, service, time, flow_id, traffic_class, receiver, rtt_delay, controller, debug):
        """
        Admit a task if there is an idle VM 
        """
        self.service_class_count[service][traffic_class] += 1
        serviceTime = self.services[service].service_time
        self.cpuInfo.update_core_status(time) #need to call before simulate
        core_indx, num_free_cores = self.cpuInfo.get_available_core(time)
        price = self.vm_prices[num_free_cores-1]
        if core_indx == None:
            #print "Time: " + repr(time) + " core indx: " + repr(core_indx) + " num_free_cores: " + repr(num_free_cores) + " Idle time: " + repr(self.cpuInfo.idleTime) + " Traffic class: " + repr(traffic_class) + " REJECTED CONG"
            controller.reject_service(time, flow_id, service, self.is_cloud, traffic_class, self.node, price)
            return [False, CONGESTION]
        else:
            utility = self.utilities[service][traffic_class]
            if utility < price: # reject the request
                #print "Time: " + repr(time) + " core indx: " + repr(core_indx) + " num_free_cores: " + repr(num_free_cores) + " Idle time: " + repr(self.cpuInfo.idleTime) + " Traffic class: " + repr(traffic_class) + " REJECTED UTIL"
                controller.reject_service(time, flow_id, service, self.is_cloud, traffic_class, self.node, price)
                return [False, CONGESTION]
            finishTime = time + serviceTime
            self.cpuInfo.assign_task_to_core(core_indx, finishTime, service)
            controller.add_event(finishTime, receiver, service, self.node, flow_id, traffic_class, rtt_delay, TASK_COMPLETE) 
            controller.execute_service(time, flow_id, service, self.is_cloud, traffic_class, self.node, price*serviceTime)
            #print "Time: " + repr(time) + " core indx: " + repr(core_indx) + " num_free_cores: " + repr(num_free_cores) + " Idle time: " + repr(self.cpuInfo.idleTime) + "Traffic class: " + repr(traffic_class) + "ACCEPTED"
            return [True, SUCCESS]
    
    def admit_self_tuning(self, service, time, flow_id, traffic_class, receiver, rtt_delay, controller, debug):
        """
        Admit a task if there is an idle VM for the given service
        This strategy allocates VMs for specific service types (i,e., service + traffic_class)
        """
        self.service_class_count[service][traffic_class] += 1
        serviceTime = self.services[service].service_time
        self.cpuInfo.update_core_status(time) #need to call before simulate
        price = 0
        #if self.numberOfInstances[service][traffic_class] > 0:
        if self.numberOfInstances[service] > 0:
            price = self.service_class_price[service][traffic_class]
        else:
            controller.reject_service(time, flow_id, service, self.is_cloud, traffic_class, self.node, price)
            return [False, CONGESTION]

        core_indx, num_free_cores = self.cpuInfo.get_available_core(time)
        if core_indx == None:
            controller.reject_service(time, flow_id, service, self.is_cloud, traffic_class, self.node, price)
            return [False, CONGESTION]
        else:
            #if self.cpuInfo.count_running_service_type(service, traffic_class) >= self.numberOfInstances[service][traffic_class]:
            if self.cpuInfo.count_running_service(service) >= self.numberOfInstances[service]:
                controller.reject_service(time, flow_id, service, self.is_cloud, traffic_class, self.node, price)
                return [False, CONGESTION]

            #utility = self.utilities[service][traffic_class]
            finishTime = time + serviceTime
            #self.cpuInfo.assign_task_to_core(core_indx, finishTime, service, traffic_class)
            self.cpuInfo.assign_task_to_core(core_indx, finishTime, service)
            controller.add_event(finishTime, receiver, service, self.node, flow_id, traffic_class, rtt_delay, TASK_COMPLETE) 
            controller.execute_service(time, flow_id, service, self.is_cloud, traffic_class, self.node, price*serviceTime)
            return [True, SUCCESS]
    
    def admit_static_provisioning(self, service, time, flow_id, traffic_class, receiver, rtt_delay, controller, debug):
        """
        Admit a task if there is an idle VM for the given service
        This strategy allocates VMs for specific service types (i,e., service + traffic_class)
        """
        serviceTime = self.services[service].service_time
        self.cpuInfo.update_core_status(time) #need to call before simulate
        price = 0
        if self.numberOfInstances[service] > 0:
            price = self.service_class_price[service][traffic_class]
        else:
            controller.reject_service(time, flow_id, service, self.is_cloud, traffic_class, self.node, price)
            return [False, CONGESTION]

        core_indx, num_free_cores = self.cpuInfo.get_available_core(time)
        if core_indx == None:
            controller.reject_service(time, flow_id, service, self.is_cloud, traffic_class, self.node, price)
            return [False, CONGESTION]
        else:
            if self.cpuInfo.count_running_service(service) >= self.numberOfInstances[service]:
                controller.reject_service(time, flow_id, service, self.is_cloud, traffic_class, self.node, price)
                return [False, CONGESTION]

            #utility = self.utilities[service][traffic_class]
            finishTime = time + serviceTime
            self.cpuInfo.assign_task_to_core(core_indx, finishTime, service, traffic_class)
            controller.add_event(finishTime, receiver, service, self.node, flow_id, traffic_class, rtt_delay, TASK_COMPLETE) 
            controller.execute_service(time, flow_id, service, self.is_cloud, traffic_class, self.node, price*serviceTime)
            return [True, SUCCESS]

    def compute_utilities(self):
        u_max = 100.0
        service_max_delay = 0.0
        service_min_delay = float('inf')
        class_max_delay = [0.0] * self.num_classes

        for c in range(self.num_classes):
            class_max_delay[c] = self.model.topology.node[self.node]['max_delay'][c]

        service_max_delay = self.model.topology.graph['max_delay']
        service_min_delay = self.model.topology.graph['min_delay']
        class_u_min = [0.0]*self.num_classes
        
        print "Service max delay: " + repr(service_max_delay)
        print "Service min delay: " + repr(service_min_delay)

        for s in range(self.service_population):
            #print ("For service: " + repr(s))
            for c in range(self.num_classes):
                class_u_min = pow((service_max_delay - class_max_delay[c] + service_min_delay)/service_max_delay, 1/self.services[s].alpha)*(u_max - self.services[s].u_min) + self.services[s].u_min
                #print ("\tFor class: " + repr(c))
                #print ("\t\tclass_max_delay: " + repr(class_max_delay[c]))
                #print ("\t\tclass_u_min: " + repr(class_u_min))
                #print ("\t\tmin_delay: " + repr(self.model.topology.node[self.node]['min_delay'][c]))
                self.utilities[s][c] = pow((service_max_delay - self.model.topology.node[self.node]['min_delay'][c] + service_min_delay)/service_max_delay, 1/self.services[s].alpha)*(u_max - self.services[s].u_min) + self.services[s].u_min - class_u_min # QoS gain
                self.qos[s][c] = pow((service_max_delay - self.model.topology.node[self.node]['min_delay'][c] + service_min_delay)/service_max_delay, 1/self.services[s].alpha)*(u_max - self.services[s].u_min) + self.services[s].u_min

    def compute_prices(self, time, s=1.0,ControlPrint=False): #u,L,phi,gamma,mu_s,capacity):
        
        num_positive_indices = 0
        for s in range(self.service_population):
            for c in range(self.num_classes):
                if self.service_class_rate[s][c] < pow(10, -8):
                    self.service_class_rate[s][c] = 0.0
                else:
                    num_positive_indices += 1
        if num_positive_indices == 0:
            print "Arrival rate of all services are ~0 - Returning default price 100"
            self.vm_prices = [100.0]*cs.n_services
            self.rate_times[time] = [0.0] * self.service_population
            self.eff_rate_times[time] = [0.0] * self.service_population
            return
            
        #U,L,M,X,P,Y     = returnAppSPsInfoForThisMarket(incpID,options)
        Y               = [1.0] * self.service_population
        M               = [1.0/x.service_time for x in self.services]
        X               = [0.0] * self.service_population
        L               = self.service_class_rate
        U               = self.utilities
        u_sorted        = sorted(self.utilities, reverse=True)
        u_sorted_all_services = [j for i in u_sorted for j in i]
        u_sorted_all_services = sorted(u_sorted_all_services, reverse=True)
        P               = 100.0
        vmPrices        = []
        phi = 0.0

        if self.debugMode:
            ControlPrint = True

        if ControlPrint:
            print "Y = " + repr(Y)
            print "M = " + repr(M)
            print "X = " + repr(X)
            print "L = " + repr(L)
            print "U = " + repr(U)
            print "P = " + repr(P)
            print "u_sorted: " + repr(u_sorted)

        gainPerPrice  = {}
        utilisationPerPrice  = {}
        ratePerPrice  = {}
        effRatePerPrice = {}
        for P in u_sorted_all_services:
            if ControlPrint:
                print '------------------------------'
                print 'current price per service: ',P,', ',len(vmPrices)
            #estimate requested and admitted traffic
            for appSPID in range(self.service_population):
                X[appSPID],x = self.stage1AppSPCompactRequestedTraffic(P,U[appSPID],L[appSPID],M[appSPID], ControlPrint)
            flagMonetaryLosses,utilisation,monetaryGain,rates,effective_rates = self.incpMonetaryLoss(X,M,P,phi,self.n_services, ControlPrint)
            ratePerPrice[P] = rates
            effRatePerPrice[P] = effective_rates
            utilisationPerPrice[P] = utilisation
            gainPerPrice[P]        = monetaryGain
            #if market has negative profit break
            if flagMonetaryLosses:
                if ControlPrint:
                    print "Inside flagMonetaryLosses"
                break
            #update prices
            #P  -=s
            if P<0.0:
                 break
        #identify best value for money
        maxObjectiveValue  = -pow(10, -8)
        priceToAssign       = 100.0
        listOfPrices = sorted(list(gainPerPrice.keys()),reverse=True)
        for price in listOfPrices:
            if self.monetaryFocus:
                candidateObjectiveValue  =  gainPerPrice[price]
            else:
                candidateObjectiveValue  =  round(utilisationPerPrice[price],4)
            if ControlPrint:
                print  'candidate price: ',price,', with objective value: ',candidateObjectiveValue
            if candidateObjectiveValue>maxObjectiveValue:
                if ControlPrint:
                    print 'inside if max objective ',maxObjectiveValue,', candidate: ',candidateObjectiveValue
                maxObjectiveValue  = candidateObjectiveValue
                priceToAssign      = price

            if ControlPrint:
                print '\t\t\tprice: ',price,', utilisation: ',candidateObjectiveValue
                print '\t\t\tselected price: ',priceToAssign
        while True:
            if len(vmPrices)<self.n_services:
                vmPrices.append(priceToAssign)#vmPrices.append(0.0)
            else:
                break
        if priceToAssign in ratePerPrice.keys(): #dont remove this
            self.rate_times[time] = ratePerPrice[priceToAssign]
            self.eff_rate_times[time] = effRatePerPrice[priceToAssign]
        self.vm_prices = vmPrices 
        #print ("VM prices @node: " +  repr(self.node) + ": " + repr(self.vm_prices))
        if ControlPrint:
            print "Eff. rate at node: " + repr(self.node) + ": " +  repr(self.eff_rate_times[time])
        
        #update arrival rates for the next INCP
        for appSPID in range(self.service_population):
            X[appSPID],x = self.stage1AppSPCompactRequestedTraffic(P,U[appSPID],L[appSPID],M[appSPID], ControlPrint)
            if X[appSPID] != 0:
                if type(x) == float:
                    self.admitted_service_class_rate[appSPID] = [x]
                else:
                    self.admitted_service_class_rate[appSPID] = [x[c].tolist()[0][0] for c in range(self.num_classes)]
            else:
                self.admitted_service_class_rate[appSPID] = [0.0 for c in range(self.num_classes)]
        for appSPID in range(self.service_population):
            self.admitted_service_rate[appSPID] = X[appSPID]

        if ControlPrint:
            print "Done computing prices!"
    #stage1 app traffic requested 
    def stage1AppSPCompactRequestedTraffic(self, p, u, L, mu_s, ControlPrint=False):
        x = Variable(len(u))
                
        r_1 = x <= L
        r_2 = x >=0.0
        #constraints = [r_1,r_3,r_4]
        constraints = [r_1,r_2]
        p_s=[p]*len(u)
        param  = numpy.subtract(u,p_s)
        objective  = Maximize((1/mu_s)*sum_entries(mul_elemwise(param,x)))
        lp1 = Problem(objective,constraints)
        try:
            result = lp1.solve()
        except SolverError:
            print "Warning: Solver failed due to convergence problem!"
            return self.exceptionProblem(u)

             #result = lp1.solve(kktsolver=ROBUST_KKTSOLVER)
            #return 0, x
        #try:
        #    result = lp1.solve()
        #except:
        #    print 'L= ', L
        #    print 'p_s= ', p_s
        #    print 'u= ', u
        #    print 'param= ', param
        #    raise RuntimeError("Solver failed\n")
        #    sys.exit(0)

        Xsum  = 0.0
        if result<0 or math.fabs(result)<0.00001:#solver error estimation
            return Xsum,x
        for element in x:
            Xsum  += element.value
        if ControlPrint:
            print '\tAppSP gain: ',result,', ',x.value[0],', ',x.value[1], ', ', x.value[9]
            #print '\t\tlamda_1: ',r_1.dual_value
            print '\t\tsum of Xs: ',Xsum
        return Xsum,x.value

    def exceptionProblem(self, u):
        x = Variable(len(u))
                
        r_1 = x <= 0.0
        r_2 = x >=0.0
        #constraints = [r_1,r_3,r_4]
        constraints = [r_1,r_2]
        objective  = Maximize(x)
        lp1 = Problem(objective,constraints)
        result = lp1.solve()
        
        return 0.0, x.value

    def appSPTrafficRequestedForThisClass(self,p,classU,classLambda,mu_s):
        x = Variable()
        r_1 = x <=classLambda
        r_2 = x >=0.0
        constraints = [r_1,r_2]
        objective   = Maximize((1.0/mu_s)*(classU-p)*x)
        lp1 = Problem(objective,constraints)
        result = lp1.solve()
        return result,x.value
    #effective lambda-------------------------------------
    def estimateMaximumEffectiveL2(self,L,mu_s,totalCapacity, ControlPrint=False):
        #print "L = " + repr(L)
        #print "mu_s = " + repr(mu_s)
        rho  = float(L)/float(mu_s)
        totalCapacity  = int(totalCapacity)
        P_s,P_0  = self.P_sEstimation(totalCapacity,totalCapacity,rho)
        if ControlPrint:
            print ("Lambda = " + repr(L))
        effectiveLambda  = L*(1.0-P_s)
        if ControlPrint:
            print ("Effective Lambda = " + repr(effectiveLambda))
        return effectiveLambda

    def estimateMaximumEffectiveL(self, passingRates,exponentialTimeOfExecution,C_d,PrintFlag=False):
        if PrintFlag:
            print ("C_d: " + repr(C_d))
        #estimate rho
        rho    = passingRates/exponentialTimeOfExecution
        #Estimate P_d(n=0)
        P_0DS  = []
        for index in xrange(C_d+1):
            element = math.pow(rho,index)/math.factorial(index)
            P_0DS.append(element)
        P_Cd  = element/sum(P_0DS)
        #lambda effective
        lambdaEffective  = passingRates*(1.0-P_Cd)
        return lambdaEffective

    def P_sEstimation(self,s,totalCapacity,rho):
        #estimate P_0----------------------
        sumOfP_0Denominator  = 0.0
        #print "Rho is " + repr(rho)
        for index in xrange(totalCapacity+1):
            numerator    = float(math.pow(rho,index))
            #numerator    = rho**index
            denominator  = self.fact(index)
            sumOfP_0Denominator +=(numerator/denominator)
        P_0  = 1.0/sumOfP_0Denominator
        P_s  = float(math.pow(rho,s))/self.fact(s)
        P_s *= P_0
        return P_s,P_0
    """#
    def P_sEstimation(self,s,totalCapacity,rho):
        #estimate P_0----------------------
        sumOfP_0Denominator  = 0.0
        for index in xrange(totalCapacity+1):
            try:
                numerator    = float(math.pow(rho,index))
            except OverflowError:
                numerator = float('inf')
            denominator  = self.fact(index)
            sumOfP_0Denominator +=(numerator/denominator)
        P_0  = 1.0/sumOfP_0Denominator
        try:
            P_s  = float(math.pow(rho,s))/self.fact(s)
        except OverflowError:
            P_s = float('inf')
        P_s *= P_0
        return P_s,P_0
    """# 
    def fact(self, n):
        f = 1.0
        for x in range(1, n +1):
            f *= x
        return float(f)
    
    def incpMonetaryLoss(self,X,M,P,phi,vmsCapacity,ControlPrint=False):
        x      = []
        objective  = 0.0
        y  = 0.0
        price  = 0.0
        rho_thres = 0.95 #0.9999999999999
        effective_rates = [0.0] * self.service_population
        rates = [0.0] * self.service_population
        for appSPID in range(self.service_population):
            m  = 1.0/M[appSPID]
            p  = P-phi
            price  = P
            x  = self.estimateMaximumEffectiveL(X[appSPID],M[appSPID],vmsCapacity)
            rates[appSPID] = X[appSPID]
            effective_rates[appSPID] = x
            if ControlPrint:
                print '-------------'
                print '\tX ',X[appSPID],', M: ',M[appSPID]
                print '\t\tappSPID: ',appSPID,', Capacity: ',vmsCapacity
                print '\t\t\trate: ',X[appSPID],', eff. rate: ',x
                print '-------------'
            y +=m*x
            objective+=m*p*x #-options.gamma*pow(x,2)
        if ControlPrint:
            print 'price: ',price
            print '\tobjective monetary gain: ',objective
            print '\tadmitted traffic per service: ',X
            print '\taverage number of occupied VMs: ',y
        if objective<0 and math.fabs(objective)>0.001:
            return True,y/vmsCapacity,objective, rates, effective_rates
        elif y >= vmsCapacity*rho_thres:
            return True,y/vmsCapacity,objective, rates, effective_rates
        else:
            return False,y/vmsCapacity,objective, rates, effective_rates
    #other functions--------------------------------------
    #def admit_task_FIFO(self, service, time, flow_id, deadline, receiver, rtt_delay, controller, debug):
    def admit_task_FIFO(self, service, time, flow_id, traffic_class, receiver, rtt_delay, controller, debug):
        """
        Parameters
        ----------
        service : index of the service requested
        time    : current time (arrival time of the service job)

        Return
        ------
        comp_time : is when the task is going to be finished (after queuing + execution)
        vm_index : index of the VM that will execute the task
        """
        # Onur modified this method for auction paper
        self.service_class_count[service][traffic_class] += 1
        serviceTime = self.services[service].service_time
        self.cpuInfo.update_core_status(time) #need to call before simulate
        core_indx, num_free_cores = self.cpuInfo.get_available_core(time)
        if core_indx == None:
            #print "Time: " + repr(time) + " core indx: " + repr(core_indx) + " num_free_cores: " + repr(num_free_cores) + " Idle time: " + repr(self.cpuInfo.idleTime) + " Traffic class: " + repr(traffic_class) + " REJECTED CONG"
            controller.reject_service(time, flow_id, service, self.is_cloud, traffic_class, self.node, 0.0)
            return [False, CONGESTION]
        else:
            finishTime = time + serviceTime
            self.cpuInfo.assign_task_to_core(core_indx, finishTime, service)
            controller.add_event(finishTime, receiver, service, self.node, flow_id, traffic_class, rtt_delay, TASK_COMPLETE) 
            controller.execute_service(time, flow_id, service, self.is_cloud, traffic_class, self.node, 0.0)
            #print "Time: " + repr(time) + " core indx: " + repr(core_indx) + " num_free_cores: " + repr(num_free_cores) + " Idle time: " + repr(self.cpuInfo.idleTime) + "Traffic class: " + repr(traffic_class) + "ACCEPTED"
            return [True, SUCCESS]

    def admit_task_EDF(self, service, time, flow_id, deadline, receiver, rtt_delay, controller, debug):
        """
        Parameters
        ----------
        service : index of the service requested
        time    : current time (arrival time of the service job)

        Return
        ------
        comp_time : is when the task is going to be finished (after queuing + execution)
        vm_index : index of the VM that will execute the task
        """

        serviceTime = self.services[service].service_time
        if self.is_cloud:
            #aTask = Task(time, deadline, self.node, service, serviceTime, flow_id, receiver)
            controller.add_event(time+serviceTime, receiver, service, self.node, flow_id, deadline, rtt_delay, TASK_COMPLETE)
            controller.execute_service(flow_id, service, self.node, time, self.is_cloud)
            if debug:
                print ("CLOUD: Accepting TASK")
            return [True, CLOUD]
        
        if self.numberOfInstances[service] == 0:
            #print ("ERROR no instances in admit_task_EDF")
            return [False, NO_INSTANCES]
        
        if deadline - time - rtt_delay - serviceTime < 0:
            if debug:
                print ("Refusing TASK: deadline already missed")
            return [False, DEADLINE_MISSED]
            
        aTask = Task(time, deadline, rtt_delay, self.node, service, serviceTime, flow_id, receiver)
        self.taskQueue.append(aTask)
        self.taskQueue = sorted(self.taskQueue, key=lambda x: x.expiry) #smaller to larger (absolute) deadline
        self.cpuInfo.update_core_status(time) #need to call before simulate
        self.simulate_execution(aTask, time, debug)
        for task in self.taskQueue:
            if debug:
                print("After simulate:")
                task.print_task()
            if (task.expiry - task.rtt_delay) < task.finishTime:
                self.missed_requests[service] += 1
                if debug:
                    print("Refusing TASK: Congestion")
                self.taskQueue.remove(aTask)
                return [False, CONGESTION]
        
        # New task can be admitted, add to service Queue
        self.running_requests[service] += 1
        # Run the next task (if there is any)
        newTask = self.schedule(time) 
        if newTask is not None:
            controller.add_event(newTask.finishTime, newTask.receiver, newTask.service, self.node, newTask.flow_id, newTask.expiry, newTask.rtt_delay, TASK_COMPLETE) 
            controller.execute_service(newTask.flow_id, newTask.service, self.node, time, self.is_cloud)

        if self.numberOfInstances[service] == 0:
            print "Error: this should not happen in admit_task_EDF()"
       
        if debug:
            print ("Accepting Task")
        return [True, SUCCESS]

    def admit_task(self, service, time, flow_id, deadline, receiver, rtt_delay, controller, debug):
        ret = None
        if self.sched_policy == "EDF":
            ret = self.admit_task_EDF(service, time, flow_id, deadline, receiver, rtt_delay, controller, debug)
        elif self.sched_policy == "FIFO":
            ret = self.admit_task_FIFO(service, time, flow_id, deadline, receiver, rtt_delay, controller, debug)
        else:
            print ("Error: This should not happen in admit_task(): " +repr(self.sched_policy))
            
        return ret
    
    def reassign_vm(self, serviceToReplace, newService, debug):
        """
        Instantiate service at the given vm
        """
        if self.numberOfInstances[serviceToReplace] == 0:
            raise ValueError("Error in reassign_vm: the service to replace has no instances")
            
        if debug:
            print "Replacing service: " + repr(serviceToReplace) + " with: " + repr(newService) + " at node: " + repr(self.node)
        self.numberOfInstances[newService] += 1
        self.numberOfInstances[serviceToReplace] -= 1

    def getIdleTime(self, time):
        """
        Get the total idle time of the node
        """
        return self.cpuInfo.get_idleTime(time)

