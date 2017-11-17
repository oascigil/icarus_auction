# -*- coding: utf-8 -*-
"""Computational Spot implementation
This module contains the implementation of a set of VMs residing at a node. Each VM is abstracted as a FIFO queue. 
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
        # Currently running service instance at each cpu (scheduled earlier)
        self.coreService = [None]*self.numOfCores
        # Idle time of the server
        self.idleTime = 0.0

    def get_idleTime(self, time):
        
        # update the idle times
        for indx in range(0, self.numOfCores):
            if self.coreFinishTime[indx] < time:
                self.idleTime += time - self.coreFinishTime[indx]
                self.coreFinishTime[indx] = time
        
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
        if self.coreFinishTime[core_indx] == time:
            return core_indx
        else:
            return None
    
    def get_next_available_core(self):
        indx = self.coreFinishTime.index(min(self.coreFinishTime)) 
        
        return indx

    def assign_task_to_core(self, core_indx, fin_time, service):
        
        if self.coreFinishTime[core_indx] > fin_time:
            raise ValueError("Error in assign_task_to_core: there is a running task")
        
        self.coreService[core_indx] = service
        self.coreFinishTime[core_indx] = fin_time

    def update_core_status(self, time):
        
        for indx in range(0, self.numOfCores):
            if self.coreFinishTime[indx] <= time:
                self.idleTime += time - self.coreFinishTime[indx]
                self.coreFinishTime[indx] = time
                self.coreService[indx] = None

    def print_core_status(self):
        for indx in range(0, len(self.coreService)):
            print ("Core: " + repr(indx) + " finish time: " + repr(self.coreFinishTime[indx]) + " service: " + repr(self.coreService[indx]))

class ComputationalSpot(object):
    """ 
    A set of computational resources, where the basic unit of computational resource 
    is a VM. Each VM is bound to run a specific service instance and abstracted as a 
    Queue. The service time of the Queue is extracted from the service properties. 
    """

    def __init__(self, model, numOfCores, n_services, services, node, sched_policy = "EDF", dist=None):
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

        self.service_population = len(services)
        self.model = model
        self.num_classes = self.model.topology.graph['n_classes']

        print ("Number of VMs @node: " + repr(node) + " " + repr(n_services))
        print ("Number of cores @node: " + repr(node) + " " + repr(numOfCores))

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
        
        # num. of instances of each service in the memory
        self.numberOfInstances = [0]*self.service_population 

        # server missed requests (due to congestion)
        self.missed_requests = [0] * self.service_population
        
        # service request count (per service)
        self.running_requests = [0 for x in range(0, self.service_population)] #correct!
        
        # delegated service request counts (per service)
        self.delegated_requests = [0 for x in range(0, self.service_population)]
        
        self.services = services
        self.view = None
        self.node = node
        height = self.model.topology.graph['height']
        link_delay = self.model.topology.graph['link_delay']
        self.depth = self.model.topology.node[self.node]['depth']
        self.delay_to_cs = (height - self.depth)*link_delay 

        # Price of each VM
        self.vm_prices = None
        self.service_class_rate = [[0.0 for x in range(self.num_classes)] for y in range(self.service_population)]
        self.utilities = [[0.0 for x in range(self.num_classes)] for y in range(self.service_population)] 
        print "Computing utilities"
        self.compute_utilities() # compute the utilities of each service and class
        print "Utilities: " + repr(self.utilities)
        print "Done Computing utilities"
        # Outputs from the get_prices() call:
        self.admitted_service_rate = [0.0]*self.service_population
        self.admitted_service_class_rate = [[0.0 for x in range(self.num_classes)] for y in range(self.service_population)]

        # TODO setup all the variables: numberOfInstances, cpuInfo, etc. ...
        #num_services = 0
        #if dist is None and self.is_cloud == False:
            # setup a random set of services to run in the memory
        #    while num_services < self.n_services:
        #        service_index = random.choice(range(0, self.service_population))
        #        if self.numberOfInstances[service_index] == 0:
        #            self.numberOfInstances[service_index] = 1
        #            num_services += 1
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
                        print "Time: " + repr(time) + " core indx: " + repr(core_indx) + " num_free_cores: " + repr(num_free_cores) + " Idle time: " + repr(self.cpuInfo.idleTime) + " Traffic class: " + repr(traffic_class) + " REJECTED CONG " + "queue len: " + repr(len(self.taskQueue))
                        return [False, CONGESTION]
                print "Time: " + repr(time) + " core indx: " + repr(core_indx) + " num_free_cores: " + repr(num_free_cores) + " Idle time: " + repr(self.cpuInfo.idleTime) + " Traffic class: " + repr(traffic_class) + " ACCEPTED QUEUED " + "queue len: " + repr(len(self.taskQueue))
                # New task can be admitted, add to service Queue
                self.running_requests[service] += 1
                # Run the next task (if there is any)
                newTask = self.schedule(time) 
                if newTask is not None:
                    controller.add_event(newTask.finishTime, newTask.receiver, newTask.service, self.node, newTask.flow_id, newTask.traffic_class, newTask.rtt_delay, TASK_COMPLETE) 
                    controller.execute_service(newTask.finishTime, newTask.service, self.is_cloud, newTask.traffic_class, self.utilities, price) 
                return [True, SUCCESS]
            else:
                return [False, CONGESTION]
        else:
            utility = self.utilities[service][traffic_class]
            price = self.vm_prices[num_free_cores-1]
            if utility < price: # reject the request
                print "Time: " + repr(time) + " core indx: " + repr(core_indx) + " num_free_cores: " + repr(num_free_cores) + " Idle time: " + repr(self.cpuInfo.idleTime) + " Traffic class: " + repr(traffic_class) + " REJECTED UTIL " + "queue len: " + repr(len(self.taskQueue))
                return [False, CONGESTION]
            finishTime = time + serviceTime
            self.cpuInfo.assign_task_to_core(core_indx, finishTime, service)
            controller.add_event(finishTime, receiver, service, self.node, flow_id, traffic_class, rtt_delay, TASK_COMPLETE) 
            controller.execute_service(time, service, self.is_cloud, traffic_class, self.utilities, price)
            print "Time: " + repr(time) + " core indx: " + repr(core_indx) + " num_free_cores: " + repr(num_free_cores) + " Idle time: " + repr(self.cpuInfo.idleTime) + "Traffic class: " + repr(traffic_class) + "ACCEPTED " + "queue len: " + repr(len(self.taskQueue))
            return [True, SUCCESS]


    def admit_task_auction(self, service, time, flow_id, traffic_class, receiver, rtt_delay, controller, debug):
        """
        Admit a task if there is an idle VM 
        """
        serviceTime = self.services[service].service_time
        self.cpuInfo.update_core_status(time) #need to call before simulate
        core_indx, num_free_cores = self.cpuInfo.get_available_core(time)
        if core_indx == None:
            print "Time: " + repr(time) + " core indx: " + repr(core_indx) + " num_free_cores: " + repr(num_free_cores) + " Idle time: " + repr(self.cpuInfo.idleTime) + " Traffic class: " + repr(traffic_class) + " REJECTED CONG"
            return [False, CONGESTION]
        else:
            utility = self.utilities[service][traffic_class]
            price = self.vm_prices[num_free_cores-1]
            if utility < price: # reject the request
                print "Time: " + repr(time) + " core indx: " + repr(core_indx) + " num_free_cores: " + repr(num_free_cores) + " Idle time: " + repr(self.cpuInfo.idleTime) + " Traffic class: " + repr(traffic_class) + " REJECTED UTIL"
                return [False, CONGESTION]
            finishTime = time + serviceTime
            self.cpuInfo.assign_task_to_core(core_indx, finishTime, service)
            controller.add_event(finishTime, receiver, service, self.node, flow_id, traffic_class, rtt_delay, TASK_COMPLETE) 
            controller.execute_service(time, service, self.is_cloud, traffic_class, self.utilities, price)
            print "Time: " + repr(time) + " core indx: " + repr(core_indx) + " num_free_cores: " + repr(num_free_cores) + " Idle time: " + repr(self.cpuInfo.idleTime) + "Traffic class: " + repr(traffic_class) + "ACCEPTED"
            return [True, SUCCESS]
    
    def compute_utilities(self):
        u_max = 100.0
        service_max_delay = 0.0
        service_min_delay = float('inf')
        class_max_delay = [0.0] * self.num_classes
        class_min_delay = [0.0] * self.num_classes

        for c in range(self.num_classes):
            class_max_delay[c] = self.model.topology.graph['max_delay'][c]
            class_min_delay[c] = 0
            service_max_delay = max(service_max_delay, class_max_delay[c])
            service_min_delay = 0

        class_u_min = [0.0]*self.num_classes
        
        for s in range(self.service_population):
            for c in range(self.num_classes):
                class_u_min = pow((service_max_delay - class_max_delay[c] + service_min_delay)/service_max_delay, 1/self.services[s].alpha)*u_max

                self.utilities[s][c] = class_u_min + (u_max - class_u_min) * pow((class_max_delay[c] - (self.delay_to_cs + self.model.topology.graph['min_delay'][c]))/class_max_delay[c], 1/self.services[s].alpha) 

    def compute_prices(self, s=0.25,ControlPrint=False): #u,L,phi,gamma,mu_s,capacity):
        #U,L,M,X,P,Y     = returnAppSPsInfoForThisMarket(incpID,options)
        Y               = [1.0] * self.service_population
        M               = [1.0/x.service_time for x in self.services]
        X               = [0.0] * self.service_population
        L               = self.service_class_rate
        U               = self.utilities
        #P               = [float('inf')]*self.service_population
        P               = 100.0
        vmPrices        = []
        X_old           = 0.0
        requestedCapacity  = {}
        phi = 0.2
        for appSPID in range(self.service_population):
            requestedCapacity[appSPID]  = []
            for classIndex in xrange(len(L[appSPID])):
                requestedCapacity[appSPID].append(0.0)
        print "Y = " + repr(Y)
        print "M = " + repr(M)
        print "X = " + repr(X)
        print "L = " + repr(L)
        print "U = " + repr(U)
        print "P = " + repr(P)
        while True:#iteration
            if ControlPrint:
                print '------------------------------'
                print 'current price per service: ',P
            endProcess  = False
            X_current       = 0.0
            totalAdditionalVMsRequired  = 0.0
            #estimate requested and admitted traffic
            for appSPID in range(self.service_population):
                X[appSPID],x,additionalVMsRequired = self.stage1AppSPCompactRequestedTraffic(P,U[appSPID],L[appSPID],M[appSPID], requestedCapacity[appSPID])
                X_current   += X[appSPID]
                totalAdditionalVMsRequired+=additionalVMsRequired
            #if market has negative profit break
            flagMonetaryLosses  = self.incpMonetaryLoss(X,M,P, phi)
            if flagMonetaryLosses:
                break
            #if capacity of effective ration reached for some classes update the VM prices
            while totalAdditionalVMsRequired>0:
                if len(vmPrices)==self.n_services:#if cloudlet capacity is reached
                    endProcess  = True
                    break
                vmPrices.append(P)
                totalAdditionalVMsRequired-=1.0
            #if there is no further allocation at this price, decrease price by s -> Dutch auction
            if math.fabs(X_current-X_old)<0.0001:
                P  -=s
                if P<0.0:
                    endProcess  = True
            X_old  = X_current
            #print '\t\t\t\tVM prices ',vmPrices
            if endProcess:
                break
        while True:
            if len(vmPrices)<self.n_services:
                vmPrices.append(0.0)
            else:
                break
        #delta_C=math.ceil(rho*self.n_services)-self.n_services
        #print "delta_C: " + repr(delta_C)
        #vmPrices  = vmPrices[int(delta_C):]
        self.vm_prices = vmPrices
        #self.vm_prices = [0.0 for x in range(len(vmPrices))]
        print 'VM prices ', vmPrices
        #update arrival rates for the next INCP
        for appSPID in range(self.service_population):
            X[appSPID],x,additionalVMsRequired = self.stage1AppSPCompactRequestedTraffic(P,U[appSPID],L[appSPID],M[appSPID], requestedCapacity[appSPID])
            if X[appSPID] != 0:
                if type(x) == float:
                    self.admitted_service_class_rate[appSPID] = [x]
                else:
                    self.admitted_service_class_rate[appSPID] = [x[c] for c in range(self.num_classes)]
            else:
                self.admitted_service_class_rate[appSPID] = [0.0 for c in range(self.num_classes)]
        for appSPID in range(self.service_population):
            self.admitted_service_rate[appSPID] = X[appSPID]
        print 'Accepted rates: ', X
        #        continue
        #    index  = 0
        #    for classID in listOfClasses:
        #        appSpLamdaPerClass[appSPID,classID] -=x[index].item(0)
        #        index+=1

    def stage1AppSPCompactRequestedTraffic(self, p,u,L,mu_s,requestedCapacity,ControlPrint=False):
        #r_1 = sum_entries(x)<=Y
        L_new  = self.estimateMaximumEffectiveL(L,mu_s,requestedCapacity)
        result  = 0.0
        X_sum   = 0.0
        x       = []
        additionalVMsRequired  = 0.0
        for classIndex in xrange(len(L_new)):
            classU       = u[classIndex]
            originalLambda  = L[classIndex]
            classLambda  = L_new[classIndex]
            classResult,classX  = self.appSPTrafficRequestedForThisClass(p,classU,classLambda,mu_s)
            if math.fabs(classX-classLambda)<0.0001 and math.fabs(float(originalLambda-classLambda)/classLambda)>0.01:
                additionalVMsRequired  += 1.0
                requestedCapacity[classIndex] +=1
            result  += classResult
            X_sum   += classX
            x.append(classX)
        if result<0 or math.fabs(result)<0.00001:#solver error estimation
            return X_sum,x,0.0
        if ControlPrint:
            print '\tAppSP gain: ',result
            print '\t\tp ',p
            print '\t\tx: ',x
            print '\t\tL_new: ',L_new
            print '\t\tL: ',L
            print '\t\tAdditional VMs: ',additionalVMsRequired
            #print '\t\tlamda_1: ',r_1.dual_value
            print '\t\tsum of Xs: ',X_sum
        return X_sum,x,additionalVMsRequired
    
    def estimateMaximumEffectiveL(self, L,mu_s,requestedCapacity):
        effectiveLambda  = []
        for classIndex in xrange(len(L)):
            nextAvailableCapacity  = int(requestedCapacity[classIndex]+1)
            classLambda  = L[classIndex]
            #estimate probability 0---------
            rho  = classLambda/mu_s
            sumOfP_0Denominator  = 0.0
            for s in xrange(nextAvailableCapacity+1):
                numerator    = float(math.pow(rho,s))
                denominator  = self.fact(s)
                sumOfP_0Denominator +=(numerator/denominator)
            P_0  = 1.0/sumOfP_0Denominator
            P_s  = float(math.pow(rho,nextAvailableCapacity))/self.fact(nextAvailableCapacity)
            P_s*=P_0
            effectiveMaximumLambda  = classLambda*(1.0-P_s)
            effectiveLambda.append(effectiveMaximumLambda)
        return effectiveLambda

    def fact(self, n):
        f = 1.0
        for x in range(1, n +1):
            f *= x
        return float(f)

    def appSPTrafficRequestedForThisClass(self,p,classU,classLambda,mu_s):
        x = Variable()
        r_1 = x <=classLambda
        r_2 = x >=0.0
        constraints = [r_1,r_2]
        objective   = Maximize((1/mu_s)*(classU-p)*x)
        lp1 = Problem(objective,constraints)
        result = lp1.solve()
        return result,x.value

    def incpMonetaryLoss(self,X,M,P,phi,ControlPrint=False):
        x      = []
        objective  = 0.0
        y  = 0.0
        price  = 0.0
        for appSPID in range(self.service_population):
            m  = 1.0/M[appSPID]
            p  = P-phi
            price  = P
            x  = X[appSPID]
            y +=m*x
            objective+=m*p*x #-options.gamma*pow(x,2)
        if ControlPrint:
            print 'price: ',price
            print '\tobjective: ',objective
            print '\tadmitted traffic per service: ',X
            print '\taverage number of occupied VMs: ',y
        if objective<0 and math.fabs(objective)>0.001:
            return True
        else:
            return False
    def admit_task_FIFO(self, service, time, flow_id, deadline, receiver, rtt_delay, controller, debug):
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
            controller.add_event(time+serviceTime, receiver, service, self.node, flow_id, deadline, rtt_delay, TASK_COMPLETE)
            controller.execute_service(flow_id, service, self.node, time, self.is_cloud)
            if debug:
                print ("CLOUD: Accepting TASK")
            return [True, CLOUD]
        
        if self.numberOfInstances[service] == 0:
            return [False, NO_INSTANCES]
        
        aTask = Task(time, deadline, rtt_delay, self.node, service, serviceTime, flow_id, receiver)
        self.taskQueue.append(aTask)
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

