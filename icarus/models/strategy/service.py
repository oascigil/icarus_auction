# -*- coding: utf-8 -*-
"""Implementations of all service-based strategies"""
from __future__ import division
from __future__ import print_function

import networkx as nx
import random
import math

from icarus.registry import register_strategy
from icarus.util import inheritdoc, path_links
from .base import Strategy

__all__ = [
       'StrictestDeadlineFirst',
       'MostFrequentlyUsed',
       'Hybrid',
       'Lru', 
       'DoubleAuction',
       'DoubleAuctionTrace',
       'SelfTuningTrace',
       'LFUTrace',
       'StaticTrace'
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

# Auction
@register_strategy('DOUBLE_AUCTION')
class DoubleAuction(Strategy):
    """A distributed approach for service-centric routing
    """
    def __init__(self, view, controller, replacement_interval=5.0, debug=False, **kwargs):
        super(DoubleAuction, self).__init__(view,controller)
        self.receivers = view.topology().receivers()
        self.compSpots = self.view.service_nodes()
        self.num_nodes = len(self.compSpots.keys())
        self.num_services = self.view.num_services()
        self.debug = debug
        self.replacement_interval = replacement_interval
        self.last_replacement = 0.0
        for node in self.compSpots.keys():
            cs = self.compSpots[node]
            self.controller.set_vm_prices(node, cs.vm_prices)
            self.controller.set_node_util(node, cs.utilities)
            
    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log, node, flow_id, traffic_class, rtt_delay, status):
        if time - self.last_replacement > self.replacement_interval:
            #print("Evaluation interval over at time: " + repr(time))
            self.controller.replacement_interval_over(self.replacement_interval, time)
            self.last_replacement = time
        service = content
        cloud = self.view.content_source(service)
        if receiver == node and status == REQUEST:
            self.controller.start_session(time, receiver, service, log, flow_id, traffic_class)
            path = self.view.shortest_path(node, cloud)
            next_node = path[1]
            delay = self.view.path_delay(node, next_node)
            self.controller.add_event(time+delay, receiver, service, next_node, flow_id, traffic_class, rtt_delay, REQUEST)
            return
        
        if self.debug:
            print ("\nEvent\n time: " + repr(time) + " receiver  " + repr(receiver) + " service " + repr(service) + " node " + repr(node) + " flow_id " + repr(flow_id) + " traffic class " + repr(traffic_class) + " status " + repr(status)) 
        
        compSpot = None
        if self.view.has_computationalSpot(node):
            compSpot = self.view.compSpot(node)
        
        if status == RESPONSE: 
            # response is on its way back to the receiver
            if node == receiver:
                self.controller.end_session(True, time, flow_id) #TODO add flow_time
                return
            else:
                path = self.view.shortest_path(node, receiver)
                next_node = path[1]
                delay = self.view.link_delay(node, next_node)
                self.controller.add_event(time+delay, receiver, service, next_node, flow_id, traffic_class, rtt_delay, RESPONSE)

        elif status == TASK_COMPLETE:
            #schedule the next queued task (if this is not the cloud)
            if node != cloud:
                task = compSpot.schedule(time)
                if task is not None:
                    self.controller.add_event(task.finishTime, task.receiver, task.service, node, task.flow_id, task.traffic_class, task.rtt_delay, TASK_COMPLETE)
                    print ("Task service: " + repr(task.service) + " traffic class: " + repr(task.traffic_class))
                    self.controller.execute_service(task.finishTime, flow_id, task.service, False, task.traffic_class, node, compSpot.vm_prices[0]) 
            # forward the completed task
            path = self.view.shortest_path(node, receiver)
            next_node = path[1]
            delay = self.view.link_delay(node, next_node)
            self.controller.add_event(time+delay, receiver, service, next_node, flow_id, traffic_class, rtt_delay, RESPONSE)
            
        elif status == REQUEST:
            # Processing a request
            if node == cloud: # request reached the cloud
                service_time = self.view.get_service_time(service)
                self.controller.add_event(time+service_time, receiver, service, node, flow_id, traffic_class, rtt_delay, TASK_COMPLETE)
                self.controller.execute_service(time, flow_id, service, True, traffic_class, node, 0)
            else:    
                path = self.view.shortest_path(node, cloud)
                next_node = path[1]
                delay = self.view.path_delay(node, next_node)
                ret, reason = compSpot.admit_task_auction(service, time, flow_id, traffic_class, receiver, rtt_delay, self.controller, self.debug)
                if ret == False:
                    delay = self.view.path_delay(node, next_node)
                    rtt_delay += 2*delay
                    self.controller.add_event(time+delay, receiver, service, next_node, flow_id, traffic_class, rtt_delay, REQUEST)

# LRU
@register_strategy('LRU')
class Lru(Strategy):
    """A distributed approach for service-centric routing
    """
    def __init__(self, view, controller, replacement_interval=10, debug=False, p = 0.5, **kwargs):
        super(Lru, self).__init__(view,controller)
        self.last_replacement = 0
        self.replacement_interval = replacement_interval
        self.receivers = view.topology().receivers()
        self.compSpots = self.view.service_nodes()
        self.num_nodes = len(self.compSpots.keys())
        self.num_services = self.view.num_services()
        self.debug = debug
        self.p = p

    def initialise_metrics(self):
        """
        Initialise metrics/counters to 0
        """
        for node in self.compSpots.keys():
            cs = self.compSpots[node]
            cs.running_requests = [0 for x in range(0, self.num_services)]
            cs.missed_requests = [0 for x in range(0, self.num_services)]
            cs.cpuInfo.idleTime = 0.0

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log, node, flow_id, deadline, rtt_delay, status):
        """
        response : True, if this is a response from the cloudlet/cloud
        deadline : deadline for the request 
        flow_id : Id of the flow that the request/response is part of
        node : the current node at which the request/response arrived
        """
        
        service = content

        if time - self.last_replacement > self.replacement_interval:
            self.controller.replacement_interval_over(flow_id, self.replacement_interval, time)
            self.last_replacement = time
            self.initialise_metrics()
        if receiver == node and status == REQUEST:
            self.controller.start_session(time, receiver, service, log, flow_id, deadline)
            source = self.view.content_source(service)
            path = self.view.shortest_path(node, source)
            next_node = path[1]
            delay = self.view.path_delay(node, next_node)
            rtt_delay += delay*2
            self.controller.add_event(time+delay, receiver, service, next_node, flow_id, deadline, rtt_delay, REQUEST)
            return

        if self.debug:
            print ("\nEvent\n time: " + repr(time) + " receiver  " + repr(receiver) + " service " + repr(service) + " node " + repr(node) + " flow_id " + repr(flow_id) + " deadline " + repr(deadline) + " status " + repr(status)) 

        compSpot = None
        if self.view.has_computationalSpot(node):
            compSpot = self.view.compSpot(node)
        else: # the node has no computational spots (0 services)
            if status is not RESPONSE:
                source = self.view.content_source(service)
                if node == source:
                    print ("Error: reached the source node: " + repr(node) + " this should not happen!")
                    return
                path = self.view.shortest_path(node, source)
                next_node = path[1]
                delay = self.view.link_delay(node, next_node)
                if self.debug:
                    print ("Pass upstream (no compSpot) to node: " + repr(next_node) + " " + repr(time+delay))
                self.controller.add_event(time+delay, receiver, service, next_node, flow_id, deadline, rtt_delay+2*delay, REQUEST)
                return
        
        if status == RESPONSE: 
            # response is on its way back to the receiver
            if node == receiver:
                self.controller.end_session(True, time, flow_id) #TODO add flow_time
                return
            else:
                path = self.view.shortest_path(node, receiver)
                next_node = path[1]
                delay = self.view.link_delay(node, next_node)
                self.controller.add_event(time+delay, receiver, service, next_node, flow_id, deadline, rtt_delay, RESPONSE)

        elif status == TASK_COMPLETE:
            task = compSpot.schedule(time)
            #schedule the next queued task at this node
            if task is not None:
                self.controller.add_event(task.finishTime, task.receiver, task.service, node, task.flow_id, task.expiry, task.rtt_delay, TASK_COMPLETE)

            # forward the completed task
            path = self.view.shortest_path(node, receiver)
            next_node = path[1]
            delay = self.view.link_delay(node, next_node)
            self.controller.add_event(time+delay, receiver, service, next_node, flow_id, deadline, rtt_delay, RESPONSE)
            
        elif status == REQUEST:
            # Processing a request
            source = self.view.content_source(service)
            path = self.view.shortest_path(node, source)
            next_node = path[1]
            delay = self.view.path_delay(node, next_node)
            ret, reason = compSpot.admit_task(service, time, flow_id, deadline, receiver, rtt_delay, self.controller, self.debug)
            if ret == False:
                if reason == NO_INSTANCES:
                    # is upstream possible to execute
                    if deadline - time - rtt_delay - 2*delay < compSpot.services[service].service_time:
                        evicted = self.controller.put_content(node, service)
                        compSpot.reassign_vm(evicted, service, self.debug)
                    elif self.p == 1.0 or random.random() <= self.p:
                        evicted = self.controller.put_content(node, service)
                        compSpot.reassign_vm(evicted, service, self.debug)
                rtt_delay += 2*delay
                self.controller.add_event(time+delay, receiver, service, next_node, flow_id, deadline, rtt_delay, REQUEST)
            else:
                self.controller.get_content(node, service)

@register_strategy('HYBRID')
class Hybrid(Strategy):
    """A distributed approach for service-centric routing
    """
    
    def __init__(self, view, controller, replacement_interval=10, debug=False, n_replacements=1, **kwargs):
        super(Hybrid, self).__init__(view,controller)
        self.replacement_interval = replacement_interval
        self.n_replacements = n_replacements
        self.last_replacement = 0
        self.receivers = view.topology().receivers()
        self.compSpots = self.view.service_nodes()
        self.num_nodes = len(self.compSpots.keys())
        self.num_services = self.view.num_services()
        self.debug = debug
        # metric to rank each VM of Comp. Spot
        self.deadline_metric = {x : {} for x in range(0, self.num_nodes)}
        self.cand_deadline_metric = {x : {} for x in range(0, self.num_nodes)}
        for node in self.compSpots.keys():
            cs = self.compSpots[node]
            for vm_indx in range(0, self.num_services):
                self.deadline_metric[node][vm_indx] = 0.0
            for service_indx in range(0, self.num_services):
                self.cand_deadline_metric[node][service_indx] = 0.0
    
    # Hybrid
    def initialise_metrics(self):
        """
        Initialise metrics/counters to 0
        """
        for node in self.compSpots.keys():
            cs = self.compSpots[node]
            cs.running_requests = [0 for x in range(0, self.num_services)]
            cs.missed_requests = [0 for x in range(0, self.num_services)]
            cs.cpuInfo.idleTime = 0.0
            for vm_indx in range(0, self.num_services):
                self.deadline_metric[node][vm_indx] = 0.0
            for service_indx in range(0, self.num_services):
                self.cand_deadline_metric[node][service_indx] = 0.0
    #HYBRID 
    def replace_services(self, k, time):
        """
        This method does the following:
        1. Evaluate instantiated and stored services at each computational spot for the past time interval, ie, [t-interval, t]. 
        2. Decide which services to instantiate in the next time interval [t, t+interval].
        Parameters:
        k : max number of instances to replace at each computational spot
        interval: the length of interval
        """
        # First sort services by deadline strictness
        for node, cs in self.compSpots.items():
            if cs.is_cloud:
                continue
            n_replacements = k
            service_deadlines = []
            service_util = [] 
            n_requests = {}
            utilisation_metrics = []
            cand_services = []
            vm_metrics = self.deadline_metric[node]
            cand_metric = self.cand_deadline_metric[node]
            if self.debug:
                print ("Replacement at node " + repr(node))
            for service in range(0, self.num_services):
                d_metric = 0.0
                u_metric = 0.0
                if cs.numberOfInstances[service] == 0 and cs.missed_requests[service] > 0:
                    d_metric = cand_metric[service]/cs.missed_requests[service]
                    u_metric = cs.missed_requests[service] * cs.services[service].service_time
                elif cs.numberOfInstances[service] > 0 and cs.running_requests[service] > 0: 
                    d_metric = vm_metrics[service]/cs.running_requests[service]
                    u_metric = cs.running_requests[service] * cs.services[service].service_time
                if d_metric > 0.0:
                    service_deadlines.append([service, d_metric])
                if u_metric > 0.0:
                   service_util.append([service, u_metric])
                n_requests[service] = cs.running_requests[service] + cs.missed_requests[service]
            service_deadlines = sorted(service_deadlines, key=lambda x: x[1]) # small to large
            services = []
            # Fill the memory with time critical services 
            for indx in range(0, len(service_deadlines)):
                service = service_deadlines[indx][0]
                metric = service_deadlines[indx][1]
                if cs.rtt_upstream[service] == 0:
                    #compute rtt to upstream node
                    source = self.view.content_source(service)
                    path = self.view.shortest_path(node, source)
                    next_node = path[1]
                    cs.rtt_upstream[service] = self.view.path_delay(node, next_node)
                if metric > cs.rtt_upstream[service]:
                    break
                services.append([service, n_requests[service]])
            
            # services that are not delegatable upstream and have decreasing utilisations
            services = sorted(services, key=lambda x: x[1], reverse=True) #large to small
            n_services = 0
            cs.numberOfInstances = [0]*cs.service_population 
            for indx in range(0, len(services)):
                service = services[indx][0]
                if n_requests[service] == 1:
                    break
                cs.numberOfInstances[service] = 1
                n_services += 1
                if n_services == cs.n_services:
                    break

            # fill the remaining slots with high utilisation services
            if (cs.n_services > n_services):
                print(repr(cs.n_services-n_services) + " slots filled with popular services")
                service_util = sorted(service_util, key=lambda x: x[1], reverse=True) # large to small
            else:
                print ("No slots for popular services")
            indx = 0
            while (cs.n_services > n_services) and (indx < len(service_util)):
                service = service_util[indx][0]
                indx += 1
                cs.numberOfInstances[service] += 1
                n_services += 1

    #HYBRID 
    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log, node, flow_id, deadline, rtt_delay, status):
        """
        response : True, if this is a response from the cloudlet/cloud
        deadline : deadline for the request 
        flow_id : Id of the flow that the request/response is part of
        node : the current node at which the request/response arrived
        """
        #self.debug = False
        #if node == 12:
        #    self.debug = True

        service = content

        if time - self.last_replacement > self.replacement_interval:
            #self.print_stats()
            self.controller.replacement_interval_over(flow_id, self.replacement_interval, time)
            self.replace_services(self.n_replacements, time)
            self.last_replacement = time
            self.initialise_metrics()
        
        if receiver == node and status == REQUEST:
            self.controller.start_session(time, receiver, service, log, flow_id, deadline)
            source = self.view.content_source(service)
            path = self.view.shortest_path(node, source)
            next_node = path[1]
            delay = self.view.path_delay(node, next_node)
            rtt_delay += delay*2
            self.controller.add_event(time+delay, receiver, service, next_node, flow_id, deadline, rtt_delay, REQUEST)
            return

        if self.debug:
            print ("\nEvent\n time: " + repr(time) + " receiver  " + repr(receiver) + " service " + repr(service) + " node " + repr(node) + " flow_id " + repr(flow_id) + " deadline " + repr(deadline) + " status " + repr(status)) 

        compSpot = None
        if self.view.has_computationalSpot(node):
            compSpot = self.view.compSpot(node)
        else: # the node has no computational spots (0 services)
            if status is not RESPONSE:
                source = self.view.content_source(service)
                if node == source:
                    print ("Error: reached the source node: " + repr(node) + " this should not happen!")
                    return
                path = self.view.shortest_path(node, source)
                next_node = path[1]
                delay = self.view.link_delay(node, next_node)
                if self.debug:
                    print ("Pass upstream (no compSpot) to node: " + repr(next_node) + " " + repr(time+delay))
                self.controller.add_event(time+delay, receiver, service, next_node, flow_id, deadline, rtt_delay+2*delay, REQUEST)
                return
            
        if status == RESPONSE: 
            # response is on its way back to the receiver
            if node == receiver:
                self.controller.end_session(True, time, flow_id) #TODO add flow_time
                return
            else:
                path = self.view.shortest_path(node, receiver)
                next_node = path[1]
                delay = self.view.link_delay(node, next_node)
                self.controller.add_event(time+delay, receiver, service, next_node, flow_id, deadline, rtt_delay, RESPONSE)

        elif status == TASK_COMPLETE:
            task = compSpot.schedule(time)
            #schedule the next queued task at this node
            if task is not None:
                self.controller.add_event(task.finishTime, task.receiver, task.service, node, task.flow_id, task.expiry, task.rtt_delay, TASK_COMPLETE)

            # forward the completed task
            path = self.view.shortest_path(node, receiver)
            next_node = path[1]
            delay = self.view.link_delay(node, next_node)
            self.controller.add_event(time+delay, receiver, service, next_node, flow_id, deadline, rtt_delay, RESPONSE)
            
        elif status == REQUEST:
            # Processing a request
            source = self.view.content_source(service)
            path = self.view.shortest_path(node, source)
            next_node = path[1]
            delay = self.view.path_delay(node, next_node)
            deadline_metric = (deadline - time - rtt_delay - compSpot.services[service].service_time)/deadline
            if self.debug:
                print ("Deadline metric: " + repr(deadline_metric))
            if self.view.has_service(node, service):
                if self.debug:
                    print ("Calling admit_task")
                ret, reason = compSpot.admit_task(service, time, flow_id, deadline, receiver, rtt_delay, self.controller, self.debug)
                if self.debug:
                    print ("Done Calling admit_task")
                if ret is False:    
                    # Pass the Request upstream
                    rtt_delay += delay*2
                    self.controller.add_event(time+delay, receiver, service, next_node, flow_id, deadline, rtt_delay, REQUEST)
                    if deadline_metric > 0:
                        self.cand_deadline_metric[node][service] += deadline_metric
                    if self.debug:
                        print ("Pass upstream to node: " + repr(next_node))
                else:
                    if deadline_metric > 0:
                        self.deadline_metric[node][service] += deadline_metric
            else: #Not running the service
                compSpot.missed_requests[service] += 1
                if self.debug:
                    print ("Not running the service: Pass upstream to node: " + repr(next_node))
                rtt_delay += delay*2
                self.controller.add_event(time+delay, receiver, service, next_node, flow_id, deadline, rtt_delay, REQUEST)
                if deadline_metric > 0:
                    self.cand_deadline_metric[node][service] += deadline_metric
        else:
            print ("Error: unrecognised status value : " + repr(status))


# Highest Utilisation First Strategy 
@register_strategy('MFU')
class MostFrequentlyUsed(Strategy):
    """A distributed approach for service-centric routing
    """
    
    def __init__(self, view, controller, replacement_interval=10, debug=False, n_replacements=1, **kwargs):
        super(MostFrequentlyUsed, self).__init__(view,controller)
        self.replacement_interval = replacement_interval
        self.n_replacements = n_replacements
        self.last_replacement = 0
        self.receivers = view.topology().receivers()
        self.compSpots = self.view.service_nodes()
        self.num_nodes = len(self.compSpots.keys())
        self.num_services = self.view.num_services()
        self.debug = debug
        # metric to rank each VM of Comp. Spot
        self.deadline_metric = {x : {} for x in range(0, self.num_nodes)}
        self.cand_deadline_metric = {x : {} for x in range(0, self.num_nodes)}
        for node in self.compSpots.keys():
            cs = self.compSpots[node]
            for vm_indx in range(0, self.num_services):
                self.deadline_metric[node][vm_indx] = 0.0
            for service_indx in range(0, self.num_services):
                self.cand_deadline_metric[node][service_indx] = 0.0
    
    # MFU
    def initialise_metrics(self):
        """
        Initialise metrics/counters to 0
        """
        for node in self.compSpots.keys():
            cs = self.compSpots[node]
            cs.running_requests = [0 for x in range(0, self.num_services)]
            cs.missed_requests = [0 for x in range(0, self.num_services)]
            cs.cpuInfo.idleTime = 0.0
            for vm_indx in range(0, self.num_services):
                self.deadline_metric[node][vm_indx] = 0.0
            for service_indx in range(0, self.num_services):
                self.cand_deadline_metric[node][service_indx] = 0.0
    # MFU
    def replace_services(self, k, time):
        """
        This method does the following:
        1. Evaluate instantiated and stored services at each computational spot for the past time interval, ie, [t-interval, t]. 
        2. Decide which services to instantiate in the next time interval [t, t+interval].
        Parameters:
        k : max number of instances to replace at each computational spot
        interval: the length of interval
        """

        for node, cs in self.compSpots.items():
            if cs.is_cloud:
                continue
            n_replacements = k
            vms = []
            cand_services = []
            vm_metrics = self.deadline_metric[node]
            cand_metric = self.cand_deadline_metric[node]
            if self.debug:
                print ("Replacement at node " + repr(node))
            for service in range(0, self.num_services):
                usage_metric = 0.0
                if cs.numberOfInstances[service] == 0:
                    if self.debug:
                        print ("No running instances of service: " + repr(service)) 
                    continue
                if cs.running_requests[service] == 0:
                    usage_metric = 0.0 
                    if self.debug:
                        print ("No scheduled requests for service: " + repr(service)) 
                else:
                    usage_metric = cs.running_requests[service] * cs.services[service].service_time
                    if self.debug:
                        print ("Usage metric for service: " + repr(service) + " is " + repr(usage_metric))

                vms.append([service, usage_metric])
            
            for service in range(0, self.num_services):
                #if cs.numberOfInstances[service] > 0:
                #    continue
                if cs.missed_requests[service] == 0:
                    usage_metric = 0.0
                else:
                    usage_metric = cs.missed_requests[service] * cs.services[service].service_time
                    if self.debug:
                        print ("Usage metric for Upstream Service: " + repr(service) + " is " + repr(usage_metric))
                cand_services.append([service, usage_metric])

            # sort vms and virtual_vms arrays according to metric
            vms = sorted(vms, key=lambda x: x[1]) #small to large
            cand_services = sorted(cand_services, key=lambda x: x[1], reverse=True) #large to small
            if self.debug:
                print ("VMs: " + repr(vms))
                print ("Cand. Services: " + repr(cand_services))
            # Small metric is better
            indx = 0
            for vm in vms: 
                if vm[1] > cand_services[indx][1]:
                    break
                else:
                    # Are they the same service? This should not happen really
                    if vm[0] != cand_services[indx][0]:
                        cs.reassign_vm(vm[0], cand_services[indx][0], self.debug)
                        n_replacements -= 1
                
                if n_replacements == 0 or indx == len(cand_services):
                    break
                indx += 1

    # MFU
    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log, node, flow_id, deadline, rtt_delay, status):
        """
        response : True, if this is a response from the cloudlet/cloud
        deadline : deadline for the request 
        flow_id : Id of the flow that the request/response is part of
        node : the current node at which the request/response arrived
        """
        service = content

        if time - self.last_replacement > self.replacement_interval:
            #self.print_stats()
            self.controller.replacement_interval_over(flow_id, self.replacement_interval, time)
            self.replace_services(self.n_replacements, time)
            self.last_replacement = time
            self.initialise_metrics()
        
        if receiver == node and status == REQUEST:
            self.controller.start_session(time, receiver, service, log, flow_id, deadline)
            source = self.view.content_source(service)
            path = self.view.shortest_path(node, source)
            next_node = path[1]
            delay = self.view.path_delay(node, next_node)
            rtt_delay += delay*2
            self.controller.add_event(time+delay, receiver, service, next_node, flow_id, deadline, rtt_delay, REQUEST)
            return

        if self.debug:
            print ("\nEvent\n time: " + repr(time) + " receiver  " + repr(receiver) + " service " + repr(service) + " node " + repr(node) + " flow_id " + repr(flow_id) + " deadline " + repr(deadline) + " status " + repr(status)) 

        # MFU
        compSpot = None
        if self.view.has_computationalSpot(node):
            compSpot = self.view.compSpot(node)
        else: # the node has no computational spots (0 services)
            if status is not RESPONSE:
                source = self.view.content_source(service)
                if node == source:
                    print ("Error: reached the source node: " + repr(node) + " this should not happen!")
                    return
                path = self.view.shortest_path(node, source)
                next_node = path[1]
                delay = self.view.link_delay(node, next_node)
                if self.debug:
                    print ("Pass upstream (no compSpot) to node: " + repr(next_node) + " " + repr(time+delay))
                self.controller.add_event(time+delay, receiver, service, next_node, flow_id, deadline, rtt_delay+2*delay, REQUEST)
                return
            
        if status == RESPONSE: 
            # response is on its way back to the receiver
            if node == receiver:
                self.controller.end_session(True, time, flow_id) #TODO add flow_time
                return
            else:
                path = self.view.shortest_path(node, receiver)
                next_node = path[1]
                delay = self.view.link_delay(node, next_node)
                self.controller.add_event(time+delay, receiver, service, next_node, flow_id, deadline, rtt_delay, RESPONSE)

        elif status == TASK_COMPLETE:
            task = compSpot.schedule(time)
            #schedule the next queued task at this node
            if task is not None:
                self.controller.add_event(task.finishTime, task.receiver, task.service, node, task.flow_id, task.expiry, task.rtt_delay, TASK_COMPLETE)

            # forward the completed task
            path = self.view.shortest_path(node, receiver)
            next_node = path[1]
            delay = self.view.link_delay(node, next_node)
            self.controller.add_event(time+delay, receiver, service, next_node, flow_id, deadline, rtt_delay, RESPONSE)
            
        # MFU
        elif status == REQUEST:
            # Processing a request
            source = self.view.content_source(service)
            path = self.view.shortest_path(node, source)
            next_node = path[1]
            delay = self.view.path_delay(node, next_node)
            deadline_metric = (deadline - time - rtt_delay - compSpot.services[service].service_time)/deadline
            if self.debug:
                print ("Deadline metric: " + repr(deadline_metric))
            if self.view.has_service(node, service):
                if self.debug:
                    print ("Calling admit_task")
                ret, reason = compSpot.admit_task(service, time, flow_id, deadline, receiver, rtt_delay, self.controller, self.debug)
                if self.debug:
                    print ("Done Calling admit_task")
                if ret is False:    
                    # Pass the Request upstream
                    rtt_delay += delay*2
                    self.controller.add_event(time+delay, receiver, service, next_node, flow_id, deadline, rtt_delay, REQUEST)
                    if deadline_metric > 0:
                        self.cand_deadline_metric[node][service] += deadline_metric
                    if self.debug:
                        print ("Pass upstream to node: " + repr(next_node))
                else:
                    if deadline_metric > 0:
                        self.deadline_metric[node][service] += deadline_metric
            else: #Not running the service
                compSpot.missed_requests[service] += 1
                if self.debug:
                    print ("Not running the service: Pass upstream to node: " + repr(next_node))
                rtt_delay += delay*2
                self.controller.add_event(time+delay, receiver, service, next_node, flow_id, deadline, rtt_delay, REQUEST)
                if deadline_metric > 0:
                    self.cand_deadline_metric[node][service] += deadline_metric
        else:
            print ("Error: unrecognised status value : " + repr(status))

# Strictest Deadline First Strategy
@register_strategy('SDF')
class StrictestDeadlineFirst(Strategy):
    """ A distributed approach for service-centric routing
    """
    # SDF  
    def __init__(self, view, controller, replacement_interval=10, debug=False, n_replacements=1, **kwargs):
        super(StrictestDeadlineFirst, self).__init__(view,controller)
        self.replacement_interval = replacement_interval
        self.n_replacements = n_replacements
        self.last_replacement = 0
        self.receivers = view.topology().receivers()
        self.compSpots = self.view.service_nodes()
        self.num_nodes = len(self.compSpots.keys())
        self.num_services = self.view.num_services()
        self.debug = debug
        # metric to rank each VM of Comp. Spot
        self.deadline_metric = {x : {} for x in range(0, self.num_nodes)}
        self.cand_deadline_metric = {x : {} for x in range(0, self.num_nodes)}
        for node in self.compSpots.keys():
            cs = self.compSpots[node]
            for vm_indx in range(0, self.num_services):
                self.deadline_metric[node][vm_indx] = 0.0
            for service_indx in range(0, self.num_services):
                self.cand_deadline_metric[node][service_indx] = 0.0
    
    # SDF  
    def initialise_metrics(self):
        """
        Initialise metrics/counters to 0
        """
        for node in self.compSpots.keys():
            cs = self.compSpots[node]
            cs.running_requests = [0 for x in range(0, self.num_services)]
            cs.missed_requests = [0 for x in range(0, self.num_services)]
            cs.cpuInfo.idleTime = 0.0
            for vm_indx in range(0, self.num_services):
                self.deadline_metric[node][vm_indx] = 0.0
            for service_indx in range(0, self.num_services):
                self.cand_deadline_metric[node][service_indx] = 0.0
    # SDF  
    def replace_services(self, k, time):
        """
        This method does the following:
        1. Evaluate instantiated and stored services at each computational spot for the past time interval, ie, [t-interval, t]. 
        2. Decide which services to instantiate in the next time interval [t, t+interval].
        Parameters:
        k : max number of instances to replace at each computational spot
        interval: the length of interval
        """

        for node, cs in self.compSpots.items():
            if cs.is_cloud:
                continue
            n_replacements = k
            vms = []
            cand_services = []
            vm_metrics = self.deadline_metric[node]
            cand_metric = self.cand_deadline_metric[node]
            if self.debug:
                print ("Replacement at node " + repr(node))
            for service in range(0, self.num_services):
                if cs.numberOfInstances[service] == 0:
                    if self.debug:
                        print ("No running instances of service: " + repr(service)) 
                    continue
                d_metric = 0.0
                if cs.running_requests[service] == 0:
                    d_metric = 1.0 
                    if self.debug:
                        print ("No scheduled requests for service: " + repr(service)) 
                else:
                    d_metric = vm_metrics[service]/cs.running_requests[service]
                    if self.debug:
                        print ("Deadline metric for service: " + repr(service) + " is " + repr(d_metric))

                vms.append([service, d_metric])
            # SDF  
            for service in range(0, self.num_services):
                #if self.debug:
                #    print ("\tNumber of Requests for upstream service " + repr(service) + " is "  + repr(cs.missed_requests[service]))
                d_metric = 0.0
                if cs.missed_requests[service] == 0:
                    d_metric = 1.0
                else:
                    d_metric = cand_metric[service]/cs.missed_requests[service]
                    if self.debug:
                        print ("Deadline metric for Upstream Service: " + repr(service) + " is " + repr(d_metric))
                cand_services.append([service, d_metric])
            # sort vms and virtual_vms arrays according to metric
            vms = sorted(vms, key=lambda x: x[1], reverse=True) #larger to smaller
            cand_services = sorted(cand_services, key=lambda x: x[1]) #smaller to larger
            if self.debug:
                print ("VMs: " + repr(vms))
                print ("Cand. Services: " + repr(cand_services))
            # Small metric is better
            indx = 0
            for vm in vms:
                # vm's metric is worse than the cand. and they are not the same service
                if vm[1] < cand_services[indx][1]:
                    break
                else:
                    if vm[0] != cand_services[indx][0]:
                       cs.reassign_vm(vm[0], cand_services[indx][0], self.debug)
                       n_replacements -= 1
                
                if n_replacements == 0 or indx == len(cand_services):
                    break
                indx += 1
    # SDF  
    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log, node, flow_id, deadline, rtt_delay, status):
        """
        response : True, if this is a response from the cloudlet/cloud
        deadline : deadline for the request 
        flow_id : Id of the flow that the request/response is part of
        node : the current node at which the request/response arrived
        """
        service = content

        if time - self.last_replacement > self.replacement_interval:
            #self.print_stats()
            self.controller.replacement_interval_over(flow_id, self.replacement_interval, time)
            self.replace_services(self.n_replacements, time)
            self.last_replacement = time
            self.initialise_metrics()
        
        if receiver == node and status == REQUEST:
            self.controller.start_session(time, receiver, service, log, flow_id, deadline)
            source = self.view.content_source(service)
            path = self.view.shortest_path(node, source)
            next_node = path[1]
            delay = self.view.path_delay(node, next_node)
            rtt_delay += delay*2
            self.controller.add_event(time+delay, receiver, service, next_node, flow_id, deadline, rtt_delay, REQUEST)
            return

        if self.debug:
            print ("\nEvent\n time: " + repr(time) + " receiver  " + repr(receiver) + " service " + repr(service) + " node " + repr(node) + " flow_id " + repr(flow_id) + " deadline " + repr(deadline) + " status " + repr(status)) 

        compSpot = None
        if self.view.has_computationalSpot(node):
            compSpot = self.view.compSpot(node)
        else: # the node has no computational spots (0 services)
            if status is not RESPONSE:
                source = self.view.content_source(service)
                if node == source:
                    print ("Error: reached the source node: " + repr(node) + " this should not happen!")
                    return
                path = self.view.shortest_path(node, source)
                next_node = path[1]
                delay = self.view.link_delay(node, next_node)
                if self.debug:
                    print ("Pass upstream (no compSpot) to node: " + repr(next_node) + " " + repr(time+delay))
                self.controller.add_event(time+delay, receiver, service, next_node, flow_id, deadline, rtt_delay+2*delay, REQUEST)
                return
        # SDF  
        if status == RESPONSE: 
            # response is on its way back to the receiver
            if node == receiver:
                self.controller.end_session(True, time, flow_id) #TODO add flow_time
                return
            else:
                path = self.view.shortest_path(node, receiver)
                next_node = path[1]
                delay = self.view.link_delay(node, next_node)
                self.controller.add_event(time+delay, receiver, service, next_node, flow_id, deadline, rtt_delay, RESPONSE)

        elif status == TASK_COMPLETE:
            task = compSpot.schedule(time)
            #schedule the next queued task at this node
            if task is not None:
                self.controller.add_event(task.finishTime, task.receiver, task.service, node, task.flow_id, task.expiry, task.rtt_delay, TASK_COMPLETE)

            # forward the completed task
            path = self.view.shortest_path(node, receiver)
            next_node = path[1]
            delay = self.view.link_delay(node, next_node)
            self.controller.add_event(time+delay, receiver, service, next_node, flow_id, deadline, rtt_delay, RESPONSE)
            
        # SDF  
        elif status == REQUEST:
            # Processing a request
            source = self.view.content_source(service)
            path = self.view.shortest_path(node, source)
            next_node = path[1]
            delay = self.view.path_delay(node, next_node)
            deadline_metric = (deadline - time - rtt_delay - compSpot.services[service].service_time)/deadline
            if self.debug:
                print ("Deadline metric: " + repr(deadline_metric))
            if self.view.has_service(node, service):
                if self.debug:
                    print ("Calling admit_task")
                ret, reason = compSpot.admit_task(service, time, flow_id, deadline, receiver, rtt_delay, self.controller, self.debug)
                if self.debug:
                    print ("Done Calling admit_task")
                if ret is False:    
                    # Pass the Request upstream
                    rtt_delay += delay*2
                    self.controller.add_event(time+delay, receiver, service, next_node, flow_id, deadline, rtt_delay, REQUEST)
                    if deadline_metric > 0:
                        self.cand_deadline_metric[node][service] += deadline_metric
                    if self.debug:
                        print ("Pass upstream to node: " + repr(next_node))
                else:
                    if deadline_metric > 0:
                        self.deadline_metric[node][service] += deadline_metric
            else: #Not running the service
                compSpot.missed_requests[service] += 1
                if self.debug:
                    print ("Not running the service: Pass upstream to node: " + repr(next_node))
                rtt_delay += delay*2
                self.controller.add_event(time+delay, receiver, service, next_node, flow_id, deadline, rtt_delay, REQUEST)
                if deadline_metric > 0:
                    self.cand_deadline_metric[node][service] += deadline_metric
        else:
            print ("Error: unrecognised status value : " + repr(status))


# Auction
@register_strategy('DOUBLE_AUCTION_TRACE')
class DoubleAuctionTrace(Strategy):
    """A distributed approach for service-centric routing
    """
    def __init__(self, view, controller, replacement_interval=5.0, debug=False, **kwargs):
        super(DoubleAuctionTrace, self).__init__(view,controller)
        self.receivers = view.topology().receivers()
        self.compSpots = self.view.service_nodes()
        self.num_nodes = len(self.compSpots.keys())
        self.num_services = self.view.num_services()
        self.debug = debug
        self.replacement_interval = replacement_interval
        self.last_replacement = 0.0
        for node in self.compSpots.keys():
            cs = self.compSpots[node]
            self.controller.set_vm_prices(node, cs.vm_prices, 0)
            self.controller.set_node_util(node, cs.utilities, 0)
            for s in range(cs.service_population):
                cs.service_class_rate[s] = 0.0
            
    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log, node, flow_id, traffic_class, rtt_delay, status):
        if time - self.last_replacement > self.replacement_interval:
            if self.debug:
                print("Replacement interval is over at time: " + repr(time))
            #print("Evaluation interval over at time: " + repr(time))
            self.controller.replacement_interval_over(self.replacement_interval, time)
            self.last_replacement = time
            for n in self.compSpots.keys():
                if self.debug: 
                    print ("Computing prices @" + repr(n))
                cs = self.compSpots[n]
                for s in range(cs.service_population):
                    if self.debug:
                        print ("Printing count node:" + repr(n) + " for service: " + repr(s) + " is: " + repr(cs.service_class_count[s]))
                    cs.service_class_rate[s] = [(1.0*cs.service_class_count[s][c])/self.replacement_interval for c in range(cs.num_classes)]
                    cs.service_class_count[s] = [0 for c in range(cs.num_classes)]
                if self.debug:
                    print ("Printing service-class rates for node:" + repr(n) + " " + repr(cs.service_class_rate))
                cs.compute_prices()
                self.controller.set_vm_prices(n, cs.vm_prices, time)
                self.controller.set_node_util(n, cs.utilities, time)
        
        service = content
        cloud = self.view.content_source(service)

        if self.debug:
            print ("\nEvent\n time: " + repr(time) + " receiver  " + repr(receiver) + " service " + repr(service) + " node " + repr(node) + " flow_id " + repr(flow_id) + " traffic class " + repr(traffic_class) + " status " + repr(status)) 

        if receiver == node and status == REQUEST:
            self.controller.start_session(time, receiver, service, log, flow_id, traffic_class)
            path = self.view.shortest_path(node, cloud)
            next_node = path[1]
            delay = self.view.path_delay(node, next_node)
            self.controller.add_event(time+delay, receiver, service, next_node, flow_id, traffic_class, rtt_delay, REQUEST)
            return
        
        compSpot = None
        if self.view.has_computationalSpot(node):
            compSpot = self.view.compSpot(node)
        
        if status == RESPONSE: 
            # response is on its way back to the receiver
            if node == receiver:
                self.controller.end_session(True, time, flow_id) #TODO add flow_time
                return
            else:
                path = self.view.shortest_path(node, receiver)
                next_node = path[1]
                delay = self.view.link_delay(node, next_node)
                self.controller.add_event(time+delay, receiver, service, next_node, flow_id, traffic_class, rtt_delay, RESPONSE)

        elif status == TASK_COMPLETE:
            #schedule the next queued task (if this is not the cloud)
            if node != cloud:
                task = compSpot.schedule(time)
                if task is not None:
                    self.controller.add_event(task.finishTime, task.receiver, task.service, node, task.flow_id, task.traffic_class, task.rtt_delay, TASK_COMPLETE)
                    print ("Task service: " + repr(task.service) + " traffic class: " + repr(task.traffic_class))
                    self.controller.execute_service(task.finishTime, flow_id, task.service, False, task.traffic_class, node, compSpot.vm_prices[0]) 
            # forward the completed task
            path = self.view.shortest_path(node, receiver)
            next_node = path[1]
            delay = self.view.link_delay(node, next_node)
            self.controller.add_event(time+delay, receiver, service, next_node, flow_id, traffic_class, rtt_delay, RESPONSE)
            
        elif status == REQUEST:
            # Processing a request
            if node == cloud: # request reached the cloud
                service_time = self.view.get_service_time(service)
                self.controller.add_event(time+service_time, receiver, service, node, flow_id, traffic_class, rtt_delay, TASK_COMPLETE)
                self.controller.execute_service(time, flow_id, service, True, traffic_class, node, 0)
            else:    
                path = self.view.shortest_path(node, cloud)
                next_node = path[1]
                delay = self.view.path_delay(node, next_node)
                ret, reason = compSpot.admit_task_auction(service, time, flow_id, traffic_class, receiver, rtt_delay, self.controller, self.debug)
                if ret == False:
                    delay = self.view.path_delay(node, next_node)
                    rtt_delay += 2*delay
                    self.controller.add_event(time+delay, receiver, service, next_node, flow_id, traffic_class, rtt_delay, REQUEST)


@register_strategy('SELF_TUNING_TRACE')
class SelfTuningTrace(Strategy):
    """A distributed approach for service-centric routing
    """
    def __init__(self, view, controller, replacement_interval=5.0, debug=False, **kwargs):
        super(SelfTuningTrace, self).__init__(view,controller)
        self.receivers = view.topology().receivers()
        self.compSpots = self.view.service_nodes()
        self.num_nodes = len(self.compSpots.keys())
        self.num_services = self.view.num_services()
        self.num_classes = self.view.num_traffic_classes()
        print ("Number of services: " + repr(self.num_services))
        print ("Number of classes: " + repr(self.num_classes))
        self.debug = debug
        self.replacement_interval = replacement_interval
        self.last_replacement = 0.0
        for node in self.compSpots.keys():
            cs = self.compSpots[node]
            self.controller.set_vm_prices(node, cs.vm_prices, 0)
            self.controller.set_node_util(node, cs.utilities, 0)
            for s in range(cs.service_population):
                cs.service_class_rate[s] = 0.0

    # SELF_TUNING_TRACE
    def replace_services(self, debug = True):
        """
        This method does the following:
        1. Evaluate instantiated and stored services at each computational spot for the past time interval, ie, [t-interval, t]. 
        2. Decide which services to instantiate in the next time interval [t, t+interval].
        """
        print ("Replacing services...")
        for node, cs in self.compSpots.items():
            if debug:
                print ("Replacement @node: " + repr(node))
            if cs.is_cloud:
                continue
            cs.service_class_price = [[None for x in range(self.num_classes)] for y in range(self.num_services)]
            cs.numberOfInstances = [[0 for x in range(self.num_classes)] for y in range(self.num_services)]
            cs.vm_prices = []
            cs.service_class_rate = [[0.0 for x in range(self.num_classes)] for y in range(self.num_services)]
            service_rank = []
            for s in range(self.num_services):
                for c in range(self.num_classes):
                    if cs.service_class_count[s][c] > 0:
                        util = cs.utilities[s][c]
                        service_rank.append([s, c, util])
        
            service_rank = sorted(service_rank, key=lambda x: x[2], reverse=True) #larger to smaller util
            if debug:
                print("Sorted ranking for services: " + repr(service_rank))
            vm_cap = cs.n_services
            s_prev = None
            c_prev = None
            for s,c,u in service_rank:
                if vm_cap == 0:
                    break
                l = (cs.service_class_count[s][c]*1.0)/self.replacement_interval
                if debug:
                    print("Number of requests for service: " + repr(s) + " class: " + repr(c) + ": " + repr(cs.service_class_count[s][c]))
                cs.service_class_rate[s][c] = l
                l_eff = cs.estimateMaximumEffectiveL(l, 1.0/cs.services[s].service_time, l*cs.services[s].service_time)
                if debug:
                    print("Lambda = " + repr(l) + " Lambda_effective = " + repr(l_eff))
                num_vms = l_eff*(cs.services[s].service_time)
                num_vms = int(math.floor(num_vms))
                if num_vms > 0:
                    num_vms = min(num_vms, vm_cap)
                    cs.numberOfInstances[s][c] = num_vms
                    if debug:
                        print (repr(num_vms) + " VMs assigned to service: " + repr(s) + " class: " + repr(c))
                    vm_cap -= num_vms
                
                    if s_prev is not None and c_prev is not None:
                        cs.service_class_price[s_prev][c_prev] = u
                        cs.vm_prices.append(u)
                    s_prev = s 
                    c_prev = c

            if s_prev is not None and c_prev is not None:
                cs.service_class_price[s_prev][c_prev] = 0.0
            cs.vm_prices.append(0.0)

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log, node, flow_id, traffic_class, rtt_delay, status):
        if time - self.last_replacement > self.replacement_interval:
            if self.debug:
                print("Replacement interval is over at time: " + repr(time))
            #print("Evaluation interval over at time: " + repr(time))
            self.controller.replacement_interval_over(self.replacement_interval, time)
            self.last_replacement = time
            self.replace_services()
            for n,cs in self.compSpots.items():
                if cs.is_cloud:
                    continue
                if self.debug: 
                    print ("Computing prices @" + repr(n))
                for s in range(cs.service_population):
                    cs.service_class_count[s] = [0 for c in range(cs.num_classes)]
                self.controller.set_vm_prices(n, cs.vm_prices, time)
                self.controller.set_node_util(n, cs.utilities, time)
        
        service = content
        cloud = self.view.content_source(service)

        if self.debug:
            print ("\nEvent\n time: " + repr(time) + " receiver  " + repr(receiver) + " service " + repr(service) + " node " + repr(node) + " flow_id " + repr(flow_id) + " traffic class " + repr(traffic_class) + " status " + repr(status)) 

        if receiver == node and status == REQUEST:
            self.controller.start_session(time, receiver, service, log, flow_id, traffic_class)
            path = self.view.shortest_path(node, cloud)
            next_node = path[1]
            delay = self.view.path_delay(node, next_node)
            self.controller.add_event(time+delay, receiver, service, next_node, flow_id, traffic_class, rtt_delay, REQUEST)
            return
        
        compSpot = None
        if self.view.has_computationalSpot(node):
            compSpot = self.view.compSpot(node)
        
        if status == RESPONSE: 
            # response is on its way back to the receiver
            if node == receiver:
                self.controller.end_session(True, time, flow_id) #TODO add flow_time
                return
            else:
                path = self.view.shortest_path(node, receiver)
                next_node = path[1]
                delay = self.view.link_delay(node, next_node)
                self.controller.add_event(time+delay, receiver, service, next_node, flow_id, traffic_class, rtt_delay, RESPONSE)

        elif status == TASK_COMPLETE:
            #schedule the next queued task (if this is not the cloud)
            if node != cloud:
                task = compSpot.schedule(time)
                if task is not None:
                    self.controller.add_event(task.finishTime, task.receiver, task.service, node, task.flow_id, task.traffic_class, task.rtt_delay, TASK_COMPLETE)
                    print ("Task service: " + repr(task.service) + " traffic class: " + repr(task.traffic_class))
                    self.controller.execute_service(task.finishTime, flow_id, task.service, False, task.traffic_class, node, compSpot.vm_prices[0]) 
            # forward the completed task
            path = self.view.shortest_path(node, receiver)
            next_node = path[1]
            delay = self.view.link_delay(node, next_node)
            self.controller.add_event(time+delay, receiver, service, next_node, flow_id, traffic_class, rtt_delay, RESPONSE)
            
        elif status == REQUEST:
            # Processing a request
            if node == cloud: # request reached the cloud
                service_time = self.view.get_service_time(service)
                self.controller.add_event(time+service_time, receiver, service, node, flow_id, traffic_class, rtt_delay, TASK_COMPLETE)
                self.controller.execute_service(time, flow_id, service, True, traffic_class, node, 0)
            else:    
                path = self.view.shortest_path(node, cloud)
                next_node = path[1]
                delay = self.view.path_delay(node, next_node)
                ret, reason = compSpot.admit_self_tuning(service, time, flow_id, traffic_class, receiver, rtt_delay, self.controller, self.debug)
                if ret == False:
                    delay = self.view.path_delay(node, next_node)
                    rtt_delay += 2*delay
                    self.controller.add_event(time+delay, receiver, service, next_node, flow_id, traffic_class, rtt_delay, REQUEST)

@register_strategy('LFU_TRACE')
class LFUTrace(Strategy):
    """A distributed approach for service-centric routing
    """
    def __init__(self, view, controller, replacement_interval=5.0, debug=False, **kwargs):
        super(LFUTrace, self).__init__(view,controller)
        self.receivers = view.topology().receivers()
        self.compSpots = self.view.service_nodes()
        self.num_nodes = len(self.compSpots.keys())
        self.num_services = self.view.num_services()
        self.num_classes = self.view.num_traffic_classes()
        print ("Number of services: " + repr(self.num_services))
        print ("Number of classes: " + repr(self.num_classes))
        self.debug = debug
        self.replacement_interval = replacement_interval
        self.last_replacement = 0.0
        for node in self.compSpots.keys():
            cs = self.compSpots[node]
            self.controller.set_vm_prices(node, cs.vm_prices, 0)
            self.controller.set_node_util(node, cs.utilities, 0)
            for s in range(cs.service_population):
                cs.service_class_rate[s] = 0.0
    
    def replace_services(self, debug = True):
        """
        This method does the following:
        1. Evaluate instantiated and stored services at each computational spot for the past time interval, ie, [t-interval, t]. 
        2. Decide which services to instantiate in the next time interval [t, t+interval].
        """
        print ("Replacing services...")
        for node, cs in self.compSpots.items():
            if debug:
                print ("Replacement @node: " + repr(node))
            if cs.is_cloud:
                continue
            cs.service_class_price = [[None for x in range(self.num_classes)] for y in range(self.num_services)]
            cs.numberOfInstances = [[0 for x in range(self.num_classes)] for y in range(self.num_services)]
            cs.vm_prices = []
            cs.service_class_rate = [[0.0 for x in range(self.num_classes)] for y in range(self.num_services)]
            service_utility_rank = []
            service_rate_rank = []
            total_rate = 0.0
            for s in range(self.num_services):
                for c in range(self.num_classes):
                    if cs.service_class_count[s][c] > 0:
                        util = cs.utilities[s][c]
                        rate = (cs.service_class_count[s][c]*1.0)/self.replacement_interval
                        rate_eff = cs.estimateMaximumEffectiveL(rate, 1.0/cs.services[s].service_time, rate*cs.services[s].service_time)
                        if debug:
                            print("Lambda = " + repr(rate) + " Lambda_effective = " + repr(rate_eff))
                        total_rate += rate_eff
                        if debug:
                            print("Number of requests for service: " + repr(s) + " class: " + repr(c) + ": " + repr(cs.service_class_count[s][c]))
                        service_rate_rank.append([s, c, rate_eff])
                        service_utility_rank.append([s, c, util])
        
            service_rate_rank = sorted(service_rate_rank, key=lambda x: x[2], reverse=True) #larger to smaller rate
            service_utility_rank = sorted(service_utility_rank, key=lambda x: x[2], reverse=True) #larger to smaller utility
            if debug:
                print("Sorted ranking for services: " + repr(service_rate_rank))
                print("Sorted ranking for services: " + repr(service_utility_rank))
            remaining_cap = cs.n_services
            for s,c,r in service_rate_rank:
                if remaining_cap == 0:
                    break
                num_vms = (r/total_rate)*cs.n_services
                num_vms = int(math.ceil(num_vms))
                if num_vms > 0:
                    num_vms = min(num_vms, remaining_cap)
                    remaining_cap -= num_vms
                    cs.numberOfInstances[s][c] = num_vms
                    if debug:
                        print (repr(num_vms) + " VMs assigned to service: " + repr(s) + " class: " + repr(c))
                
            if debug:
                print("Remaining VM capacity unallocated is: " + repr(remaining_cap))
            
            s_prev = None
            c_prev = None
            for s,c,u in service_utility_rank:
                if s_prev is not None and c_prev is not None:
                    cs.service_class_price[s_prev][c_prev] = u
                    cs.vm_prices.append(u)
                s_prev = s 
                c_prev = c

            if s_prev is not None and c_prev is not None:
                cs.service_class_price[s_prev][c_prev] = 0.0
            cs.vm_prices.append(0.0)

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log, node, flow_id, traffic_class, rtt_delay, status):
        if time - self.last_replacement > self.replacement_interval:
            if self.debug:
                print("Replacement interval is over at time: " + repr(time))
            #print("Evaluation interval over at time: " + repr(time))
            self.controller.replacement_interval_over(self.replacement_interval, time)
            self.last_replacement = time
            self.replace_services()
            for n,cs in self.compSpots.items():
                if self.debug: 
                    print ("Computing prices @" + repr(n))
                for s in range(cs.service_population):
                    cs.service_class_count[s] = [0 for c in range(cs.num_classes)]
                self.controller.set_vm_prices(n, cs.vm_prices, time)
                self.controller.set_node_util(n, cs.utilities, time)
        
        service = content
        cloud = self.view.content_source(service)

        if self.debug:
            print ("\nEvent\n time: " + repr(time) + " receiver  " + repr(receiver) + " service " + repr(service) + " node " + repr(node) + " flow_id " + repr(flow_id) + " traffic class " + repr(traffic_class) + " status " + repr(status)) 

        if receiver == node and status == REQUEST:
            self.controller.start_session(time, receiver, service, log, flow_id, traffic_class)
            path = self.view.shortest_path(node, cloud)
            next_node = path[1]
            delay = self.view.path_delay(node, next_node)
            self.controller.add_event(time+delay, receiver, service, next_node, flow_id, traffic_class, rtt_delay, REQUEST)
            return
        
        compSpot = None
        if self.view.has_computationalSpot(node):
            compSpot = self.view.compSpot(node)
        
        if status == RESPONSE: 
            # response is on its way back to the receiver
            if node == receiver:
                self.controller.end_session(True, time, flow_id) #TODO add flow_time
                return
            else:
                path = self.view.shortest_path(node, receiver)
                next_node = path[1]
                delay = self.view.link_delay(node, next_node)
                self.controller.add_event(time+delay, receiver, service, next_node, flow_id, traffic_class, rtt_delay, RESPONSE)

        elif status == TASK_COMPLETE:
            #schedule the next queued task (if this is not the cloud)
            if node != cloud:
                task = compSpot.schedule(time)
                if task is not None:
                    self.controller.add_event(task.finishTime, task.receiver, task.service, node, task.flow_id, task.traffic_class, task.rtt_delay, TASK_COMPLETE)
                    print ("Task service: " + repr(task.service) + " traffic class: " + repr(task.traffic_class))
                    self.controller.execute_service(task.finishTime, flow_id, task.service, False, task.traffic_class, node, compSpot.vm_prices[0]) 
            # forward the completed task
            path = self.view.shortest_path(node, receiver)
            next_node = path[1]
            delay = self.view.link_delay(node, next_node)
            self.controller.add_event(time+delay, receiver, service, next_node, flow_id, traffic_class, rtt_delay, RESPONSE)
            
        elif status == REQUEST:
            # Processing a request
            if node == cloud: # request reached the cloud
                service_time = self.view.get_service_time(service)
                self.controller.add_event(time+service_time, receiver, service, node, flow_id, traffic_class, rtt_delay, TASK_COMPLETE)
                self.controller.execute_service(time, flow_id, service, True, traffic_class, node, 0)
            else:    
                path = self.view.shortest_path(node, cloud)
                next_node = path[1]
                delay = self.view.path_delay(node, next_node)
                ret, reason = compSpot.admit_self_tuning(service, time, flow_id, traffic_class, receiver, rtt_delay, self.controller, self.debug)
                if ret == False:
                    delay = self.view.path_delay(node, next_node)
                    rtt_delay += 2*delay
                    self.controller.add_event(time+delay, receiver, service, next_node, flow_id, traffic_class, rtt_delay, REQUEST)

@register_strategy('STATIC_TRACE')
class StaticTrace(Strategy):
    """Provision VM resources statically. The assignment of VMs to services is based on the number of requests for each service - 
       Assuming an oracle that knows all the future queries.
    """
    
    def __init__(self, view, controller, trace_file = '', n_measured_requests = 0, replacement_interval=5.0, debug=True, **kwargs):
        super(StaticTrace, self).__init__(view,controller)
        self.receivers = view.topology().receivers()
        self.compSpots = self.view.service_nodes()
        self.num_nodes = len(self.compSpots.keys())
        self.num_services = self.view.num_services()
        self.num_classes = self.view.num_traffic_classes()
        print ("Number of services: " + repr(self.num_services))
        print ("Number of classes: " + repr(self.num_classes))
        self.debug = debug
        self.replacement_interval = replacement_interval
        self.last_replacement = 0.0
        
        n_requests = 0
        service_counts = {}
        try:
            aFile = open(trace_file, 'r')
        except IOError:
            print ("Could not read the workload trace file:", trace_file)
            sys.exit()
        while True: 
            line = aFile.readline()
            if (not line) or (n_requests == n_measured_requests):
                break
            service = int(line)
            if service in service_counts.keys():
                service_counts[service] += 1
            else:
                service_counts[service] = 1
            n_requests += 1
        aFile.close()
        
        if debug:
            print("Service counts: " + repr(service_counts))
        service_counts_list = []
        for key, value in service_counts.iteritems():
            service_counts_list.append([key,value])

        if debug:
            print("Service counts list: " + repr(service_counts_list))
        service_counts_list_sorted = sorted(service_counts_list, key=lambda x: x[1], reverse=True)
        if debug:
            print("Service counts sorted: " + repr(service_counts_list_sorted))
            
        for node, cs in self.compSpots.items():
            service_utility_rank = []
            cs.numberOfInstances = [0 for x in range(cs.service_population)]
            remaining_vms = cs.n_services
            for s, count in service_counts_list_sorted:
                num_vms = math.ceil((1.0*count*cs.n_services)/n_requests)
                num_vms = min(num_vms, remaining_vms)
                remaining_vms -= num_vms
                cs.numberOfInstances[s] = num_vms
                if num_vms > 0:
                    for c in range(cs.num_classes):
                        service_utility_rank.append([s, c, cs.utilities[s][c]])
                if remaining_vms == 0:
                    break
            service_utility_rank = sorted(service_utility_rank, key=lambda x: x[2], reverse=True)
            s_prev = None
            c_prev = None
            cs.vm_prices = []
            for s,c,u in service_utility_rank:
                if s_prev is not None:
                    cs.service_class_price[s_prev][c_prev] = u
                    if debug:
                        print("Price of service: " + repr(s_prev) + " class " + repr(c_prev) + ": " + repr(u))
                    cs.vm_prices.append(u)
                s_prev = s
                c_prev = c
        
    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log, node, flow_id, traffic_class, rtt_delay, status):
        if time - self.last_replacement > self.replacement_interval:
            if self.debug:
                print("Replacement interval is over at time: " + repr(time))
            #print("Evaluation interval over at time: " + repr(time))
            self.controller.replacement_interval_over(self.replacement_interval, time)
            self.last_replacement = time
            for n,cs in self.compSpots.items():
                if self.debug: 
                    print ("Computing prices @" + repr(n))
                for s in range(cs.service_population):
                    cs.service_class_count[s] = [0 for c in range(cs.num_classes)]
                self.controller.set_vm_prices(n, cs.vm_prices, time)
                self.controller.set_node_util(n, cs.utilities, time)
        
        service = content
        cloud = self.view.content_source(service)

        #if self.debug:
        #    print ("\nEvent\n time: " + repr(time) + " receiver  " + repr(receiver) + " service " + repr(service) + " node " + repr(node) + " flow_id " + repr(flow_id) + " traffic class " + repr(traffic_class) + " status " + repr(status)) 

        if receiver == node and status == REQUEST:
            self.controller.start_session(time, receiver, service, log, flow_id, traffic_class)
            path = self.view.shortest_path(node, cloud)
            next_node = path[1]
            delay = self.view.path_delay(node, next_node)
            self.controller.add_event(time+delay, receiver, service, next_node, flow_id, traffic_class, rtt_delay, REQUEST)
            return
        
        compSpot = None
        if self.view.has_computationalSpot(node):
            compSpot = self.view.compSpot(node)
        
        if status == RESPONSE: 
            # response is on its way back to the receiver
            if node == receiver:
                self.controller.end_session(True, time, flow_id) #TODO add flow_time
                return
            else:
                path = self.view.shortest_path(node, receiver)
                next_node = path[1]
                delay = self.view.link_delay(node, next_node)
                self.controller.add_event(time+delay, receiver, service, next_node, flow_id, traffic_class, rtt_delay, RESPONSE)

        elif status == TASK_COMPLETE:
            #schedule the next queued task (if this is not the cloud)
            if node != cloud:
                task = compSpot.schedule(time)
                if task is not None:
                    self.controller.add_event(task.finishTime, task.receiver, task.service, node, task.flow_id, task.traffic_class, task.rtt_delay, TASK_COMPLETE)
                    #print ("Task service: " + repr(task.service) + " traffic class: " + repr(task.traffic_class))
                    self.controller.execute_service(task.finishTime, flow_id, task.service, False, task.traffic_class, node, compSpot.vm_prices[0]) 
            # forward the completed task
            path = self.view.shortest_path(node, receiver)
            next_node = path[1]
            delay = self.view.link_delay(node, next_node)
            self.controller.add_event(time+delay, receiver, service, next_node, flow_id, traffic_class, rtt_delay, RESPONSE)
            
        elif status == REQUEST:
            # Processing a request
            if node == cloud: # request reached the cloud
                service_time = self.view.get_service_time(service)
                self.controller.add_event(time+service_time, receiver, service, node, flow_id, traffic_class, rtt_delay, TASK_COMPLETE)
                self.controller.execute_service(time, flow_id, service, True, traffic_class, node, 0)
            else:    
                path = self.view.shortest_path(node, cloud)
                next_node = path[1]
                delay = self.view.path_delay(node, next_node)
                ret, reason = compSpot.admit_static_provisioning(service, time, flow_id, traffic_class, receiver, rtt_delay, self.controller, self.debug)
                if ret == False:
                    delay = self.view.path_delay(node, next_node)
                    rtt_delay += 2*delay
                    self.controller.add_event(time+delay, receiver, service, next_node, flow_id, traffic_class, rtt_delay, REQUEST)

