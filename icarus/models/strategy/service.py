# -*- coding: utf-8 -*-
"""Implementations of all service-based strategies"""
from __future__ import division
from __future__ import print_function

import networkx as nx
#import random
import math

from icarus.registry import register_strategy
from icarus.util import inheritdoc, path_links
from .base import Strategy

__all__ = [
       'DoubleAuction',
       'Fifo',
       'DoubleAuctionTrace',
       'SelfTuningTrace',
       'LFUTrace',
       'StaticTrace',
       'Static' #is this used?
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
        self.topology = view.topology()
        for node in self.compSpots.keys():
            cs = self.compSpots[node]
            self.controller.set_vm_prices(node, cs.vm_prices)
            print ("Price @node: " + str(node) + " is: " + str(cs.vm_prices))
            self.controller.set_node_util(node, cs.utilities)
            self.controller.set_node_qos(node, cs.qos)
            self.controller.set_node_traffic_rates(cs.node, 0.0, cs.rate_times[0.0], cs.eff_rate_times[0.0])

    def map_traffic_class(self, curr_node, upstream_node, traffic_class):
        """
        This method retrieves the traffic class of the upstream node, given the 
        current node's traffic class
        """
        #if self.topology.graph['parent'][curr_node] != upstream_node: #sanity check
        #    raise ValueError('Parent node does not match upstream')
        if curr_node in self.receivers:
            return self.topology.node[curr_node]['parent_class'][0]
        else:
            return self.topology.node[curr_node]['parent_class'][traffic_class]

        #return self.topology.node[curr_node]['parent_class'][traffic_class]
            
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
            # map traffic class in the second node
            if next_node != cloud:
                traffic_class = self.map_traffic_class(node, next_node, traffic_class)
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
                    # map traffic class in the second node
                    if next_node != cloud:
                        traffic_class = self.map_traffic_class(node, next_node, traffic_class)
                    self.controller.add_event(time+delay, receiver, service, next_node, flow_id, traffic_class, rtt_delay, REQUEST)

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
        self.topology = view.topology()
        self.last_replacement = 0.0
        for node in self.compSpots.keys():
            cs = self.compSpots[node]
            self.controller.set_vm_prices(node, cs.vm_prices, 0)
            self.controller.set_node_util(node, cs.utilities, 0)
            self.controller.set_node_qos(node, cs.qos, 0)
            self.controller.set_node_traffic_rates(cs.node, 0.0, cs.rate_times[0.0], cs.eff_rate_times[0.0])
            for s in range(cs.service_population):
                cs.service_class_rate[s] = [0.0 for c in range(cs.num_classes)]
            
    def map_traffic_class(self, curr_node, upstream_node, traffic_class):
        """
        This method retrieves the traffic class of the upstream node, given the 
        current node's traffic class
        """
        #if self.topology.graph['parent'][curr_node] != upstream_node: #sanity check
        #    raise ValueError('Parent node does not match upstream')
        if curr_node in self.receivers:
            return self.topology.node[curr_node]['parent_class'][0]
        else:
            return self.topology.node[curr_node]['parent_class'][traffic_class]

        
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
                if self.topology.node[n]['n_classes'] == 0:
                    continue
                cs = self.compSpots[n]
                for s in range(cs.service_population):
                    if self.debug:
                        print ("Printing count node:" + repr(n) + " for service: " + repr(s) + " is: " + repr(cs.service_class_count[s]))
                    cs.service_class_rate[s] = [(1.0*cs.service_class_count[s][c])/self.replacement_interval for c in range(cs.num_classes)]
                    cs.service_class_count[s] = [0 for c in range(cs.num_classes)]
                if self.debug:
                    print ("Printing service-class rates for node:" + repr(n) + " " + repr(cs.service_class_rate))
                cs.compute_prices(time)
                self.controller.set_vm_prices(n, cs.vm_prices, time)
                self.controller.set_node_util(n, cs.utilities, time)
                self.controller.set_node_qos(n, cs.qos, time)
                if time in cs.rate_times.keys():
                    self.controller.set_node_traffic_rates(cs.node, time, cs.rate_times[time], cs.eff_rate_times[time])
        
        service = content
        cloud = self.view.content_source(service)

        if self.debug:
            print ("\nEvent\n time: " + repr(time) + " receiver  " + repr(receiver) + " service " + repr(service) + " node " + repr(node) + " flow_id " + repr(flow_id) + " traffic class " + repr(traffic_class) + " status " + repr(status)) 

        if receiver == node and status == REQUEST:
            self.controller.start_session(time, receiver, service, log, flow_id, traffic_class)
            path = self.view.shortest_path(node, cloud)
            next_node = path[1]
            # map traffic class in the second node
            if next_node != cloud:
                traffic_class = self.map_traffic_class(node, next_node, traffic_class)
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
                    self.controller.execute_service(task.finishTime, flow_id, task.service, False, task.traffic_class, node, compSpot.vm_prices[0]) # never gets executed when there is no queuing (schedule call above does not return anything)
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
                    # map traffic class in the second node
                    if next_node != cloud:
                        traffic_class = self.map_traffic_class(node, next_node, traffic_class)
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
        self.topology = view.topology()
        for node in self.compSpots.keys():
            cs = self.compSpots[node]
            self.controller.set_vm_prices(node, cs.vm_prices, 0)
            self.controller.set_node_util(node, cs.utilities, 0)
            self.controller.set_node_qos(node, cs.qos, 0)
            for s in range(cs.service_population):
                cs.service_class_rate[s] = [0.0 for c in range(cs.num_classes)] 

    # SELF_TUNING_TRACE
    def map_traffic_class(self, curr_node, upstream_node, traffic_class):
        """
        This method retrieves the traffic class of the upstream node, given the 
        current node's traffic class
        """
        #if self.topology.graph['parent'][curr_node] != upstream_node: #sanity check
        #    raise ValueError('Parent node does not match upstream')
        if curr_node in self.receivers:
            return self.topology.node[curr_node]['parent_class'][0]
        else:
            return self.topology.node[curr_node]['parent_class'][traffic_class]

        #return self.topology.node[curr_node]['parent_class'][traffic_class]

    # SELF_TUNING_TRACE
    def replace_services(self, time, debug = False):
        """
        This method does the following:
        1. Evaluate instantiated and stored services at each computational spot for the past time interval, ie, [t-interval, t]. 
        2. Decide which services to instantiate in the next time interval [t, t+interval].
        """
        if debug:
            print ("Replacing services...")
        for node, cs in self.compSpots.items():
            if debug:
                print ("Replacement @node: " + repr(node))
            if cs.is_cloud:
                continue
            cs.service_class_price = [[None for x in range(cs.num_classes)] for y in range(self.num_services)]
            #cs.numberOfInstances = [[0 for x in range(cs.num_classes)] for y in range(self.num_services)]
            cs.numberOfInstances = [0 for x in range(cs.service_population)]
            cs.vm_prices = []
            cs.service_class_rate = [[0.0 for x in range(cs.num_classes)] for y in range(self.num_services)]
            service_rank = []
            service_util_rank = []
            for s in range(self.num_services):
                for c in range(cs.num_classes):
                    util = cs.utilities[s][c]
                    if cs.service_class_count[s][c] > 0:
                        service_rank.append([s, c, util])
                    service_util_rank.append([s, c, util])
        
            service_rank = sorted(service_rank, key=lambda x: x[2], reverse=True) #larger to smaller util
            service_util_rank = sorted(service_util_rank, key=lambda x: x[2], reverse=True) #larger to smaller utility
            if debug:
                print("Sorted ranking for services: " + repr(service_rank))
            vm_cap = cs.n_services
            s_prev = None
            c_prev = None
            # Rank service,class pairs based on their QoS gain (utility) and then assign VM instances
            # to service,class pairs based on their observed rate in the previous time interval. 
            total_count = 0
            for s,c,u in service_rank:
                if vm_cap == 0:
                    break
                l = (cs.service_class_count[s][c]*1.0)/self.replacement_interval
                total_count += cs.service_class_count[s][c]
                if debug:
                    print("Number of requests for service: " + repr(s) + " class: " + repr(c) + ": " + repr(cs.service_class_count[s][c]))
                cs.service_class_rate[s][c] = l
                l_eff = cs.estimateMaximumEffectiveL(l, 1.0/cs.services[s].service_time, vm_cap)
                if debug:
                    print("Lambda = " + repr(l) + " Lambda_effective = " + repr(l_eff))
                num_vms = l_eff*(cs.services[s].service_time)
                #num_vms = l*(cs.services[s].service_time)
                num_vms = int(math.ceil(num_vms))
                if num_vms > 0:
                    num_vms = min(num_vms, vm_cap)
                    cs.numberOfInstances[s] += num_vms
                    if debug:
                        print (repr(num_vms) + " VMs assigned to service: " + repr(s) + " class: " + repr(c))
                    vm_cap -= num_vms
                
            # Assign prices
            s_prev = None
            c_prev = None
            for s,c,u in service_util_rank:
                if s_prev is not None and c_prev is not None:
                    cs.service_class_price[s_prev][c_prev] = u
                    cs.vm_prices.append(u)
                s_prev = s 
                c_prev = c

            if s_prev is not None and c_prev is not None:
                cs.service_class_price[s_prev][c_prev] = 1.0
            cs.vm_prices.append(1.0)

            additional_vms = 0
            if vm_cap != 0:
                while vm_cap != 0:
                    if total_count == 0:
                        for s,c,u in service_util_rank:
                            cs.numberOfInstances[s] += 1
                            vm_cap -= 1
                            additional_vms += 1
                            if vm_cap == 0:
                                break

                    else:
                       for s,c,u in service_rank:
                            #if cs.numberOfInstances[s][c] > 0:
                            if cs.numberOfInstances[s] > 0:
                                #cs.numberOfInstances[s][c] += 1
                                cs.numberOfInstances[s] += 1
                                vm_cap -= 1
                                additional_vms += 1
                                if vm_cap == 0:
                                    break
                    #print ("Added: " + str(additional_vms) + " VMs in Self-Tuning Strategy at time: " + str(time))
                    

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log, node, flow_id, traffic_class, rtt_delay, status):
        if time - self.last_replacement > self.replacement_interval:
            if self.debug:
                print("Replacement interval is over at time: " + repr(time))
            #print("Evaluation interval over at time: " + repr(time))
            self.controller.replacement_interval_over(self.replacement_interval, time)
            self.last_replacement = time
            self.replace_services(time)
            for n,cs in self.compSpots.items():
                if cs.is_cloud:
                    continue
                if self.debug: 
                    print ("Computing prices @" + repr(n))
                for s in range(cs.service_population):
                    cs.service_class_count[s] = [0 for c in range(cs.num_classes)]
                self.controller.set_vm_prices(n, cs.vm_prices, time)
                self.controller.set_node_qos(n, cs.qos, time)
                self.controller.set_node_util(n, cs.utilities, time)
        
        service = content
        cloud = self.view.content_source(service)

        if self.debug:
            print ("\nEvent\n time: " + repr(time) + " receiver  " + repr(receiver) + " service " + repr(service) + " node " + repr(node) + " flow_id " + repr(flow_id) + " traffic class " + repr(traffic_class) + " status " + repr(status)) 

        if receiver == node and status == REQUEST:
            self.controller.start_session(time, receiver, service, log, flow_id, traffic_class)
            path = self.view.shortest_path(node, cloud)
            next_node = path[1]
            # map traffic class in the second node
            if next_node != cloud:
                traffic_class = self.map_traffic_class(node, next_node, traffic_class)
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
                    #self.controller.execute_service(task.finishTime, flow_id, task.service, False, task.traffic_class, node, compSpot.vm_prices[0]) 
                    self.controller.execute_service(task.finishTime, flow_id, task.service, False, task.traffic_class, node, cs.service_class_price[task.service][task.traffic_class])#this never gets executed when there is no queuing
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
                    # map traffic class in the second node
                    if next_node != cloud:
                        traffic_class = self.map_traffic_class(node, next_node, traffic_class)
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
        self.topology = view.topology()
        print ("Number of services: " + repr(self.num_services))
        print ("Number of classes at each leaf node: " + repr(self.num_classes))
        self.debug = debug
        self.replacement_interval = replacement_interval
        self.last_replacement = 0.0
        for node in self.compSpots.keys():
            cs = self.compSpots[node]
            self.controller.set_vm_prices(node, cs.vm_prices, 0)
            self.controller.set_node_util(node, cs.utilities, 0)
            self.controller.set_node_qos(node, cs.qos, 0)
            for s in range(cs.service_population):
                cs.service_class_rate[s] = [0.0 for c in range(cs.num_classes)] 
    
    def map_traffic_class(self, curr_node, upstream_node, traffic_class):
        """
        This method retrieves the traffic class of the upstream node, given the 
        current node's traffic class
        """
        #if self.topology.graph['parent'][curr_node] != upstream_node: #sanity check
        #    raise ValueError('Parent node does not match upstream')

        if curr_node in self.receivers:
            return self.topology.node[curr_node]['parent_class'][0]
        else:
            return self.topology.node[curr_node]['parent_class'][traffic_class]

        #return self.topology.node[curr_node]['parent_class'][traffic_class]
    
    def replace_services(self, time, debug = False):
        """
        This method does the following:
        1. Evaluate instantiated and stored services at each computational spot for the past time interval, ie, [t-interval, t]. 
        2. Decide which services to instantiate in the next time interval [t, t+interval].
        """
        if debug:
            print ("LFU replacing services @ time: " + str(time))
        for node, cs in self.compSpots.items():
            if debug:
                print ("Replacement @node: " + repr(node))
            if cs.is_cloud:
                continue
            #cs.numberOfInstances = [[0 for x in range(cs.num_classes)] for y in range(self.num_services)]
            cs.vm_prices = []
            cs.service_class_rate = [[0.0 for x in range(cs.num_classes)] for y in range(self.num_services)]
            service_utility_rank = []
            service_all_rank = []
            service_rate_rank = []
            total_count = 0
            for s in range(self.num_services):
                for c in range(cs.num_classes):
                    util = cs.utilities[s][c]
                    if cs.service_class_count[s][c] > 0:
                        rate = (cs.service_class_count[s][c]*1.0)/self.replacement_interval
                        service_rate_rank.append([s, c, rate])
                        service_utility_rank.append([s, c, util])
                        total_count += 1
                    service_all_rank.append([s, c, util])

            cs.numberOfInstances = [0 for x in range(cs.service_population)]
            cs.service_class_price = [[None for x in range(cs.num_classes)] for y in range(self.num_services)]
            # Sort (service, class) pairs by rate and utility
            service_rate_rank = sorted(service_rate_rank, key=lambda x: x[2], reverse=True) #larger to smaller rate
            service_utility_rank = sorted(service_utility_rank, key=lambda x: x[2], reverse=True) #larger to smaller utility
            service_all_rank = sorted(service_all_rank, key=lambda x: x[2], reverse=True) #larger to smaller utility
            #if debug:
            #    print("Sorted ranking for services: " + repr(service_rate_rank))
            #    print("Sorted ranking for services: " + repr(service_utility_rank))

            remaining_cap = cs.n_services
            for s,c,r in service_rate_rank:
                if remaining_cap == 0:
                    break
                rate_eff = cs.estimateMaximumEffectiveL(r, 1.0/cs.services[s].service_time, remaining_cap)
                num_of_vms = math.ceil(rate_eff*cs.services[s].service_time)
                num_of_vms = min(int(num_of_vms), remaining_cap)
                remaining_cap -= num_of_vms
                #cs.numberOfInstances[s][c] = num_of_vms
                cs.numberOfInstances[s] += num_of_vms
                if debug:
                    print (repr(num_of_vms) + " VMs assigned to service: " + repr(s) + " class: " + repr(c))

            if debug:
                print("Remaining VM capacity unallocated is: " + repr(remaining_cap))

            # Assign prices
            s_prev = None
            c_prev = None
            for s,c,u in service_all_rank:
                if s_prev is not None and c_prev is not None:
                    cs.service_class_price[s_prev][c_prev] = u
                    cs.vm_prices.append(u)
                s_prev = s 
                c_prev = c

            if s_prev is not None and c_prev is not None:
                cs.service_class_price[s_prev][c_prev] = 1.0
            cs.vm_prices.append(1.0)

            additional_vms = 0
            if total_count == 0:
                    while remaining_cap != 0:
                        for s,c,u in service_all_rank:
                            cs.numberOfInstances[s] += 1
                            additional_vms += 1
                            remaining_cap -= 1
                            if remaining_cap == 0:
                                break
                    #print ("Added: " + str(additional_vms) + " VMs in LFU Strategy at time: " + str(time))
            else:
                if remaining_cap != 0:
                    while remaining_cap != 0:
                        for s,c,u in service_rate_rank:
                            #if cs.numberOfInstances[s][c] > 0:
                            #if cs.numberOfInstances[s] > 0:
                                #cs.numberOfInstances[s][c] += 1
                            cs.numberOfInstances[s] += 1
                            additional_vms += 1
                            remaining_cap -= 1
                            if remaining_cap == 0:
                                break
                    #print ("Added: " + str(additional_vms) + " VMs in LFU Strategy at time: " + str(time))


    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log, node, flow_id, traffic_class, rtt_delay, status):
        if time - self.last_replacement > self.replacement_interval:
            if self.debug:
                print("Replacement interval is over at time: " + repr(time))
            #print("Evaluation interval over at time: " + repr(time))
            self.controller.replacement_interval_over(self.replacement_interval, time)
            self.last_replacement = time
            self.replace_services(time)
            for n,cs in self.compSpots.items():
                if self.debug: 
                    print ("Computing prices @" + repr(n))
                for s in range(cs.service_population):
                    cs.service_class_count[s] = [0 for c in range(cs.num_classes)]
                self.controller.set_vm_prices(n, cs.vm_prices, time)
                self.controller.set_node_util(n, cs.utilities, time)
                self.controller.set_node_qos(n, cs.qos, time)
        
        service = content
        cloud = self.view.content_source(service)

        if self.debug:
            print ("\nEvent\n time: " + repr(time) + " receiver  " + repr(receiver) + " service " + repr(service) + " node " + repr(node) + " flow_id " + repr(flow_id) + " traffic class " + repr(traffic_class) + " status " + repr(status)) 

        if receiver == node and status == REQUEST:
            self.controller.start_session(time, receiver, service, log, flow_id, traffic_class)
            path = self.view.shortest_path(node, cloud)
            next_node = path[1]
            # map traffic class in the second node
            if next_node != cloud:
                traffic_class = self.map_traffic_class(node, next_node, traffic_class)
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
                    self.controller.execute_service(task.finishTime, flow_id, task.service, False, task.traffic_class, node, cs.service_class_price[task.service][task.traffic_class]) #this never gets executed when there is no queuing
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
                    # map traffic class in the second node
                    if next_node != cloud:
                        traffic_class = self.map_traffic_class(node, next_node, traffic_class)
                    self.controller.add_event(time+delay, receiver, service, next_node, flow_id, traffic_class, rtt_delay, REQUEST)

@register_strategy('STATIC_TRACE')
class StaticTrace(Strategy):
    """Provision VM resources statically. The assignment of VMs to services is based on the number of requests for each service - 
       Assuming an oracle that knows all the future queries.
    """
    def __init__(self, view, controller, trace_file = '', n_measured_requests = 0, replacement_interval=5.0, debug=False, **kwargs):
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
        self.topology = view.topology()
        
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
            self.controller.set_vm_prices(node, cs.vm_prices)
            self.controller.set_node_util(node, cs.utilities)
            self.controller.set_node_qos(node, cs.qos)
            self.controller.set_node_traffic_rates(cs.node, 0.0, cs.rate_times[0.0], cs.eff_rate_times[0.0])
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
        
            while remaining_vms != 0:
                for s,c,u in service_counts_list_sorted:
                    if cs.numberOfInstances[s] > 0:
                        cs.numberOfInstances[s] += 1
                        remaining_vms -= 1
    
    def map_traffic_class(self, curr_node, upstream_node, traffic_class):
        """
        This method retrieves the traffic class of the upstream node, given the 
        current node's traffic class
        """
        #if self.topology.graph['parent'][curr_node] != upstream_node: #sanity check
        #    raise ValueError('Parent node does not match upstream')
        if curr_node in self.receivers:
            return self.topology.node[curr_node]['parent_class'][0]
        else:
            return self.topology.node[curr_node]['parent_class'][traffic_class]

        #return self.topology.node[curr_node]['parent_class'][traffic_class]
        
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
                self.controller.set_node_qos(n, cs.qos, time)
        
        service = content
        cloud = self.view.content_source(service)

        #if self.debug:
        #    print ("\nEvent\n time: " + repr(time) + " receiver  " + repr(receiver) + " service " + repr(service) + " node " + repr(node) + " flow_id " + repr(flow_id) + " traffic class " + repr(traffic_class) + " status " + repr(status)) 

        if receiver == node and status == REQUEST:
            self.controller.start_session(time, receiver, service, log, flow_id, traffic_class)
            path = self.view.shortest_path(node, cloud)
            next_node = path[1]
            delay = self.view.path_delay(node, next_node)
            # map traffic class in the second node
            if next_node != cloud:
                traffic_class = self.map_traffic_class(node, next_node, traffic_class)
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
                    #self.controller.execute_service(task.finishTime, flow_id, task.service, False, task.traffic_class, node, compSpot.vm_prices[0]) 
                    self.controller.execute_service(task.finishTime, flow_id, task.service, False, task.traffic_class, node, cs.service_class_price[task.service][task.traffic_class]) 
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
                    # map traffic class in the second node
                    if next_node != cloud:
                        traffic_class = self.map_traffic_class(node, next_node, traffic_class)
                    self.controller.add_event(time+delay, receiver, service, next_node, flow_id, traffic_class, rtt_delay, REQUEST)

@register_strategy('FIFO')
class Fifo(Strategy):
    """A distributed approach for service-centric routing
    """
    def __init__(self, view, controller, replacement_interval=5.0, debug=False, **kwargs):
        super(Fifo, self).__init__(view,controller)
        self.receivers = view.topology().receivers()
        self.compSpots = self.view.service_nodes()
        self.num_nodes = len(self.compSpots.keys())
        self.num_services = self.view.num_services()
        self.debug = debug
        self.replacement_interval = replacement_interval
        self.last_replacement = 0.0
        self.topology = view.topology()
        for node in self.compSpots.keys():
            cs = self.compSpots[node]
            self.controller.set_vm_prices(node, cs.vm_prices)
            self.controller.set_node_util(node, cs.utilities)
            self.controller.set_node_qos(node, cs.qos)
    
    def map_traffic_class(self, curr_node, upstream_node, traffic_class):
        """
        This method retrieves the traffic class of the upstream node, given the 
        current node's traffic class
        """
        #if self.topology.graph['parent'][curr_node] != upstream_node: #sanity check
        #    raise ValueError('Parent node does not match upstream')
        if curr_node in self.receivers:
            return self.topology.node[curr_node]['parent_class'][0]
        else:
            return self.topology.node[curr_node]['parent_class'][traffic_class]

        #return self.topology.node[curr_node]['parent_class'][traffic_class]
    
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
            # map traffic class in the second node
            if next_node != cloud:
                traffic_class = self.map_traffic_class(node, next_node, traffic_class)
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
                ret, reason = compSpot.admit_task_FIFO(service, time, flow_id, traffic_class, receiver, rtt_delay, self.controller, self.debug)
                if ret == False:
                    delay = self.view.path_delay(node, next_node)
                    rtt_delay += 2*delay
                    # map traffic class in the second node
                    if next_node != cloud:
                        traffic_class = self.map_traffic_class(node, next_node, traffic_class)
                    self.controller.add_event(time+delay, receiver, service, next_node, flow_id, traffic_class, rtt_delay, REQUEST)

@register_strategy('STATIC')
class Static(Strategy):
    """Provision VM resources statically. The assignment of VMs to services is based on the number of requests for each service - 
       Assuming an oracle that knows all the future queries.
    """
    
    def __init__(self, view, controller, trace_file = '', n_measured_requests = 0, replacement_interval=5.0, debug=False, **kwargs):
        super(Static, self).__init__(view,controller)
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
        self.topology = view.topology()
        
        for node, cs in self.compSpots.items():
            service_utility_rank = []
            cs.numberOfInstances = [0 for x in range(cs.service_population)]
            self.controller.set_vm_prices(node, cs.vm_prices)
            self.controller.set_node_util(node, cs.utilities)
            self.controller.set_node_qos(node, cs.qos)
            self.controller.set_node_traffic_rates(cs.node, 0.0, cs.rate_times[0.0], cs.eff_rate_times[0.0])
            remaining_vms = cs.n_services
            for s in range(self.num_services):
                num_vms = math.ceil((cs.n_services)/self.num_services)
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
    
    def map_traffic_class(self, curr_node, upstream_node, traffic_class):
        """
        This method retrieves the traffic class of the upstream node, given the 
        current node's traffic class
        """
        #if self.topology.graph['parent'][curr_node] != upstream_node: #sanity check
        #    raise ValueError('Parent node does not match upstream')
        #print ("Curr node: " + repr(curr_node))
        #print ("traffic_class: " + repr(traffic_class))
        
        if curr_node in self.receivers:
            return self.topology.node[curr_node]['parent_class'][0]
        else:
            return self.topology.node[curr_node]['parent_class'][traffic_class]
        
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
                self.controller.set_node_qos(n, cs.qos, time)
        
        service = content
        cloud = self.view.content_source(service)

        if receiver == node and status == REQUEST:
            self.controller.start_session(time, receiver, service, log, flow_id, traffic_class)
            path = self.view.shortest_path(node, cloud)
            next_node = path[1]
            delay = self.view.path_delay(node, next_node)
            # map traffic class in the second node
            if next_node != cloud:
                traffic_class = self.map_traffic_class(node, next_node, traffic_class)
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
                    #self.controller.execute_service(task.finishTime, flow_id, task.service, False, task.traffic_class, node, compSpot.vm_prices[0]) 
                    self.controller.execute_service(task.finishTime, flow_id, task.service, False, task.traffic_class, node, cs.service_class_price[task.service][task.traffic_class]) 
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
                    # map traffic class in the second node
                    if next_node != cloud:
                        traffic_class = self.map_traffic_class(node, next_node, traffic_class)
                    self.controller.add_event(time+delay, receiver, service, next_node, flow_id, traffic_class, rtt_delay, REQUEST)
