# -*- coding: utf-8 -*-
"""Performance metrics loggers

This module contains all data collectors that record events while simulations
are being executed and compute performance metrics.

Currently implemented data collectors allow users to measure cache hit ratio,
latency, path stretch and link load.

To create a new data collector, it is sufficient to create a new class
inheriting from the `DataCollector` class and override all required methods.
"""
from __future__ import division
import collections

from icarus.registry import register_data_collector
from icarus.tools import cdf
from icarus.util import Tree, inheritdoc


__all__ = [
    'DataCollector',
    'CollectorProxy',
    'CacheHitRatioCollector',
    'LinkLoadCollector',
    'LatencyCollector',
    'PathStretchCollector',
    'DummyCollector'
           ]


class DataCollector(object):
    """Object collecting notifications about simulation events and measuring
    relevant metrics.
    """

    def __init__(self, view, **params):
        """Constructor

        Parameters
        ----------
        view : NetworkView
            An instance of the network view
        params : keyworded parameters
            Collector parameters
        """
        self.view = view

    def start_session(self, timestamp, receiver, content, flow_id=0, traffic_class=0):
        """Notifies the collector that a new network session started.

        A session refers to the retrieval of a content from a receiver, from
        the issuing of a content request to the delivery of the content.

        Parameters
        ----------
        timestamp : int
            The timestamp of the event
        receiver : any hashable type
            The receiver node requesting a content
        content : any hashable type
            The content identifier requested by the receiver
        """
        pass

    def cache_hit(self, node):
        """Reports that the requested content has been served by the cache at
        node *node*.

        Parameters
        ----------
        node : any hashable type
            The node whose cache served the content
        """
        pass

    def cache_miss(self, node):
        """Reports that the cache at node *node* has been looked up for
        requested content but there was a cache miss.

        Parameters
        ----------
        node : any hashable type
            The node whose cache served the content
        """
        pass

    def server_hit(self, node):
        """Reports that the requested content has been served by the server at
        node *node*.

        Parameters
        ----------
        node : any hashable type
            The server node which served the content
        """
        pass
    
    def replacement_interval_over(self, replacement_interval, timestamp):
        """ Reports the end of a replacement interval for services
        """

        pass

    def execute_service(self, time, service, is_cloud, traffic_class, node, price):
        """ Reports the end of a replacement interval for services
        """

        pass
    
    def reject_service(self, time, service, is_cloud, traffic_class, node, price):
        """ Reports the end of a replacement interval for services
        """

        pass
    
    def set_vm_prices(self, node, vm_prices, time=0):
        """ Reports the end of a replacement interval for services
        """
        
        pass

    def set_node_traffic_rates(self, node, time, rates, eff_rates):
        """ Reports the end of a replacement interval for services
        """
        
        pass

    def set_node_util(self, node, utilities):
        """ Reports the end of a replacement interval for services
        """
        
        pass

    def request_hop(self, u, v, main_path=True):
        """Reports that a request has traversed the link *(u, v)*

        Parameters
        ----------
        u : any hashable type
            Origin node
        v : any hashable type
            Destination node
        main_path : bool, optional
            If *True*, indicates that link link is on the main path that will
            lead to hit a content. It is normally used to calculate latency
            correctly in multicast cases. Default value is *True*
        """
        pass

    def content_hop(self, u, v, main_path=True):
        """Reports that a content has traversed the link *(u, v)*

        Parameters
        ----------
        u : any hashable type
            Origin node
        v : any hashable type
            Destination node
        main_path : bool, optional
            If *True*, indicates that this link is being traversed by content
            that will be delivered to the receiver. This is needed to
            calculate latency correctly in multicast cases. Default value is
            *True*
        """
        pass

    def end_session(self, success=True, timestamp=0, flow_id=0):
        """Reports that the session is closed, i.e. the content has been
        successfully delivered to the receiver or a failure blocked the
        execution of the request

        Parameters
        ----------
        success : bool, optional
            *True* if the session was completed successfully, *False* otherwise
        """
        pass

    def results(self):
        """Returns the aggregated results measured by the collector.

        Returns
        -------
        results : dict
            Dictionary mapping metric with results.
        """
        pass

# Note: The implementation of CollectorProxy could be improved to avoid having
# to rewrite almost identical methods, for example by playing with __dict__
# attribute. However, it was implemented this way to make it more readable and
# easier to understand.
class CollectorProxy(DataCollector):
    """This class acts as a proxy for all concrete collectors towards the
    network controller.

    An instance of this class registers itself with the network controller and
    it receives notifications for all events. This class is responsible for
    dispatching events of interests to concrete collectors.
    """

    EVENTS = ('start_session', 'end_session', 'cache_hit', 'cache_miss', 'server_hit',
              'request_hop', 'content_hop', 'results', 'replacement_interval_over', 
              'execute_service', 'reject_service', 'set_vm_prices', 'set_node_traffic_rates', 'set_node_util')

    def __init__(self, view, collectors):
        """Constructor

        Parameters
        ----------
        view : NetworkView
            An instance of the network view
        collector : list of DataCollector
            List of instances of DataCollector that will be notified of events
        """
        self.view = view
        self.collectors = {e: [c for c in collectors if e in type(c).__dict__]
                           for e in self.EVENTS}

    @inheritdoc(DataCollector)
    def start_session(self, timestamp, receiver, content, flow_id=0, traffic_class=0):
        for c in self.collectors['start_session']:
            c.start_session(timestamp, receiver, content, flow_id, traffic_class)

    @inheritdoc(DataCollector)
    def cache_hit(self, node):
        for c in self.collectors['cache_hit']:
            c.cache_hit(node)

    @inheritdoc(DataCollector)
    def cache_miss(self, node):
        for c in self.collectors['cache_miss']:
            c.cache_miss(node)

    @inheritdoc(DataCollector)
    def server_hit(self, node):
        for c in self.collectors['server_hit']:
            c.server_hit(node)

    @inheritdoc(DataCollector)
    def request_hop(self, u, v, main_path=True):
        for c in self.collectors['request_hop']:
            c.request_hop(u, v, main_path)

    @inheritdoc(DataCollector)
    def content_hop(self, u, v, main_path=True):
        for c in self.collectors['content_hop']:
            c.content_hop(u, v, main_path)

    @inheritdoc(DataCollector)
    def replacement_interval_over(self, replacement_interval, timestamp):
        for c in self.collectors['replacement_interval_over']:
            c.replacement_interval_over(replacement_interval, timestamp)

    @inheritdoc(DataCollector)
    def execute_service(self, time, service, is_cloud, traffic_class, node, price):
        for c in self.collectors['execute_service']:
            c.execute_service(time, service, is_cloud, traffic_class, node, price)
    
    @inheritdoc(DataCollector)
    def reject_service(self, time, service, is_cloud, traffic_class, node, price):
        for c in self.collectors['execute_service']:
            c.reject_service(time, service, is_cloud, traffic_class, node, price)
    
    @inheritdoc(DataCollector)
    def set_vm_prices(self, node, vm_prices, time=0):
        for c in self.collectors['set_vm_prices']:
            c.set_vm_prices(node, vm_prices, time)

    @inheritdoc(DataCollector)
    def set_node_traffic_rates(self, node, time, rates, eff_rates):
        for c in self.collectors['set_vm_prices']:
            c.set_node_traffic_rates(node, time, rates, eff_rates)

    @inheritdoc(DataCollector)
    def set_node_util(self, node, utilities, time):
        for c in self.collectors['set_node_util']:
            c.set_node_util(node, utilities, time)

    @inheritdoc(DataCollector)
    def end_session(self, success=True, time=0, flow_id=0):
        for c in self.collectors['end_session']:
            c.end_session(success, time, flow_id)

    @inheritdoc(DataCollector)
    def results(self):
        return Tree(**{c.name: c.results() for c in self.collectors['results']})


@register_data_collector('LINK_LOAD')
class LinkLoadCollector(DataCollector):
    """Data collector measuring the link load
    """

    def __init__(self, view, req_size=150, content_size=1500):
        """Constructor

        Parameters
        ----------
        view : NetworkView
            The network view instance
        req_size : int
            Average size (in bytes) of a request
        content_size : int
            Average size (in byte) of a content
        """
        self.view = view
        self.req_count = collections.defaultdict(int)
        self.cont_count = collections.defaultdict(int)
        if req_size <= 0 or content_size <= 0:
            raise ValueError('req_size and content_size must be positive')
        self.req_size = req_size
        self.content_size = content_size
        self.t_start = -1
        self.t_end = 1

    @inheritdoc(DataCollector)
    def start_session(self, timestamp, receiver, content):
        if self.t_start < 0:
            self.t_start = timestamp
        self.t_end = timestamp

    @inheritdoc(DataCollector)
    def request_hop(self, u, v, main_path=True):
        self.req_count[(u, v)] += 1

    @inheritdoc(DataCollector)
    def content_hop(self, u, v, main_path=True):
        self.cont_count[(u, v)] += 1

    @inheritdoc(DataCollector)
    def results(self):
        duration = self.t_end - self.t_start
        used_links = set(self.req_count.keys()).union(set(self.cont_count.keys()))
        link_loads = dict((link, (self.req_size * self.req_count[link] +
                                  self.content_size * self.cont_count[link]) / duration)
                          for link in used_links)
        link_loads_int = dict((link, load)
                              for link, load in link_loads.items()
                              if self.view.link_type(*link) == 'internal')
        link_loads_ext = dict((link, load)
                              for link, load in link_loads.items()
                              if self.view.link_type(*link) == 'external')
        mean_load_int = sum(link_loads_int.values()) / len(link_loads_int) \
                        if len(link_loads_int) > 0 else 0
        mean_load_ext = sum(link_loads_ext.values()) / len(link_loads_ext) \
                        if len(link_loads_ext) > 0 else 0
        return Tree({'MEAN_INTERNAL':     mean_load_int,
                     'MEAN_EXTERNAL':     mean_load_ext,
                     'PER_LINK_INTERNAL': link_loads_int,
                     'PER_LINK_EXTERNAL': link_loads_ext})


@register_data_collector('LATENCY')
class LatencyCollector(DataCollector):
    """Data collector measuring latency, i.e. the delay taken to delivery a
    content.
    """

    def __init__(self, view, cdf=False):
        """Constructor

        Parameters
        ----------
        view : NetworkView
            The network view instance
        cdf : bool, optional
            If *True*, also collects a cdf of the latency
        """
        self.cdf = cdf
        self.view = view
        self.sess_count = 0
        self.interval_sess_count = 0
        self.css = self.view.service_nodes()
        self.n_services = self.css.items()[0][1].service_population
        self.num_classes = self.css.items()[0][1].num_classes

        # Time series for various metrics
        self.idle_times = {}
        self.sat_times = {}
        self.price_times = {}
        self.node_idle_times = {}
        self.qos_times = {}
        self.revenue_times = {}
        self.node_rate_times = {}
        self.node_eff_rate_times = {}
        # price for each node
        self.node_prices = {}
        # total qos for each class 
        self.qos_class = [0.0 for x in range(self.num_classes)]
        self.qos_service = [0.0 for x in range(self.n_services)]
        # number of requests for each class 
        self.class_requests = [0 for x in range(self.num_classes)]
        self.class_executed_requests = [0 for x in range(self.num_classes)]
        self.service_requests = [0 for x in range(self.n_services)]
        self.service_executed_requests = [0 for x in range(self.n_services)]
        self.class_revenue = [0.0 for x in range(self.num_classes)]
        self.service_revenue = [0.0 for x in range(self.n_services)]
        self.node_utilities = {}
        self.class_sat_rate = [0.0 for x in range(self.num_classes)]
        self.service_sat_rate = [0.0 for x in range(self.n_services)]
        self.sat_requests_nodes = {x:0 for x in self.css.keys()}
        self.rejected_requests_nodes = {x:0 for x in self.css.keys()}
        self.qos_total = 0.0
        self.revenue_total = 0.0
        self.num_executed = 0
    
    @inheritdoc(DataCollector)
    def execute_service(self, time, service, is_cloud, traffic_class, node, price):

        if not is_cloud:
            utility = self.node_utilities[node][service][traffic_class]
            self.qos_class[traffic_class] += utility
            self.class_executed_requests[traffic_class] += 1
            self.service_executed_requests[service] += 1
            self.qos_service[service] += utility
            self.class_revenue[traffic_class] += price
            self.service_revenue[service] += price
            self.sat_requests_nodes[node] += 1
            self.qos_total += utility
            self.revenue_total += price
            self.num_executed += 1
        
    @inheritdoc(DataCollector)
    def reject_service(self, time, service, is_cloud, traffic_class, node, price):
        if not is_cloud:
            self.rejected_requests_nodes[node] += 1

    @inheritdoc(DataCollector)
    def set_vm_prices(self, node, vm_prices, time):
        self.node_prices[node] = vm_prices
        if time in self.price_times.keys():
            self.price_times[time].append((1.0*sum(vm_prices))/len(vm_prices))
        else:
            self.price_times[time] = [(1.0*sum(vm_prices))/len(vm_prices)]
        
    @inheritdoc(DataCollector)
    def set_node_traffic_rates(self, node, time, rates, eff_rates):
        if time in self.node_eff_rate_times.keys():
            self.node_eff_rate_times[time].append([node, eff_rates])
        else:
            self.node_eff_rate_times[time] = [[node, eff_rates]]

        if time in self.node_rate_times.keys():
            self.node_rate_times[time].append([node, rates])
        else:
            self.node_rate_times[time] = [[node, rates]]

    @inheritdoc(DataCollector)
    def set_node_util(self, node, utilities, time):
        self.node_utilities[node] = utilities

    @inheritdoc(DataCollector)
    def replacement_interval_over(self, replacement_interval, timestamp):
        total_idle_time = 0.0
        self.node_idle_times[timestamp] = []
        if self.num_executed > 0:
            self.qos_times[timestamp] = (1.0*self.qos_total)/self.num_executed
        else:
            self.qos_times[timestamp] = 0.0
        self.revenue_times[timestamp] = (1.0*self.revenue_total)
        self.num_executed = 0
        self.qos_total = 0.0
        self.revenue_total = 0.0
        numberOfnodes = 0
        for node, cs in self.css.items():
            if cs.is_cloud:
                continue
            
            idle_time = cs.getIdleTime(timestamp)
            total_idle_time += idle_time
            total_idle_time /= cs.numOfCores

            self.node_idle_times[timestamp].append(1.0*idle_time/(cs.numOfCores*replacement_interval))
            cs.cpuInfo.idleTime = 0.0
            numberOfnodes += 1

        #print "Timestamp: " + repr(timestamp) + " Idle time : " + repr(total_idle_time)
        self.idle_times[timestamp] = (1.0*total_idle_time)/(1.0*replacement_interval*numberOfnodes)
        
        self.sat_times[timestamp] = []
        for node, cs in self.css.items():
            if cs.is_cloud:
                continue
            node_sat = (1.0*self.sat_requests_nodes[node])/(self.rejected_requests_nodes[node] + self.sat_requests_nodes[node])
            self.sat_times[timestamp].append(node_sat)
            print ("Accepted requests @node: " + repr(node) + " is " + repr(self.sat_requests_nodes[node]))
            print ("Rejected requests @node: " + repr(node) + " is " + repr(self.rejected_requests_nodes[node]))
            self.sat_requests_nodes[node] = 0
            self.rejected_requests_nodes[node] = 0

        # Initialise interval counts
        self.interval_sess_count = 0
        self.qos_interval = 0.0

    @inheritdoc(DataCollector)
    def start_session(self, timestamp, receiver, content, flow_id=0, traffic_class=0):
        self.sess_count += 1
        self.interval_sess_count += 1
        self.class_requests[traffic_class] += 1
        self.service_requests[content] += 1

    @inheritdoc(DataCollector)
    def request_hop(self, u, v, main_path=True):
        if main_path:
            self.sess_latency += self.view.link_delay(u, v)

    @inheritdoc(DataCollector)
    def content_hop(self, u, v, main_path=True):
        if main_path:
            self.sess_latency += self.view.link_delay(u, v)

    @inheritdoc(DataCollector)
    def results(self):
        for c in range(self.num_classes):
            if self.class_executed_requests[c] > 0:
                self.qos_class[c] = self.qos_class[c] #/self.class_executed_requests[c]
                self.class_revenue[c] = self.class_revenue[c] #/self.class_executed_requests[c]
            else:
                self.qos_class[c] = 0.0
                self.class_revenue[c] = 0.0
            
            if self.class_requests[c] > 0:
                self.class_sat_rate[c] = (1.0*self.class_executed_requests[c]) / self.class_requests[c]
            else:
                self.class_sat_rate[c] = 0.0

            #print "QoS for class: " + repr(c) + " is " + repr(self.qos_class[c])
            #print "Per-request revenue from class: " + repr(c) + " is " + repr(self.class_revenue[c])
        results = Tree({'QoS_CLASS' : self.qos_class})
        per_service_sats = {}
        for s in range(self.n_services):
            if self.service_requests[s] == 0:
                self.qos_service[s] = 0.0
            else:
                self.qos_service[s] = self.qos_service[s]/self.service_requests[s]
            self.service_revenue[s] = self.service_revenue[s] #/self.service_executed_requests[s]
            if self.service_requests[s] > 0:
                self.service_sat_rate[s] = (1.0*self.service_executed_requests[s]) / self.service_requests[s]
            else:
                self.service_sat_rate[s] = 0.0

            #print "QoS for service: " + repr(s) + " is " + repr(self.qos_service[s])
            #print "Per-request revenue from service: " + repr(s) + " is " + repr(self.service_revenue[s])

        results['IDLE_TIMES'] = self.idle_times #sum(self.idle_times.values())/len(self.idle_times.keys())
        #print "Idle times: " + repr(self.idle_times)
        results['QoS_SERVICE'] = self.qos_service
        results['QOS_TIMES'] = self.qos_times
        results['REVENUE_TIMES'] = self.revenue_times
        results['NODE_VM_PRICES'] = self.node_prices
        results['CLASS_REVENUE'] = self.class_revenue
        results['SERVICE_REVENUE'] = self.service_revenue
        results['CLASS_SAT_RATE'] = self.class_sat_rate
        results['SERVICE_SAT_RATE'] = self.service_sat_rate
        results['SERVICE_RATE'] = self.service_requests
        results['CLASS_RATE'] = self.class_requests
        results['SAT_TIMES'] = self.sat_times #{x:1.0*sum(self.sat_times[x])/len(self.sat_times[x]) for x in self.sat_times.keys()}
        results['PRICE_TIMES'] = self.price_times
        results['NODE_IDLE_TIMES'] = self.node_idle_times
        results['NODE_UTILITIES'] = self.node_utilities
        results['IDLE_TIMES_AVG'] = sum(self.idle_times.values())/len(self.idle_times.keys())
        results['REVENUE_TIMES_AVG'] = sum(self.revenue_times.values())/len(self.revenue_times.keys())
        results['PRICE_TIMES_AVG'] = sum([x[0] for x in self.price_times.values()])/len(self.price_times.keys())
        results['QOS_TIMES_AVG'] = sum(self.qos_times.values())/len(self.qos_times.keys())
        results['NODE_RATE_TIMES'] = self.node_rate_times
        results['NODE_EFF_RATE_TIMES'] = self.node_eff_rate_times
        
        """
        print "Printing Idle times:"
        for key in sorted(self.idle_times):
            print (repr(key) + " " + repr(self.idle_times[key]))
        #results['VMS_PER_SERVICE'] = self.vms_per_service       
        print "\nPrinting Node Idle times:"
        for key in sorted(self.node_idle_times):
            print (repr(key) + " " + repr(self.node_idle_times[key]))
        """

        return results

@register_data_collector('CACHE_HIT_RATIO')
class CacheHitRatioCollector(DataCollector):
    """Collector measuring the cache hit ratio, i.e. the portion of content
    requests served by a cache.
    """

    def __init__(self, view, off_path_hits=False, per_node=True, content_hits=False):
        """Constructor

        Parameters
        ----------
        view : NetworkView
            The NetworkView instance
        off_path_hits : bool, optional
            If *True* also records cache hits from caches not on located on the
            shortest path. This metric may be relevant only for some strategies
        content_hits : bool, optional
            If *True* also records cache hits per content instead of just
            globally
        """
        self.view = view
        self.off_path_hits = off_path_hits
        self.per_node = per_node
        self.cont_hits = content_hits
        self.sess_count = 0
        self.cache_hits = 0
        self.serv_hits = 0
        if off_path_hits:
            self.off_path_hit_count = 0
        if per_node:
            self.per_node_cache_hits = collections.defaultdict(int)
            self.per_node_server_hits = collections.defaultdict(int)
        if content_hits:
            self.curr_cont = None
            self.cont_cache_hits = collections.defaultdict(int)
            self.cont_serv_hits = collections.defaultdict(int)

    @inheritdoc(DataCollector)
    def start_session(self, timestamp, receiver, content):
        self.sess_count += 1
        if self.off_path_hits:
            source = self.view.content_source(content)
            self.curr_path = self.view.shortest_path(receiver, source)
        if self.cont_hits:
            self.curr_cont = content

    @inheritdoc(DataCollector)
    def cache_hit(self, node):
        self.cache_hits += 1
        if self.off_path_hits and node not in self.curr_path:
            self.off_path_hit_count += 1
        if self.cont_hits:
            self.cont_cache_hits[self.curr_cont] += 1
        if self.per_node:
            self.per_node_cache_hits[node] += 1

    @inheritdoc(DataCollector)
    def server_hit(self, node):
        self.serv_hits += 1
        if self.cont_hits:
            self.cont_serv_hits[self.curr_cont] += 1
        if self.per_node:
            self.per_node_server_hits[node] += 1

    @inheritdoc(DataCollector)
    def results(self):
        n_sess = self.cache_hits + self.serv_hits
        hit_ratio = self.cache_hits / n_sess
        results = Tree(**{'MEAN': hit_ratio})
        if self.off_path_hits:
            results['MEAN_OFF_PATH'] = self.off_path_hit_count / n_sess
            results['MEAN_ON_PATH'] = results['MEAN'] - results['MEAN_OFF_PATH']
        if self.cont_hits:
            cont_set = set(list(self.cont_cache_hits.keys()) + list(self.cont_serv_hits.keys()))
            cont_hits = dict((i, (self.cont_cache_hits[i] / (self.cont_cache_hits[i] + self.cont_serv_hits[i])))
                            for i in cont_set)
            results['PER_CONTENT'] = cont_hits
        if self.per_node:
            for v in self.per_node_cache_hits:
                self.per_node_cache_hits[v] /= n_sess
            for v in self.per_node_server_hits:
                self.per_node_server_hits[v] /= n_sess
            results['PER_NODE_CACHE_HIT_RATIO'] = self.per_node_cache_hits
            results['PER_NODE_SERVER_HIT_RATIO'] = self.per_node_server_hits
        return results


@register_data_collector('PATH_STRETCH')
class PathStretchCollector(DataCollector):
    """Collector measuring the path stretch, i.e. the ratio between the actual
    path length and the shortest path length.
    """

    def __init__(self, view, cdf=False):
        """Constructor

        Parameters
        ----------
        view : NetworkView
            The network view instance
        cdf : bool, optional
            If *True*, also collects a cdf of the path stretch
        """
        self.view = view
        self.cdf = cdf
        self.req_path_len = collections.defaultdict(int)
        self.cont_path_len = collections.defaultdict(int)
        self.sess_count = 0
        self.mean_req_stretch = 0.0
        self.mean_cont_stretch = 0.0
        self.mean_stretch = 0.0
        if self.cdf:
            self.req_stretch_data = collections.deque()
            self.cont_stretch_data = collections.deque()
            self.stretch_data = collections.deque()

    @inheritdoc(DataCollector)
    def start_session(self, timestamp, receiver, content):
        self.receiver = receiver
        self.source = self.view.content_source(content)
        self.req_path_len = 0
        self.cont_path_len = 0
        self.sess_count += 1

    @inheritdoc(DataCollector)
    def request_hop(self, u, v, main_path=True):
        self.req_path_len += 1

    @inheritdoc(DataCollector)
    def content_hop(self, u, v, main_path=True):
        self.cont_path_len += 1

    @inheritdoc(DataCollector)
    def end_session(self, success=True):
        if not success:
            return
        req_sp_len = len(self.view.shortest_path(self.receiver, self.source))
        cont_sp_len = len(self.view.shortest_path(self.source, self.receiver))
        req_stretch = self.req_path_len / req_sp_len
        cont_stretch = self.cont_path_len / cont_sp_len
        stretch = (self.req_path_len + self.cont_path_len) / (req_sp_len + cont_sp_len)
        self.mean_req_stretch += req_stretch
        self.mean_cont_stretch += cont_stretch
        self.mean_stretch += stretch
        if self.cdf:
            self.req_stretch_data.append(req_stretch)
            self.cont_stretch_data.append(cont_stretch)
            self.stretch_data.append(stretch)

    @inheritdoc(DataCollector)
    def results(self):
        results = Tree({'MEAN': self.mean_stretch / self.sess_count,
                        'MEAN_REQUEST': self.mean_req_stretch / self.sess_count,
                        'MEAN_CONTENT': self.mean_cont_stretch / self.sess_count})
        if self.cdf:
            results['CDF'] = cdf(self.stretch_data)
            results['CDF_REQUEST'] = cdf(self.req_stretch_data)
            results['CDF_CONTENT'] = cdf(self.cont_stretch_data)
        return results


@register_data_collector('DUMMY')
class DummyCollector(DataCollector):
    """Dummy collector to be used for test cases only."""

    def __init__(self, view):
        """Constructor

        Parameters
        ----------
        view : NetworkView
            The network view instance
        output : stream
            Stream on which debug collector writes
        """
        self.view = view

    @inheritdoc(DataCollector)
    def start_session(self, timestamp, receiver, content):
        self.session = dict(timestamp=timestamp, receiver=receiver,
                            content=content, cache_misses=[],
                            request_hops=[], content_hops=[])

    @inheritdoc(DataCollector)
    def cache_hit(self, node):
        self.session['serving_node'] = node

    @inheritdoc(DataCollector)
    def cache_miss(self, node):
        self.session['cache_misses'].append(node)

    @inheritdoc(DataCollector)
    def server_hit(self, node):
        self.session['serving_node'] = node

    @inheritdoc(DataCollector)
    def request_hop(self, u, v, main_path=True):
        self.session['request_hops'].append((u, v))

    @inheritdoc(DataCollector)
    def content_hop(self, u, v, main_path=True):
        self.session['content_hops'].append((u, v))

    @inheritdoc(DataCollector)
    def end_session(self, success=True):
        self.session['success'] = success

    def session_summary(self):
        """Return a summary of latest session

        Returns
        -------
        session : dict
            Summary of session
        """
        return self.session
