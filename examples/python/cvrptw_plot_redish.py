# Modified from cvrptw_plot.py
# To play with special case for Re-dish
#
#
# This Python file uses the following encoding: utf-8
# Copyright 2015 Tin Arm Engineering AB
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Capacitated Vehicle Routing Problem with Time Windows (and optional orders).

   This is a sample using the routing library python wrapper to solve a
   CVRPTW problem.
   A description of the problem can be found here:
   http://en.wikipedia.org/wiki/Vehicle_routing_problem.
   The variant which is tackled by this model includes a capacity dimension,
   time windows and optional orders, with a penalty cost if orders are not
   performed.
   To help explore the problem, two classes are provided Customers() and
   Vehicles(): used to randomly locate orders and depots, and to randomly
   generate demands, time-window constraints and vehicles.
   Distances are computed using the Great Circle distances. Distances are in km
   and times in seconds.

   A function for the displaying of the vehicle plan
   display_vehicle_output

   The optimization engine uses local search to improve solutions, first
   solutions being generated using a cheapest addition heuristic.
   Numpy and Matplotlib are required for the problem creation and display.

"""
import os
import numpy as np
from matplotlib import pyplot as plt
from collections import namedtuple
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
from datetime import datetime, timedelta
import string
import folium

class Customers():
    """
        A class that generates and holds customers information.

        Randomly normally distribute a number of customers and locations within
        a region described by a rectangle.  Generate a random demand for each
        customer. Generate a random time window for each customer.
        May either be initiated with the extents, as a dictionary describing
        two corners of a rectangle in latitude and longitude OR as a center
        point (lat, lon), and box_size in km.  The default arguments are for a
        10 x 10 km square centered in Sheffield).

        Args: extents (Optional[Dict]): A dictionary describing a rectangle in
        latitude and longitude with the keys 'llcrnrlat', 'llcrnrlon' &
        'urcrnrlat' & 'urcrnrlat'  center (Optional(Tuple): A tuple of
        (latitude, longitude) describing the centre of the rectangle.  box_size
        (Optional float: The length in km of the box's sides.  num_stops (int):
        The number of customers, including the depots that are placed normally
        distributed in the rectangle.  min_demand (int): Lower limit on the
        randomly generated demand at each customer.  max_demand (int): Upper
        limit on the randomly generated demand at each customer.
            min_tw: shortest random time window for a customer, in hours.
            max_tw: longest random time window for a customer, in hours.
        Examples: To place 100 customers randomly within 100 km x 100 km
        rectangle, centered in the default location, with a random demand of
        between 5 and 10 units:  >>> customers = Customers(num_stops=100,
        box_size=100, ...                 min_demand=5, max_demand=10)
        alternatively, to place 75 customers in the same area with default
        arguments for demand:  >>> extents = {'urcrnrlon': 0.03403, 'llcrnrlon':
        -2.98325, ...     'urcrnrlat': 54.28127, 'llcrnrlat': 52.48150} >>>
        customers = Customers(num_stops=75, extents=extents)
  """

    def __init__(self,
                 extents=None,
                 center=(53.381393, -1.474611),
                 box_size=10,
                 num_stops=100,
                 min_demand=0,
                 max_demand=25,
                 min_tw=1,
                 max_tw=5,
                 fromList=False,
                 custNames=[],
                 custLats=[],
                 custLons=[],
                 custDemands=[],
                 custTwsOpen=[],
                 custTwwClose=[]):
        if fromList:
            self.number = len(custNames)
            stops = np.array(range(0, self.number))
            demands = custDemands
            lats = custLats
            lons = custLons
            start_times = custTwsOpen
            stop_times = custTwwClose
            names = custNames
            self.time_horizon = 24 * 60**2  # A 24 hour period.

        else:
            self.number = num_stops  #: The number of customers and depots
            #: Location, a named tuple for locations.
            Location = namedtuple('Location', ['lat', 'lon'])
            if extents is not None:
                self.extents = extents  #: The lower left and upper right points
                #: Location[lat,lon]: the centre point of the area.
                self.center = Location(
                    extents['urcrnrlat'] - 0.5 *
                    (extents['urcrnrlat'] - extents['llcrnrlat']),
                    extents['urcrnrlon'] - 0.5 *
                    (extents['urcrnrlon'] - extents['llcrnrlon']))
            else:
                #: Location[lat,lon]: the centre point of the area.
                (clat, clon) = self.center = Location(center[0], center[1])
                rad_earth = 6367  # km
                circ_earth = np.pi * rad_earth
                #: The lower left and upper right points
                self.extents = {
                    'llcrnrlon': (clon - 180 * box_size /
                                  (circ_earth * np.cos(np.deg2rad(clat)))),
                    'llcrnrlat':
                    clat - 180 * box_size / circ_earth,
                    'urcrnrlon': (clon + 180 * box_size /
                                  (circ_earth * np.cos(np.deg2rad(clat)))),
                    'urcrnrlat':
                    clat + 180 * box_size / circ_earth
                }
            # The 'name' of the stop, indexed from 0 to num_stops-1
            # create a numpy array of 5 strings, each of length 10
            str_length = 10
            characters = string.ascii_letters + string.digits
            names = np.array(
                [''.join(np.random.choice(list(characters), size=str_length)) for _ in range(num_stops)])
            stops = np.array(range(0, num_stops))
            # normaly distributed random distribution of stops within the box
            stdv = 6  # the number of standard deviations 99.9% will be within +-3
            lats = (self.extents['llcrnrlat'] + np.random.randn(num_stops) *
                    (self.extents['urcrnrlat'] - self.extents['llcrnrlat']) / stdv)
            lons = (self.extents['llcrnrlon'] + np.random.randn(num_stops) *
                    (self.extents['urcrnrlon'] - self.extents['llcrnrlon']) / stdv)
            # uniformly distributed integer demands.
            demands = np.random.randint(min_demand, max_demand, num_stops)

            self.time_horizon = 24 * 60**2  # A 24 hour period.

            # The customers demand min_tw to max_tw hour time window for each
            # delivery
            time_windows = np.random.randint(min_tw * 3600, max_tw * 3600,
                                                     num_stops)
            # The last time a delivery window can start
            latest_time = self.time_horizon - time_windows
            start_times = np.array([datetime.timedelta(0,6*3600) for o in time_windows], dtype=datetime)
            stop_times = np.array([datetime.timedelta(0,18*3600) for o in time_windows], dtype=datetime)
            # Make random timedeltas, nominally from the start of the day.
            # for idx in range(self.number):
            #     stime = int(np.random.randint(0, latest_time[idx]))
            #     start_times[idx] = datetime.timedelta(seconds=6*3600) # stime)
            #     stop_times[idx] = (
            #         start_times[idx] + timedelta(seconds=86399)) # int(time_windows[idx])))
            #     stop_times[idx] = datetime.timedelta(seconds=18*3600)
            print("init")

        # A named tuple for the customer
        Customer = namedtuple(
            'Customer',
            [
                'index',  # the index of the stop
                'demand',  # the demand for the stop
                'lat',  # the latitude of the stop
                'lon',  # the longitude of the stop
                'tw_open',  # timedelta window open
                'tw_close',
                'name'  # the customer's name
            ])  # timedelta window cls

        self.customers = [
            Customer(idx, dem, lat, lon, tw_open, tw_close, name)
            for idx, dem, lat, lon, tw_open, tw_close, name in zip(
                stops, demands, lats, lons, start_times, stop_times, names)
        ]

        # The number of seconds needed to 'unload' 1 unit of goods.
        self.service_time_per_dem = 300  # seconds

    def set_manager(self, manager):
        self.manager = manager

    def central_start_node(self, invert=False):
        """
        Return a random starting node, with probability weighted by distance
        from the centre of the extents, so that a central starting node is
        likely.

        Args: invert (Optional bool): When True, a peripheral starting node is
        most likely.

        Returns:
            int: a node index.

        Examples:
        #    >>> customers.central_start_node(invert=True)
            42
        """
        num_nodes = len(self.customers)
        dist = np.empty((num_nodes, 1))
        for idx_to in range(num_nodes):
            dist[idx_to] = self._haversine(self.center.lon, self.center.lat,
                                           self.customers[idx_to].lon,
                                           self.customers[idx_to].lat)
        furthest = np.max(dist)

        if invert:
            prob = dist * 1.0 / sum(dist)
        else:
            prob = (furthest - dist * 1.0) / sum(furthest - dist)
        indexes = np.array([range(num_nodes)])
        start_node = np.random.choice(
            indexes.flatten(), size=1, replace=True, p=prob.flatten())
        return start_node[0]

    def make_distance_mat(self, method='haversine'):
        """
        Return a distance matrix and make it a member of Customer, using the
        method given in the call. Currently only Haversine (GC distance) is
        implemented, but Manhattan, or using a maps API could be added here.
        Raises an AssertionError for all other methods.

        Args: method (Optional[str]): method of distance calculation to use. The
        Haversine formula is the only method implemented.

        Returns:
            Numpy array of node to node distances.

        Examples:
        #    >>> dist_mat = customers.make_distance_mat(method='haversine')
        #    >>> dist_mat = customers.make_distance_mat(method='manhattan')
            AssertionError
        """
        self.distmat = np.zeros((self.number, self.number))
        methods = {'haversine': self._haversine}
        assert (method in methods)
        for frm_idx in range(self.number):
            for to_idx in range(self.number):
                if frm_idx != to_idx:
                    frm_c = self.customers[frm_idx]
                    to_c = self.customers[to_idx]
                    self.distmat[frm_idx, to_idx] = self._haversine(
                        frm_c.lon, frm_c.lat, to_c.lon, to_c.lat)
        # Calculate my own distance matrix from testGoogAPI.py - need to refactor to read in from file...
        #distance
        self.distmat = np.array([
[0.000000, 14.502000, 14.531000, 11.288000, 4.228000, 4.035000, 5.548000, 20.082000, 9.519000, 41.720000],
[13.408000, 0.000000, 0.740000, 3.322000, 27.310000, 12.042000, 14.709000, 27.093000, 31.648000, 44.075000],
[13.621000, 1.384000, 0.000000, 4.000000, 27.522000, 12.254000, 14.921000, 27.306000, 31.860000, 44.979000],
[13.960000, 2.460000, 3.202000, 0.000000, 27.862000, 8.627000, 12.501000, 22.557000, 19.920000, 42.864000],
[4.133000, 28.293000, 28.323000, 28.914000, 0.000000, 7.459000, 8.444000, 30.067000, 5.978000, 54.641000],
[3.808000, 12.518000, 12.535000, 8.444000, 8.036000, 0.000000, 2.668000, 16.357000, 10.688000, 38.875000],
[6.142000, 17.822000, 17.839000, 12.540000, 7.593000, 2.996000, 0.000000, 14.237000, 8.962000, 38.820000],
[21.133000, 28.632000, 28.650000, 23.351000, 31.336000, 19.004000, 17.216000, 0.000000, 27.669000, 26.843000],
[9.787000, 25.259000, 25.277000, 19.978000, 5.978000, 10.979000, 9.422000, 26.265000, 0.000000, 50.839000],
[42.156000, 42.328000, 43.371000, 42.164000, 54.658000, 39.239000, 40.537000, 26.756000, 50.990000, 0.000000],
])
        #time
#         self.distmat = np.array([
# [0.000000, 0.526389, 0.514167, 0.536111, 0.270833, 0.235556, 0.386389, 0.490556, 0.501944, 0.864722],
# [0.463889, 0.000000, 0.074444, 0.232778, 0.610000, 0.450556, 0.600833, 0.604444, 0.849167, 0.731111],
# [0.464167, 0.134167, 0.000000, 0.288056, 0.610556, 0.450833, 0.601389, 0.604722, 0.849444, 0.782222],
# [0.536111, 0.198056, 0.241667, 0.000000, 0.682500, 0.448611, 0.517500, 0.535278, 0.823056, 0.584167],
# [0.281667, 0.692778, 0.680556, 0.722500, 0.000000, 0.498056, 0.466389, 0.699722, 0.385556, 1.132778],
# [0.186667, 0.482500, 0.471389, 0.390833, 0.457778, 0.000000, 0.201667, 0.393889, 0.560278, 0.719444],
# [0.356944, 0.646667, 0.635278, 0.516667, 0.428611, 0.255278, 0.000000, 0.373333, 0.452222, 0.805278],
# [0.538611, 0.678611, 0.667500, 0.548889, 0.712778, 0.481667, 0.416667, 0.000000, 0.525556, 0.562222],
# [0.526667, 0.914722, 0.903611, 0.784722, 0.370556, 0.607500, 0.493056, 0.485833, 0.000000, 0.918889],
# [0.888889, 0.726389, 0.768611, 0.632222, 1.123333, 0.781944, 0.827222, 0.556389, 0.936111, 0.000000],
# ])

        return (self.distmat)

    def _haversine(self, lon1, lat1, lon2, lat2):
        """
        Calculate the great circle distance between two points
        on the earth specified in decimal degrees of latitude and longitude.
        https://en.wikipedia.org/wiki/Haversine_formula

        Args:
            lon1: longitude of pt 1,
            lat1: latitude of pt 1,
            lon2: longitude of pt 2,
            lat2: latitude of pt 2

        Returns:
            the distace in km between pt1 and pt2
        """
        # convert decimal degrees to radians
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

        # haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = (np.sin(dlat / 2)**2 +
             np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2)
        c = 2 * np.arcsin(np.sqrt(a))

        # 6367 km is the radius of the Earth
        km = 6367 * c
        return km

    def get_total_demand(self):
        """
        Return the total demand of all customers.
        """
        return (sum([c.demand for c in self.customers]))

    def return_dist_callback(self, **kwargs):
        """
        Return a callback function for the distance matrix.

        Args: **kwargs: Arbitrary keyword arguments passed on to
        make_distance_mat()

        Returns:
            function: dist_return(a,b) A function that takes the 'from' node
                index and the 'to' node index and returns the distance in km.
        """
        self.make_distance_mat(**kwargs)

        def dist_return(from_index, to_index):
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = self.manager.IndexToNode(from_index)
            to_node = self.manager.IndexToNode(to_index)
            return (self.distmat[from_node][to_node])

        return dist_return

    def return_dem_callback(self):
        """
        Return a callback function that gives the demands.

        Returns:
            function: dem_return(a) A function that takes the 'from' node
                index and returns the distance in km.
        """

        def dem_return(from_index):
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = self.manager.IndexToNode(from_index)
            return (self.customers[from_node].demand)

        return dem_return

    def zero_depot_demands(self, depot):
        """
        Zero out the demands and time windows of depot.  The Depots do not have
        demands or time windows so this function clears them.

        Args:  depot (int): index of the stop to modify into a depot.
        Examples:  >>> customers.zero_depot_demands(5) >>>
        customers.customers[5].demand == 0 True
    """
        start_depot = self.customers[depot]
        self.customers[depot] = start_depot._replace(
            demand=0, tw_open=None, tw_close=None)

    def make_service_time_call_callback(self):
        """
        Return a callback function that provides the time spent servicing the
        customer.  Here is it proportional to the demand given by
        self.service_time_per_dem, default 300 seconds per unit demand.

        Returns:
            function [dem_return(a, b)]: A function that takes the from/a node
                index and the to/b node index and returns the service time at a

        """

        def service_time_return(a, b):
            return (self.customers[a].demand * self.service_time_per_dem)

        return service_time_return

    def make_transit_time_callback(self, speed_kmph=60):
        """
        Creates a callback function for transit time. Assuming an average
        speed of speed_kmph
        Args:
            speed_kmph: the average speed in km/h

        Returns:
            function [transit_time_return(a, b)]: A function that takes the
                from/a node index and the to/b node index and returns the
                transit time from a to b.
        """

        def transit_time_return(a, b):
            return (self.distmat[a][b] / (speed_kmph * 1.0 / 60**2))

        return transit_time_return


class Vehicles():
    """
    A Class to create and hold vehicle information.

    The Vehicles in a CVRPTW problem service the customers and belong to a
    depot. The class Vehicles creates a list of named tuples describing the
    Vehicles.  The main characteristics are the vehicle capacity, fixed cost,
    and cost per km.  The fixed cost of using a certain type of vehicles can be
    higher or lower than others. If a vehicle is used, i.e. this vehicle serves
    at least one node, then this cost is added to the objective function.

    Note:
        If numpy arrays are given for capacity and cost, then they must be of
        the same length, and the number of vehicles are inferred from them.
        If scalars are given, the fleet is homogeneous, and the number of
        vehicles is determined by number.

    Args: capacity (scalar or numpy array): The integer capacity of demand
    units.  cost (scalar or numpy array): The fixed cost of the vehicle.  number
    (Optional [int]): The number of vehicles in a homogeneous fleet.
  """

    def __init__(self, capacity=100, cost=100, number=None):

        Vehicle = namedtuple('Vehicle', ['index', 'capacity', 'cost'])

        if number is None:
            self.number = np.size(capacity)
        else:
            self.number = number
        idxs = np.array(range(0, self.number))

        if np.isscalar(capacity):
            capacities = capacity * np.ones_like(idxs)
        elif np.size(capacity) != self.number:
            print('capacity is neither scalar, nor the same size as num!')
        else:
            capacities = capacity

        if np.isscalar(cost):
            costs = cost * np.ones_like(idxs)
        elif np.size(cost) != self.number:
            print(np.size(cost))
            print('cost is neither scalar, nor the same size as num!')
        else:
            costs = cost

        self.vehicles = [
            Vehicle(idx, capacity, cost)
            for idx, capacity, cost in zip(idxs, capacities, costs)
        ]

    def get_total_capacity(self):
        return (sum([c.capacity for c in self.vehicles]))

    def return_starting_callback(self, customers, sameStartFinish=False):
        # create a different starting and finishing depot for each vehicle
        self.starts = [
            #int(customers.central_start_node()) for o in range(self.number)
            0 for o in range(self.number) # customer 0 is depot
        ]
        if sameStartFinish:
            self.ends = self.starts
        else:
            self.ends = [
                int(customers.central_start_node(invert=True))
                for o in range(self.number)
            ]
        # the depots will not have demands, so zero them.
        for depot in self.starts:
            customers.zero_depot_demands(depot)
        for depot in self.ends:
            customers.zero_depot_demands(depot)

        def start_return(v):
            return (self.starts[v])

        return start_return


def discrete_cmap(N, base_cmap=None):
    """
    Create an N-bin discrete colormap from the specified input map
    """
    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


def vehicle_output_string(manager, routing, plan, customers):
    """
    Return a string displaying the output of the routing instance and
    assignment (plan).

    Args: routing (ortools.constraint_solver.pywrapcp.RoutingModel): routing.
    plan (ortools.constraint_solver.pywrapcp.Assignment): the assignment.

    Returns:
        (string) plan_output: describing each vehicle's plan.

        (List) dropped: list of dropped orders.

    """
    dropped = []
    for order in range(routing.Size()):
        if (plan.Value(routing.NextVar(order)) == order):
            dropped.append(str(order))

    capacity_dimension = routing.GetDimensionOrDie('Capacity')
    time_dimension = routing.GetDimensionOrDie('Time')
    plan_output = ''

    for route_number in range(routing.vehicles()):
        order = routing.Start(route_number)
        plan_output += 'Route {0}:'.format(route_number)
        if routing.IsEnd(plan.Value(routing.NextVar(order))):
            plan_output += ' Empty \n'
        else:
            while True:
                load_var = capacity_dimension.CumulVar(order)
                time_var = time_dimension.CumulVar(order)
                node = manager.IndexToNode(order)
                # plan_output += \
                #     ' {node} Load({load}) Time({tmin}, {tmax}) -> '.format(
                #         node=customers.customers[node].name,
                #         load=plan.Value(load_var),
                #         tmin=str(timedelta(seconds=plan.Min(time_var))),
                #         tmax=str(timedelta(seconds=plan.Max(time_var))))
                plan_output += \
                    ' {node} Load({load}) -> '.format(
                        node=customers.customers[node].name,
                        load=plan.Value(load_var))
                if routing.IsEnd(order):
                    plan_output += ' EndRoute {0}. \n'.format(route_number)
                    break
                order = plan.Value(routing.NextVar(order))
        plan_output += '\n'

    return (plan_output, dropped)


def build_vehicle_route(manager, routing, plan, customers, veh_number):
    """
    Build a route for a vehicle by starting at the strat node and
    continuing to the end node.

    Args: routing (ortools.constraint_solver.pywrapcp.RoutingModel): routing.
    plan (ortools.constraint_solver.pywrapcp.Assignment): the assignment.
    customers (Customers): the customers instance.  veh_number (int): index of
    the vehicle

    Returns:
        (List) route: indexes of the customers for vehicle veh_number
    """
    veh_used = routing.IsVehicleUsed(plan, veh_number)
    print('Vehicle {0} is used {1}'.format(veh_number, veh_used))
    if veh_used:
        route = []
        node = routing.Start(veh_number)  # Get the starting node index
        route.append(customers.customers[manager.IndexToNode(node)])
        while not routing.IsEnd(node):
            route.append(customers.customers[manager.IndexToNode(node)])
            node = plan.Value(routing.NextVar(node))

        route.append(customers.customers[manager.IndexToNode(node)])
        return route
    else:
        return None


def plot_vehicle_routes(veh_route, ax1, customers, vehicles):
    """
    Plot the vehicle routes on matplotlib axis ax1.

    Args: veh_route (dict): a dictionary of routes keyed by vehicle idx.  ax1
    (matplotlib.axes._subplots.AxesSubplot): Matplotlib axes  customers
    (Customers): the customers instance.  vehicles (Vehicles): the vehicles
    instance.
  """
    veh_used = [v for v in veh_route if veh_route[v] is not None]

    cmap = discrete_cmap(vehicles.number + 2, 'nipy_spectral')

    for veh_number in veh_used:

        lats, lons = zip(*[(c.lat, c.lon) for c in veh_route[veh_number]])
        lats = np.array(lats)
        lons = np.array(lons)
        s_dep = customers.customers[vehicles.starts[veh_number]]
        s_fin = customers.customers[vehicles.ends[veh_number]]
        ax1.annotate(
            'v({veh}) S @ {node}'.format(
                veh=veh_number, node=vehicles.starts[veh_number]),
            xy=(s_dep.lon, s_dep.lat),
            xytext=(10, 10),
            xycoords='data',
            textcoords='offset points',
            arrowprops=dict(
                arrowstyle='->',
                connectionstyle='angle3,angleA=90,angleB=0',
                shrinkA=0.05),
        )
        ax1.annotate(
            'v({veh}) F @ {node}'.format(
                veh=veh_number, node=vehicles.ends[veh_number]),
            xy=(s_fin.lon, s_fin.lat),
            xytext=(10, -20),
            xycoords='data',
            textcoords='offset points',
            arrowprops=dict(
                arrowstyle='->',
                connectionstyle='angle3,angleA=-90,angleB=0',
                shrinkA=0.05),
        )
        ax1.plot(lons, lats, 'o', mfc=cmap(veh_number + 1))
        ax1.quiver(
            lons[:-1],
            lats[:-1],
            lons[1:] - lons[:-1],
            lats[1:] - lats[:-1],
            scale_units='xy',
            angles='xy',
            scale=1,
            color=cmap(veh_number + 1))


def main():
    # Create a set of customer, (and depot) stops.
    # depot is customer 0.  so...
    customers = Customers(
        fromList = True,
        custNames = ["Re-Dish", "White Case", "MOMA", "Loreal", "SIS Cooper", "JP Morgan", \
                     "Berkeley Carroll", "SIS Wadsworth", "Brearly", "L'Oreal NJ"],
        custLats = [40.69998097056833, 40.7594104, 40.7618038, 40.7527315, 40.6865353, 40.6926286, \
                    40.67402795, 40.60478, 40.63973545, 40.62745318],
        custLons = [-73.94819913666936, -73.9828299, -73.9775202, -74.00152786, -73.9083109, -73.98384265, \
                    -73.97790482, -74.0643273, -73.91455675, -74.32094301],
        custDemands =  [0, 2, 25, 30, 8, 35, 8, 8, 12, 20],
        custTwsOpen =  np.array([timedelta(0,6*3600) for o in range(0, 10)], dtype=datetime),
        custTwwClose = np.array([timedelta(0,18*3600) for o in range(0, 10)], dtype=datetime)
    )
    # customers = Customers(
    #     num_stops=50,
    #     min_demand=1,
    #     max_demand=15,
    #     box_size=40,
    #     min_tw=3,
    #     max_tw=6)
    for c in customers.customers:
        print(c.index, c.demand)
    # Create a list of inhomgenious vehicle capacities as integer units.
    #capacity = [50, 75, 100, 125, 150, 175, 200, 250]
    capacity = [70, 70, 36] # from redish
    #capacity = [5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000]
    # Create a list of inhomogeneous fixed vehicle costs.
    #cost = [int(100 + 2 * np.sqrt(c)) for c in capacity]
    cost = [int(100) for c in capacity]

    # Create a set of vehicles, the number set by the length of capacity.
    vehicles = Vehicles(capacity=capacity, cost=cost)

    # check to see that the problem is feasible, if we don't have enough
    # vehicles to cover the demand, there is no point in going further.
    assert (customers.get_total_demand() < vehicles.get_total_capacity())

    # Set the starting nodes, and create a callback fn for the starting node.
    #start_fn = vehicles.return_starting_callback(
    #    customers, sameStartFinish=False)
    start_fn = vehicles.return_starting_callback(
        customers, sameStartFinish=True)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(
        customers.number,  # int number
        vehicles.number,  # int number
        vehicles.starts,  # List of int start depot
        vehicles.ends)  # List of int end depot

    customers.set_manager(manager)

    # Set model parameters
    model_parameters = pywrapcp.DefaultRoutingModelParameters()

    # The solver parameters can be accessed from the model parameters. For example :
    #   model_parameters.solver_parameters.CopyFrom(
    #       pywrapcp.Solver.DefaultSolverParameters())
    #    model_parameters.solver_parameters.trace_propagation = True

    # Make the routing model instance.
    routing = pywrapcp.RoutingModel(manager, model_parameters)

    parameters = pywrapcp.DefaultRoutingSearchParameters()
    # Setting first solution heuristic (cheapest addition).
    parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    # Routing: forbids use of TSPOpt neighborhood, (this is the default behaviour)
    parameters.local_search_operators.use_tsp_opt = pywrapcp.BOOL_FALSE
    # Disabling Large Neighborhood Search, (this is the default behaviour)
    parameters.local_search_operators.use_path_lns = pywrapcp.BOOL_FALSE
    parameters.local_search_operators.use_inactive_lns = pywrapcp.BOOL_FALSE

    parameters.time_limit.seconds = 100
    parameters.use_full_propagation = True
    #parameters.log_search = True

    # Create callback fns for distances, demands, service and transit-times.
    dist_fn = customers.return_dist_callback()
    dist_fn_index = routing.RegisterTransitCallback(dist_fn)

    dem_fn = customers.return_dem_callback()
    dem_fn_index = routing.RegisterUnaryTransitCallback(dem_fn)

    # Create and register a transit callback.
    serv_time_fn = customers.make_service_time_call_callback()
    transit_time_fn = customers.make_transit_time_callback()
    def tot_time_fn(from_index, to_index):
        """
        The time function we want is both transit time and service time.
        """
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return serv_time_fn(from_node, to_node) + transit_time_fn(from_node, to_node)

    tot_time_fn_index = routing.RegisterTransitCallback(tot_time_fn)

    # Set the cost function (distance callback) for each arc, homogeneous for
    # all vehicles.
    routing.SetArcCostEvaluatorOfAllVehicles(dist_fn_index)

    # Set vehicle costs for each vehicle, not homogeneous.
    for veh in vehicles.vehicles:
        routing.SetFixedCostOfVehicle(veh.cost, int(veh.index))

    # Add a dimension for vehicle capacities
    null_capacity_slack = 0
    routing.AddDimensionWithVehicleCapacity(
        dem_fn_index,  # demand callback
        null_capacity_slack,
        capacity,  # capacity array
        True,
        'Capacity')
    # Add a dimension for time and a limit on the total time_horizon
    routing.AddDimension(
        tot_time_fn_index,  # total time function callback
        customers.time_horizon,
        customers.time_horizon,
        True,
        'Time')

    time_dimension = routing.GetDimensionOrDie('Time')
    for cust in customers.customers:
        if cust.tw_open is not None:
            time_dimension.CumulVar(manager.NodeToIndex(cust.index)).SetRange(
                cust.tw_open.seconds, cust.tw_close.seconds)
    """
     To allow the dropping of orders, we add disjunctions to all the customer
    nodes. Each disjunction is a list of 1 index, which allows that customer to
    be active or not, with a penalty if not. The penalty should be larger
    than the cost of servicing that customer, or it will always be dropped!
    """
    # To add disjunctions just to the customers, make a list of non-depots.
    non_depot = set(range(customers.number))
    non_depot.difference_update(vehicles.starts)
    non_depot.difference_update(vehicles.ends)
    penalty = 4000  # The cost for dropping a node from the plan.
    nodes = [routing.AddDisjunction([manager.NodeToIndex(c)], penalty) for c in non_depot]

    # This is how you would implement partial routes if you already knew part
    # of a feasible solution for example:
    # partial = np.random.choice(list(non_depot), size=(4,5), replace=False)

    # routing.CloseModel()
    # partial_list = [partial[0,:].tolist(),
    #                 partial[1,:].tolist(),
    #                 partial[2,:].tolist(),
    #                 partial[3,:].tolist(),
    #                 [],[],[],[]]
    # print(routing.ApplyLocksToAllVehicles(partial_list, False))

    # Solve the problem !
    assignment = routing.SolveWithParameters(parameters)

    # The rest is all optional for saving, printing or plotting the solution.
    if assignment:
        ## save the assignment, (Google Protobuf format)
        #save_file_base = os.path.realpath(__file__).split('.')[0]
        #if routing.WriteAssignment(save_file_base + '_assignment.ass'):
        #    print('succesfully wrote assignment to file ' + save_file_base +
        #          '_assignment.ass')

        print('The Objective Value is {0}'.format(assignment.ObjectiveValue()))

        plan_output, dropped = vehicle_output_string(manager, routing, assignment, customers)
        print(plan_output)
        print('dropped nodes: ' + ', '.join(dropped))

        # you could print debug information like this:
        # print(routing.DebugOutputAssignment(assignment, 'Capacity'))

        vehicle_routes = {}
        for veh in range(vehicles.number):
            vehicle_routes[veh] = build_vehicle_route(manager, routing, assignment,
                                                      customers, veh)

        # make map of routes
        nyc_map = folium.Map(location=[40.7128, -74.0060], zoom_start=11)
        colors = ['red', 'green', 'blue']

        for i in range(len(vehicle_routes)):
            stop = 0
            for j in vehicle_routes[i]:
                if j.index != 0:
                    stop+=1
                    folium.Marker(location=[j.lat,j.lon], popup=j.name+" "+str(stop), icon=folium.Icon(color=colors[i])).add_to(nyc_map)

        # add the redish location
        # 40.69998097056833, -73.94819913666936
        folium.Marker(location=[40.69998097056833, -73.94819913666936], popup="Re-dish",
                      icon=folium.Icon(color='black')).add_to(nyc_map)
        nyc_map.save("map.html")

        # Plotting of the routes in matplotlib.
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # Plot all the nodes as black dots.
        clon, clat = zip(*[(c.lon, c.lat) for c in customers.customers])
        ax.plot(clon, clat, 'k.')
        # plot the routes as arrows
        plot_vehicle_routes(vehicle_routes, ax, customers, vehicles)
        plt.show()

    else:
        print('No assignment')


if __name__ == '__main__':
    main()
