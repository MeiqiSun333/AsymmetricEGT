# main file

import random
import numpy as np
import networkx as nx
from mesa import Model, Agent
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import multiprocessing
import os
import json
import time
from experiment import *
from experiment_stratch import *


class Player(Agent):
    def __init__(self, unique_id, model, pos, config, eta=None, ration=None):
        self.unique_id = unique_id
        self.model = model
        self.pos = pos
        self.U = np.random.uniform(low=config.U_distribution[0], high=config.U_distribution[1])
        self.V = np.random.uniform(low=config.V_distribution[0], high=config.V_distribution[1])
        self.wealth = config.wealth
        self.recent_wealth = config.recent_wealth
        self.eta = eta if eta is not None else np.random.uniform(low=config.eta_distribution[0], high=config.eta_distribution[1])
        self.ration = ration if ration is not None else np.random.lognormal(mean=config.ration_distribution[0], sigma=config.ration_distribution[1])
        self.V_with_belief = 0
        self.U_with_belief = 0
        self.history = np.zeros(5, dtype=float)
        self.has_played = False

        # parameters for belief
        self.alpha_epsilon = config.alpha_epsilon

        # parameters for UV update
        self.alpha_uv = config.alpha_uv
        self.beta_uv = config.beta_uv

        # parameters for rewiring
        self.rewiring_prob = config.rewiring_prob
        self.alpha_con = config.alpha_con
        self.beta_con = config.beta_con
        self.epsilon = config.epsilon


    def choose_neighbor(self):
        neighbors = self.model.adjacency[self.pos]  # set of neighbors
        candidates = []
        for node_id in neighbors:
            ag = self.model.id_to_agent[node_id]
            if not ag.has_played:
                candidates.append(ag)
        return random.choice(candidates) if candidates else None


    def add_belief(self, other):
        dv = self.alpha_epsilon * (self.V - other.V)
        p_epsilon_v = (np.exp(dv) - np.exp(-dv)) / (np.exp(dv) + np.exp(-dv))
        self.V_with_belief = self.V - p_epsilon_v * self.V

        du = self.alpha_epsilon * (self.U - other.U)
        p_epsilon_u = (np.exp(du) - np.exp(-du)) / (np.exp(du) + np.exp(-du))
        self.U_with_belief = self.U - p_epsilon_u * self.U


    @staticmethod
    def indicator_function(x, A):
        return 1 if x > A else 0


    def choose_strategy(self, other, p_c1=0.5, p_c2=0.5, max_iter=5, tol=0.5):
        for _ in range(max_iter):
            old_c1, old_c2 = p_c1, p_c2

            # best-response to old_c2
            ec = np.exp(self.ration * (old_c2 + (1 - old_c2) * self.U_with_belief))
            ed = np.exp(self.ration * (old_c2 * self.V_with_belief))
            new_c1 = ec / (ec + ed)

            # best-response to old_c1
            ec2 = np.exp(other.ration * (old_c1 + (1 - old_c1) * other.V_with_belief))
            ed2 = np.exp(other.ration * (old_c1 * other.U_with_belief))
            new_c2 = ec2 / (ec2 + ed2)

            delta = abs(new_c1 - old_c1) + abs(new_c2 - old_c2)
            p_c1, p_c2 = new_c1, new_c2

            if delta < tol: break

        r1 = random.random()
        r2 = random.random()

        if r1 <= p_c1 and r2 <= p_c2:
            E1, E2 = 1, 1
        elif r1 <= p_c1 and r2 > p_c2:
            E1, E2 = self.U_with_belief, other.V_with_belief
        elif r1 > p_c1 and r2 <= p_c2:
            E1, E2 = self.V_with_belief, other.U_with_belief
        else:
            E1, E2 = 0, 0

        return E1, E2


    def update_wealth(self, payoff):
        np.roll(self.history, -1)
        self.history[-1] = payoff
        self.wealth = self.wealth * (1.0 + self.model.discount) + payoff

        # force wealth equal to 0 if negative
        if self.wealth < 0:
            self.wealth = 0

        discount_factors = (1.0 + self.model.discount) ** np.arange(5)[::-1]
        self.recent_wealth = np.dot(self.history, discount_factors)


    def update_payoff_mx(self, other):
        indicator = Player.indicator_function(other.wealth, self.wealth)
        diff_wealth = (other.wealth - self.wealth)
        denom = np.tanh(self.beta_uv * diff_wealth)
        dV = self.eta * indicator * (other.V - self.V) * denom
        dU = self.eta * indicator * (other.U - self.U) * denom
        self.V += dV
        self.U += dU
        return self.V, self.U


    def rewire_connection(self, other):
        if self.unique_id > other.unique_id:
            return

        if random.random() < self.rewiring_prob:
            # remove edge
            if other.pos in self.model.adjacency[self.pos]:
                self.model.adjacency[self.pos].remove(other.pos)
            if self.pos in self.model.adjacency[other.pos]:
                self.model.adjacency[other.pos].remove(self.pos)

            possible_neighbors = set(self.model.adjacency[other.pos]) - {self.pos}

            # remove nodes that are already the neighbor
            possible_new_neighbors = [
                nbr for nbr in possible_neighbors
                if nbr not in self.model.adjacency[self.pos]
            ]

            rewired = False
            if possible_new_neighbors:
                for new_neighbor_pos in possible_new_neighbors:
                    new_agent = self.model.id_to_agent[new_neighbor_pos]
                    temp_value = self.beta_con * max(abs(self.wealth - new_agent.wealth), self.epsilon)
                    p_con = 0.0
                    if temp_value < 1.0:
                        p_con = 1.0 / (1.0 + (1.0 - temp_value) ** (-self.alpha_con))

                    if random.random() < p_con:
                        self.model.adjacency[self.pos].add(new_neighbor_pos)
                        self.model.adjacency[new_neighbor_pos].add(self.pos)
                        rewired = True
                        break

            if not rewired:
                self.model.adjacency[self.pos].add(other.pos)
                self.model.adjacency[other.pos].add(self.pos)
        else:
            pass

    def reset(self):
        self.has_played = False

    def step(self):
        other = self.choose_neighbor()
        if other:
            self.add_belief(other)
            other.add_belief(self)
            my_payoff, other_payoff = self.choose_strategy(other)
            self.update_wealth(my_payoff)
            other.update_wealth(other_payoff)
            self.update_payoff_mx(other)
            self.has_played = True
            other.has_played = True
            self.rewire_connection(other)

        self.reset()


class NetworkModel(Model):
    def __init__(self, config):
        self.config = config
        self.discount = config.discount
        self.random = random.Random()
        self.schedule = RandomActivation(self)

        self.G = self._create_network_nx()
        self.adjacency = self._build_adjacency_dict(self.G)

        self.steps = 0

        self.id_to_agent = {}

        for i, node in enumerate(self.G.nodes()):
            agent = Player(i, self, node, config)
            self.id_to_agent[node] = agent
            self.schedule.add(agent)

        self.datacollector = DataCollector(
            model_reporters={
                "Gini": lambda m: NetworkModel.compute_gini([a.wealth for a in m.schedule.agents]),
                "Average Degree": lambda m: NetworkModel.avg_degree(m),
                "Clustering Coefficient": lambda m: nx.average_clustering(m.G),
                "Average Path Length": lambda m: NetworkModel.compute_average_path_length(m.G),
                "Wealth Distribution": lambda m: [a.wealth for a in m.schedule.agents]
            },
            agent_reporters={
                "Wealth": "wealth",
                "Recent Wealth": "recent_wealth",
                "Risk Aversion": "eta",
                "Rationality": "ration",
                "U": "U",
                "V": "V"
            }
        )

    def _create_network_nx(self):
        if self.config.network_type == 'watts-strogatz':
            return nx.watts_strogatz_graph(self.config.num_agents, k=self.config.k, p=self.config.p)
        elif self.config.network_type == 'scale-free':
            return nx.barabasi_albert_graph(self.config.num_agents, m=self.config.m)
        else:
            return nx.random_regular_graph(d=self.config.d, n=self.config.num_agents)


    def _build_adjacency_dict(self, G):
        adjacency = {}
        for node in G.nodes():
            adjacency[node] = set()
        for u, v in G.edges():
            adjacency[u].add(v)
            adjacency[v].add(u)
        return adjacency


    @staticmethod
    def compute_gini(wealths):
        wealths = np.array(wealths, dtype=float)
        n = len(wealths)
        if n == 0 or np.sum(wealths) == 0:
            return 0.0
        sorted_wealths = np.sort(wealths)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_wealths) / (n * np.sum(sorted_wealths))) - (n + 1) / n
        return gini


    @staticmethod
    def compute_average_path_length(G):
        if nx.is_connected(G):
            return nx.average_shortest_path_length(G)
        else:
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            return nx.average_shortest_path_length(subgraph)


    @staticmethod
    def avg_degree(model):
        total_edges = 0
        for node, neighbors in model.adjacency.items():
            total_edges += len(neighbors)
        return total_edges / len(model.adjacency)


    def step(self):
        self.steps += 1

        for agent in self.schedule.agents:
            agent.reset()

        self.schedule.step()

        self._sync_to_networkx()

        if self.steps % 10 == 0:
            self.datacollector.collect(self)


    def _sync_to_networkx(self):
        self.G.clear()
        self.G.add_nodes_from(self.adjacency.keys())
        for node, neighbors in self.adjacency.items():
            for nbr in neighbors:
                self.G.add_edge(node, nbr)


    def run_model(self, steps):
        for _ in range(steps):
            self.step()


class DefaultConfig:
    def __init__(self):
        self.num_agents = 300
        self.num_steps = 480

        # attributes
        self.U_distribution = (-2, 2)
        self.V_distribution = (-2, 2)
        self.wealth = 0
        self.recent_wealth = 0
        self.eta_distribution = (0, 1)
        self.ration_distribution = (0, 1)

        # parameters for belief
        self.alpha_epsilon = 0.5

        # parameters for UV update
        self.alpha_uv = 0.5
        self.beta_uv = 0.5

        # parameters for rewiring
        self.rewiring_prob = 0.6
        self.alpha_con = 0.5
        self.beta_con = 0.5
        self.epsilon = 0.005

        # model
        self.discount = 0

        # network
        self.network_type = 'scale-free'
        self.k = 6
        self.p = 0.1
        # scale-free
        self.m = 3
        # random regular
        self.d = 6

    def update_parameters(self, params):
        for key, value in params.items():
            setattr(self, key, value)


def _parallel_run(args):

    experiment_id, network_type, avg_degree, rewiring_prob, num_steps, results_dir = args
    if experiment_id == 1:
        res = run_experiment1(
            network_type=network_type,
            num_steps=num_steps,
            avg_degree=avg_degree,
            rewiring_prob=rewiring_prob
        )
        file_name = f"{network_type}_degree_{avg_degree}.json"

    elif experiment_id == 2:
        res = run_experiment2(num_steps=num_steps, repetitions=10, distinct_samples=16)
        file_name = "experiment2_sobol.json"

    elif experiment_id == 3:
        res = run_experiment_extreme_check(
            network_type=network_type,
            num_steps=num_steps,
            avg_degree=avg_degree,
            rewiring_prob=rewiring_prob
        )
        file_name = f"{network_type}_deg_{avg_degree}_rew_{rewiring_prob}.json"


    file_path = os.path.join(results_dir, file_name)
    with open(file_path, 'w') as f:
        json.dump(res, f, indent=4)
    print(f"Saved results for experiment {experiment_id}")


def main():
    num_steps = 480

    # experiment 1: network structure with no rewiring
    results_dir1 = "experiment1"
    experiment_id = 1
    os.makedirs(results_dir1, exist_ok=True)
    tasks1 = []
    for network_type in ['watts-strogatz', 'scale-free', 'regular']:
        tasks1.append((experiment_id, network_type, 8, 0, num_steps, results_dir1))

    # experiment 2
    results_dir2 = "experiment_multi"
    experiment_id = 2
    os.makedirs(results_dir2, exist_ok=True)
    tasks2 = [(experiment_id, '', 0, 0, num_steps, results_dir2)]


    # experiment extre: check the params of top rich
    results_dir3 = "experiment_extre"
    experiment_id = 3
    os.makedirs(results_dir3, exist_ok=True)
    tasks3 = []
    deg = 8
    rew = 0
    for net in ['watts-strogatz', 'scale-free', 'regular']:
        tasks3.append((experiment_id, net, deg, rew, num_steps, results_dir3))


    # tasks = tasks1 + tasks2
    with multiprocessing.Pool() as pool:
        pool.map(_parallel_run, tasks1)

    # for task in tasks1:
    #     _parallel_run(task)


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print("Total time taken:", end_time - start_time)
