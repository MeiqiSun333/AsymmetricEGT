from mesa import Model, Agent
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import random
import numpy as np
import networkx as nx


class Player(Agent):
    def __init__(self, unique_id, model, pos, config):
        self.unique_id = unique_id
        self.model = model
        # super().__init__()
        self.pos = pos
        self.U = np.random.uniform(low=config.U_distribution[0], high=config.U_distribution[1])
        self.V = np.random.uniform(low=config.V_distribution[0], high=config.V_distribution[1])
        self.wealth = config.wealth
        self.recent_wealth = config.recent_wealth
        self.eta = np.random.normal(loc=config.eta_distribution[0], scale=config.eta_distribution[1])
        self.ration = np.random.lognormal(mean=config.ration_distribution[0], sigma=config.ration_distribution[1])
        self.V_with_belief = 0
        self.U_with_belief = 0
        self.history = np.zeros(5)
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
        neighbors_nodes = list(self.model.G.neighbors(self.pos))
        candidates = []
        for nn in neighbors_nodes:
            other_agent = self.model.G.nodes[nn].get("agent", None)
            if other_agent and (not other_agent.has_played):
                candidates.append(other_agent)
        if candidates:
            return self.random.choice(candidates)
        else:
            return None


    def add_belief(self, other):
        z_alpha_v = self.alpha_epsilon * (self.V - other.V)
        p_epsilon_v = np.tanh(z_alpha_v)
        z_alpha_u = self.alpha_epsilon * (self.U - other.U)
        p_epsilon_u = np.tanh(z_alpha_u)
        self.V_with_belief = self.V - p_epsilon_v * self.V
        self.U_with_belief = self.U - p_epsilon_u * self.U

    @staticmethod
    def indicator_function(x, A):
        return 1 if x > A else 0

    def compute_probabilities(self, own_payoff_c, own_payoff_d, ration):
        exp_c = np.exp(ration * own_payoff_c)
        exp_d = np.exp(ration * own_payoff_d)
        total = exp_c + exp_d
        return exp_c / total, exp_d / total

    def choose_strategy(self, other):
        p1_C_if_C, p1_D_if_C = self.compute_probabilities(1, self.V_with_belief, self.ration)
        p1_C_if_D, p1_D_if_D = self.compute_probabilities(self.U_with_belief, 0, self.ration)

        p2_C_if_C, p2_D_if_C = other.compute_probabilities(1, other.V_with_belief, other.ration)
        p2_C_if_D, p2_D_if_D = other.compute_probabilities(other.U_with_belief, 0, other.ration)

        joint_CC = p1_C_if_C * p2_C_if_C
        joint_CD = p1_C_if_D * p2_D_if_C
        joint_DC = p1_D_if_C * p2_C_if_D
        joint_DD = p1_D_if_D * p2_D_if_D

        strategies = ['CC', 'CD', 'DC', 'DD']
        probabilities = [joint_CC, joint_CD, joint_DC, joint_DD]
        probabilities = np.array(probabilities) / np.sum(probabilities)
        chosen_combination = np.random.choice(strategies, p=probabilities)

        my_payoff_matrix = np.array([[1, self.U_with_belief], [self.V_with_belief, 0]])
        other_payoff_matrix = np.array([[1, other.U_with_belief], [other.V_with_belief, 0]])
        if chosen_combination == 'CC':
            my_payoff = my_payoff_matrix[0, 0]
            other_payoff = other_payoff_matrix[0, 0]
        elif chosen_combination == 'CD':
            my_payoff = my_payoff_matrix[0, 1]
            other_payoff = other_payoff_matrix[0, 1]
        elif chosen_combination == 'DC':
            my_payoff = my_payoff_matrix[1, 0]
            other_payoff = other_payoff_matrix[1, 0]
        else:
            my_payoff = my_payoff_matrix[1, 1]
            other_payoff = other_payoff_matrix[1, 1]

        return chosen_combination, my_payoff, other_payoff

    def update_wealth(self, payoff):
        self.history = np.roll(self.history, -1)
        self.history[-1] = payoff
        self.wealth = self.wealth * (1 + self.model.discount) + payoff
        if len(self.history) < 5:
            self.recent_wealth = np.dot(self.history, (1 + self.model.discount) ** np.arange(len(self.history))[::-1])
        else:
            self.recent_wealth = np.dot(self.history, (1 + self.model.discount) ** np.arange(5)[::-1])

    def update_payoff_mx(self, other):
        z_uv = self.eta * (other.recent_wealth - self.recent_wealth)
        p_uv = np.tanh(z_uv)
        if random.random() < p_uv:
            indicator = Player.indicator_function(other.wealth, self.wealth)
            self.V += self.alpha_uv * (other.V - self.V) / (1 + np.exp(-self.beta_uv * indicator * (other.wealth - self.wealth)))
            self.U += self.alpha_uv * (other.U - self.U) / (1 + np.exp(-self.beta_uv * indicator * (other.wealth - self.wealth)))
            return self.V, self.U

    def rewire_connection(self, other):
        if random.random() < self.rewiring_prob:
            self.model.G.remove_edge(self.pos, other.pos)
            possible_neighbors = set(self.model.G.neighbors(other.pos)) - {self.pos}
            if possible_neighbors:
                for new_neighbor_pos in list(possible_neighbors):
                    new_neighbor = self.model.G.nodes[new_neighbor_pos]["agent"]
                    temp_value = self.beta_con * max(np.abs(self.wealth - new_neighbor.wealth), self.epsilon)
                    if temp_value < 1:
                        p_con = 1 / (1 + (1 - temp_value) ** (-self.alpha_con))
                    else:
                        p_con = 0
                    if random.random() < p_con:
                        self.model.G.add_edge(self.pos, new_neighbor_pos)
                        break
                else:
                    # All possible second neighbors were not connected successfully, randomly selected another node
                    random_neighbor = random.choice(list(set(self.model.G.nodes) - {self.pos}))
            else:
                # No possible second neighbors, randomly selected another node
                random_neighbor = random.choice(list(set(self.model.G.nodes) - {self.pos}))
                self.model.G.add_edge(self.pos, random_neighbor)

    def reset(self):
            self.has_played = False

    def step(self):
        other = self.choose_neighbor()
        if other:
            self.add_belief(other)
            chosen_strategy, my_payoff, other_payoff = Player.choose_strategy(self, other)
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
        self.G = self.create_network()
        self.steps = 0
        self.datacollector = DataCollector(
            model_reporters={
                "Gini": lambda m: NetworkModel.compute_gini([a.wealth for a in m.schedule.agents]),
                "Average Degree": lambda m: np.mean([d for n, d in m.G.degree()]),
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

        for i, node in enumerate(self.G.nodes()):
            agent = Player(i, self, node, config)
            self.schedule.add(agent)
            self.G.nodes[node]["agent"] = agent

    @staticmethod
    def compute_gini(wealths):
        wealths = np.array(wealths)
        n = len(wealths)
        if n == 0 or np.sum(wealths) == 0:
            return 0
        sorted_wealths = np.sort(wealths)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_wealths) / (n * np.sum(sorted_wealths))) - (n + 1) / n
        return gini

    @staticmethod
    def compute_average_path_length(G):
        if nx.is_connected(G):
            return nx.average_shortest_path_length(G)
        else:
            # Calculate the avg path length of the largest connected subgraph
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            return nx.average_shortest_path_length(subgraph)


    def create_network(self):
        if self.config.network_type == 'watts-strogatz':
            return nx.watts_strogatz_graph(self.config.num_agents, k=self.config.k, p=self.config.p)
        elif self.config.network_type == 'scale-free':
            return nx.barabasi_albert_graph(self.config.num_agents, m=self.config.m)
        else:
            return nx.random_regular_graph(d=self.config.d, n=self.config.num_agents)

    def step(self):
        self.steps += 1
        for agent in self.schedule.agents:
            agent.reset()
        self.datacollector.collect(self)
        self.schedule.step()


    def run_model(self, steps):
        for i in range(steps):
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
        self.eta_distribution = (1, 0.5)
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
        self.discount = 0.05

        #network
        self.network_type = 'watts-strogatz'
        # watts-strogatz
        self.k = 6
        self.p = 0.1
        # scale-free
        self.m = 3
        # regular random
        self.d = 6

    # network parameters update
    def update_parameters(self, params):
        for key, value in params.items():
            setattr(self, key, value)