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
        neighbors = list(self.model.G.neighbors(self.pos))
        candidates = []
        for node_id in neighbors:
            ag = self.model.G.nodes[node_id].get("agent", None)
            if ag is not None and (not ag.has_played):
                candidates.append(ag)
        if candidates:
            return random.choice(candidates)
        else:
            return None

    def add_belief(self, other):
        dv = self.alpha_epsilon * (self.V - other.V)
        p_epsilon_v = np.tanh(dv)
        self.V_with_belief = self.V - p_epsilon_v * self.V

        du = self.alpha_epsilon * (self.U - other.U)
        p_epsilon_u = np.tanh(du)
        self.U_with_belief = self.U - p_epsilon_u * self.U

    @staticmethod
    def indicator_function(x, A):
        return 1 if x > A else 0

    def choose_strategy(self, other, p_c1 = 0.5, p_c2 = 0.5, max_iter=50, tol=1e-5):

        def safe_exp(x, clip=40.0):
            if x > clip:
                return np.exp(clip)
            elif x < -clip:
                return np.exp(-clip)
            else:
                return np.exp(x)


        for _ in range(max_iter):
            old_p_c1 = p_c1
            old_p_c2 = p_c2

            ec = safe_exp(self.ration * (old_p_c2 + (1 - old_p_c2) * self.U_with_belief))
            ed = safe_exp(self.ration * (old_p_c2 * self.V_with_belief))
            new_p_c1 = ec / (ec + ed)

            ec2 = safe_exp(other.ration * (p_c1 + (1 - p_c1) * other.V_with_belief))
            ed2 = safe_exp(other.ration * (p_c1 * other.U_with_belief))
            new_p_c2 = ec2 / (ec2 + ed2)

            delta = abs(old_p_c1 - new_p_c1) + abs(old_p_c2 - new_p_c2)
            p_c1, p_c2 = new_p_c1, new_p_c2
            if delta < tol:
                break

        E1 = p_c1 * p_c2 + p_c1 * (1 - p_c2) * self.U_with_belief + (1 - p_c1) * p_c2 * self.V_with_belief
        E2 = p_c1 * p_c2 + p_c1 * (1 - p_c2) * other.V_with_belief + (1 - p_c1) * p_c2 * other.U_with_belief

        return E1, E2


    def update_wealth(self, payoff):
        np.roll(self.history, -1)
        self.history[-1] = payoff
        self.wealth = self.wealth * (1.0 + self.model.discount) + payoff

        discount_factors = (1.0 + self.model.discount) ** np.arange(5)[::-1]
        self.recent_wealth = np.dot(self.history, discount_factors)

    def update_payoff_mx(self, other):
        z_uv = self.eta * (other.recent_wealth - self.recent_wealth)
        p_uv = np.tanh(z_uv)
        if random.random() < p_uv:
            indicator = Player.indicator_function(other.wealth, self.wealth)
            diff_wealth = (other.wealth - self.wealth)
            denom = (1.0 + np.exp(-self.beta_uv * indicator * diff_wealth))
            dV = self.alpha_uv * (other.V - self.V) / denom
            dU = self.alpha_uv * (other.U - self.U) / denom
            self.V += dV
            self.U += dU
            return self.V, self.U

    def rewire_connection(self, other):
        if random.random() < self.rewiring_prob:
            # remove edge
            self.model.G.remove_edge(self.pos, other.pos)
            # gather possible neighbors
            possible_neighbors = set(self.model.G.neighbors(other.pos)) - {self.pos}
            done = False
            if possible_neighbors:
                for new_neighbor_pos in possible_neighbors:
                    new_agent = self.model.G.nodes[new_neighbor_pos]["agent"]
                    temp_value = self.beta_con * max(abs(self.wealth - new_agent.wealth), self.epsilon)
                    p_con = 0.0
                    if temp_value < 1.0:
                        p_con = 1.0 / (1.0 + (1.0 - temp_value) ** (-self.alpha_con))
                    if random.random() < p_con:
                        self.model.G.add_edge(self.pos, new_neighbor_pos)
                        done = True
                        break

            if not done:
                # if not possible or no success => random
                all_nodes = list(self.model.all_node_ids_minus_self[self.pos])
                if all_nodes:
                    rand_node = random.choice(all_nodes)
                    self.model.G.add_edge(self.pos, rand_node)

    def reset(self):
        self.has_played = False

    def step(self):
        other = self.choose_neighbor()
        if other:
            self.add_belief(other)
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
        self.G = self.create_network()
        self.steps = 0

        self.all_node_ids_minus_self = {}
        all_nodes_list = list(self.G.nodes())
        for node_id in all_nodes_list:
            tmp = all_nodes_list.copy()
            tmp.remove(node_id)
            self.all_node_ids_minus_self[node_id] = tmp

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
            agent.model = self
            self.schedule.add(agent)
            self.G.nodes[node]["agent"] = agent

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
        if self.steps%5==0:
            self.datacollector.collect(self)
        # self.datacollector.collect(self)
        self.schedule.step()

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