# from process_files.main_faster import *
from main_parallel import *

def run_experiment(network_type, num_steps, avg_degree, rewiring_prob=0, repetitions=10, **kwargs):
    results = []
    for _ in range(repetitions):
        config = DefaultConfig()
        config.num_agents = 300
        if network_type == 'watts-strogatz':
            config.network_type = 'watts-strogatz'
            config.k = avg_degree
            config.p = rewiring_prob
        elif network_type == 'scale-free':
            config.network_type = 'scale-free'
            config.m = avg_degree // 2
        elif network_type == 'regular':
            config.network_type = 'regular'
            config.d = avg_degree

        config.rewiring_prob = rewiring_prob

        for key, value in kwargs.items():
            setattr(config, key, value)

        model = NetworkModel(config)
        all_steps_data = []
        for step in range(num_steps):
            model.step()
            if step%10 == 0:
                U_distribution = [agent.U for agent in model.schedule.agents]
                V_distribution = [agent.V for agent in model.schedule.agents]
                wealth_distribution = [agent.wealth for agent in model.schedule.agents]
            else:
                U_distribution = None
                V_distribution = None
                wealth_distribution = None

            if step % 50 == 0:
                model._sync_to_networkx()
                edges_count = model.G.number_of_edges()
                cluster = nx.average_clustering(model.G)
            else:
                edges_count = None
                cluster = None

            step_data = {
                'Step': step,
                'Gini': model.compute_gini([agent.wealth for agent in model.schedule.agents]),
                'U_distribution': U_distribution,
                'V_distribution': V_distribution,
                "Wealth_distribution": wealth_distribution,
                "Edges": edges_count,
                "Clustering": cluster
            }

            all_steps_data.append(step_data)
        results.append(all_steps_data)
    return results

