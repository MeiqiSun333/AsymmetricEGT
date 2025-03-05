from main_parallel import *

def run_experiment1(network_type, num_steps, avg_degree, rewiring_prob=0, repetitions=10, top_k=10, **kwargs):
    results = []
    for _ in range(repetitions):
        config = DefaultConfig()
        config.num_agents = 500
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
            if step % 10 == 0:
                U_distribution = [agent.U for agent in model.schedule.agents]
                V_distribution = [agent.V for agent in model.schedule.agents]
                wealth_distribution = [agent.wealth for agent in model.schedule.agents]
                sorted_nodes = sorted(range(config.num_agents),
                                      key=lambda i: wealth_distribution[i],
                                      reverse=True)
                top_nodes = sorted_nodes[:top_k]

                top_rich_uv = []
                for node in top_nodes:
                    top_rich_uv.append({
                        "node_id": node,
                        "wealth": wealth_distribution[node],
                        "U": U_distribution[node],
                        "V": V_distribution[node]
                    })

                model._sync_to_networkx()
                edges_list = list(model.G.edges())
                edges_count = model.G.number_of_edges()
                cluster = nx.average_clustering(model.G)
            else:
                U_distribution = None
                V_distribution = None
                wealth_distribution = None
                top_rich_uv = None
                edges_list = None
                edges_count = None
                cluster = None

            step_data = {
                'Step': step,
                'Gini': model.compute_gini([agent.wealth for agent in model.schedule.agents]),
                'U_distribution': U_distribution,
                'V_distribution': V_distribution,
                "Wealth_distribution": wealth_distribution,
                'Top_Rich_UV': top_rich_uv,
                'Edges': edges_list,
                "Edges_count": edges_count,
                "Clustering": cluster
            }

            all_steps_data.append(step_data)
        results.append(all_steps_data)
    return results

