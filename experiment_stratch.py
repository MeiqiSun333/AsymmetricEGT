from main_parallel import *
import math
import random
import os, json
import numpy as np
from statistics import mean
import matplotlib.pyplot as plt
from SALib.sample import saltelli
from SALib.analyze import sobol
from itertools import combinations
from functools import partial
from multiprocessing import Pool


def plot_index(s, params, i, title=''):
    if i == '2':
        p = len(params)
        params = list(combinations(params, 2))
        indices = s['S2'].reshape((p ** 2))
        indices = indices[~np.isnan(indices)]
        errors = s['S2_conf'].reshape((p ** 2))
        errors = errors[~np.isnan(errors)]
    else:
        indices = s['S' + i]
        errors = s['S' + i + '_conf']
        plt.figure()

    l = len(indices)
    plt.title(title)
    plt.ylim([-0.2, l - 1 + 0.2])
    plt.yticks(range(l), params)
    plt.errorbar(indices, range(l), xerr=errors, linestyle='None', marker='o')
    plt.axvline(0, c='k')


def run_experiment2(num_steps, repetitions=5, distinct_samples=16, out_dir="experiment_sobol", parallel=True):
    os.makedirs(out_dir, exist_ok=True)
    problem = {
        'num_vars': 8,
        'names': [
            'network_type_index',
            'avg_degree',
            'rewiring_prob',
            'alpha_epsilon',
            'beta_uv',
            'alpha_con',
            'beta_con',
            'discount'
        ],
        'bounds': [
            [0, 2],  # network_type_index
            [4, 12],  # avg_degree
            [0, 1],  # rewiring_prob
            [0, 1],  # alpha_epsilon
            [0, 1],  # beta_uv
            [0, 1],  # alpha_con
            [0, 1],  # beta_con
            [0, 0.005]  # discount
        ]
    }

    param_values = saltelli.sample(problem, distinct_samples, calc_second_order=True)

    param_list = param_values.tolist()
    param_file = os.path.join(out_dir, "sobol_params.json")
    with open(param_file, 'w') as f:
        json.dump(param_list, f, indent=2)

    # if parallel:
    pool = multiprocessing.Pool()
    func = partial(_simulate_param_set, num_steps=num_steps, repetitions=repetitions)
    Y = pool.map(func, param_list)
    pool.close();
    pool.join()
    # else:
    #     Y = []
    #     for pset in param_list:
    #         yval = _simulate_param_set(pset, num_steps=num_steps, repetitions=repetitions)
    #         Y.append(yval)

    Y = np.array(Y)
    results_file = os.path.join(out_dir, "sobol_outputs.json")
    with open(results_file, 'w') as f:
        json.dump(Y.tolist(), f)
    si = sobol.analyze(problem, Y, calc_second_order=True)
    si_file = os.path.join(out_dir, "sobol_indices.json")
    si_dump = {}
    for k, v in si.items():
        if isinstance(v, np.ndarray):
            si_dump[k] = v.tolist()
        else:
            si_dump[k] = v
    with open(si_file, 'w') as f:
        json.dump(si_dump, f, indent=2)

    plot_index(si, problem['names'], '1', 'First order sensitivity')
    plt.savefig("first_order.png", dpi=150)
    plot_index(si, problem['names'], '2', 'Second order sensitivity')
    plt.savefig("second_order.png", dpi=150)
    plot_index(si, problem['names'], 'T', 'Total sensitivity')
    plt.savefig("total_order.png", dpi=150)

    print("All tasks done.")


def _simulate_param_set(param_set, num_steps=480, repetitions=5):

    network_types = ['watts-strogatz', 'scale-free', 'regular']
    net_idx_f, avg_deg_f, rew_p, alpha_e, beta_uv, alpha_c, beta_c, disc = param_set

    net_idx = int(round(net_idx_f))
    if net_idx < 0: net_idx = 0
    if net_idx > 2: net_idx = 2
    net_type = network_types[net_idx]
    possible_degs = [4, 6, 8, 10, 12]
    deg_raw = int(round(avg_deg_f))
    avg_deg = min(possible_degs, key=lambda x: abs(x - deg_raw))

    final_gini = run_one_simulation(
        network_type=net_type,
        num_steps=num_steps,
        avg_degree=avg_deg,
        rewiring_prob=rew_p,
        repetitions=repetitions,
        alpha_epsilon=alpha_e,
        beta_uv=beta_uv,
        alpha_con=alpha_c,
        beta_con=beta_c,
        discount=disc
    )
    return final_gini


def run_one_simulation(network_type, num_steps, avg_degree, rewiring_prob, repetitions=5, **kwargs):

    final_ginis = []
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
        for step in range(num_steps):
            model.step()

        final_g = model.compute_gini([ag.wealth for ag in model.schedule.agents])
        final_ginis.append(final_g)

    return mean(final_ginis)


def run_experiment_extreme_check(network_type, num_steps, avg_degree, rewiring_prob=0, repetitions=30, extreme_threshold=1400, **kwargs):
    results = []

    for rep_i in range(repetitions):
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
        for step in range(num_steps):
            model.step()

        model._sync_to_networkx()

        agent_list = model.schedule.agents
        extreme_agents = []
        for ag in agent_list:
            if ag.wealth > extreme_threshold:
                node_id = ag.pos
                # Check if the node degree can be fetched without error
                node_deg = len(model.G[node_id]) if node_id in model.G else 0
                # Fetch clustering for node if possible
                node_clust = nx.clustering(model.G, node_id) if node_id in model.G else 0
                # Calculate shortest path length if the graph is not empty
                spl = nx.single_source_shortest_path_length(model.G, node_id)
                extreme_agents.append({
                    "agent_id": ag.unique_id,
                    "wealth": ag.wealth,
                    "U": ag.U,
                    "V": ag.V,
                    "eta": ag.eta,
                    "ration": ag.ration,
                    "recent_wealth": ag.recent_wealth,
                    "deg": node_deg,
                    "cluster": node_clust,
                    "path_length": max(spl.values()) if spl else 0
                })

        results.extend(extreme_agents)  # Append each extreme agent directly to results

    return results

def run_experiment_increase_degree(network_type, num_steps=600, avg_degree=8, rewiring_prob=0.6, low_degree_threshold=10, pick_count=30):

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

    model = NetworkModel(config)

    # 准备存 wealth_ts[step][agent_id]
    wealth_ts = [None]*(num_steps+1)  # 0..num_steps inclusive if you want final
    for s in range(num_steps+1):
        wealth_ts[s] = [0]*config.num_agents

    # run
    for step in range(num_steps+1):
        if step>0:
            model.step()

        # record wealth for all
        for ag in model.schedule.agents:
            wealth_ts[step][ag.unique_id] = ag.wealth


        if step==100:

            model._sync_to_networkx()
            degs = [(node, len(model.G[node])) for node in model.G.nodes()]

            low_deg_nodes = [nd for (nd,dv) in degs if dv<low_degree_threshold]

            if len(low_deg_nodes)>pick_count:
                boosted_nodes = random.sample(low_deg_nodes, pick_count)
            else:
                boosted_nodes = low_deg_nodes

            needed_sum = 0
            needed_dict = {}
            for nd in boosted_nodes:
                cur_deg = len(model.G[nd])
                needed = 30 - cur_deg
                if needed<0:
                    needed=0
                needed_sum += needed
                needed_dict[nd]=needed

            candidate_edges = []
            for (u,v) in model.G.edges():
                if (u not in boosted_nodes) and (v not in boosted_nodes):
                    candidate_edges.append((u,v))

            removed_edges = random.sample(candidate_edges, needed_sum) if len(candidate_edges)>=needed_sum else candidate_edges

            # remove them
            for (u,v) in removed_edges:
                model.adjacency[u].discard(v)
                model.adjacency[v].discard(u)

            for nd in boosted_nodes:
                needed = needed_dict[nd]
                if needed>0:
                    possible_new = set(model.adjacency.keys()) - model.adjacency[nd] - {nd}
                    add_partners = random.sample(list(possible_new), needed) if len(possible_new) >= needed else list(
                        possible_new)
                    for partner in add_partners:
                        model.adjacency[nd].add(partner)
                        model.adjacency[partner].add(nd)

            model._sync_to_networkx()


    final_boosted = boosted_nodes if step>=100 else []

    series = []
    for s in range(num_steps+1):
        if final_boosted:
            wvals = [wealth_ts[s][nd] for nd in final_boosted]
            avg_w = sum(wvals)/len(wvals)
        else:
            avg_w=0
        series.append(avg_w)

    return series

if __name__ == "__main__":
    run_experiment2(num_steps=480)




