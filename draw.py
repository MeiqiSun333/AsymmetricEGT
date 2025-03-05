# draw plots for report

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt


def read_json_results(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def get_gini_dicts_for_all_reps(data):
    steps = [item['Step'] for item in data[0]]
    steps_list = sorted(steps)

    rep2dict = []
    for rep_i in range(len(data)):
        step_gini_map = {}
        for item in data[rep_i]:
            st = item['Step']
            gi = item['Gini']
            step_gini_map[st] = gi
        rep2dict.append(step_gini_map)

    return rep2dict, steps_list


def compute_diff_stats(rep2dict_A, rep2dict_B, steps_list):

    n_rep = len(rep2dict_A)
    result = {}
    for st in steps_list:
        diffs = []
        for i in range(n_rep):
            gA = rep2dict_A[i].get(st, 0)
            gB = rep2dict_B[i].get(st, 0)
            diffs.append(gA - gB)
        diffs = np.array(diffs)
        mean_diff = diffs.mean()
        std_diff = diffs.std(ddof=1)
        ci = 1.96 * std_diff / sqrt(n_rep)
        lower = mean_diff - ci
        upper = mean_diff + ci
        result[st] = (mean_diff, lower, upper)
    return result


def plot_gini_diffs(results_dir = "experiment1"):

    fig, ax = plt.subplots(figsize=(8, 5))

    file_reg = os.path.join(results_dir, f"regular_degree_8.json")
    file_ws = os.path.join(results_dir, f"watts-strogatz_degree_8.json")
    file_sf = os.path.join(results_dir, f"scale-free_degree_8.json")

    data_reg = read_json_results(file_reg)
    data_ws = read_json_results(file_ws)
    data_sf = read_json_results(file_sf)

    reg_rep2dict, steps_list = get_gini_dicts_for_all_reps(data_reg)
    ws_rep2dict, _ = get_gini_dicts_for_all_reps(data_ws)
    sf_rep2dict, _ = get_gini_dicts_for_all_reps(data_sf)

    ws_diff_stats = compute_diff_stats(ws_rep2dict, reg_rep2dict, steps_list)
    sf_diff_stats = compute_diff_stats(sf_rep2dict, reg_rep2dict, steps_list)

    ws_mean, ws_lo, ws_hi = [], [], []
    sf_mean, sf_lo, sf_hi = [], [], []

    for st in steps_list:
        m1, l1, u1 = ws_diff_stats[st]
        ws_mean.append(m1)
        ws_lo.append(l1)
        ws_hi.append(u1)

        m2, l2, u2 = sf_diff_stats[st]
        sf_mean.append(m2)
        sf_lo.append(l2)
        sf_hi.append(u2)

    ax.plot(steps_list, ws_mean, label='WS - Regular', color='tab:blue')
    ax.fill_between(steps_list, ws_lo, ws_hi, color='tab:blue', alpha=0.2)
    ax.plot(steps_list, sf_mean, label='SF - Regular', color='tab:orange')
    ax.fill_between(steps_list, sf_lo, sf_hi, color='tab:orange', alpha=0.2)

    ax.set_xlabel("Step")
    ax.set_ylabel("Gini difference")
    ax.legend()

    plt.title("Difference in Gini Coefficient")
    plt.tight_layout()
    plt.show()


def compute_avg_wealth_per_step(data):
    n_rep = len(data)
    steps_list = sorted(item["Step"] for item in data[0])

    step_means_map = {st: [] for st in steps_list}

    for rep_i in range(n_rep):
        rep_data = data[rep_i]
        step2wealthDist = {}
        for item in rep_data:
            st = item["Step"]
            wdist = item.get("Wealth_distribution", None)
            if wdist is not None:
                step2wealthDist[st] = wdist

        for st in steps_list:
            if st in step2wealthDist:
                wdist = step2wealthDist[st]
                mean_w = np.mean(wdist)
                # wdist_sorted = np.sort(wdist)
                # top_25_count = max(1, int(len(wdist) * 0.25))  # top 25%, at least 1
                # filtered_wdist = wdist_sorted[-top_25_count:]
                # mean_w = np.mean(filtered_wdist)
            else:
                mean_w = 0
            step_means_map[st].append(mean_w)

    step2finalmean = {}
    for st in steps_list:
        arr = np.array(step_means_map[st])
        step2finalmean[st] = arr.mean()  # across repetition

    return steps_list, step2finalmean


def plot_avg_wealths(results_dir = "experiment1"):
    fig, ax = plt.subplots(figsize=(8, 5))

    file_reg = os.path.join(results_dir, f"regular_degree_8.json")
    file_ws = os.path.join(results_dir, f"watts-strogatz_degree_8.json")
    file_sf = os.path.join(results_dir, f"scale-free_degree_8.json")

    data_reg = read_json_results(file_reg)
    data_ws = read_json_results(file_ws)
    data_sf = read_json_results(file_sf)

    # compute average wealth vs step
    steps_reg, reg_wealth_map = compute_avg_wealth_per_step(data_reg)
    steps_ws, ws_wealth_map = compute_avg_wealth_per_step(data_ws)
    steps_sf, sf_wealth_map = compute_avg_wealth_per_step(data_sf)

    steps_union = sorted(set(steps_reg) | set(steps_ws) | set(steps_sf))

    reg_line = [reg_wealth_map.get(s, 0) for s in steps_union]
    ws_line = [ws_wealth_map.get(s, 0) for s in steps_union]
    sf_line = [sf_wealth_map.get(s, 0) for s in steps_union]

    ax.plot(steps_union, reg_line, label='Regular', color='tab:red')
    ax.plot(steps_union, ws_line, label='WattsStrogatz', color='tab:blue')
    ax.plot(steps_union, sf_line, label='ScaleFree', color='tab:orange')

    ax.set_xlabel("Step")
    ax.set_ylabel("Average Wealth")
    ax.legend()

    plt.title("Avg Degree vs Wealth")
    plt.tight_layout()
    plt.show()


def plot_UV(results_dir = "experiment2", file_name = "scale-free_degree_8.json"):
    file_path = os.path.join(results_dir, file_name)
    data = read_json_results(file_path)

    steps_of_interest = [0, 60, 120, 180, 240, 300, 360, 420, 470]
    # steps_of_interest = [0, 20, 50, 80, 100, 120, 150, 170, 200]
    fig, axes = plt.subplots(3, 3, figsize=(8, 8), sharex=True, sharey=True)

    for i, step_val in enumerate(steps_of_interest):
        r = i // 3
        c = i % 3

        U_all = []
        V_all = []
        for rep_i in range(len(data)):
            step_data = data[rep_i][step_val]
            U_rep = step_data['U_distribution']
            V_rep = step_data['V_distribution']
            U_all.extend(U_rep)  # combine all reps
            V_all.extend(V_rep)

        ax = axes[r][c]

        hist, xedges, yedges = np.histogram2d(U_all, V_all, bins=30, range=[[-2, 2], [-2, 2]])
        # imshow默认 x是cols => U, y是rows => V
        # 但 histogram2d 返回 shape (bins_y, bins_x),
        # 要转置 hist.T if we want X-> horizontal axis, Y->vertical
        ax.imshow(hist.T,
                  origin='lower',
                  extent=[-2, 2, -2, 2],  # U in [-2,2], V in [-2,2]
                  cmap='viridis',
                  aspect='auto')

        ax.set_title(f"Step={step_val}")
        ax.set_xlabel("U")
        ax.set_ylabel("V")

    plt.tight_layout()
    plt.show()


def build_degree_from_edge_list(edge_list, num_nodes=500):
    degrees = [0]*num_nodes
    for (u,v) in edge_list:
        degrees[u]+=1
        degrees[v]+=1
    return degrees


def plot_degree_vs_wealth(results_dir = "experiment2", filename = "scale-free_degree_8.json"):
    file_path = os.path.join(results_dir, filename)
    data = read_json_results(file_path)

    single_run = data[5]

    steps_of_interest = [0, 60, 120, 180, 240, 300, 360, 420, 470]
    fig, axes = plt.subplots(3, 3, figsize=(8,8), sharex=True, sharey=True)
    axes = axes.flatten()

    num_nodes = 500

    for i, step_val in enumerate(steps_of_interest):
        ax = axes[i]

        step_data = None
        for item in single_run:
            if item["Step"] == step_val:
                step_data = item
                break
        if not step_data:
            ax.set_title(f"Step={step_val} NO DATA")
            continue


        edge_list = step_data.get("Edges", None)
        wdist = step_data.get("Wealth_distribution", None)


        degrees = build_degree_from_edge_list(edge_list, num_nodes)

        ax.scatter(degrees, wdist, s=5, alpha=0.6, c='b')
        ax.set_title(f"Step={step_val}")
        ax.set_xlabel("Degree")
        ax.set_ylabel("Wealth")

    plt.tight_layout()
    plt.show()


def plot_U_vs_wealth(results_dir = "experiment2", filename="scale-free_degree_8.json"):
    file_path = os.path.join(results_dir, filename)
    data = read_json_results(file_path)

    single_run = data[0]

    steps_of_interest = [0, 60, 120, 180, 240, 300, 360, 420, 470]

    fig, axes = plt.subplots(3, 3, figsize=(8,8), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, step_val in enumerate(steps_of_interest):
        ax = axes[i]

        step_data = None
        for item in single_run:
            if item["Step"] == step_val:
                step_data = item
                break

        u_dist = step_data.get("U_distribution", None)
        w_dist = step_data.get("Wealth_distribution", None)

        ax.scatter(u_dist, w_dist, s=5, alpha=0.6, c='b')
        ax.set_title(f"Step={step_val}")
        ax.set_xlabel("U")
        ax.set_ylabel("Wealth")

    plt.tight_layout()
    plt.show()


def plot_top_rich_boxplot_U(filename="scale-free_degree_8.json"):

    results_dir = "experiment2"
    file_path = os.path.join(results_dir, filename)
    data = read_json_results(file_path)

    single_run = data[0]

    step2U = {}
    for step_data in single_run:
        st = step_data["Step"]
        top_rich = step_data.get("Top_Rich_UV", None)
        if top_rich is None:
            continue

        U_list = [ item["U"] for item in top_rich ]
        step2U[st] = U_list


    steps_sorted = sorted(step2U.keys())

    data_for_box = []
    labels = []
    for st in steps_sorted:
        data_for_box.append(step2U[st])
        labels.append(st)

    plt.figure(figsize=(10,5))
    plt.boxplot(data_for_box, labels=labels, showfliers=True)
    plt.title("Top Players' U Distribution across steps")
    plt.xlabel("Step")
    plt.ylabel("U")
    plt.show()


def plot_top_rich_mean_U(results_dir = "experiment2", filename="scale-free_degree_8.json"):
    file_path = os.path.join(results_dir, filename)
    data = read_json_results(file_path)

    single_run = data[0]

    step_vals = []
    meanU_vals = []
    stdU_vals = []

    for step_data in single_run:
        st = step_data["Step"]
        top_rich = step_data.get("Top_Rich_UV", None)
        if not top_rich:
            continue
        U_list = [item["U"] for item in top_rich]
        arr = np.array(U_list)
        step_vals.append(st)
        meanU_vals.append(arr.mean())
        stdU_vals.append(arr.std(ddof=1))


    idx_sorted = np.argsort(step_vals)
    step_vals = np.array(step_vals)[idx_sorted]
    meanU_vals = np.array(meanU_vals)[idx_sorted]
    stdU_vals = np.array(stdU_vals)[idx_sorted]

    plt.figure(figsize=(8, 5))
    plt.plot(step_vals, meanU_vals, marker='o')

    plt.fill_between(step_vals, meanU_vals - stdU_vals, meanU_vals + stdU_vals,
                     alpha=0.2, color='tab:blue')

    plt.title("Mean payoff of top players")
    plt.xlabel("Step")
    plt.ylabel("U")
    plt.show()


def plot_final_top10pct_mean_U(results_dir = "experiment2", filename="scale-free_degree_8.json"):
    file_path = os.path.join(results_dir, filename)
    data = read_json_results(file_path)

    single_run = data[0]

    max_step_found = -1
    final_data = None
    for sd in single_run:
        if sd.get("Top_Rich_UV") is not None:
            if sd["Step"] > max_step_found:
                max_step_found = sd["Step"]
                final_data = sd


    final_top_rich_uv = final_data["Top_Rich_UV"]

    top_ids = [item["node_id"] for item in final_top_rich_uv]


    step2U = {}
    for sd in single_run:
        st = sd["Step"]
        u_dist = sd.get("U_distribution", None)
        if u_dist is not None:
            subU = [u_dist[i] for i in top_ids]
            step2U[st] = subU


    steps_sorted = sorted(step2U.keys())
    step_vals = []
    mean_vals = []
    lower_vals = []
    upper_vals = []

    for st in steps_sorted:
        arr = np.array(step2U[st])
        m = arr.mean()
        s = arr.std(ddof=1)
        N = len(arr)
        sem = s / np.sqrt(N)
        ci = 1.96 * sem
        lower = m - ci
        upper = m + ci

        step_vals.append(st)
        mean_vals.append(m)
        lower_vals.append(lower)
        upper_vals.append(upper)


    plt.figure(figsize=(10, 6))
    plt.plot(step_vals, mean_vals, marker='o')
    plt.fill_between(step_vals, lower_vals, upper_vals, alpha=0.2, color='tab:blue')
    plt.title(f"Payoff of top {len(top_ids)} players")
    plt.xlabel("Step")
    plt.ylabel("U")
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # plot_gini_diffs(results_dir = "experiment1")

    # plot_avg_wealths(results_dir = "experiment1")
    #
    # plot_UV(results_dir = "experiment1", file_name="scale-free_degree_8.json")

    # plot_degree_vs_wealth(results_dir = "experiment1", filename="scale-free_degree_8.json")
    #
    # plot_U_vs_wealth(results_dir = "experiment1", filename="scale-free_degree_8.json")

    # plot_top_rich_mean_U(results_dir = "experiment1", filename="scale-free_degree_8.json")
    #
    plot_final_top10pct_mean_U(results_dir = "experiment1", filename="scale-free_degree_8.json")


