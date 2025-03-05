# Check the output structure and the convergence of UV

import json
import numpy as np
import os

def network_structure_check(file_dir="experiment1", file_name = "regular_degree_8.json"):
    # Check the structure of the networks in experiment 1

    os.chdir(file_dir)
    with open(file_name, 'r') as f:
        data = json.load(f)

    single_run = data[0]  # pick the first repetition
    steps_with_data = []
    for step, sd in enumerate(single_run):
        if sd['Edges_count'] is not None:
            steps_with_data.append(step)
    print("We have Edges at steps:", steps_with_data)

    for rep_i, step_val in enumerate(steps_with_data):
        step_data =  single_run[step_val]
        edge_rep = step_data['Edges']
        clustering_rep = step_data['Clustering']
        print("Edges:", edge_rep)
        print("cluster:", clustering_rep)


def gather_UV_all(data, step):
    U_all = []
    V_all = []
    for rep_i in range(len(data)):
        step_data = data[rep_i][step]
        U_rep = step_data.get("U_distribution") or []
        V_rep = step_data.get("V_distribution") or []
        U_all.extend(U_rep)
        V_all.extend(V_rep)
    return U_all, V_all


def distribution_diff_2d(U_old, V_old, U_new, V_new,
                        bins=30, range_=[[-2, 2], [-2, 2]]):
    H_old, xedges, yedges = np.histogram2d(U_old, V_old, bins=bins, range=range_)
    H_new, _, _ = np.histogram2d(U_new, V_new, bins=bins, range=range_)
    diff = np.sum(np.abs(H_old - H_new))
    total = H_old.sum() + H_new.sum()
    if total>0:
        return diff / total
    else:
        return 0.0


def UV_convergence_check(file_dir="experiment2", file_name="regular_degree_8.json", threshold=0.05):
    os.chdir(file_dir)
    with open(file_name, 'r') as f:
        data = json.load(f)

    rep0_data = data[0]
    steps_all = [sd["Step"] for sd in rep0_data]
    steps_sorted = sorted(steps_all)

    valid_steps = []
    for st in steps_sorted:
        step_data = rep0_data[st]
        if step_data.get("U_distribution") is not None:
            valid_steps.append(st)


    converged = False
    for i in range(1, len(valid_steps)):
        t_prev = valid_steps[i-1]
        t_curr = valid_steps[i]
        U_prev, V_prev = gather_UV_all(data, t_prev)
        U_curr, V_curr = gather_UV_all(data, t_curr)

        dist2d = distribution_diff_2d(U_prev, V_prev, U_curr, V_curr, bins=30, range_=[[-2,2],[-2,2]])

        if dist2d < threshold:
            print(f"Converged at step={t_curr}, 2D-dist={dist2d:.4f} < {threshold}")
            converged = True
            break

    if not converged:
        print("No convergence detected by the final valid step.")


if __name__ == "__main__":
    # network_structure_check(file_dir="experiment1", file_name="watts-strogatz_degree_8.json")

    UV_convergence_check(file_dir="experiment2", file_name="scale-free_degree_8.json", threshold=0.05)