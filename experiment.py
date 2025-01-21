from main2 import *
import os
import json
import time

def run_experiment(network_type, num_steps, avg_degree, rewiring_prob=0, repetitions=10):
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

        model = NetworkModel(config)
        all_steps_data = []
        for step in range(num_steps):
            model.step()
            step_data = {
                'Step': step,
                'Gini': model.compute_gini([agent.wealth for agent in model.schedule.agents])
            }
            all_steps_data.append(step_data)
        results.append(all_steps_data)
    return results

def main():
    start_time = time.time()
    network_types = ['watts-strogatz', 'scale-free', 'regular']
    num_steps = 480
    avg_degrees = [6, 8, 10]

    results_dir = "experiment1"
    os.makedirs(results_dir, exist_ok=True)

    for avg_degree in avg_degrees:
        for network_type in network_types:
            results = run_experiment(network_type, num_steps, avg_degree)
            file_name = f"{network_type}_avg_degree_{avg_degree}.json"
            file_path = os.path.join(results_dir, file_name)
            with open(file_path, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"Results for {network_type} with average degree {avg_degree} saved to {file_name}")

    end_time = time.time()
    print("Total time taken:", end_time - start_time)

if __name__ == '__main__':
    main()

