from main import *
import os
import json

# Experiment1: All network structures lead to inequality.
def run_experiment(network_type, num_steps, params, repetitions=5):
    results = []
    for _ in range(repetitions):
        config = DefaultConfig()
        config.network_type = network_type
        config.num_agents = 30
        config.update_parameters(params)
        model = NetworkModel(config)
        all_steps_data = []
        for step in range(num_steps):
            model.step()
            step_data = {
                'Step': model.steps,
                'Gini': model.compute_gini([agent.wealth for agent in model.schedule.agents])
            }
            all_steps_data.append(step_data)
        results.append(all_steps_data)
    return results

def main():
    network_types = ['watts-strogatz', 'scale-free', 'regular']
    num_steps = 80
    parameter_variations = {
        'watts-strogatz': [{'k': k, 'p': p} for k in range(4, 10, 2) for p in [0.1, 0.3, 0.5]],
        'scale-free': [{'m': m} for m in range(3, 6)],
        'regular': [{'d': d} for d in range(3, 7)]
    }

    results_dir = "experiment1"
    os.makedirs(results_dir, exist_ok=True)

    for network_type in network_types:
        for params in parameter_variations[network_type]:
            results = run_experiment(network_type, num_steps, params)
            file_name = f"{network_type}_params_{params}.json"
            file_path = os.path.join(results_dir, file_name)
            with open(file_path, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"Results for {network_type} with params {params} saved to {file_name}")

if __name__ == '__main__':
    main()
