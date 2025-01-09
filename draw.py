import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_1samp


def read_data(results_dir):
    data_frames = []
    for filename in os.listdir(results_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(results_dir, filename)
            with open(file_path, 'r') as f:
                data = json.load(f)
            for experiment in data:
                df = pd.DataFrame(experiment)
                df['network_type'] = filename.split('_')[0]
                df['params'] = str(filename.split('params_')[1].split('.json')[0])
                data_frames.append(df)
    return pd.concat(data_frames, ignore_index=True)


def plot_gini_over_time(data, results_dir):
    fig, axs = plt.subplots(nrows=1, ncols=len(data['network_type'].unique()), figsize=(18, 6))

    network_types = data['network_type'].unique()
    for ax, network_type in zip(axs, network_types):
        subset = data[data['network_type'] == network_type]
        params = subset['params'].unique()

        for param in params:
            param_data = subset[subset['params'] == param]
            numeric_data = param_data.select_dtypes(include=[np.number])
            mean_data = numeric_data.groupby('Step').mean()
            ci_lower = param_data.groupby('Step').apply(lambda x: np.percentile(x['Gini'], 2.5), include_groups=False)
            ci_upper = param_data.groupby('Step').apply(lambda x: np.percentile(x['Gini'], 97.5), include_groups=False)
            ax.plot(mean_data.index, mean_data['Gini'], label=f'{param}')
            ax.fill_between(mean_data.index, ci_lower, ci_upper, alpha=0.3)

        ax.axhline(y=0.4, color='black', linestyle='--', label='Inequality Threshold')
        ax.set_title(f'{network_type.replace("-", " ").title()}')
        ax.legend(loc='upper right')

    fig.supxlabel('Time Step')
    fig.supylabel('Gini Coefficient')
    fig.suptitle('Gini Coefficient Over Time', fontsize=16)

    plt.tight_layout(rect=[0.02, 0, 1, 1])
    plt.savefig(os.path.join(results_dir, 'gini_coefficient_over_time.png'))
    plt.show()


def perform_statistical_test(data, last_n_steps=10):
    inequality_threshold = 0.4
    results = {}
    for network_type in data['network_type'].unique():
        subset = data[data['network_type'] == network_type]

        last_steps_data = subset[subset['Step'] >= subset['Step'].max() - last_n_steps + 1]
        mean_gini_last_steps = last_steps_data.groupby(['network_type', 'params']).mean()['Gini']
        t_stat, p_value = ttest_1samp(mean_gini_last_steps, inequality_threshold)
        mean_gini = mean_gini_last_steps.mean()
        significant = p_value < 0.05 and mean_gini > inequality_threshold
        results[network_type] = {
            'T-statistic': t_stat,
            'P-value': p_value,
            'Mean Gini': mean_gini,
            'Significantly Higher than Threshold': significant
        }
    return results


def format_statistical_results(results):
    formatted_results = "Statistical Test Results:\n"
    for network_type, stats in results.items():
        formatted_results += f"\n{network_type.title()}\n"
        formatted_results += f"  T-statistic: {stats['T-statistic']:.2f}\n"
        formatted_results += f"  P-value: {stats['P-value']:.2e}\n"
        formatted_results += f"  Mean Gini: {stats['Mean Gini']:.3f}\n"
        formatted_results += f"  Significantly Higher than Threshold: {'Yes' if stats['Significantly Higher than Threshold'] else 'No'}\n"
    return formatted_results


def main():
    results_dir = "experiment1"
    data = read_data(results_dir)
    plot_gini_over_time(data, results_dir)
    stat_results = perform_statistical_test(data)
    print(format_statistical_results(stat_results))


if __name__ == '__main__':
    main()
