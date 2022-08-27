import wandb
import csv

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


"""
This file should be executed after the 5 runs for each of the 12 EA considered. Then repeat this file execution for each Dataset that
has been taken in consideration, and for each GNN model.

INPUT
    - a list of Group Names, which were given as input to WandB, each one representing an experiment, performed internally 5 times.


OUTPUT
    - 12 single plots, each one averaged over 5 runs, representing the Accuracy vs Diversity curves of the EA performed;
    - 1 global plot, representing the overall comparison of the 12 EA in terms of Test Accuracy, where each algorithm is 
      categorized as: Exploitative-Explorative-Neutral, depending on its specific EA hyperparameters. 
"""

WANDB_ENTITY = 'bio-ai-2022'
WANDB_PROJECT = 'AutoGL-EA'

def getAlgTag(group:str, config:dict):
    final = {}

    if 'ga' in group.lower():

        if (config['crossover_rate'] >= 0.95) and (config['mutation_rate'] <= 0.05):
            final['Name'] = 'GA-Exploitative'
            final['Tag'] = 'Exploitative'
        elif (0.85 <= config['crossover_rate'] < 0.95) and (0.05 < config['mutation_rate'] <= 0.1):
            final['Name'] = 'Ga'
            final['Tag'] = 'Neutral'
        else:
            final['Name'] = 'GA-Exlporative'
            final['Tag'] = 'Explorative'

    elif 'de' in group.lower():

        if (config['crossover_rate'] >= 0.95) and (config['mutation_rate'] <= 0.05):
            final['Name'] = 'DE-Exploitative'
            final['Tag'] = 'Exploitative'
        elif (0.85 <= config['crossover_rate'] < 0.95) and (0.05 < config['mutation_rate'] <= 0.1):
            final['Name'] = 'DE'
            final['Tag'] = 'Neutral'
        else:
            final['Name'] = 'DE-Explorative'
            final['Tag'] = 'Explorative'

    elif 'es' in group.lower() and 'cma' not in group.lower():

        if 'plus' in group.lower():
            final['Name'] = 'ES-plus'
            final['Tag'] = 'Exploitative'
        else:
            final['Name'] = 'ES-comma'
            final['Tag'] = 'Explorative'

    elif 'pso' in group.lower():

        if config['cognitive_v'] == 0.0:
            final['Name'] = 'PSO-social'
            final['Tag'] = 'Exploitative'
        elif config['social_v'] == 0.0:
            final['Name'] = 'PSO-cognitive'
            final['Tag'] = 'Explorative'
        else:
            final['Name'] = 'PSO'
            final['Tag'] = 'Neutral'

    else:

        final['Name'] = 'CMA-ES'
        final['Tag'] = 'Neutral'

    return final


def wandb_api(group:str):

    #WandB API call
    api = wandb.Api()
    runs = api.runs(WANDB_ENTITY + '/' + WANDB_PROJECT, filters={'group': group})

    #store experiment configuration
    config = runs[0].config

    #store the best test accuracy within the group
    #store the multiple DFs corresponding to different runs
    best_test_acc = 0
    list_df_original = []
    for run in runs:
        list_df_original.append(run.history())
        if run.summary['test_acc:'] > best_test_acc:
            best_test_acc = run.summary['test_acc:']

    #delete unuseful rows and columns
    list_df_final = []
    for df in list_df_original:
        df_final = df.iloc[:15, [0, 9, 7, 11]]

        if 'CMA' in group:
            df_final['mean_fit'] = df_final['mean_fit'].abs()

        df_final = df_final.rename(columns={"mean_fit": "avg_fit"})
        list_df_final.append(df_final)

    #merge the multiple DFs into one, averaging element-wise values
    df_concat = pd.concat(list_df_final)
    df_means = df_concat.groupby(df_concat.index).mean()
    df_means.index = df_means.index + 1

    #create qualitative variable
    alg_tag = getAlgTag(group, config)
    alg_tag['Accuracy'] = best_test_acc

    return df_means, alg_tag


def plot_acc_div(alg_list:list, run_names:list, title:str):

    n_cols = 3
    n_rows = 4

    fig, axes = plt.subplots(n_cols, n_rows, figsize=(32, 18), dpi=50)
    plt.subplots_adjust(wspace=0.25, hspace=0.4)

    #alg_list[count].plot(ax=axes[r,c])

    count=0
    for r in range(n_rows):
        for c in range(n_cols):

            df = alg_list[count]
            group = run_names[count]

            x = df.index
            y1 = df.avg_fit
            y2 = df.avg_div

            axes[r,c].plot(x, y1, color="tab:red")
            axes[r,c].fill_between(x, y1 + (df.std_fit*0.2), y1 - (df.std_fit*0.2), alpha = 0.2, color="tab:red")

            ax2 = axes[r,c].twinx()
            ax2.plot(x, y2, color="tab:blue")
            ax2.fill_between(x, y2 + (df.std_div*0.1), y2 - (df.std_div*0.1), alpha = 0.2, color="tab:blue")

            # ax1 (left y axis)
            axes[r,c].set_xlabel("Generations", fontsize=20)
            axes[r,c].set_ylabel("Accuracy", color="tab:red", fontsize=20)
            axes[r,c].tick_params(axis="y", rotation=0, labelcolor="tab:red")

            # ax2 (right Y axis)
            ax2.set_ylabel("Diversity", color="tab:blue", fontsize=20)
            ax2.tick_params(axis="y", labelcolor="tab:blue")
            ax2.set_title(
                "{group} - Accuracy vs Diversity".format(group=group), fontsize=20
            )
            ax2.set_xticks(np.arange(1, len(x) + 1, 1))
            ax2.set_xticklabels(x[::1], rotation=90, fontdict={"fontsize": 10})

            count+=1

    plt.savefig('{group}.png'.format(group=title))
    plt.show()

    return


def comparison_plot(alg_tags:list, title:str):
    df = pd.DataFrame(alg_tags)
    df.sort_values(by='Accuracy', ascending=False, inplace=True)

    labels = df.Tag.unique()

    x = np.arange(len(labels))
    width = 0.1

    fig, ax = plt.subplots(figsize=(18,9), dpi=50)

    for i, tag in enumerate(labels):
        df_small = df.loc[df.Tag == tag, :]
        count = 0
        for alg in df_small.values:
            if count == 0:
                rect1 = ax.bar(x[i] - 1.5*width, alg[2], width, label=alg[0])
                plt.text(x[i] - 1.5*width, alg[2]+0.0005, s = alg[0], horizontalalignment='left', fontsize='medium')
            elif count == 1:
                rect2 = ax.bar(x[i] - width/2, alg[2], width, label=alg[0])
                plt.text(x[i] - width/2, alg[2]+0.0005, s = alg[0], horizontalalignment='left', fontsize='medium')
            elif count == 2:
                rect3 = ax.bar(x[i] + width/2, alg[2], width, label=alg[0])
                plt.text(x[i] + width/2, alg[2]+0.0005, s = alg[0], horizontalalignment='left', fontsize='medium')
            else:
                rect4 = ax.bar(x[i] + 1.5*width, alg[2], width, label=alg[0])
                plt.text(x[i] + 1.5*width, alg[2]+0.0005, s = alg[0], horizontalalignment='left', fontsize='medium')
            count += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Algorithm Aprroach')
    ax.set_ylabel('Accuracy')
    ax.set_title("Cora Dataset, GCN problem : Explorative vs Neutral vs Exploitative", fontsize=18)
    ax.set_xticks(x, labels)
    ax.set_ylim(0.78, 0.83)
    ax.legend()

    fig.tight_layout()

    plt.savefig('comparison_{group}.png'.format(group=title))
    plt.show()

    return

def save_best_decoded_individual(group_name: str, test_acc: int, trainer: dict, model: dict):
    final = {'group_name': group_name, 'Test_Acc': test_acc}
    final.update(trainer)
    final.update(model)

    final = {'early_stopping' if k == 'early_stopping_round' else k: v for k, v in final.items()}
    final = {'hidden_nodes' if k == 'hidden' else k: v for k, v in final.items()}
    final['hidden_nodes'] = final['hidden_nodes'][0]

    with open('report/runs.csv'.format(name=group_name), 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=final.keys())
        writer.writeheader()
        writer.writerow(final)

    return


def main(run_names:list()):

    """
    Example of the input: 
        run_names = ['GA Test - Neutral Setting', 'ES-plus Test - Explorative Setting', 'ES-comma Test - Explorative Setting']
    """

    alg_list = []
    alg_tags = []
    
    for exp in run_names:
        df, alg_tag = wandb_api(exp)
        alg_list.append(df)
        alg_tags.append(alg_tag)

    plot_acc_div(alg_list, run_names, 'CORA-GCN')    
    comparison_plot(alg_tags, 'CORA-GCN')
    


if __name__ == '__main__':
    list_name = ['Ste_test_CORA-GCN_GA-Exploitative', 'CORA-GCN_GA-Neutral', 'CORA-GCN_GA-Explorative', 'CORA-GCN_DE-Exploitative', 
                 'CORA-GCN_DE-Neutral', 'CORA-GCN_DE-Explorative', 'CORA-GCN_ES-comma', 'CORA-GCN_ES-plus', 'CORA-GCN_CMA-ES', 'CORA-GCN_PSO-Social', 'CORA-GCN_PSO', 'CORA-GCN_PSO-Cognitive']
    main(list_name)