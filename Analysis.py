import wandb

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


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
            final['Name'] = 'DE-Exlporative'
            final['Tag'] = 'Explorative'

    elif 'es' in group.lower():

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
    runs = api.runs("bio-ai-2022/AutoGL-EA", filters={"group":group})

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
        df_final = df.iloc[:15, [0, 5, 8, 11]]
        list_df_final.append(df_final)

    #merge the multiple DFs into one, averaging element-wise values
    df_concat = pd.concat(list_df_final)
    df_means = df_concat.groupby(df_concat.index).mean()
    df_means.index = df_means.index + 1

    #create qualitative variable
    alg_tag = getAlgTag(group, config)
    alg_tag['Accuracy'] = best_test_acc

    return df_means, alg_tag


def plot_acc_div(df:pd.DataFrame, group:str):

    x = df.index
    y1 = df.avg_fit
    y2 = df.avg_div

    fig, ax1 = plt.subplots(1, 1, figsize=(16, 7), dpi=50)
    ax1.plot(x, y1, color="tab:red")
    ax1.fill_between(x, y1 + (df.std_fit*0.1), y1 - (df.std_fit*0.1), alpha = 0.2, color="tab:red")

    ax2 = ax1.twinx()
    ax2.plot(x, y2, color="tab:blue")
    ax2.fill_between(x, y2 + (df.std_div*0.1), y2 - (df.std_div*0.1), alpha = 0.2, color="tab:blue")

    # ax1 (left y axis)
    ax1.set_xlabel("Generations", fontsize=20)
    ax1.set_ylabel("Accuracy", color="tab:red", fontsize=20)
    ax1.tick_params(axis="y", rotation=0, labelcolor="tab:red")

    # ax2 (right Y axis)
    ax2.set_ylabel("Diversity", color="tab:blue", fontsize=20)
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax2.set_title(
        "{group} - Accuracy vs Diversity".format(group=group), fontsize=20
    )
    ax2.set_xticks(np.arange(1, len(x) + 1, 1))
    ax2.set_xticklabels(x[::1], rotation=90, fontdict={"fontsize": 10})
    
    plt.savefig('{group}.png'.format(group=group))
    plt.show()

    return


def comparison_plot(alg_list:list):

    df = pd.DataFrame(alg_list)

    # Data preparation
    # Create as many colors as there are unique midwest['category']
    categories = np.unique(df['Name'])
    colors = [plt.cm.tab10(i/float(len(categories)-1)) for i in range(len(categories))]

    # Draw Plot for Each Category
    plt.figure(figsize=(10, 8), dpi=72, facecolor='w', edgecolor='k')
    for i, category in enumerate(categories):
        plt.scatter('Tag', 'Accuracy', 
                    data=df.loc[df.Name==category, :], 
                    s=50, c=np.array(colors[i]).reshape((1,4)), label=str(category))

    # Decorations
    plt.gca().set(xlabel='Algorithm Aprroach', ylabel='Accuracy')

    plt.xticks(fontsize=12); plt.yticks(fontsize=12)
    plt.title("EA Approach - Explorative vs Neutral vs Exploitative", fontsize=22)
    plt.legend(fontsize=12)    
    plt.show()  



def main():

    alg_list = []
    run_name = ['GA Test - Neutral Setting', 'ES-plus Test - Explorative Setting', 'ES-comma Test - Explorative Setting']

    for exp in run_name:
        _, alg_tag = wandb_api(exp)
        alg_list.append(alg_tag)
        
        
    comparison_plot(alg_list)
    return


if __name__ == '__main__':
    main()