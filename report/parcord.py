import pandas as pd
import plotly.graph_objs as go
import matplotlib.pyplot as plt

df_original = pd.read_csv('Bio-AI_runs.csv')

df_final = df_original.loc[df_original.groupby('group_name')['Test_Acc'].idxmax()]
df_final.drop('num_layers', axis=1, inplace=True)

df = df_final.copy()
keys = df_final.keys()
categorical_columns = ['group_name', 'act']
col_list = []

for col in df.keys():
    if col in categorical_columns:
        # Categorical columns.
        values = df[col].unique()
        # Works if values are strings, otherwise we probably need to convert them.
        value2dummy = dict(zip(values, range(len(values))))
        df[col] = [value2dummy[v] for v in df[col]]
        col_dict = dict(
            label=col,
            tickvals=list(value2dummy.values()),
            ticktext=list(value2dummy.keys()),
            values=df[col],
        )
    else:
        # Continuous columns.
        col_dict = dict(
            range=(df[col].min(), df[col].max()),
            label=col,
            values=df[col],
        )
    col_list.append(col_dict)

fig = go.Figure(data=go.Parcoords(
    line=dict(color=df_final['Test_Acc'],
              colorbar={'title': 'Accuracy'},
              showscale=True),
    dimensions=col_list))

# fig.write_image('Parallel-Plot_CORA-GCN.png')
fig.show()
