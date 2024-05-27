import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set_style('white')
sns.set_context("paper", font_scale=1.8
                , rc={"lines.linewidth": 3})

df = pd.read_csv('results/SR25.txt')
df['dataset'] = df['dataset'].str.upper()

ax = sns.barplot(df[df['mark_neighbors'] == True], color='orange', x='dataset', y='mean_failure_rate')
ax.axhline(y=1.0, label='3-WL')
ax.set(xlabel='', ylabel='Failure rate')

plt.savefig('results/sr_graphs.pdf', bbox_inches='tight')
