import pandas as pd  # data processing, CSV file I/O
survey = pd.read_csv('cleanedDatasets/OSMIcleaned.csv')

notes = survey[['Disorder Notes', 'Describe Past Experience']]
survey.drop(notes, axis =1, inplace=True)

import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
          
flatUI = ["#2c2c54", "#34ace0", "#84817a", "#ff3f34", "#05c46b", "#ffa801"]
sns.set_palette(flatUI)
sns.palplot(sns.color_palette())
plt.show()

def showFigure(fig, x=16, y=4):
    fig = plt.gcf()
    fig.set_size_inches(x, y)
    plt.show()
    
# countValues = ['Family History of Mental Illness',
#        'Company Size', 'year',
#        'Age', 'Age-Group', 'Gender', 'Sought Treatment', 'Prefer Anonymity',
#        'Rate Reaction to Problems', 'Negative Consequences',
#        'Access to information', 'Insurance', 'Diagnosis',
#        'Discuss Mental Health Problems', 'Responsible Employer', 'Disorder',
#        'Tech Employer']    
 

        
fig = sns.distplot(survey['Age']);    
showFigure(fig)    
fig = sns.countplot(x='Age-Group', data=survey)    
showFigure(fig)
fig = sns.boxenplot(x='Age', y='Gender', data=survey)
showFigure(fig)

fig = sns.catplot(x='Insurance', kind="count", hue='Gender', data=survey)
showFigure(fig)
fig = sns.catplot(x='Prefer Anonymity', kind="count", hue='Gender', data=survey)
showFigure(fig)
fig = sns.catplot(x='Diagnosis', kind="count", hue='Gender', data=survey)
showFigure(fig)
fig = sns.catplot(x='Family History of Mental Illness', kind="count", hue='Gender', data=survey)
showFigure(fig)
fig = sns.catplot(x='Age-Group', y='Disorder', kind="bar", data=survey)
showFigure(fig)
fig = sns.swarmplot(x='Age', y='Family History of Mental Illness', hue='Gender',  palette="bright", data=survey)
showFigure(fig, y=6)

fig = sns.catplot(x='Company Size', hue='Gender', data=survey, kind="count")
showFigure(fig)
fig = sns.catplot(x='Company Size', hue='Responsible Employer', data=survey, kind="count")
showFigure(fig)
fig = sns.catplot(x='Company Size', hue='Discuss Mental Health Problems', data=survey, kind="count")
showFigure(fig)
fig = sns.catplot(x='Company Size', hue='Negative Consequences', data=survey, kind="count")
showFigure(fig)
fig = sns.catplot(x='Company Size', hue='Access to information', data=survey, kind="count")
showFigure(fig)
fig = sns.catplot(x='Company Size', hue='Rate Reaction to Problems', data=survey, kind="count")
showFigure(fig)
fig = sns.swarmplot(x='Age', y='Company Size', hue='Gender', palette="bright", data=survey)
showFigure(fig, y=6)

fig = sns.catplot(y='Sought Treatment', hue='Gender', kind="count", data=survey)
showFigure(fig)
fig = sns.catplot(x='year', y='Sought Treatment', hue='Gender', kind="bar", data=survey)
showFigure(fig)

fig = sns.catplot(x='Gender', y='Sought Treatment', hue='Age-Group', kind="point", data=survey)
showFigure(fig)
fig = sns.lmplot(x='Disorder', y='Sought Treatment', col='Gender', hue='Age-Group', data=survey)
showFigure(fig)

corr=survey.corr()
fig = sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
showFigure(fig, y=6)