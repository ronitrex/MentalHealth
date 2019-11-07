import pandas as pd  # data processing, CSV file I/O

survey2014 = pd.read_csv('datasets/OSMIMentalHealthinTechSurvey2014.csv').assign(Year=2014)
survey2016 = pd.read_csv('datasets/OSMIMentalHealthinTechSurvey2016.csv').assign(Year=2016)
survey2017 = pd.read_csv('datasets/OSMIMentalHealthinTechSurvey2017.csv').assign(Year=2017)
survey2018 = pd.read_csv('datasets/OSMIMentalHealthinTechSurvey2018.csv').assign(Year=2018)

survey1416 =  pd.concat([survey2014, survey2016], ignore_index=True, sort=True)
survey1718 = pd.concat([survey2017, survey2018], ignore_index=True, sort=True)
survey = pd.concat([survey1416, survey1718], ignore_index=True, sort=True)
print(survey.shape)

import re 

def cleanColumns(dataframe):
    dataframe.columns = map(str.lower, dataframe.columns)

    # Remove HTML artifacts
    dataframe.rename(columns=lambda colname: re.sub('</\w+>', '', colname), inplace=True)
    dataframe.rename(columns=lambda colname: re.sub('<\w+>', '', colname), inplace=True)
    dataframe.rename(columns = {'how many employees does your company or organization have?':'Company Size', 
                                'do you have a family history of mental illness?' : 'Family History of Mental Illness'}, inplace = True) 
    
    dataframe.drop(columns=['#', 'start date (utc)', 'submit date (utc)', 'network id', 'timestamp'], inplace=True)
    dataframe.reset_index(drop=True)

    return dataframe

survey = cleanColumns(survey)
print(survey.shape)

import numpy as np; 
import matplotlib.pyplot as plt

def ShowNullValues(dataframe):
    total = dataframe.isnull().sum().sort_values(ascending=False)
    nullValues = dataframe.isnull().sum()
    totalValues = dataframe.isnull().count()
    percent = (nullValues/totalValues).sort_values(ascending=False)
    missingData = pd.concat([total, percent*100], axis=1, keys=['Total missing', 'Percent'])
    print(missingData.head(20))
    plt.figure(figsize=(25,5))
    total.plot.bar()
    y = ((lambda x: str(x)) (x) for x in range(len(dataframe.columns)))
    plt.xticks(np.arange(len(dataframe.columns)), (y))
    plt.ylabel("No. of missing or empty values")
    plt.xlabel("Dataset features")
   
    plt.show()
    return missingData

noisyData = ShowNullValues(survey)

# ageDistribution = survey.loc[:, survey.columns.str.contains('age', regex=True)]
ageDistribution = survey.loc[:, ['age', 'what is your age?']]
ageDistribution.fillna(0, inplace=True)
survey.loc[:,'Age'] = ageDistribution.sum(axis=1)
survey.loc[survey['Age']>100, 'Age'] = 30
survey.loc[survey['Age']<10, 'Age'] = 30
survey['Age-Group'] = pd.cut(survey['Age'], [0, 20, 30, 40, 65, 100], labels=["0-20", "21-30", "31-40", "41-65", "66-100"], include_lowest=True)
survey.drop(ageDistribution, axis=1, inplace=True)
showAge = survey['Age']
print(showAge.unique())

genderDistribution = survey.loc[:, survey.columns.str.contains('gender|Gender', regex=True)]
survey['Gender'] = genderDistribution.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
survey.loc[survey['Gender'].str.contains('Trans|them|trans|Undecided|Contextual|transgender|nb|unicorn|Unicorn|queer|NB|binary|Enby|Human|little|androgynous|Androgyne|Neutral|Agender|Androgynous|Androgynous|Fluid|GenderFluid|Genderflux|genderqueer|Genderqueer' , regex=True), 'Gender'] = 'Undecided'
survey.loc[survey['Gender'].str.contains('Female|female|FEMALE|Woman|woman|w|womail|W|Cis female| Female (cis)|Cis Female|cis female|cis woman|F|f' , regex=True), 'Gender'] = 'Female'
cond1 = survey['Gender']!='Female'
cond2 = survey['Gender']!='Undecided'
survey.loc[cond1 & cond2, 'Gender'] = 'Male'
survey.drop(genderDistribution, axis=1, inplace=True)
showGender = survey['Gender']
print(showGender.unique())

soughtTreatment = survey.loc[:, survey.columns.str.contains('sought treatment')]
soughtTreatment.fillna('', inplace=True)
survey['Sought Treatment'] = soughtTreatment.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
survey.loc[survey['Sought Treatment'].str.contains('yes|1.0|1|Yes' , regex=True, na=False), 'Sought Treatment'] = 1
survey.loc[survey['Sought Treatment'].str.contains('no|0.0|0|No' , regex=True, na=False), 'Sought Treatment'] = 0
survey.drop(soughtTreatment, axis=1, inplace=True)
showSoughtTreatment = survey['Sought Treatment']
print(showSoughtTreatment.unique())

noisyData = ShowNullValues(survey)

describethe = survey.loc[:, survey.columns.str.contains('describe the')]
describethe.fillna('', inplace=True)
survey['Describe Past Experience'] = describethe.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
survey.drop(describethe, axis=1, inplace=True)
showPastExperience = survey['Describe Past Experience']
print(showPastExperience)

anon = survey.loc[:, survey.columns.str.contains('anonymous')]
anon.fillna('', inplace=True)
survey['Prefer Anonymity'] = anon.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
survey.loc[survey['Prefer Anonymity'].str.contains('yes|1.0|1|Yes', regex=True, na=False), 'Prefer Anonymity'] = 1
survey.loc[survey['Prefer Anonymity'].str.contains('no|0.0|0|No' , regex=True, na=False), 'Prefer Anonymity'] = 0
survey.drop(anon, axis=1, inplace=True)
showPreferAnonymity = survey['Prefer Anonymity']
print(showPreferAnonymity.unique())

react = survey.loc[:, survey.columns.str.contains('react')]
react.fillna('', inplace=True)
survey['Rate Reaction to Problems'] = react.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
survey.loc[survey['Rate Reaction to Problems'].str.contains('0.0|1.0|2.0|3.0|4.0|5.0', regex=True), 'Rate Reaction to Problems'] = 'Below Average'
survey.loc[survey['Rate Reaction to Problems'].str.contains('6.0|7.0|8.0|9.0|10.0', regex=True), 'Rate Reaction to Problems'] = 'Above Average'
survey.drop(react, axis=1, inplace=True)
showReaction = survey['Rate Reaction to Problems']
print(showReaction.unique())

neg = survey.loc[:, survey.columns.str.contains('negative|badly', regex=True)]
neg.fillna(' ', inplace=True)
survey['Negative Consequences'] = neg.apply(lambda row: ''.join(row.values.astype(str)), axis=1)
survey.loc[survey['Negative Consequences'].str.contains('yes|1.0|1|Yes' , regex=True), 'Negative Consequences'] = 'Yes'
survey.loc[survey['Negative Consequences'].str.contains('maybe|Maybe|1' , regex=True), 'Negative Consequences'] = 'Maybe'
survey.loc[survey['Negative Consequences'].str.contains('no|No|0' , regex=True), 'Negative Consequences'] = 'No'
survey.loc[survey['Negative Consequences'].str.contains('self-employed' , regex=True), 'Negative Consequences'] = 'Self-Employed'
survey.drop(neg, axis=1, inplace=True)
showNegativeConsequnces = survey['Negative Consequences']
print(showNegativeConsequnces.unique())


work = survey.loc[:, survey.columns.str.contains('work in', regex=True)]
survey.drop(work, axis=1, inplace=True)
state = survey.loc[:, survey.columns.str.contains('country', regex=True)]
state.fillna('', inplace=True)
survey['Location'] = state.apply(lambda row: ''.join(row.values.astype(str)), axis=1)
survey.loc[survey['Location'].str.contains('[A-Z][A-Z]|United States' , regex=True), 'Location'] = 'USA'
showLocation = survey['Location']
survey.drop(state, axis=1, inplace=True)
print(showLocation.unique())


resources = survey.loc[:, survey.columns.str.contains('resources', regex=True)]
resources.fillna('', inplace=True)
survey['Access to information'] = resources.apply(lambda row: ''.join(row.values.astype(str)), axis=1)
survey.loc[survey['Access to information'].str.contains('yes|Yes' , regex=True, na=False), 'Access to information'] = 1
survey.loc[survey['Access to information'].str.contains('no|No' , regex=True, na=False), 'Access to information'] = 0
survey.drop(resources, axis=1, inplace=True)
showAccessToInformation = survey['Access to information']
print(showAccessToInformation.unique())

noisyData = ShowNullValues(survey)

insurance = survey.loc[:, survey.columns.str.contains('insurance', regex=True)]
insurance.fillna('', inplace=True)
survey['Insurance'] = insurance.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
survey.loc[survey['Insurance'].str.contains('1.0|1' , regex=True, na=False), 'Insurance'] = 1
survey.loc[survey['Insurance'].str.contains('0.0|0' , regex=True, na=False), 'Insurance'] = 0
survey.drop(insurance, axis=1, inplace=True)
showInsurance = survey['Insurance']
print(showInsurance.unique())


diagnosis = survey.loc[:, survey.columns.str.contains('diagnosed|Diagnosed|diagnose|Diagnose', regex=True)]
diagnosis.fillna(' ', inplace=True)
survey['Diagnosis'] = diagnosis.apply(lambda row: ''.join(row.values.astype(str)), axis=1)
survey.loc[survey['Diagnosis'].str.contains('yes|Yes' , regex=True), 'Diagnosis'] = 'Yes'
survey.loc[survey['Diagnosis'].str.contains('no|No' , regex=True), 'Diagnosis'] = 'No'
survey.loc[survey['Diagnosis'].str.contains('sometimes|Sometimes' , regex=True), 'Diagnosis'] = 'Sometimes'
survey.drop(diagnosis, axis=1, inplace=True)
showDiagnosis = survey['Diagnosis']
print(showDiagnosis.unique())


discuss = survey.loc[:, survey.columns.str.contains('discuss|Discuss', regex=True)]
discuss.fillna('', inplace=True)
survey['Discuss Mental Health Problems'] = discuss.apply(lambda row: ''.join(row.values.astype(str)), axis=1)
survey.loc[survey['Discuss Mental Health Problems'].str.contains('some|Some' , regex=True), 'Discuss Mental Health Problems'] = 'Maybe'
survey.loc[survey['Discuss Mental Health Problems'].str.contains('yes|Yes' , regex=True), 'Discuss Mental Health Problems'] = 'Yes'
survey.loc[survey['Discuss Mental Health Problems'].str.contains('no|No' , regex=True), 'Discuss Mental Health Problems'] = 'No'
survey.drop(discuss, axis=1, inplace=True)
showDiscussMentalHealthProblems = survey['Discuss Mental Health Problems']
print(showDiscussMentalHealthProblems.unique())

response = survey.loc[:, survey.columns.str.contains('handled|provided|serious', regex=True)]
response.fillna(' ', inplace=True)
survey['Responsible Employer'] = response.apply(lambda row: ''.join(row.values.astype(str)), axis=1)
survey.loc[survey['Responsible Employer'].str.contains('some|Some' , regex=True), 'Responsible Employer'] = 'Maybe'
survey.loc[survey['Responsible Employer'].str.contains('yes|Yes' , regex=True), 'Responsible Employer'] = 'Yes'
survey.loc[survey['Responsible Employer'].str.contains('no|No' , regex=True), 'Responsible Employer'] = 'No'
survey.loc[survey['Responsible Employer'].str.contains('self-employed' , regex=True), 'Responsible Employer'] = 'Self-Employed'
survey.drop(response, axis=1, inplace=True)
showResposibleEmployer = survey['Responsible Employer']
print(showResposibleEmployer.unique())

noisyData = ShowNullValues(survey)


Disorder = survey.loc[:, survey.columns.str.contains('Disorder|disorder|syndrome|other', regex=True)]
Disorder.fillna('', inplace=True)
DisorderNotes = Disorder.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
survey['Disorder Notes'] = DisorderNotes
survey['Disorder'] = DisorderNotes
disorderTerms = 'disorder|Disorder|negative|Negative|syndrome|Syndrome|bipolar|Bipolar|depression|Depression|autism|PTSD|Yes|yes'
survey.loc[survey['Disorder'].str.contains(disorderTerms , regex=True), 'Disorder'] = 1
survey.loc[survey['Disorder']!=1, 'Disorder'] = 0
survey.drop(Disorder, axis=1, inplace=True)
showDisorder = survey[['Disorder', 'Disorder Notes']]
print(survey['Disorder'].unique())

techEmployer = survey.loc[:, survey.columns.str.contains('tech company|tech/IT', regex=True)]
techEmployer.fillna(' ', inplace=True)
survey['Tech Employer'] = techEmployer.apply(lambda row: ''.join(row.values.astype(str)), axis=1)
survey.loc[survey['Tech Employer'].str.contains('yes|Yes|1|1.0' , regex=True, na=False), 'Tech Employer'] = 1
survey.loc[survey['Tech Employer'].str.contains('no|No|0|0.0' , regex=True, na=False), 'Tech Employer'] = 0
survey.drop(techEmployer, axis=1, inplace=True)
showTechEmployer = survey['Tech Employer']
print(showTechEmployer.unique())

noisyData = ShowNullValues(survey)

survey = survey.loc[:, ~survey.columns.duplicated()]
survey.replace('', np.nan, inplace=True)   

emptyColumns = survey.isnull().sum() 
for column in emptyColumns.index:
      if emptyColumns[column]>1000:
          survey.drop(column, axis=1, inplace=True)

for feature in survey:
    try: 
        survey[feature] = pd.to_numeric(survey[feature], errors='coerce').astype(int)
        print('numeric cast\t\t', feature)
    except:
       try:
           survey[feature] = survey[feature].astype(str)
           survey.loc[survey[feature].str.contains('^\s+$|nan' , regex=True), feature] = np.nan
           print('str cast\t\t', feature)
       except:
           continue
                
survey.to_csv('cleanedDatasets/OSMIcleaned.csv', index=False)

noisyData = ShowNullValues(survey)

print(survey.shape)