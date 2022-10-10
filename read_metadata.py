from codecs import ignore_errors
import pandas as pd
import json
import os
import numpy as np


def main():
    """
    find findings in mimic (done), open-i, pubmed, and covid_chestxray
    """
    # file = f'metadata.csv'
    # pwd = os.getcwd()
    # filepath = os.path.join('/', *pwd.split(os.path.sep)[:-1], f'bert-AAD/data/nih_train.csv')
    file = f'indiana_reports.csv'
    # pubmed has no intersection with covid_chestxray
    # intersection between covid_chestxray and open-i is ['tuberculosis', 'pneumonia'] + normal/no finding
    covid_findings = ['Aspergillosis', 'Aspiration', 'Bacterial', 'COVID-19',
       'Chlamydophila', 'E.Coli', 'Fungal', 'H1N1', 'Herpes ',
       'Influenza', 'Klebsiella', 'Legionella', 'Lipoid', 'MERS-CoV',
       'MRSA', 'Mycoplasma', 'No Finding', 'Nocardia', 'Pneumocystis',
       'Pneumonia', 'SARS', 'Staphylococcus', 'Streptococcus',
       'Tuberculosis', 'Unknown', 'Varicella', 'Viral', 'todo']
    covid_findings_split = []
    haha = [covid_findings_split.extend(item.split("/")) for item in covid_findings]
    covid_findings_unique = np.unique(covid_findings_split)
    covid_findings_unique = [item.lower() for item in covid_findings_unique]
    df = pd.read_csv(file)  # [930, 30]
    findings = df['Problems']
    reports = df['findings']  # if nan use df['impression']
    all_findings = pd.unique(findings)
    unique_findings = []
    hh = [unique_findings.extend(item.split(';')) for item in all_findings]
    unique_findings = np.unique(unique_findings)
    unique_findings = [item.lower() for item in unique_findings]
    intersect = list(set(covid_findings_unique).intersection(set(unique_findings)))
    print(intersect)


def filter_covid():
    """
    intersection between covid_chestxray and open-i is ['tuberculosis', 'pneumonia'] + normal/no finding
    In covid_chestxray, find report + label in above findings ['No Finding']
    In open-i, find report + label in above findings ['normal']
    done
    """
    covid = f'metadata.csv'
    covid_df = pd.read_csv(covid)
    findings = covid_df['finding']
    reports = covid_df['clinical_notes']
    columns=['tuberculosis', 'pneumonia', 'no finding', 'report']
    covid_new_df = pd.DataFrame(columns=columns)
    for i, row in covid_df.iterrows():
        fins = row['finding'].lower().split('/')  # findings
        new_row = [1 if item in fins else 0 for item in columns[:-1]]
        if sum(new_row) > 0:
            new_row.append(row['clinical_notes'])
            new_row = pd.Series(new_row, index=covid_new_df.columns)
            covid_new_df = covid_new_df.append(new_row, ignore_index=True)
    # remove duplicated rows
    covid_new_df = covid_new_df.drop_duplicates(subset=['report'])
    covid_new_df.dropna(subset=['report'], inplace=True)  # [612, 4]
    covid_new_df.to_csv(f'data/covid_all.csv')
    covid_train_df = covid_new_df.sample(frac=0.81, replace=False)
    covid_train_df.to_csv(f'data/covid_train.csv')


def filter_openi():
    """
    intersection between covid_chestxray and open-i is ['tuberculosis', 'pneumonia'] + normal/no finding
    In covid_chestxray, find report + label in above findings ['No Finding']
    In open-i, find report + label in above findings ['normal']
    """
    openi = f'data/indiana_reports.csv'
    df = pd.read_csv(openi)
    findings = df['Problems']
    reports = df['findings']
    df['report'] = (df['findings'] + df['impression']).astype(str)
    columns=['tuberculosis', 'pneumonia', 'normal', 'report']
    openi_new_df = pd.DataFrame(columns=columns)
    for i, row in df.iterrows():
        fins = row['Problems'].lower().split(';')
        new_row = [1 if item in fins else 0 for item in columns[:-1]]
        if sum(new_row) > 0:
            new_row.append(row['report'])
            new_row = pd.Series(new_row, index=openi_new_df.columns)
            openi_new_df = openi_new_df.append(new_row, ignore_index=True)

    openi_new_df = openi_new_df.drop_duplicates(subset=['report'])
    openi_new_df.dropna(subset=['report'], inplace=True)
    openi_new_df.to_csv(f'data/openi_all.csv')
    openi_train_df = openi_new_df.sample(frac=0.8, replace=False)
    openi_train_df.to_csv(f'data/openi_train.csv')


def read_json():
    file = f'06102020.litcovid.released.image.json'
    with open(file) as f:
        data = json.load(f)
        print(data)


if __name__ == '__main__':
    # main()
    # filter_covid()
    filter_openi()
