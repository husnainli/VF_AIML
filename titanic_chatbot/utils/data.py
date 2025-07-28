import pandas as pd
import streamlit as st
import re

def extract_title(name):
    # Extracts title such as Mr, Mrs, Miss, Master, etc.
    match = re.search(r',\s*([^\.]+)\.', name)
    return match.group(1).strip() if match else None

def extract_surname(name):
    # Surname is before the comma
    return name.split(',')[0].strip()

def extract_fullname(name):
    # Inside parentheses = real name
    if '(' in name and ')' in name:
        return re.search(r'\((.*?)\)', name).group(1).strip()
    else:
        # Otherwise use the part after the title
        after_comma = name.split(',')[1]
        after_title = after_comma.split('.')[1].strip() if '.' in after_comma else after_comma.strip()
        return after_title

def infer_is_married(row):
    title = row['Title']
    name = row['Name']
    age = row['Age']
    sex = row['Sex']
    sibsp = row['SibSp']

    # Case 1: Explicit indicator — Mrs with real name in parentheses
    if title == 'Mrs' and '(' in name:
        return 1
    
    # Case 2: Woman titled 'Mrs' (even without parentheses)
    if title == 'Mrs':
        return 1
    
    # Case 3: Man titled 'Mr', adult (age >= 18), and has exactly 1 SibSp
    # Reason: Likely traveling with wife
    if title == 'Mr' and age >= 18 and sibsp == 1:
        return 1

    # Case 4: Rare titles for married people
    rare_married_titles = ['Lady', 'Sir', 'Countess', 'Don', 'Dona', 'Jonkheer']
    if title in rare_married_titles:
        return 1

    # Case 5: Miss/Master — young and single
    # Miss could be married, but assume single unless evidence (e.g., parentheses or age > 25 + SibSp > 0)
    if title in ['Miss', 'Master']:
        return 0

    # Default: Assume unmarried
    return 0


@st.cache_data(show_spinner=False)
def load_and_prepare_data():
    df = pd.read_csv('data/titanic.csv')

    # Handle missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['Deck'] = df['Cabin'].str[0]
    df.drop(columns=['Cabin'], inplace=True)

    # Encode sex
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    # Add derived columns
    df['Title'] = df['Name'].apply(extract_title)
    df['Surname'] = df['Name'].apply(extract_surname)
    df['FullName'] = df['Name'].apply(extract_fullname)
    df['IsMarried'] = df.apply(infer_is_married, axis=1)

    return df