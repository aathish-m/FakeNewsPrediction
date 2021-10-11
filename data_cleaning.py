# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 11:15:31 2021

@author: he
"""
import pandas as pd

df = pd.read_csv('news.csv')

df.columns

data = df.drop('Unnamed: 0', axis=1)

data.to_csv('final_news_data.csv', index=False)

new = pd.read_csv('final_news_data.csv')