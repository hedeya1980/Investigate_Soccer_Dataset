
# Project: Investigate a Dataset - [Soccer Database]

## Table of Contents
<ul>
<li><a href="#intro">Introduction</a></li>
<li><a href="#wrangling">Data Wrangling</a></li>
<li><a href="#eda">Exploratory Data Analysis</a></li>
<li><a href="#conclusions">Conclusions</a></li>
</ul>

<a id='intro'></a>
## Introduction

### Dataset Description 

This soccer database comes from Kaggle and is well suited for data analysis and machine learning. It contains data for soccer matches, players, and teams from several European countries from 2008 to 2016.

The database contains the following tables:
* Country
* League
* Match
* Player
* Player Attributes
* Team
* Team Attributes

List of columns of each table are shown in the Wrangling and EDA sections below.


### Question(s) for Analysis

## Here are my questions:

### Q1: Which players scored penalties the most (penalty top-scorers)? 
### Q2: Which players had the most penalties (the fouled player)?
### Q3: What team attributes lead to the most victories?
### Q4: What teams improved the most over the time period?


```python
# Use this cell to set up import statements for all of the packages that you
#   plan to use.

# Remember to include a 'magic word' so that your visualizations are plotted
#   inline with the notebook. See this page for more:
#   http://ipython.readthedocs.io/en/stable/interactive/magics.html
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
% matplotlib inline

from sqlalchemy import create_engine

from bs4 import BeautifulSoup
import os

import math
```


```python
# Upgrade pandas to use dataframe.explode() function. 
!pip install --upgrade pandas==0.25.0
```

    Collecting pandas==0.25.0
    [?25l  Downloading https://files.pythonhosted.org/packages/1d/9a/7eb9952f4b4d73fbd75ad1d5d6112f407e695957444cb695cbb3cdab918a/pandas-0.25.0-cp36-cp36m-manylinux1_x86_64.whl (10.5MB)
    [K    100% |████████████████████████████████| 10.5MB 1.6MB/s eta 0:00:01   33% |██████████▊                     | 3.5MB 28.6MB/s eta 0:00:01    97% |███████████████████████████████▏| 10.2MB 28.3MB/s eta 0:00:01
    [?25hRequirement already satisfied, skipping upgrade: pytz>=2017.2 in /opt/conda/lib/python3.6/site-packages (from pandas==0.25.0) (2017.3)
    Requirement already satisfied, skipping upgrade: python-dateutil>=2.6.1 in /opt/conda/lib/python3.6/site-packages (from pandas==0.25.0) (2.6.1)
    Collecting numpy>=1.13.3 (from pandas==0.25.0)
    [?25l  Downloading https://files.pythonhosted.org/packages/45/b2/6c7545bb7a38754d63048c7696804a0d947328125d81bf12beaa692c3ae3/numpy-1.19.5-cp36-cp36m-manylinux1_x86_64.whl (13.4MB)
    [K    100% |████████████████████████████████| 13.4MB 2.5MB/s eta 0:00:01  5% |█▊                              | 716kB 29.1MB/s eta 0:00:01    53% |█████████████████▏              | 7.2MB 27.1MB/s eta 0:00:01    76% |████████████████████████▍       | 10.2MB 27.0MB/s eta 0:00:01
    [?25hRequirement already satisfied, skipping upgrade: six>=1.5 in /opt/conda/lib/python3.6/site-packages (from python-dateutil>=2.6.1->pandas==0.25.0) (1.11.0)
    [31mtensorflow 1.3.0 requires tensorflow-tensorboard<0.2.0,>=0.1.0, which is not installed.[0m
    Installing collected packages: numpy, pandas
      Found existing installation: numpy 1.12.1
        Uninstalling numpy-1.12.1:
          Successfully uninstalled numpy-1.12.1
      Found existing installation: pandas 0.23.3
        Uninstalling pandas-0.23.3:
          Successfully uninstalled pandas-0.23.3
    Successfully installed numpy-1.19.5 pandas-0.25.0


<a id='wrangling'></a>
## Data Wrangling

### General Properties


```python
#engine = create_engine('sqlite:///bestofrt.db')
engine = create_engine('sqlite:///database.sqlite')
```


```python
# Load your data and print out a few lines. Perform operations to inspect data
#   types and look for instances of missing or possibly errant data.
df_country = pd.read_sql('SELECT * FROM Country', engine, index_col='id')
#df_gather = pd.read_sql('master', engine)
```


```python
df_country.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Belgium</td>
    </tr>
    <tr>
      <th>1729</th>
      <td>England</td>
    </tr>
    <tr>
      <th>4769</th>
      <td>France</td>
    </tr>
    <tr>
      <th>7809</th>
      <td>Germany</td>
    </tr>
    <tr>
      <th>10257</th>
      <td>Italy</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_league = pd.read_sql('SELECT * FROM League', engine)
df_league.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>country_id</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>Belgium Jupiler League</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1729</td>
      <td>1729</td>
      <td>England Premier League</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4769</td>
      <td>4769</td>
      <td>France Ligue 1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7809</td>
      <td>7809</td>
      <td>Germany 1. Bundesliga</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10257</td>
      <td>10257</td>
      <td>Italy Serie A</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_league.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>country_id</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>15722</td>
      <td>15722</td>
      <td>Poland Ekstraklasa</td>
    </tr>
    <tr>
      <th>7</th>
      <td>17642</td>
      <td>17642</td>
      <td>Portugal Liga ZON Sagres</td>
    </tr>
    <tr>
      <th>8</th>
      <td>19694</td>
      <td>19694</td>
      <td>Scotland Premier League</td>
    </tr>
    <tr>
      <th>9</th>
      <td>21518</td>
      <td>21518</td>
      <td>Spain LIGA BBVA</td>
    </tr>
    <tr>
      <th>10</th>
      <td>24558</td>
      <td>24558</td>
      <td>Switzerland Super League</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_league[df_league['id']!=df_league['country_id']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>country_id</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



#### [Comment] 'id' is always the same as 'country_id'. Hence, I'll use one of them to index the dataframe.


```python
df_league.set_index('id')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country_id</th>
      <th>name</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Belgium Jupiler League</td>
    </tr>
    <tr>
      <th>1729</th>
      <td>1729</td>
      <td>England Premier League</td>
    </tr>
    <tr>
      <th>4769</th>
      <td>4769</td>
      <td>France Ligue 1</td>
    </tr>
    <tr>
      <th>7809</th>
      <td>7809</td>
      <td>Germany 1. Bundesliga</td>
    </tr>
    <tr>
      <th>10257</th>
      <td>10257</td>
      <td>Italy Serie A</td>
    </tr>
    <tr>
      <th>13274</th>
      <td>13274</td>
      <td>Netherlands Eredivisie</td>
    </tr>
    <tr>
      <th>15722</th>
      <td>15722</td>
      <td>Poland Ekstraklasa</td>
    </tr>
    <tr>
      <th>17642</th>
      <td>17642</td>
      <td>Portugal Liga ZON Sagres</td>
    </tr>
    <tr>
      <th>19694</th>
      <td>19694</td>
      <td>Scotland Premier League</td>
    </tr>
    <tr>
      <th>21518</th>
      <td>21518</td>
      <td>Spain LIGA BBVA</td>
    </tr>
    <tr>
      <th>24558</th>
      <td>24558</td>
      <td>Switzerland Super League</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_match=pd.read_sql('Match', engine, index_col='id')
df_match.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country_id</th>
      <th>league_id</th>
      <th>season</th>
      <th>stage</th>
      <th>date</th>
      <th>match_api_id</th>
      <th>home_team_api_id</th>
      <th>away_team_api_id</th>
      <th>home_team_goal</th>
      <th>away_team_goal</th>
      <th>...</th>
      <th>SJA</th>
      <th>VCH</th>
      <th>VCD</th>
      <th>VCA</th>
      <th>GBH</th>
      <th>GBD</th>
      <th>GBA</th>
      <th>BSH</th>
      <th>BSD</th>
      <th>BSA</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>2008/2009</td>
      <td>1</td>
      <td>2008-08-17 00:00:00</td>
      <td>492473</td>
      <td>9987</td>
      <td>9993</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>4.00</td>
      <td>1.65</td>
      <td>3.40</td>
      <td>4.50</td>
      <td>1.78</td>
      <td>3.25</td>
      <td>4.00</td>
      <td>1.73</td>
      <td>3.40</td>
      <td>4.20</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>2008/2009</td>
      <td>1</td>
      <td>2008-08-16 00:00:00</td>
      <td>492474</td>
      <td>10000</td>
      <td>9994</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3.80</td>
      <td>2.00</td>
      <td>3.25</td>
      <td>3.25</td>
      <td>1.85</td>
      <td>3.25</td>
      <td>3.75</td>
      <td>1.91</td>
      <td>3.25</td>
      <td>3.60</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>2008/2009</td>
      <td>1</td>
      <td>2008-08-16 00:00:00</td>
      <td>492475</td>
      <td>9984</td>
      <td>8635</td>
      <td>0</td>
      <td>3</td>
      <td>...</td>
      <td>2.50</td>
      <td>2.35</td>
      <td>3.25</td>
      <td>2.65</td>
      <td>2.50</td>
      <td>3.20</td>
      <td>2.50</td>
      <td>2.30</td>
      <td>3.20</td>
      <td>2.75</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>2008/2009</td>
      <td>1</td>
      <td>2008-08-17 00:00:00</td>
      <td>492476</td>
      <td>9991</td>
      <td>9998</td>
      <td>5</td>
      <td>0</td>
      <td>...</td>
      <td>7.50</td>
      <td>1.45</td>
      <td>3.75</td>
      <td>6.50</td>
      <td>1.50</td>
      <td>3.75</td>
      <td>5.50</td>
      <td>1.44</td>
      <td>3.75</td>
      <td>6.50</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>1</td>
      <td>2008/2009</td>
      <td>1</td>
      <td>2008-08-16 00:00:00</td>
      <td>492477</td>
      <td>7947</td>
      <td>9985</td>
      <td>1</td>
      <td>3</td>
      <td>...</td>
      <td>1.73</td>
      <td>4.50</td>
      <td>3.40</td>
      <td>1.65</td>
      <td>4.50</td>
      <td>3.50</td>
      <td>1.65</td>
      <td>4.75</td>
      <td>3.30</td>
      <td>1.67</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 114 columns</p>
</div>




```python
df_match.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 25979 entries, 1 to 25979
    Columns: 114 entries, country_id to BSA
    dtypes: float64(96), int64(8), object(10)
    memory usage: 22.8+ MB



```python
list(df_match.columns)
```




    ['country_id',
     'league_id',
     'season',
     'stage',
     'date',
     'match_api_id',
     'home_team_api_id',
     'away_team_api_id',
     'home_team_goal',
     'away_team_goal',
     'home_player_X1',
     'home_player_X2',
     'home_player_X3',
     'home_player_X4',
     'home_player_X5',
     'home_player_X6',
     'home_player_X7',
     'home_player_X8',
     'home_player_X9',
     'home_player_X10',
     'home_player_X11',
     'away_player_X1',
     'away_player_X2',
     'away_player_X3',
     'away_player_X4',
     'away_player_X5',
     'away_player_X6',
     'away_player_X7',
     'away_player_X8',
     'away_player_X9',
     'away_player_X10',
     'away_player_X11',
     'home_player_Y1',
     'home_player_Y2',
     'home_player_Y3',
     'home_player_Y4',
     'home_player_Y5',
     'home_player_Y6',
     'home_player_Y7',
     'home_player_Y8',
     'home_player_Y9',
     'home_player_Y10',
     'home_player_Y11',
     'away_player_Y1',
     'away_player_Y2',
     'away_player_Y3',
     'away_player_Y4',
     'away_player_Y5',
     'away_player_Y6',
     'away_player_Y7',
     'away_player_Y8',
     'away_player_Y9',
     'away_player_Y10',
     'away_player_Y11',
     'home_player_1',
     'home_player_2',
     'home_player_3',
     'home_player_4',
     'home_player_5',
     'home_player_6',
     'home_player_7',
     'home_player_8',
     'home_player_9',
     'home_player_10',
     'home_player_11',
     'away_player_1',
     'away_player_2',
     'away_player_3',
     'away_player_4',
     'away_player_5',
     'away_player_6',
     'away_player_7',
     'away_player_8',
     'away_player_9',
     'away_player_10',
     'away_player_11',
     'goal',
     'shoton',
     'shotoff',
     'foulcommit',
     'card',
     'cross',
     'corner',
     'possession',
     'B365H',
     'B365D',
     'B365A',
     'BWH',
     'BWD',
     'BWA',
     'IWH',
     'IWD',
     'IWA',
     'LBH',
     'LBD',
     'LBA',
     'PSH',
     'PSD',
     'PSA',
     'WHH',
     'WHD',
     'WHA',
     'SJH',
     'SJD',
     'SJA',
     'VCH',
     'VCD',
     'VCA',
     'GBH',
     'GBD',
     'GBA',
     'BSH',
     'BSD',
     'BSA']



#### [Comment] I'll use df_match to explore and answer most of the above questions. However, it has many columns that I won't use in my analysis. Hence, in the 'cleaning' section, I'll keep only the columns that are important for my analysis.


```python
df_player = pd.read_sql('Player', engine, index_col='id')
df_player.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>player_api_id</th>
      <th>player_name</th>
      <th>player_fifa_api_id</th>
      <th>birthday</th>
      <th>height</th>
      <th>weight</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>505942</td>
      <td>Aaron Appindangoye</td>
      <td>218353</td>
      <td>1992-02-29 00:00:00</td>
      <td>182</td>
      <td>187</td>
    </tr>
    <tr>
      <th>2</th>
      <td>155782</td>
      <td>Aaron Cresswell</td>
      <td>189615</td>
      <td>1989-12-15 00:00:00</td>
      <td>170</td>
      <td>146</td>
    </tr>
    <tr>
      <th>3</th>
      <td>162549</td>
      <td>Aaron Doran</td>
      <td>186170</td>
      <td>1991-05-13 00:00:00</td>
      <td>170</td>
      <td>163</td>
    </tr>
    <tr>
      <th>4</th>
      <td>30572</td>
      <td>Aaron Galindo</td>
      <td>140161</td>
      <td>1982-05-08 00:00:00</td>
      <td>182</td>
      <td>198</td>
    </tr>
    <tr>
      <th>5</th>
      <td>23780</td>
      <td>Aaron Hughes</td>
      <td>17725</td>
      <td>1979-11-08 00:00:00</td>
      <td>182</td>
      <td>154</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_player.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 11060 entries, 1 to 11075
    Data columns (total 6 columns):
    player_api_id         11060 non-null int64
    player_name           11060 non-null object
    player_fifa_api_id    11060 non-null int64
    birthday              11060 non-null object
    height                11060 non-null int64
    weight                11060 non-null int64
    dtypes: int64(4), object(2)
    memory usage: 604.8+ KB



```python
df_player[df_player['player_name']=='Mohamed Salah']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>player_api_id</th>
      <th>player_name</th>
      <th>player_fifa_api_id</th>
      <th>birthday</th>
      <th>height</th>
      <th>weight</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7665</th>
      <td>292462</td>
      <td>Mohamed Salah</td>
      <td>209331</td>
      <td>1992-06-15 00:00:00</td>
      <td>175</td>
      <td>159</td>
    </tr>
  </tbody>
</table>
</div>



#### [Comment] 'Mohamed Salah' --> The Egyptian King


```python
df_player_attributes=pd.read_sql('Player_Attributes', engine, index_col='id')
df_player_attributes.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>player_fifa_api_id</th>
      <th>player_api_id</th>
      <th>date</th>
      <th>overall_rating</th>
      <th>potential</th>
      <th>preferred_foot</th>
      <th>attacking_work_rate</th>
      <th>defensive_work_rate</th>
      <th>crossing</th>
      <th>finishing</th>
      <th>...</th>
      <th>vision</th>
      <th>penalties</th>
      <th>marking</th>
      <th>standing_tackle</th>
      <th>sliding_tackle</th>
      <th>gk_diving</th>
      <th>gk_handling</th>
      <th>gk_kicking</th>
      <th>gk_positioning</th>
      <th>gk_reflexes</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>218353</td>
      <td>505942</td>
      <td>2016-02-18 00:00:00</td>
      <td>67.0</td>
      <td>71.0</td>
      <td>right</td>
      <td>medium</td>
      <td>medium</td>
      <td>49.0</td>
      <td>44.0</td>
      <td>...</td>
      <td>54.0</td>
      <td>48.0</td>
      <td>65.0</td>
      <td>69.0</td>
      <td>69.0</td>
      <td>6.0</td>
      <td>11.0</td>
      <td>10.0</td>
      <td>8.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>218353</td>
      <td>505942</td>
      <td>2015-11-19 00:00:00</td>
      <td>67.0</td>
      <td>71.0</td>
      <td>right</td>
      <td>medium</td>
      <td>medium</td>
      <td>49.0</td>
      <td>44.0</td>
      <td>...</td>
      <td>54.0</td>
      <td>48.0</td>
      <td>65.0</td>
      <td>69.0</td>
      <td>69.0</td>
      <td>6.0</td>
      <td>11.0</td>
      <td>10.0</td>
      <td>8.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>218353</td>
      <td>505942</td>
      <td>2015-09-21 00:00:00</td>
      <td>62.0</td>
      <td>66.0</td>
      <td>right</td>
      <td>medium</td>
      <td>medium</td>
      <td>49.0</td>
      <td>44.0</td>
      <td>...</td>
      <td>54.0</td>
      <td>48.0</td>
      <td>65.0</td>
      <td>66.0</td>
      <td>69.0</td>
      <td>6.0</td>
      <td>11.0</td>
      <td>10.0</td>
      <td>8.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>218353</td>
      <td>505942</td>
      <td>2015-03-20 00:00:00</td>
      <td>61.0</td>
      <td>65.0</td>
      <td>right</td>
      <td>medium</td>
      <td>medium</td>
      <td>48.0</td>
      <td>43.0</td>
      <td>...</td>
      <td>53.0</td>
      <td>47.0</td>
      <td>62.0</td>
      <td>63.0</td>
      <td>66.0</td>
      <td>5.0</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>7.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>218353</td>
      <td>505942</td>
      <td>2007-02-22 00:00:00</td>
      <td>61.0</td>
      <td>65.0</td>
      <td>right</td>
      <td>medium</td>
      <td>medium</td>
      <td>48.0</td>
      <td>43.0</td>
      <td>...</td>
      <td>53.0</td>
      <td>47.0</td>
      <td>62.0</td>
      <td>63.0</td>
      <td>66.0</td>
      <td>5.0</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>7.0</td>
      <td>7.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 41 columns</p>
</div>




```python
list(df_player_attributes.columns)
```




    ['player_fifa_api_id',
     'player_api_id',
     'date',
     'overall_rating',
     'potential',
     'preferred_foot',
     'attacking_work_rate',
     'defensive_work_rate',
     'crossing',
     'finishing',
     'heading_accuracy',
     'short_passing',
     'volleys',
     'dribbling',
     'curve',
     'free_kick_accuracy',
     'long_passing',
     'ball_control',
     'acceleration',
     'sprint_speed',
     'agility',
     'reactions',
     'balance',
     'shot_power',
     'jumping',
     'stamina',
     'strength',
     'long_shots',
     'aggression',
     'interceptions',
     'positioning',
     'vision',
     'penalties',
     'marking',
     'standing_tackle',
     'sliding_tackle',
     'gk_diving',
     'gk_handling',
     'gk_kicking',
     'gk_positioning',
     'gk_reflexes']




```python
df_team=pd.read_sql('Team', engine, index_col='id')
df_team.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>team_api_id</th>
      <th>team_fifa_api_id</th>
      <th>team_long_name</th>
      <th>team_short_name</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>9987</td>
      <td>673.0</td>
      <td>KRC Genk</td>
      <td>GEN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9993</td>
      <td>675.0</td>
      <td>Beerschot AC</td>
      <td>BAC</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10000</td>
      <td>15005.0</td>
      <td>SV Zulte-Waregem</td>
      <td>ZUL</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9994</td>
      <td>2007.0</td>
      <td>Sporting Lokeren</td>
      <td>LOK</td>
    </tr>
    <tr>
      <th>5</th>
      <td>9984</td>
      <td>1750.0</td>
      <td>KSV Cercle Brugge</td>
      <td>CEB</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_team.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 299 entries, 1 to 51606
    Data columns (total 4 columns):
    team_api_id         299 non-null int64
    team_fifa_api_id    288 non-null float64
    team_long_name      299 non-null object
    team_short_name     299 non-null object
    dtypes: float64(1), int64(1), object(2)
    memory usage: 11.7+ KB



```python
df_team['team_fifa_api_id'].unique()
```




    array([  6.73000000e+02,   6.75000000e+02,   1.50050000e+04,
             2.00700000e+03,   1.75000000e+03,   2.29000000e+02,
             6.74000000e+02,   1.74700000e+03,              nan,
             2.32000000e+02,   1.10724000e+05,   2.31000000e+02,
             5.46000000e+02,   1.00081000e+05,   1.11560000e+05,
             6.81000000e+02,   6.70000000e+02,   6.80000000e+02,
             2.39000000e+02,   2.01300000e+03,   1.00087000e+05,
             1.10913000e+05,   6.82000000e+02,   1.10000000e+01,
             1.30000000e+01,   1.00000000e+00,   1.09000000e+02,
             1.06000000e+02,   9.00000000e+00,   1.90000000e+01,
             1.91700000e+03,   2.00000000e+00,   1.00000000e+01,
             7.00000000e+00,   3.00000000e+00,   1.20000000e+01,
             1.80000000e+01,   4.00000000e+00,   1.80600000e+03,
             1.95200000e+03,   1.44000000e+02,   5.00000000e+00,
             1.79000000e+03,   8.80000000e+01,   1.10000000e+02,
             1.79600000e+03,   1.92600000e+03,   1.96000000e+03,
             1.50000000e+01,   1.79200000e+03,   1.70000000e+01,
             1.79300000e+03,   1.79900000e+03,   1.96100000e+03,
             9.50000000e+01,   1.94300000e+03,   1.79500000e+03,
             5.70000000e+01,   7.10000000e+01,   5.90000000e+01,
             2.10000000e+02,   1.73800000e+03,   7.20000000e+01,
             1.73900000e+03,   2.17000000e+02,   6.60000000e+01,
             1.80900000e+03,   6.90000000e+01,   7.30000000e+01,
             1.82300000e+03,   6.50000000e+01,   7.40000000e+01,
             2.19000000e+02,   2.26000000e+02,   1.80500000e+03,
             1.10456000e+05,   1.81900000e+03,   6.40000000e+01,
             7.00000000e+01,   1.11376000e+05,   1.11989000e+05,
             3.78000000e+02,   6.14000000e+02,   1.11271000e+05,
             1.10569000e+05,   3.79000000e+02,   5.80000000e+01,
             2.94000000e+02,   6.20000000e+01,   6.80000000e+01,
             1.53000000e+03,   1.10316000e+05,   2.10000000e+01,
             2.80000000e+01,   3.20000000e+01,   2.20000000e+01,
             3.40000000e+01,   4.85000000e+02,   1.75000000e+02,
             3.10000000e+01,   1.82400000e+03,   1.66000000e+02,
             1.59000000e+02,   3.80000000e+01,   1.62000000e+02,
             1.00290000e+04,   2.30000000e+01,   3.60000000e+01,
             1.83200000e+03,   1.60000000e+02,   2.50000000e+01,
             1.71000000e+02,   1.69000000e+02,   2.90000000e+01,
             1.10329000e+05,   1.00409000e+05,   1.10636000e+05,
             1.65000000e+02,   1.10500000e+05,   1.00300000e+04,
             1.11239000e+05,   1.10502000e+05,   3.90000000e+01,
             1.83800000e+03,   1.84200000e+03,   4.60000000e+01,
             1.10364000e+05,   1.10556000e+05,   1.92000000e+02,
             2.03000000e+02,   1.10374000e+05,   4.50000000e+01,
             4.70000000e+01,   1.89000000e+02,   5.20000000e+01,
             4.80000000e+01,   1.83700000e+03,   4.40000000e+01,
             5.40000000e+01,   3.47000000e+02,   5.50000000e+01,
             1.84300000e+03,   1.84800000e+03,   1.84400000e+03,
             5.00000000e+01,   1.10915000e+05,   1.90000000e+02,
             1.12225000e+05,   2.00000000e+02,   2.06000000e+02,
             1.11974000e+05,   1.74600000e+03,   1.11657000e+05,
             1.12409000e+05,   1.90900000e+03,   1.91500000e+03,
             1.90200000e+03,   1.90800000e+03,   1.90700000e+03,
             2.45000000e+02,   1.91000000e+03,   6.35000000e+02,
             1.90300000e+03,   2.47000000e+02,   1.00634000e+05,
             2.46000000e+02,   1.00646000e+05,   6.50000000e+02,
             1.91300000e+03,   1.90600000e+03,   1.90400000e+03,
             1.90500000e+03,   1.00651000e+05,   1.97100000e+03,
             1.91400000e+03,   6.47000000e+02,   1.00632000e+05,
             1.00626000e+05,   1.87300000e+03,   1.11429000e+05,
             8.74000000e+02,   1.87100000e+03,   1.57000000e+03,
             1.11092000e+05,   1.11091000e+05,   3.01000000e+02,
             1.56400000e+03,   8.73000000e+02,   1.10744000e+05,
             1.11082000e+05,   1.10745000e+05,   1.11086000e+05,
             1.10747000e+05,   1.11083000e+05,   1.10749000e+05,
             1.11087000e+05,   1.10746000e+05,   1.12512000e+05,
             1.10565000e+05,   2.36000000e+02,   1.88900000e+03,
             2.37000000e+02,   1.88700000e+03,   6.65000000e+02,
             1.89200000e+03,   1.89600000e+03,   1.90100000e+03,
             7.44000000e+02,   2.34000000e+02,   1.00180000e+04,
             1.89100000e+03,   1.00737000e+05,   1.89300000e+03,
             1.89500000e+03,   1.11540000e+05,   1.89700000e+03,
             1.88800000e+03,   1.90000000e+03,   1.00200000e+04,
             1.12513000e+05,   1.00741000e+05,   1.89800000e+03,
             7.90000000e+01,   8.60000000e+01,   8.00000000e+01,
             8.30000000e+01,   8.20000000e+01,   8.10000000e+01,
             7.70000000e+01,   6.20000000e+02,   7.80000000e+01,
             1.00805000e+05,   1.84000000e+02,   1.81000000e+02,
             1.00804000e+05,   1.82000000e+02,   1.80000000e+02,
             6.31000000e+02,   1.75400000e+03,   4.61000000e+02,
             4.53000000e+02,   4.79000000e+02,   4.83000000e+02,
             2.42000000e+02,   2.43000000e+02,   4.77000000e+02,
             2.41000000e+02,   4.56000000e+02,   4.81000000e+02,
             4.59000000e+02,   1.86000000e+03,   4.49000000e+02,
             5.71000000e+02,   4.52000000e+02,   4.62000000e+02,
             4.48000000e+02,   1.86100000e+03,   2.40000000e+02,
             5.73000000e+02,   1.74200000e+03,   2.44000000e+02,
             2.60000000e+02,   1.00879000e+05,   1.85300000e+03,
             4.57000000e+02,   1.10832000e+05,   4.80000000e+02,
             4.50000000e+02,   4.68000000e+02,   4.67000000e+02,
             1.86700000e+03,   4.72000000e+02,   3.22000000e+02,
             1.71400000e+03,   9.00000000e+02,   8.96000000e+02,
             4.34000000e+02,   1.10770000e+05,   8.97000000e+02,
             2.86000000e+02,   4.35000000e+02,   8.94000000e+02,
             8.98000000e+02,   1.71500000e+03,   3.24000000e+02,
             1.86200000e+03])




```python
df_team[df_team['team_fifa_api_id'].isnull()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>team_api_id</th>
      <th>team_fifa_api_id</th>
      <th>team_long_name</th>
      <th>team_short_name</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9</th>
      <td>7947</td>
      <td>NaN</td>
      <td>FCV Dender EH</td>
      <td>DEN</td>
    </tr>
    <tr>
      <th>15</th>
      <td>4049</td>
      <td>NaN</td>
      <td>Tubize</td>
      <td>TUB</td>
    </tr>
    <tr>
      <th>26561</th>
      <td>6601</td>
      <td>NaN</td>
      <td>FC Volendam</td>
      <td>VOL</td>
    </tr>
    <tr>
      <th>34816</th>
      <td>177361</td>
      <td>NaN</td>
      <td>Termalica Bruk-Bet Nieciecza</td>
      <td>TBN</td>
    </tr>
    <tr>
      <th>35286</th>
      <td>7992</td>
      <td>NaN</td>
      <td>Trofense</td>
      <td>TRO</td>
    </tr>
    <tr>
      <th>35291</th>
      <td>10213</td>
      <td>NaN</td>
      <td>Amadora</td>
      <td>AMA</td>
    </tr>
    <tr>
      <th>36248</th>
      <td>9765</td>
      <td>NaN</td>
      <td>Portimonense</td>
      <td>POR</td>
    </tr>
    <tr>
      <th>36723</th>
      <td>4064</td>
      <td>NaN</td>
      <td>Feirense</td>
      <td>FEI</td>
    </tr>
    <tr>
      <th>38789</th>
      <td>6367</td>
      <td>NaN</td>
      <td>Uniao da Madeira</td>
      <td>MAD</td>
    </tr>
    <tr>
      <th>38791</th>
      <td>188163</td>
      <td>NaN</td>
      <td>Tondela</td>
      <td>TON</td>
    </tr>
    <tr>
      <th>51606</th>
      <td>7896</td>
      <td>NaN</td>
      <td>Lugano</td>
      <td>LUG</td>
    </tr>
  </tbody>
</table>
</div>



#### [Comment] I won't clean the 'team_fifa_api_id'. It's not important for my analysis. 'team_api_id' is the important ID for my analysis, and I don't want to lose any team records.


```python
df_team_attributes=pd.read_sql('Team_Attributes', engine, index_col='id')
df_team_attributes.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>team_fifa_api_id</th>
      <th>team_api_id</th>
      <th>date</th>
      <th>buildUpPlaySpeed</th>
      <th>buildUpPlaySpeedClass</th>
      <th>buildUpPlayDribbling</th>
      <th>buildUpPlayDribblingClass</th>
      <th>buildUpPlayPassing</th>
      <th>buildUpPlayPassingClass</th>
      <th>buildUpPlayPositioningClass</th>
      <th>...</th>
      <th>chanceCreationShooting</th>
      <th>chanceCreationShootingClass</th>
      <th>chanceCreationPositioningClass</th>
      <th>defencePressure</th>
      <th>defencePressureClass</th>
      <th>defenceAggression</th>
      <th>defenceAggressionClass</th>
      <th>defenceTeamWidth</th>
      <th>defenceTeamWidthClass</th>
      <th>defenceDefenderLineClass</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>434</td>
      <td>9930</td>
      <td>2010-02-22 00:00:00</td>
      <td>60</td>
      <td>Balanced</td>
      <td>NaN</td>
      <td>Little</td>
      <td>50</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>...</td>
      <td>55</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>50</td>
      <td>Medium</td>
      <td>55</td>
      <td>Press</td>
      <td>45</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>2</th>
      <td>434</td>
      <td>9930</td>
      <td>2014-09-19 00:00:00</td>
      <td>52</td>
      <td>Balanced</td>
      <td>48.0</td>
      <td>Normal</td>
      <td>56</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>...</td>
      <td>64</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>47</td>
      <td>Medium</td>
      <td>44</td>
      <td>Press</td>
      <td>54</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>3</th>
      <td>434</td>
      <td>9930</td>
      <td>2015-09-10 00:00:00</td>
      <td>47</td>
      <td>Balanced</td>
      <td>41.0</td>
      <td>Normal</td>
      <td>54</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>...</td>
      <td>64</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>47</td>
      <td>Medium</td>
      <td>44</td>
      <td>Press</td>
      <td>54</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>4</th>
      <td>77</td>
      <td>8485</td>
      <td>2010-02-22 00:00:00</td>
      <td>70</td>
      <td>Fast</td>
      <td>NaN</td>
      <td>Little</td>
      <td>70</td>
      <td>Long</td>
      <td>Organised</td>
      <td>...</td>
      <td>70</td>
      <td>Lots</td>
      <td>Organised</td>
      <td>60</td>
      <td>Medium</td>
      <td>70</td>
      <td>Double</td>
      <td>70</td>
      <td>Wide</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>5</th>
      <td>77</td>
      <td>8485</td>
      <td>2011-02-22 00:00:00</td>
      <td>47</td>
      <td>Balanced</td>
      <td>NaN</td>
      <td>Little</td>
      <td>52</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>...</td>
      <td>52</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>47</td>
      <td>Medium</td>
      <td>47</td>
      <td>Press</td>
      <td>52</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>




```python
list(df_team_attributes.columns)
```




    ['team_fifa_api_id',
     'team_api_id',
     'date',
     'buildUpPlaySpeed',
     'buildUpPlaySpeedClass',
     'buildUpPlayDribbling',
     'buildUpPlayDribblingClass',
     'buildUpPlayPassing',
     'buildUpPlayPassingClass',
     'buildUpPlayPositioningClass',
     'chanceCreationPassing',
     'chanceCreationPassingClass',
     'chanceCreationCrossing',
     'chanceCreationCrossingClass',
     'chanceCreationShooting',
     'chanceCreationShootingClass',
     'chanceCreationPositioningClass',
     'defencePressure',
     'defencePressureClass',
     'defenceAggression',
     'defenceAggressionClass',
     'defenceTeamWidth',
     'defenceTeamWidthClass',
     'defenceDefenderLineClass']




```python
df_team_attributes.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1458 entries, 1 to 1458
    Data columns (total 24 columns):
    team_fifa_api_id                  1458 non-null int64
    team_api_id                       1458 non-null int64
    date                              1458 non-null object
    buildUpPlaySpeed                  1458 non-null int64
    buildUpPlaySpeedClass             1458 non-null object
    buildUpPlayDribbling              489 non-null float64
    buildUpPlayDribblingClass         1458 non-null object
    buildUpPlayPassing                1458 non-null int64
    buildUpPlayPassingClass           1458 non-null object
    buildUpPlayPositioningClass       1458 non-null object
    chanceCreationPassing             1458 non-null int64
    chanceCreationPassingClass        1458 non-null object
    chanceCreationCrossing            1458 non-null int64
    chanceCreationCrossingClass       1458 non-null object
    chanceCreationShooting            1458 non-null int64
    chanceCreationShootingClass       1458 non-null object
    chanceCreationPositioningClass    1458 non-null object
    defencePressure                   1458 non-null int64
    defencePressureClass              1458 non-null object
    defenceAggression                 1458 non-null int64
    defenceAggressionClass            1458 non-null object
    defenceTeamWidth                  1458 non-null int64
    defenceTeamWidthClass             1458 non-null object
    defenceDefenderLineClass          1458 non-null object
    dtypes: float64(1), int64(10), object(13)
    memory usage: 284.8+ KB



### Data Cleaning



```python
# After discussing the structure of the data and any problems that need to be
#   cleaned, perform those cleaning steps in the second part of this section.

```


```python
df_match=df_match[['date',
 'match_api_id',
 'home_team_api_id',
 'away_team_api_id',
 'home_team_goal',
 'away_team_goal',
 'goal',
 'shoton',
 'shotoff',
 'foulcommit'
 ]]
df_match.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 25979 entries, 1 to 25979
    Data columns (total 10 columns):
    date                25979 non-null object
    match_api_id        25979 non-null int64
    home_team_api_id    25979 non-null int64
    away_team_api_id    25979 non-null int64
    home_team_goal      25979 non-null int64
    away_team_goal      25979 non-null int64
    goal                14217 non-null object
    shoton              14217 non-null object
    shotoff             14217 non-null object
    foulcommit          14217 non-null object
    dtypes: int64(5), object(5)
    memory usage: 2.2+ MB


> **I kept only the above columns, which are the columns that I'll use to address my 'Questions'.**


```python
df_match['date']=pd.to_datetime(df_match['date'])
df_match.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 25979 entries, 1 to 25979
    Data columns (total 10 columns):
    date                25979 non-null datetime64[ns]
    match_api_id        25979 non-null int64
    home_team_api_id    25979 non-null int64
    away_team_api_id    25979 non-null int64
    home_team_goal      25979 non-null int64
    away_team_goal      25979 non-null int64
    goal                14217 non-null object
    shoton              14217 non-null object
    shotoff             14217 non-null object
    foulcommit          14217 non-null object
    dtypes: datetime64[ns](1), int64(5), object(4)
    memory usage: 2.2+ MB


>**I converted the 'date' column to the datetime format. This will allow me to extract the 'year' information.**


```python
df_player.birthday=pd.to_datetime(df_player.birthday)
df_player.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 11060 entries, 1 to 11075
    Data columns (total 6 columns):
    player_api_id         11060 non-null int64
    player_name           11060 non-null object
    player_fifa_api_id    11060 non-null int64
    birthday              11060 non-null datetime64[ns]
    height                11060 non-null int64
    weight                11060 non-null int64
    dtypes: datetime64[ns](1), int64(4), object(1)
    memory usage: 604.8+ KB


>**I converted the 'birthday' column to the datetime format.**


```python
df_team_attributes.drop(columns=['buildUpPlayDribbling'], inplace=True)
```

>**Most of the data in the 'buildUpPlayDribbling' are Nulls. Hence, I dropped it to avoid losing other important data from the corresponding rows.**


```python
df_team_attributes.date=pd.to_datetime(df_team_attributes.date)
df_team_attributes.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1458 entries, 1 to 1458
    Data columns (total 23 columns):
    team_fifa_api_id                  1458 non-null int64
    team_api_id                       1458 non-null int64
    date                              1458 non-null datetime64[ns]
    buildUpPlaySpeed                  1458 non-null int64
    buildUpPlaySpeedClass             1458 non-null object
    buildUpPlayDribblingClass         1458 non-null object
    buildUpPlayPassing                1458 non-null int64
    buildUpPlayPassingClass           1458 non-null object
    buildUpPlayPositioningClass       1458 non-null object
    chanceCreationPassing             1458 non-null int64
    chanceCreationPassingClass        1458 non-null object
    chanceCreationCrossing            1458 non-null int64
    chanceCreationCrossingClass       1458 non-null object
    chanceCreationShooting            1458 non-null int64
    chanceCreationShootingClass       1458 non-null object
    chanceCreationPositioningClass    1458 non-null object
    defencePressure                   1458 non-null int64
    defencePressureClass              1458 non-null object
    defenceAggression                 1458 non-null int64
    defenceAggressionClass            1458 non-null object
    defenceTeamWidth                  1458 non-null int64
    defenceTeamWidthClass             1458 non-null object
    defenceDefenderLineClass          1458 non-null object
    dtypes: datetime64[ns](1), int64(10), object(12)
    memory usage: 273.4+ KB


>**I converted the 'date' column to the datetime format. This will allow me to extract the 'year' information.**

<a id='eda'></a>
## Exploratory Data Analysis

> **Tip**: Now that you've trimmed and cleaned your data, you're ready to move on to exploration. **Compute statistics** and **create visualizations** with the goal of addressing the research questions that you posed in the Introduction section. You should compute the relevant statistics throughout the analysis when an inference is made about the data. Note that at least two or more kinds of plots should be created as part of the exploration, and you must  compare and show trends in the varied visualizations. 



> **Tip**: - Investigate the stated question(s) from multiple angles. It is recommended that you be systematic with your approach. Look at one variable at a time, and then follow it up by looking at relationships between variables. You should explore at least three variables in relation to the primary question. This can be an exploratory relationship between three variables of interest, or looking at how two independent variables relate to a single dependent variable of interest. Lastly, you  should perform both single-variable (1d) and multiple-variable (2d) explorations.



### Research Question 1  (Which players scored penalties the most (top penalty scorer)?)


```python
df_match_goals=df_match[['match_api_id','goal']]
```


```python
df_match_goals.dropna(inplace=True)
df_match_goals.head()
```

    /opt/conda/lib/python3.6/site-packages/ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      """Entry point for launching an IPython kernel.





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>match_api_id</th>
      <th>goal</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1729</th>
      <td>489042</td>
      <td>&lt;goal&gt;&lt;value&gt;&lt;comment&gt;n&lt;/comment&gt;&lt;stats&gt;&lt;goals...</td>
    </tr>
    <tr>
      <th>1730</th>
      <td>489043</td>
      <td>&lt;goal&gt;&lt;value&gt;&lt;comment&gt;n&lt;/comment&gt;&lt;stats&gt;&lt;goals...</td>
    </tr>
    <tr>
      <th>1731</th>
      <td>489044</td>
      <td>&lt;goal&gt;&lt;value&gt;&lt;comment&gt;n&lt;/comment&gt;&lt;stats&gt;&lt;goals...</td>
    </tr>
    <tr>
      <th>1732</th>
      <td>489045</td>
      <td>&lt;goal&gt;&lt;value&gt;&lt;comment&gt;n&lt;/comment&gt;&lt;stats&gt;&lt;goals...</td>
    </tr>
    <tr>
      <th>1733</th>
      <td>489046</td>
      <td>&lt;goal&gt;&lt;value&gt;&lt;comment&gt;n&lt;/comment&gt;&lt;stats&gt;&lt;goals...</td>
    </tr>
  </tbody>
</table>
</div>



>**The 'goal' records are in HTML format, So, I'll use BeatifulSoup to parse them.
I referred to the 'European Soccer Database Supplementary' https://www.kaggle.com/jiezi2004/soccer to understand the tags and contents of the 'goal' and 'foulcommit' records.**


```python
def parse_html_match_attrib(df, parsed_column, column_list):
    df_list=[]
    for index, row in df.iterrows():
        soup=BeautifulSoup(row[parsed_column],'lxml')
        values=soup.find_all('value')
        for value in values:
            goal_details_dict={}
            #value_soup=BeautifulSoup(value,'lxml')
            goal_details_dict["match_id"]=index
            for c in column_list:
                if value.find(c):
                    goal_details_dict[c]=value.find(c).contents[0]
            df_list.append(goal_details_dict)
                
    #print(goal_details_dict)
    return df_list
```

>**I'll use the above function to parse the 'goals' and 'foulcommit' columns of 'df_match' to be able to answer the first and the second questions.**


```python
goals_df_list=parse_html_match_attrib(df_match_goals, 'goal', ['player1', 'team', 'goal_type', 'comment'])
```


```python
df_goals = pd.DataFrame(goals_df_list, columns = ['match_id', 'player1', 'team', 'goal_type', 'comment'])
df_goals.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>match_id</th>
      <th>player1</th>
      <th>team</th>
      <th>goal_type</th>
      <th>comment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1729</td>
      <td>37799</td>
      <td>10261</td>
      <td>n</td>
      <td>n</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1729</td>
      <td>24148</td>
      <td>10260</td>
      <td>n</td>
      <td>n</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1730</td>
      <td>26181</td>
      <td>9825</td>
      <td>n</td>
      <td>n</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1731</td>
      <td>30853</td>
      <td>8650</td>
      <td>n</td>
      <td>n</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1732</td>
      <td>23139</td>
      <td>8654</td>
      <td>n</td>
      <td>n</td>
    </tr>
  </tbody>
</table>
</div>



>**I used 'goals_df_list' to create the 'df_goals' dataframe that will be used to analyze the goals.**


```python
df_goals.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 62136 entries, 0 to 62135
    Data columns (total 5 columns):
    match_id     62136 non-null int64
    player1      39863 non-null object
    team         39946 non-null object
    goal_type    39946 non-null object
    comment      39980 non-null object
    dtypes: int64(1), object(4)
    memory usage: 2.4+ MB



```python
df_goals.dropna(subset=['player1', 'team'], inplace=True)
df_goals.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 39863 entries, 0 to 62135
    Data columns (total 5 columns):
    match_id     39863 non-null int64
    player1      39863 non-null object
    team         39863 non-null object
    goal_type    39863 non-null object
    comment      39863 non-null object
    dtypes: int64(1), object(4)
    memory usage: 1.8+ MB



```python
df_goals[df_goals.goal_type != df_goals.comment]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>match_id</th>
      <th>player1</th>
      <th>team</th>
      <th>goal_type</th>
      <th>comment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4064</th>
      <td>3156</td>
      <td>46700</td>
      <td>10261</td>
      <td>n</td>
      <td>dg</td>
    </tr>
    <tr>
      <th>4762</th>
      <td>3386</td>
      <td>181276</td>
      <td>8659</td>
      <td>o</td>
      <td>n</td>
    </tr>
    <tr>
      <th>5116</th>
      <td>3509</td>
      <td>46469</td>
      <td>9825</td>
      <td>o</td>
      <td>n</td>
    </tr>
    <tr>
      <th>52172</th>
      <td>22695</td>
      <td>33030</td>
      <td>8315</td>
      <td>o</td>
      <td>n</td>
    </tr>
  </tbody>
</table>
</div>



>**According to the 'Supplementary' database (referenced above), 'comment' is more accurate/reliable than 'goal_type'. However, there is no discrepency between the two columns in the 'penalty' type**


```python
df_goals['player1']=df_goals['player1'].astype(int)
df_goals['team']=df_goals['team'].astype(int)
df_goals.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 39863 entries, 0 to 62135
    Data columns (total 5 columns):
    match_id     39863 non-null int64
    player1      39863 non-null int64
    team         39863 non-null int64
    goal_type    39863 non-null object
    comment      39863 non-null object
    dtypes: int64(3), object(2)
    memory usage: 1.8+ MB


>**In the above cell, I converted the ID columns to the 'int' datatype.**


```python
penalty_score_per_player_counts_df=df_goals[df_goals['goal_type']=='p'].groupby('player1').count()
penalty_score_per_player_counts_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>match_id</th>
      <th>team</th>
      <th>goal_type</th>
      <th>comment</th>
    </tr>
    <tr>
      <th>player1</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>101041</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>101422</th>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>102572</th>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>102612</th>
      <td>17</td>
      <td>17</td>
      <td>17</td>
      <td>17</td>
    </tr>
    <tr>
      <th>102619</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



>**Players penalty scores (masking by 'goal_type', and grouping by 'player1', which is the player who scored the goal.)**


```python
maximum_scored_penalties_player=penalty_score_per_player_counts_df.max()['match_id'].item()
maximum_scored_penalties_player
```




    55



>**'maximum_scored_penalties_player' is the maximum number of penalties scored by a player.**


```python
best_penalty_scoring_player=int(penalty_score_per_player_counts_df[penalty_score_per_player_counts_df['team']==maximum_scored_penalties_player]['goal_type'].index[0])#.item()
best_penalty_scoring_player
```




    30893




```python
df_player[df_player['player_api_id']==best_penalty_scoring_player]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>player_api_id</th>
      <th>player_name</th>
      <th>player_fifa_api_id</th>
      <th>birthday</th>
      <th>height</th>
      <th>weight</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1995</th>
      <td>30893</td>
      <td>Cristiano Ronaldo</td>
      <td>20801</td>
      <td>1985-02-05</td>
      <td>185</td>
      <td>176</td>
    </tr>
  </tbody>
</table>
</div>



>**'Cristiano Ronaldo' was the top penalty scorer in the period from 2008 until 2016.**


```python
df_score_penalty_count.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>match_id</th>
      <th>team</th>
      <th>goal_type</th>
      <th>comment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>819.000000</td>
      <td>819.000000</td>
      <td>819.000000</td>
      <td>819.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.632479</td>
      <td>3.632479</td>
      <td>3.632479</td>
      <td>3.632479</td>
    </tr>
    <tr>
      <th>std</th>
      <td>4.617440</td>
      <td>4.617440</td>
      <td>4.617440</td>
      <td>4.617440</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>55.000000</td>
      <td>55.000000</td>
      <td>55.000000</td>
      <td>55.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_top_penalty_scorers=df_score_penalty_count[df_score_penalty_count['goal_type']>20]
df_top_penalty_scorers.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>match_id</th>
      <th>team</th>
      <th>goal_type</th>
      <th>comment</th>
    </tr>
    <tr>
      <th>player1</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27734</th>
      <td>28</td>
      <td>28</td>
      <td>28</td>
      <td>28</td>
    </tr>
    <tr>
      <th>30549</th>
      <td>23</td>
      <td>23</td>
      <td>23</td>
      <td>23</td>
    </tr>
    <tr>
      <th>30618</th>
      <td>25</td>
      <td>25</td>
      <td>25</td>
      <td>25</td>
    </tr>
    <tr>
      <th>30631</th>
      <td>25</td>
      <td>25</td>
      <td>25</td>
      <td>25</td>
    </tr>
    <tr>
      <th>30714</th>
      <td>29</td>
      <td>29</td>
      <td>29</td>
      <td>29</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_top_penalty_scorers.reset_index(inplace=True)
df_top_penalty_scorers.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10 entries, 0 to 9
    Data columns (total 5 columns):
    player1      10 non-null object
    match_id     10 non-null int64
    team         10 non-null int64
    goal_type    10 non-null int64
    comment      10 non-null int64
    dtypes: int64(4), object(1)
    memory usage: 480.0+ bytes



```python
df_top_penalty_scorers.player1=df_top_penalty_scorers.player1.astype(int)
```

    /opt/conda/lib/python3.6/site-packages/pandas/core/generic.py:4405: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      self[name] = value



```python
df_top_penalty_scorers_player_details=df_top_penalty_scorers.merge(df_player,left_on='player1', right_on='player_api_id', how='inner')[['player_name', 'goal_type']].sort_values(['goal_type'], ascending=False)
df_top_penalty_scorers_player_details.rename(columns={'goal_type':'scored_penalties'}, inplace=True)
```

>**'df_top_penalty_scorers_player_details' is the list of top penalty players with the number of penalties each player scored. It was obtained with a merge between df_top_penalty_scorers & df_player.**

## Research Question 2 (Which players had the most penalties (the fouled players who were granted the most penalties)?)

>**In the following cells we will apply almost the same steps as in the analysis of Question 1:**


```python
df_match_fouls=df_match[['match_api_id','foulcommit']]
```


```python
df_match_fouls.dropna(inplace=True)
df_match_fouls.head()
```

    /opt/conda/lib/python3.6/site-packages/ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      """Entry point for launching an IPython kernel.





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>match_api_id</th>
      <th>foulcommit</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1729</th>
      <td>489042</td>
      <td>&lt;foulcommit&gt;&lt;value&gt;&lt;stats&gt;&lt;foulscommitted&gt;1&lt;/f...</td>
    </tr>
    <tr>
      <th>1730</th>
      <td>489043</td>
      <td>&lt;foulcommit&gt;&lt;value&gt;&lt;stats&gt;&lt;foulscommitted&gt;1&lt;/f...</td>
    </tr>
    <tr>
      <th>1731</th>
      <td>489044</td>
      <td>&lt;foulcommit&gt;&lt;value&gt;&lt;stats&gt;&lt;foulscommitted&gt;1&lt;/f...</td>
    </tr>
    <tr>
      <th>1732</th>
      <td>489045</td>
      <td>&lt;foulcommit&gt;&lt;value&gt;&lt;stats&gt;&lt;foulscommitted&gt;1&lt;/f...</td>
    </tr>
    <tr>
      <th>1733</th>
      <td>489046</td>
      <td>&lt;foulcommit&gt;&lt;value&gt;&lt;stats&gt;&lt;foulscommitted&gt;1&lt;/f...</td>
    </tr>
  </tbody>
</table>
</div>




```python
fouls_df_list=parse_html_match_attrib(df_match_fouls, 'foulcommit', ['player1', 'team', 'player2', 'subtype'])
```


```python
df_fouls = pd.DataFrame(fouls_df_list, columns = ['match_id', 'player1', 'team', 'player2', 'subtype'])
df_fouls.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>match_id</th>
      <th>player1</th>
      <th>team</th>
      <th>player2</th>
      <th>subtype</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1729</td>
      <td>25518</td>
      <td>10261</td>
      <td>32569</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1729</td>
      <td>30929</td>
      <td>10261</td>
      <td>24157</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1729</td>
      <td>29581</td>
      <td>10261</td>
      <td>24148</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1729</td>
      <td>30373</td>
      <td>10260</td>
      <td>40565</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1729</td>
      <td>29581</td>
      <td>10261</td>
      <td>30829</td>
      <td>pushing</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_fouls.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 432886 entries, 0 to 432885
    Data columns (total 5 columns):
    match_id    432886 non-null int64
    player1     211406 non-null object
    team        218824 non-null object
    player2     190897 non-null object
    subtype     116380 non-null object
    dtypes: int64(1), object(4)
    memory usage: 16.5+ MB



```python
df_fouls.dropna(subset=['player2'], inplace=True)
df_fouls.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 190897 entries, 0 to 432880
    Data columns (total 5 columns):
    match_id    190897 non-null int64
    player1     189339 non-null object
    team        190896 non-null object
    player2     190897 non-null object
    subtype     105118 non-null object
    dtypes: int64(1), object(4)
    memory usage: 8.7+ MB



```python
df_fouls['player2']=df_fouls['player2'].astype(int)
```


```python
penalty_per_player_counts_df=df_fouls[df_fouls['subtype']=='penalty'].groupby('player2').count()
penalty_per_player_counts_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>match_id</th>
      <th>player1</th>
      <th>team</th>
      <th>subtype</th>
    </tr>
    <tr>
      <th>player2</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2983</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3512</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3520</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8985</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11496</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_given_penalty_count=df_fouls[df_fouls['subtype']=='penalty'].groupby('player2').count()
```


```python
maximum_granted_penalties_player=df_given_penalty_count.max()['match_id'].item()
maximum_granted_penalties_player
```




    18




```python
best_penalty_taking_player=penalty_per_player_counts_df[penalty_per_player_counts_df['team']==maximum_granted_penalties_player]['subtype'].index[0].item()
```


```python
df_player[df_player['player_api_id']==best_penalty_taking_player]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>player_api_id</th>
      <th>player_name</th>
      <th>player_fifa_api_id</th>
      <th>birthday</th>
      <th>height</th>
      <th>weight</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6377</th>
      <td>40636</td>
      <td>Luis Suarez</td>
      <td>176580</td>
      <td>1987-01-24</td>
      <td>182</td>
      <td>187</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_given_penalty_count.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>match_id</th>
      <th>player1</th>
      <th>team</th>
      <th>subtype</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1030.000000</td>
      <td>1030.000000</td>
      <td>1030.000000</td>
      <td>1030.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.768932</td>
      <td>1.763107</td>
      <td>1.768932</td>
      <td>1.768932</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.534509</td>
      <td>1.532986</td>
      <td>1.534509</td>
      <td>1.534509</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>18.000000</td>
      <td>18.000000</td>
      <td>18.000000</td>
      <td>18.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_top_penalty_getters=df_given_penalty_count[df_given_penalty_count['subtype']>10]
df_top_penalty_getters
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>match_id</th>
      <th>player1</th>
      <th>team</th>
      <th>subtype</th>
    </tr>
    <tr>
      <th>player2</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19533</th>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
    </tr>
    <tr>
      <th>30893</th>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
    </tr>
    <tr>
      <th>35724</th>
      <td>11</td>
      <td>11</td>
      <td>11</td>
      <td>11</td>
    </tr>
    <tr>
      <th>40636</th>
      <td>18</td>
      <td>18</td>
      <td>18</td>
      <td>18</td>
    </tr>
    <tr>
      <th>49677</th>
      <td>11</td>
      <td>11</td>
      <td>11</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_top_penalty_getters_player_details=df_top_penalty_getters.merge(df_player,left_index=True, right_on='player_api_id', how='inner')[['player_name', 'subtype']].sort_values(['subtype'], ascending=False)
df_top_penalty_getters_player_details.rename(columns={'subtype':'penalties_granted'}, inplace=True)
```

### Research Question 3  (What team attributes lead to the most victories?)


```python
df_match['year']=df_match['date'].dt.year
```


```python
df_match["home_team_win"]=df_match['home_team_goal']>df_match['away_team_goal']
df_match["home_team_loss"]=df_match['home_team_goal']<df_match['away_team_goal']
df_match["home_team_draw"]=df_match['home_team_goal']==df_match['away_team_goal']
df_match["away_team_win"]=df_match['home_team_goal']<df_match['away_team_goal']
df_match["away_team_loss"]=df_match['home_team_goal']>df_match['away_team_goal']
df_match["away_team_draw"]=df_match['home_team_goal']==df_match['away_team_goal']
df_match.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>match_api_id</th>
      <th>home_team_api_id</th>
      <th>away_team_api_id</th>
      <th>home_team_goal</th>
      <th>away_team_goal</th>
      <th>goal</th>
      <th>shoton</th>
      <th>shotoff</th>
      <th>foulcommit</th>
      <th>year</th>
      <th>home_team_win</th>
      <th>home_team_loss</th>
      <th>home_team_draw</th>
      <th>away_team_win</th>
      <th>away_team_loss</th>
      <th>away_team_draw</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2008-08-17</td>
      <td>492473</td>
      <td>9987</td>
      <td>9993</td>
      <td>1</td>
      <td>1</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>2008</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2008-08-16</td>
      <td>492474</td>
      <td>10000</td>
      <td>9994</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>2008</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2008-08-16</td>
      <td>492475</td>
      <td>9984</td>
      <td>8635</td>
      <td>0</td>
      <td>3</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>2008</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2008-08-17</td>
      <td>492476</td>
      <td>9991</td>
      <td>9998</td>
      <td>5</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>2008</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2008-08-16</td>
      <td>492477</td>
      <td>7947</td>
      <td>9985</td>
      <td>1</td>
      <td>3</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>2008</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_home_team_results=df_match.groupby(['year', 'home_team_api_id']).sum()[['home_team_win','home_team_loss','home_team_draw']].reset_index()
df_home_team_results.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>home_team_api_id</th>
      <th>home_team_win</th>
      <th>home_team_loss</th>
      <th>home_team_draw</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2008</td>
      <td>1601</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2008</td>
      <td>1957</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2008</td>
      <td>2182</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2008</td>
      <td>2183</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2008</td>
      <td>2186</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_home_team_results['total_played']=df_home_team_results['home_team_win']+df_home_team_results['home_team_loss']+df_home_team_results['home_team_draw']
df_home_team_results.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>home_team_api_id</th>
      <th>home_team_win</th>
      <th>home_team_loss</th>
      <th>home_team_draw</th>
      <th>total_played</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2008</td>
      <td>1601</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2008</td>
      <td>1957</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2008</td>
      <td>2182</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2008</td>
      <td>2183</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2008</td>
      <td>2186</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>8.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_away_team_results=df_match.groupby(['year', 'away_team_api_id']).sum()[['away_team_win','away_team_loss','away_team_draw']].reset_index()
df_away_team_results.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>away_team_api_id</th>
      <th>away_team_win</th>
      <th>away_team_loss</th>
      <th>away_team_draw</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2008</td>
      <td>1601</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2008</td>
      <td>1957</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2008</td>
      <td>2182</td>
      <td>6.0</td>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2008</td>
      <td>2183</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2008</td>
      <td>2186</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_away_team_results['total_played']=df_away_team_results['away_team_win']+df_away_team_results['away_team_loss']+df_away_team_results['away_team_draw']
df_away_team_results.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>away_team_api_id</th>
      <th>away_team_win</th>
      <th>away_team_loss</th>
      <th>away_team_draw</th>
      <th>total_played</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2008</td>
      <td>1601</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2008</td>
      <td>1957</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2008</td>
      <td>2182</td>
      <td>6.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2008</td>
      <td>2183</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2008</td>
      <td>2186</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>2.0</td>
      <td>9.0</td>
    </tr>
  </tbody>
</table>
</div>



>**In the above cells, I made two separate dataframes, one for the results of teams at home, and the other is for the results away.**


```python
df_combined_results =df_home_team_results.merge(df_away_team_results,left_on=['year','home_team_api_id'], right_on=['year','away_team_api_id'],how='outer')
df_combined_results.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>home_team_api_id</th>
      <th>home_team_win</th>
      <th>home_team_loss</th>
      <th>home_team_draw</th>
      <th>total_played_x</th>
      <th>away_team_api_id</th>
      <th>away_team_win</th>
      <th>away_team_loss</th>
      <th>away_team_draw</th>
      <th>total_played_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2008</td>
      <td>1601</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>9.0</td>
      <td>1601</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2008</td>
      <td>1957</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>8.0</td>
      <td>1957</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2008</td>
      <td>2182</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>8.0</td>
      <td>2182</td>
      <td>6.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2008</td>
      <td>2183</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>9.0</td>
      <td>2183</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2008</td>
      <td>2186</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>8.0</td>
      <td>2186</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>2.0</td>
      <td>9.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_combined_results['total_wins']=df_combined_results['home_team_win']+df_combined_results['away_team_win']
df_combined_results['total_played']=df_combined_results['home_team_win']+df_combined_results['home_team_loss']+df_combined_results['home_team_draw']+df_combined_results['away_team_win']+df_combined_results['away_team_loss']+df_combined_results['away_team_draw']
df_combined_results['win_percent']=df_combined_results['total_wins']/df_combined_results['total_played']
df_win_scores=df_combined_results[['year', 'home_team_api_id', 'total_wins', 'total_played', 'win_percent']]
```


```python
df_win_scores.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>home_team_api_id</th>
      <th>total_wins</th>
      <th>total_played</th>
      <th>win_percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2008</td>
      <td>1601</td>
      <td>6.0</td>
      <td>17.0</td>
      <td>0.352941</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2008</td>
      <td>1957</td>
      <td>4.0</td>
      <td>17.0</td>
      <td>0.235294</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2008</td>
      <td>2182</td>
      <td>11.0</td>
      <td>17.0</td>
      <td>0.647059</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2008</td>
      <td>2183</td>
      <td>10.0</td>
      <td>17.0</td>
      <td>0.588235</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2008</td>
      <td>2186</td>
      <td>3.0</td>
      <td>17.0</td>
      <td>0.176471</td>
    </tr>
  </tbody>
</table>
</div>



>**Then I combined both dataframes, and extracted the winning scores ('df_win_scores dataframe').**


```python
df_win_scores.rename(columns={'home_team_api_id':'team_api_id'}, inplace=True)
```

    /opt/conda/lib/python3.6/site-packages/pandas/core/frame.py:3781: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      return super(DataFrame, self).rename(**kwargs)



```python
df_team_attributes["year"]=df_team_attributes.date.dt.year
```


```python
df_team_attributes_plus_win_stats=df_team_attributes.merge(df_win_scores,left_on=['year','team_api_id'], right_on=['year','team_api_id'],how='inner')
df_team_attributes_plus_win_stats.head(30)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>team_fifa_api_id</th>
      <th>team_api_id</th>
      <th>date</th>
      <th>buildUpPlaySpeed</th>
      <th>buildUpPlaySpeedClass</th>
      <th>buildUpPlayDribblingClass</th>
      <th>buildUpPlayPassing</th>
      <th>buildUpPlayPassingClass</th>
      <th>buildUpPlayPositioningClass</th>
      <th>chanceCreationPassing</th>
      <th>...</th>
      <th>defencePressureClass</th>
      <th>defenceAggression</th>
      <th>defenceAggressionClass</th>
      <th>defenceTeamWidth</th>
      <th>defenceTeamWidthClass</th>
      <th>defenceDefenderLineClass</th>
      <th>year</th>
      <th>total_wins</th>
      <th>total_played</th>
      <th>win_percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>434</td>
      <td>9930</td>
      <td>2010-02-22</td>
      <td>60</td>
      <td>Balanced</td>
      <td>Little</td>
      <td>50</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>60</td>
      <td>...</td>
      <td>Medium</td>
      <td>55</td>
      <td>Press</td>
      <td>45</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>2010</td>
      <td>5.0</td>
      <td>18.0</td>
      <td>0.277778</td>
    </tr>
    <tr>
      <th>1</th>
      <td>434</td>
      <td>9930</td>
      <td>2014-09-19</td>
      <td>52</td>
      <td>Balanced</td>
      <td>Normal</td>
      <td>56</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>54</td>
      <td>...</td>
      <td>Medium</td>
      <td>44</td>
      <td>Press</td>
      <td>54</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>2014</td>
      <td>10.0</td>
      <td>37.0</td>
      <td>0.270270</td>
    </tr>
    <tr>
      <th>2</th>
      <td>434</td>
      <td>9930</td>
      <td>2015-09-10</td>
      <td>47</td>
      <td>Balanced</td>
      <td>Normal</td>
      <td>54</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>54</td>
      <td>...</td>
      <td>Medium</td>
      <td>44</td>
      <td>Press</td>
      <td>54</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>2015</td>
      <td>3.0</td>
      <td>18.0</td>
      <td>0.166667</td>
    </tr>
    <tr>
      <th>3</th>
      <td>77</td>
      <td>8485</td>
      <td>2010-02-22</td>
      <td>70</td>
      <td>Fast</td>
      <td>Little</td>
      <td>70</td>
      <td>Long</td>
      <td>Organised</td>
      <td>70</td>
      <td>...</td>
      <td>Medium</td>
      <td>70</td>
      <td>Double</td>
      <td>70</td>
      <td>Wide</td>
      <td>Cover</td>
      <td>2010</td>
      <td>11.0</td>
      <td>40.0</td>
      <td>0.275000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>77</td>
      <td>8485</td>
      <td>2011-02-22</td>
      <td>47</td>
      <td>Balanced</td>
      <td>Little</td>
      <td>52</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>53</td>
      <td>...</td>
      <td>Medium</td>
      <td>47</td>
      <td>Press</td>
      <td>52</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>2011</td>
      <td>11.0</td>
      <td>41.0</td>
      <td>0.268293</td>
    </tr>
    <tr>
      <th>5</th>
      <td>77</td>
      <td>8485</td>
      <td>2012-02-22</td>
      <td>58</td>
      <td>Balanced</td>
      <td>Little</td>
      <td>62</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>45</td>
      <td>...</td>
      <td>Medium</td>
      <td>40</td>
      <td>Press</td>
      <td>60</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>2012</td>
      <td>12.0</td>
      <td>38.0</td>
      <td>0.315789</td>
    </tr>
    <tr>
      <th>6</th>
      <td>77</td>
      <td>8485</td>
      <td>2013-09-20</td>
      <td>62</td>
      <td>Balanced</td>
      <td>Little</td>
      <td>45</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>40</td>
      <td>...</td>
      <td>Medium</td>
      <td>42</td>
      <td>Press</td>
      <td>60</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>2013</td>
      <td>14.0</td>
      <td>36.0</td>
      <td>0.388889</td>
    </tr>
    <tr>
      <th>7</th>
      <td>77</td>
      <td>8485</td>
      <td>2014-09-19</td>
      <td>58</td>
      <td>Balanced</td>
      <td>Normal</td>
      <td>62</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>56</td>
      <td>...</td>
      <td>Medium</td>
      <td>42</td>
      <td>Press</td>
      <td>60</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>2014</td>
      <td>21.0</td>
      <td>37.0</td>
      <td>0.567568</td>
    </tr>
    <tr>
      <th>8</th>
      <td>77</td>
      <td>8485</td>
      <td>2015-09-10</td>
      <td>59</td>
      <td>Balanced</td>
      <td>Normal</td>
      <td>53</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>51</td>
      <td>...</td>
      <td>Medium</td>
      <td>45</td>
      <td>Press</td>
      <td>63</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>2015</td>
      <td>24.0</td>
      <td>41.0</td>
      <td>0.585366</td>
    </tr>
    <tr>
      <th>9</th>
      <td>614</td>
      <td>8576</td>
      <td>2011-02-22</td>
      <td>65</td>
      <td>Balanced</td>
      <td>Little</td>
      <td>45</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>65</td>
      <td>...</td>
      <td>Medium</td>
      <td>45</td>
      <td>Press</td>
      <td>50</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>2011</td>
      <td>3.0</td>
      <td>19.0</td>
      <td>0.157895</td>
    </tr>
    <tr>
      <th>10</th>
      <td>614</td>
      <td>8576</td>
      <td>2012-02-22</td>
      <td>59</td>
      <td>Balanced</td>
      <td>Little</td>
      <td>52</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>48</td>
      <td>...</td>
      <td>Medium</td>
      <td>47</td>
      <td>Press</td>
      <td>53</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>2012</td>
      <td>11.0</td>
      <td>38.0</td>
      <td>0.289474</td>
    </tr>
    <tr>
      <th>11</th>
      <td>614</td>
      <td>8576</td>
      <td>2013-09-20</td>
      <td>59</td>
      <td>Balanced</td>
      <td>Little</td>
      <td>52</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>48</td>
      <td>...</td>
      <td>Medium</td>
      <td>47</td>
      <td>Press</td>
      <td>53</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>2013</td>
      <td>5.0</td>
      <td>38.0</td>
      <td>0.131579</td>
    </tr>
    <tr>
      <th>12</th>
      <td>614</td>
      <td>8576</td>
      <td>2014-09-19</td>
      <td>59</td>
      <td>Balanced</td>
      <td>Normal</td>
      <td>52</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>48</td>
      <td>...</td>
      <td>Medium</td>
      <td>47</td>
      <td>Press</td>
      <td>53</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>2014</td>
      <td>3.0</td>
      <td>19.0</td>
      <td>0.157895</td>
    </tr>
    <tr>
      <th>13</th>
      <td>47</td>
      <td>8564</td>
      <td>2010-02-22</td>
      <td>45</td>
      <td>Balanced</td>
      <td>Little</td>
      <td>30</td>
      <td>Short</td>
      <td>Free Form</td>
      <td>55</td>
      <td>...</td>
      <td>Deep</td>
      <td>35</td>
      <td>Press</td>
      <td>60</td>
      <td>Normal</td>
      <td>Offside Trap</td>
      <td>2010</td>
      <td>22.0</td>
      <td>39.0</td>
      <td>0.564103</td>
    </tr>
    <tr>
      <th>14</th>
      <td>47</td>
      <td>8564</td>
      <td>2011-02-22</td>
      <td>65</td>
      <td>Balanced</td>
      <td>Little</td>
      <td>50</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>50</td>
      <td>...</td>
      <td>Medium</td>
      <td>50</td>
      <td>Press</td>
      <td>50</td>
      <td>Normal</td>
      <td>Offside Trap</td>
      <td>2011</td>
      <td>23.0</td>
      <td>37.0</td>
      <td>0.621622</td>
    </tr>
    <tr>
      <th>15</th>
      <td>47</td>
      <td>8564</td>
      <td>2012-02-22</td>
      <td>45</td>
      <td>Balanced</td>
      <td>Little</td>
      <td>50</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>65</td>
      <td>...</td>
      <td>Medium</td>
      <td>45</td>
      <td>Press</td>
      <td>50</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>2012</td>
      <td>21.0</td>
      <td>39.0</td>
      <td>0.538462</td>
    </tr>
    <tr>
      <th>16</th>
      <td>47</td>
      <td>8564</td>
      <td>2013-09-20</td>
      <td>48</td>
      <td>Balanced</td>
      <td>Little</td>
      <td>54</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>51</td>
      <td>...</td>
      <td>Medium</td>
      <td>49</td>
      <td>Press</td>
      <td>53</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>2013</td>
      <td>17.0</td>
      <td>37.0</td>
      <td>0.459459</td>
    </tr>
    <tr>
      <th>17</th>
      <td>47</td>
      <td>8564</td>
      <td>2014-09-19</td>
      <td>48</td>
      <td>Balanced</td>
      <td>Lots</td>
      <td>52</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>66</td>
      <td>...</td>
      <td>Medium</td>
      <td>57</td>
      <td>Press</td>
      <td>49</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>2014</td>
      <td>18.0</td>
      <td>37.0</td>
      <td>0.486486</td>
    </tr>
    <tr>
      <th>18</th>
      <td>47</td>
      <td>8564</td>
      <td>2015-09-10</td>
      <td>48</td>
      <td>Balanced</td>
      <td>Lots</td>
      <td>52</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>66</td>
      <td>...</td>
      <td>Medium</td>
      <td>57</td>
      <td>Press</td>
      <td>49</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>2015</td>
      <td>15.0</td>
      <td>39.0</td>
      <td>0.384615</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1901</td>
      <td>10215</td>
      <td>2010-02-22</td>
      <td>30</td>
      <td>Slow</td>
      <td>Little</td>
      <td>30</td>
      <td>Short</td>
      <td>Organised</td>
      <td>50</td>
      <td>...</td>
      <td>Deep</td>
      <td>30</td>
      <td>Contain</td>
      <td>30</td>
      <td>Narrow</td>
      <td>Offside Trap</td>
      <td>2010</td>
      <td>10.0</td>
      <td>30.0</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1901</td>
      <td>10215</td>
      <td>2011-02-22</td>
      <td>30</td>
      <td>Slow</td>
      <td>Little</td>
      <td>50</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>50</td>
      <td>...</td>
      <td>Deep</td>
      <td>45</td>
      <td>Press</td>
      <td>65</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>2011</td>
      <td>7.0</td>
      <td>29.0</td>
      <td>0.241379</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1901</td>
      <td>10215</td>
      <td>2012-02-22</td>
      <td>45</td>
      <td>Balanced</td>
      <td>Little</td>
      <td>44</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>55</td>
      <td>...</td>
      <td>Medium</td>
      <td>38</td>
      <td>Press</td>
      <td>61</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>2012</td>
      <td>3.0</td>
      <td>29.0</td>
      <td>0.103448</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1901</td>
      <td>10215</td>
      <td>2013-09-20</td>
      <td>45</td>
      <td>Balanced</td>
      <td>Little</td>
      <td>44</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>55</td>
      <td>...</td>
      <td>Medium</td>
      <td>38</td>
      <td>Press</td>
      <td>61</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>2013</td>
      <td>9.0</td>
      <td>32.0</td>
      <td>0.281250</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1901</td>
      <td>10215</td>
      <td>2014-09-19</td>
      <td>52</td>
      <td>Balanced</td>
      <td>Normal</td>
      <td>44</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>55</td>
      <td>...</td>
      <td>Medium</td>
      <td>38</td>
      <td>Press</td>
      <td>61</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>2014</td>
      <td>6.0</td>
      <td>30.0</td>
      <td>0.200000</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1901</td>
      <td>10215</td>
      <td>2015-09-10</td>
      <td>53</td>
      <td>Balanced</td>
      <td>Normal</td>
      <td>44</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>55</td>
      <td>...</td>
      <td>Medium</td>
      <td>38</td>
      <td>Press</td>
      <td>61</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>2015</td>
      <td>5.0</td>
      <td>34.0</td>
      <td>0.147059</td>
    </tr>
    <tr>
      <th>25</th>
      <td>650</td>
      <td>10217</td>
      <td>2010-02-22</td>
      <td>30</td>
      <td>Slow</td>
      <td>Little</td>
      <td>35</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>30</td>
      <td>...</td>
      <td>Deep</td>
      <td>30</td>
      <td>Contain</td>
      <td>30</td>
      <td>Narrow</td>
      <td>Cover</td>
      <td>2010</td>
      <td>12.0</td>
      <td>35.0</td>
      <td>0.342857</td>
    </tr>
    <tr>
      <th>26</th>
      <td>650</td>
      <td>10217</td>
      <td>2011-02-22</td>
      <td>53</td>
      <td>Balanced</td>
      <td>Little</td>
      <td>53</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>48</td>
      <td>...</td>
      <td>Medium</td>
      <td>62</td>
      <td>Press</td>
      <td>65</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>2011</td>
      <td>14.0</td>
      <td>32.0</td>
      <td>0.437500</td>
    </tr>
    <tr>
      <th>27</th>
      <td>650</td>
      <td>10217</td>
      <td>2012-02-22</td>
      <td>38</td>
      <td>Balanced</td>
      <td>Little</td>
      <td>44</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>55</td>
      <td>...</td>
      <td>Medium</td>
      <td>58</td>
      <td>Press</td>
      <td>62</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>2012</td>
      <td>7.0</td>
      <td>35.0</td>
      <td>0.200000</td>
    </tr>
    <tr>
      <th>28</th>
      <td>650</td>
      <td>10217</td>
      <td>2013-09-20</td>
      <td>38</td>
      <td>Balanced</td>
      <td>Little</td>
      <td>47</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>55</td>
      <td>...</td>
      <td>Medium</td>
      <td>58</td>
      <td>Press</td>
      <td>52</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>2013</td>
      <td>9.0</td>
      <td>34.0</td>
      <td>0.264706</td>
    </tr>
    <tr>
      <th>29</th>
      <td>650</td>
      <td>10217</td>
      <td>2014-09-19</td>
      <td>58</td>
      <td>Balanced</td>
      <td>Normal</td>
      <td>55</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>55</td>
      <td>...</td>
      <td>Medium</td>
      <td>50</td>
      <td>Press</td>
      <td>52</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>2014</td>
      <td>10.0</td>
      <td>33.0</td>
      <td>0.303030</td>
    </tr>
  </tbody>
</table>
<p>30 rows × 27 columns</p>
</div>



>**Then, I merged df_win_scores with df_team_attributes, to be able to check which team attributes results in better win scores.**


```python
df_win_scores.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>team_api_id</th>
      <th>total_wins</th>
      <th>total_played</th>
      <th>win_percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1850.000000</td>
      <td>1850.000000</td>
      <td>1850.000000</td>
      <td>1850.000000</td>
      <td>1850.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2012.002703</td>
      <td>10362.916757</td>
      <td>10.477297</td>
      <td>28.085405</td>
      <td>0.360381</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.541062</td>
      <td>16805.631724</td>
      <td>6.261541</td>
      <td>9.389087</td>
      <td>0.161133</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2008.000000</td>
      <td>1601.000000</td>
      <td>0.000000</td>
      <td>8.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2010.000000</td>
      <td>8462.000000</td>
      <td>5.000000</td>
      <td>19.000000</td>
      <td>0.243243</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2012.000000</td>
      <td>8689.000000</td>
      <td>10.000000</td>
      <td>32.000000</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2014.000000</td>
      <td>9925.000000</td>
      <td>14.000000</td>
      <td>37.000000</td>
      <td>0.450000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2016.000000</td>
      <td>274581.000000</td>
      <td>33.000000</td>
      <td>44.000000</td>
      <td>0.950000</td>
    </tr>
  </tbody>
</table>
</div>



### [Comment] I'll consider the high-vecoty teams are the teams that are in Q4 of 'win_percent', i.e. the teams that have 'win_percent' of more than 0.45


```python
df_high_vectory_team_attrib=df_team_attributes_plus_win_stats[df_team_attributes_plus_win_stats['win_percent']>.45]
df_high_vectory_team_attrib.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>team_fifa_api_id</th>
      <th>team_api_id</th>
      <th>date</th>
      <th>buildUpPlaySpeed</th>
      <th>buildUpPlaySpeedClass</th>
      <th>buildUpPlayDribblingClass</th>
      <th>buildUpPlayPassing</th>
      <th>buildUpPlayPassingClass</th>
      <th>buildUpPlayPositioningClass</th>
      <th>chanceCreationPassing</th>
      <th>...</th>
      <th>defencePressureClass</th>
      <th>defenceAggression</th>
      <th>defenceAggressionClass</th>
      <th>defenceTeamWidth</th>
      <th>defenceTeamWidthClass</th>
      <th>defenceDefenderLineClass</th>
      <th>year</th>
      <th>total_wins</th>
      <th>total_played</th>
      <th>win_percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>77</td>
      <td>8485</td>
      <td>2014-09-19</td>
      <td>58</td>
      <td>Balanced</td>
      <td>Normal</td>
      <td>62</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>56</td>
      <td>...</td>
      <td>Medium</td>
      <td>42</td>
      <td>Press</td>
      <td>60</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>2014</td>
      <td>21.0</td>
      <td>37.0</td>
      <td>0.567568</td>
    </tr>
    <tr>
      <th>8</th>
      <td>77</td>
      <td>8485</td>
      <td>2015-09-10</td>
      <td>59</td>
      <td>Balanced</td>
      <td>Normal</td>
      <td>53</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>51</td>
      <td>...</td>
      <td>Medium</td>
      <td>45</td>
      <td>Press</td>
      <td>63</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>2015</td>
      <td>24.0</td>
      <td>41.0</td>
      <td>0.585366</td>
    </tr>
    <tr>
      <th>13</th>
      <td>47</td>
      <td>8564</td>
      <td>2010-02-22</td>
      <td>45</td>
      <td>Balanced</td>
      <td>Little</td>
      <td>30</td>
      <td>Short</td>
      <td>Free Form</td>
      <td>55</td>
      <td>...</td>
      <td>Deep</td>
      <td>35</td>
      <td>Press</td>
      <td>60</td>
      <td>Normal</td>
      <td>Offside Trap</td>
      <td>2010</td>
      <td>22.0</td>
      <td>39.0</td>
      <td>0.564103</td>
    </tr>
    <tr>
      <th>14</th>
      <td>47</td>
      <td>8564</td>
      <td>2011-02-22</td>
      <td>65</td>
      <td>Balanced</td>
      <td>Little</td>
      <td>50</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>50</td>
      <td>...</td>
      <td>Medium</td>
      <td>50</td>
      <td>Press</td>
      <td>50</td>
      <td>Normal</td>
      <td>Offside Trap</td>
      <td>2011</td>
      <td>23.0</td>
      <td>37.0</td>
      <td>0.621622</td>
    </tr>
    <tr>
      <th>15</th>
      <td>47</td>
      <td>8564</td>
      <td>2012-02-22</td>
      <td>45</td>
      <td>Balanced</td>
      <td>Little</td>
      <td>50</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>65</td>
      <td>...</td>
      <td>Medium</td>
      <td>45</td>
      <td>Press</td>
      <td>50</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>2012</td>
      <td>21.0</td>
      <td>39.0</td>
      <td>0.538462</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>




```python
pd.plotting.scatter_matrix(df_high_vectory_team_attrib.drop(columns=['team_fifa_api_id', 'team_api_id', 'date', 'buildUpPlaySpeedClass', 'buildUpPlayDribblingClass', 'buildUpPlayPassingClass', 'buildUpPlayPositioningClass', 'defencePressureClass', 'defenceAggressionClass', 'defenceTeamWidthClass', 'year', 'total_played']), figsize=(25,25));
```


![png](output_108_0.png)


### [Comment] The Scatter Plot is not showing a strong correlations between the 'win_percent' and the numerical team attributes


```python
attrib_class_list=[i for i in list(df_high_vectory_team_attrib.columns) if i[-5:]=='Class']
attrib_class_list
```




    ['buildUpPlaySpeedClass',
     'buildUpPlayDribblingClass',
     'buildUpPlayPassingClass',
     'buildUpPlayPositioningClass',
     'chanceCreationPassingClass',
     'chanceCreationCrossingClass',
     'chanceCreationShootingClass',
     'chanceCreationPositioningClass',
     'defencePressureClass',
     'defenceAggressionClass',
     'defenceTeamWidthClass',
     'defenceDefenderLineClass']




```python
df_p1=df_high_vectory_team_attrib['buildUpPlaySpeedClass'].value_counts()
df_p2=df_high_vectory_team_attrib['buildUpPlayDribblingClass'].value_counts()
df_p3=df_high_vectory_team_attrib['buildUpPlayPassingClass'].value_counts()
df_p4=df_high_vectory_team_attrib['buildUpPlayPositioningClass'].value_counts()
df_p5=df_high_vectory_team_attrib['chanceCreationPassingClass'].value_counts()
df_p6=df_high_vectory_team_attrib['chanceCreationCrossingClass'].value_counts()
df_p7=df_high_vectory_team_attrib['chanceCreationShootingClass'].value_counts()
df_p8=df_high_vectory_team_attrib['chanceCreationPositioningClass'].value_counts()
df_p9=df_high_vectory_team_attrib['defencePressureClass'].value_counts()
df_p10=df_high_vectory_team_attrib['defenceAggressionClass'].value_counts()
df_p11=df_high_vectory_team_attrib['defenceTeamWidthClass'].value_counts()
df_p12=df_high_vectory_team_attrib['defenceDefenderLineClass'].value_counts()
df_list=[df_p1, df_p2, df_p3, df_p4, df_p5, df_p6, df_p7, df_p8, df_p9, df_p10, df_p11, df_p12]
#len(df_list)
```


```python
num_cols=3
num_rows=math.ceil(len(df_list)/num_cols)
#print(num_cols, num_rows)
```


```python
plt.figure(figsize=(20, 20))
for i in range(len(df_list)):
    plt.subplot(num_rows,num_cols,i+1)
    df_list[i].plot(kind='pie',autopct='%.2f')
```


![png](output_113_0.png)


### [Comment] The above pie charts show the attribute classes of the high vectory teams. The 'blue' part of the above pie charts show that the majority of the high vectory teams apply the same class in each attribute.

## Research Question 4 (What teams improved the most over the time period?)


```python
# Use this, and more code cells, to explore your data. Don't forget to add
#   Markdown cells to document your observations and findings.

```


```python
years_list=list(df_win_scores.year.unique())
years_list
```




    [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016]




```python
first_year=min(years_list)
first_year
```




    2008




```python
last_year=max(years_list)
last_year
```




    2016



### [Comment] So, in the upcoming cells I'll compare the 'win_percent' of the teams in years 2008 and 2016 to explore the teams that improved the most.


```python
df_team_wins_first_year=df_win_scores[df_win_scores.year==first_year]
df_team_wins_first_year.drop(columns=['year'], inplace=True)
df_team_wins_first_year.head()
```

    /opt/conda/lib/python3.6/site-packages/pandas/core/frame.py:3697: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      errors=errors)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>team_api_id</th>
      <th>total_wins</th>
      <th>total_played</th>
      <th>win_percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1601</td>
      <td>6.0</td>
      <td>17.0</td>
      <td>0.352941</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1957</td>
      <td>4.0</td>
      <td>17.0</td>
      <td>0.235294</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2182</td>
      <td>11.0</td>
      <td>17.0</td>
      <td>0.647059</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2183</td>
      <td>10.0</td>
      <td>17.0</td>
      <td>0.588235</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2186</td>
      <td>3.0</td>
      <td>17.0</td>
      <td>0.176471</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_team_wins_last_year=df_win_scores[df_win_scores.year==last_year]
df_team_wins_last_year.drop(columns=['year'], inplace=True)
df_team_wins_last_year.head()
```

    /opt/conda/lib/python3.6/site-packages/pandas/core/frame.py:3697: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      errors=errors)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>team_api_id</th>
      <th>total_wins</th>
      <th>total_played</th>
      <th>win_percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1662</th>
      <td>1601</td>
      <td>2.0</td>
      <td>9.0</td>
      <td>0.222222</td>
    </tr>
    <tr>
      <th>1663</th>
      <td>1773</td>
      <td>2.0</td>
      <td>9.0</td>
      <td>0.222222</td>
    </tr>
    <tr>
      <th>1664</th>
      <td>1957</td>
      <td>3.0</td>
      <td>9.0</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>1665</th>
      <td>2182</td>
      <td>5.0</td>
      <td>9.0</td>
      <td>0.555556</td>
    </tr>
    <tr>
      <th>1666</th>
      <td>2186</td>
      <td>2.0</td>
      <td>9.0</td>
      <td>0.222222</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_team_wins_last_year.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>team_api_id</th>
      <th>total_wins</th>
      <th>total_played</th>
      <th>win_percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1845</th>
      <td>158085</td>
      <td>9.0</td>
      <td>20.0</td>
      <td>0.450000</td>
    </tr>
    <tr>
      <th>1846</th>
      <td>177361</td>
      <td>2.0</td>
      <td>9.0</td>
      <td>0.222222</td>
    </tr>
    <tr>
      <th>1847</th>
      <td>188163</td>
      <td>7.0</td>
      <td>20.0</td>
      <td>0.350000</td>
    </tr>
    <tr>
      <th>1848</th>
      <td>208931</td>
      <td>7.0</td>
      <td>21.0</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>1849</th>
      <td>274581</td>
      <td>2.0</td>
      <td>9.0</td>
      <td>0.222222</td>
    </tr>
  </tbody>
</table>
</div>




```python
# rename first year columns
df_team_wins_first_year.rename(columns=lambda x: x[:15] + "_first", inplace=True)
df_team_wins_first_year.head()
```

    /opt/conda/lib/python3.6/site-packages/pandas/core/frame.py:3781: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      return super(DataFrame, self).rename(**kwargs)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>team_api_id_first</th>
      <th>total_wins_first</th>
      <th>total_played_first</th>
      <th>win_percent_first</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1601</td>
      <td>6.0</td>
      <td>17.0</td>
      <td>0.352941</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1957</td>
      <td>4.0</td>
      <td>17.0</td>
      <td>0.235294</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2182</td>
      <td>11.0</td>
      <td>17.0</td>
      <td>0.647059</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2183</td>
      <td>10.0</td>
      <td>17.0</td>
      <td>0.588235</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2186</td>
      <td>3.0</td>
      <td>17.0</td>
      <td>0.176471</td>
    </tr>
  </tbody>
</table>
</div>




```python
# rename last year columns
df_team_wins_last_year.rename(columns=lambda x: x[:15] + "_last", inplace=True)
df_team_wins_last_year.head()
```

    /opt/conda/lib/python3.6/site-packages/pandas/core/frame.py:3781: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      return super(DataFrame, self).rename(**kwargs)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>team_api_id_last</th>
      <th>total_wins_last</th>
      <th>total_played_last</th>
      <th>win_percent_last</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1662</th>
      <td>1601</td>
      <td>2.0</td>
      <td>9.0</td>
      <td>0.222222</td>
    </tr>
    <tr>
      <th>1663</th>
      <td>1773</td>
      <td>2.0</td>
      <td>9.0</td>
      <td>0.222222</td>
    </tr>
    <tr>
      <th>1664</th>
      <td>1957</td>
      <td>3.0</td>
      <td>9.0</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>1665</th>
      <td>2182</td>
      <td>5.0</td>
      <td>9.0</td>
      <td>0.555556</td>
    </tr>
    <tr>
      <th>1666</th>
      <td>2186</td>
      <td>2.0</td>
      <td>9.0</td>
      <td>0.222222</td>
    </tr>
  </tbody>
</table>
</div>




```python
# merge datasets
df_wins_comparison =df_team_wins_first_year.merge(df_team_wins_last_year,left_on='team_api_id_first', right_on='team_api_id_last',how='inner')
```

### [Comment] If I divided the 'win_percent_last' over the 'win_percent_first', I'll get the percentage of improvement or declination in WINNING.


```python
df_wins_comparison["win_percent_comparison"]=df_wins_comparison["win_percent_last"]/df_wins_comparison["win_percent_first"]
df_wins_comparison.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>team_api_id_first</th>
      <th>total_wins_first</th>
      <th>total_played_first</th>
      <th>win_percent_first</th>
      <th>team_api_id_last</th>
      <th>total_wins_last</th>
      <th>total_played_last</th>
      <th>win_percent_last</th>
      <th>win_percent_comparison</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>131.000000</td>
      <td>131.000000</td>
      <td>131.000000</td>
      <td>131.000000</td>
      <td>131.000000</td>
      <td>131.000000</td>
      <td>131.000000</td>
      <td>131.000000</td>
      <td>131.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>9047.305344</td>
      <td>7.160305</td>
      <td>17.083969</td>
      <td>0.418091</td>
      <td>9047.305344</td>
      <td>7.030534</td>
      <td>17.381679</td>
      <td>0.404073</td>
      <td>1.046792</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1503.857235</td>
      <td>2.900442</td>
      <td>2.019285</td>
      <td>0.160956</td>
      <td>1503.857235</td>
      <td>3.812810</td>
      <td>3.952522</td>
      <td>0.192141</td>
      <td>0.498723</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1601.000000</td>
      <td>2.000000</td>
      <td>12.000000</td>
      <td>0.117647</td>
      <td>1601.000000</td>
      <td>0.000000</td>
      <td>9.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>8537.500000</td>
      <td>5.000000</td>
      <td>17.000000</td>
      <td>0.294118</td>
      <td>8537.500000</td>
      <td>4.500000</td>
      <td>17.000000</td>
      <td>0.277778</td>
      <td>0.687302</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>9773.000000</td>
      <td>7.000000</td>
      <td>17.000000</td>
      <td>0.411765</td>
      <td>9773.000000</td>
      <td>6.000000</td>
      <td>19.000000</td>
      <td>0.352941</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>9934.500000</td>
      <td>9.000000</td>
      <td>18.500000</td>
      <td>0.539706</td>
      <td>9934.500000</td>
      <td>8.000000</td>
      <td>20.000000</td>
      <td>0.473684</td>
      <td>1.384848</td>
    </tr>
    <tr>
      <th>max</th>
      <td>10269.000000</td>
      <td>16.000000</td>
      <td>20.000000</td>
      <td>0.812500</td>
      <td>10269.000000</td>
      <td>19.000000</td>
      <td>22.000000</td>
      <td>0.950000</td>
      <td>2.833333</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_most_improved=df_wins_comparison[df_wins_comparison['win_percent_comparison']>1.384848]
```

>**I masked the df_wins_comparison based on a win_percent improvement of 138%, because this was the 75% of 'win_percent_comparison'**


```python
df_most_improved_teams=df_most_improved.merge(df_team, left_on='team_api_id_first', right_on='team_api_id', how='inner')[['team_long_name', 'win_percent_comparison']].sort_values('win_percent_comparison', ascending=False)
df_most_improved_teams
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>team_long_name</th>
      <th>win_percent_comparison</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11</th>
      <td>Chievo Verona</td>
      <td>2.833333</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Borussia Mönchengladbach</td>
      <td>2.666667</td>
    </tr>
    <tr>
      <th>30</th>
      <td>FC Luzern</td>
      <td>2.518519</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Rio Ave FC</td>
      <td>2.400000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Tottenham Hotspur</td>
      <td>2.105263</td>
    </tr>
    <tr>
      <th>28</th>
      <td>FC Sion</td>
      <td>1.888889</td>
    </tr>
    <tr>
      <th>22</th>
      <td>CF Os Belenenses</td>
      <td>1.800000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Inverness Caledonian Thistle</td>
      <td>1.777778</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Borussia Dortmund</td>
      <td>1.714286</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Club Brugge KV</td>
      <td>1.679012</td>
    </tr>
    <tr>
      <th>16</th>
      <td>PSV</td>
      <td>1.666667</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Roda JC Kerkrade</td>
      <td>1.666667</td>
    </tr>
    <tr>
      <th>18</th>
      <td>SL Benfica</td>
      <td>1.628571</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Feyenoord</td>
      <td>1.600000</td>
    </tr>
    <tr>
      <th>23</th>
      <td>FC Vaduz</td>
      <td>1.574074</td>
    </tr>
    <tr>
      <th>27</th>
      <td>KAA Gent</td>
      <td>1.574074</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Athletic Club de Bilbao</td>
      <td>1.523810</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Roma</td>
      <td>1.523810</td>
    </tr>
    <tr>
      <th>12</th>
      <td>RCD Espanyol</td>
      <td>1.523810</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Lechia Gdańsk</td>
      <td>1.511111</td>
    </tr>
    <tr>
      <th>13</th>
      <td>KV Kortrijk</td>
      <td>1.511111</td>
    </tr>
    <tr>
      <th>31</th>
      <td>ADO Den Haag</td>
      <td>1.500000</td>
    </tr>
    <tr>
      <th>24</th>
      <td>AS Monaco</td>
      <td>1.500000</td>
    </tr>
    <tr>
      <th>29</th>
      <td>BSC Young Boys</td>
      <td>1.444444</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Real Madrid CF</td>
      <td>1.439153</td>
    </tr>
    <tr>
      <th>7</th>
      <td>KV Mechelen</td>
      <td>1.416667</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Piast Gliwice</td>
      <td>1.416667</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Jagiellonia Białystok</td>
      <td>1.416667</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Manchester City</td>
      <td>1.403509</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Paris Saint-Germain</td>
      <td>1.400000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>FC Paços de Ferreira</td>
      <td>1.400000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>SM Caen</td>
      <td>1.400000</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Juventus</td>
      <td>1.398268</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_team_attributes.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>team_fifa_api_id</th>
      <th>team_api_id</th>
      <th>buildUpPlaySpeed</th>
      <th>buildUpPlayPassing</th>
      <th>chanceCreationPassing</th>
      <th>chanceCreationCrossing</th>
      <th>chanceCreationShooting</th>
      <th>defencePressure</th>
      <th>defenceAggression</th>
      <th>defenceTeamWidth</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1458.000000</td>
      <td>1458.000000</td>
      <td>1458.000000</td>
      <td>1458.000000</td>
      <td>1458.000000</td>
      <td>1458.000000</td>
      <td>1458.000000</td>
      <td>1458.000000</td>
      <td>1458.000000</td>
      <td>1458.000000</td>
      <td>1458.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>17706.982167</td>
      <td>9995.727023</td>
      <td>52.462277</td>
      <td>48.490398</td>
      <td>52.165295</td>
      <td>53.731824</td>
      <td>53.969136</td>
      <td>46.017147</td>
      <td>49.251029</td>
      <td>52.185871</td>
      <td>2012.506859</td>
    </tr>
    <tr>
      <th>std</th>
      <td>39179.857739</td>
      <td>13264.869900</td>
      <td>11.545869</td>
      <td>10.896101</td>
      <td>10.360793</td>
      <td>11.086796</td>
      <td>10.327566</td>
      <td>10.227225</td>
      <td>9.738028</td>
      <td>9.574712</td>
      <td>1.709201</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1601.000000</td>
      <td>20.000000</td>
      <td>20.000000</td>
      <td>21.000000</td>
      <td>20.000000</td>
      <td>22.000000</td>
      <td>23.000000</td>
      <td>24.000000</td>
      <td>29.000000</td>
      <td>2010.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>110.000000</td>
      <td>8457.750000</td>
      <td>45.000000</td>
      <td>40.000000</td>
      <td>46.000000</td>
      <td>47.000000</td>
      <td>48.000000</td>
      <td>39.000000</td>
      <td>44.000000</td>
      <td>47.000000</td>
      <td>2011.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>485.000000</td>
      <td>8674.000000</td>
      <td>52.000000</td>
      <td>50.000000</td>
      <td>52.000000</td>
      <td>53.000000</td>
      <td>53.000000</td>
      <td>45.000000</td>
      <td>48.000000</td>
      <td>52.000000</td>
      <td>2013.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1900.000000</td>
      <td>9904.000000</td>
      <td>62.000000</td>
      <td>55.000000</td>
      <td>59.000000</td>
      <td>62.000000</td>
      <td>61.000000</td>
      <td>51.000000</td>
      <td>55.000000</td>
      <td>58.000000</td>
      <td>2014.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>112513.000000</td>
      <td>274581.000000</td>
      <td>80.000000</td>
      <td>80.000000</td>
      <td>80.000000</td>
      <td>80.000000</td>
      <td>80.000000</td>
      <td>72.000000</td>
      <td>72.000000</td>
      <td>73.000000</td>
      <td>2015.000000</td>
    </tr>
  </tbody>
</table>
</div>



### [Comment] From the above table, it's clear that 'Team Attributes' are recorded from 2010 until 2015, so I cannot merge the above list of improved teams with their attributes, because the years are different.

### [Comment] I'll repeat the above steps using the df_team_attributes_plus_win_stats to explore the attributes of the most improved teams from 2010 until 2015.


```python
years_list=list(df_team_attributes.year.unique())
years_list
```




    [2010, 2014, 2015, 2011, 2012, 2013]




```python
first_year=min(years_list)
first_year
```




    2010




```python
last_year=max(years_list)
last_year
```




    2015




```python
df_team_attributes_first_year=df_team_attributes_plus_win_stats[df_team_attributes_plus_win_stats.year==first_year]
df_team_attributes_first_year.drop(columns=['date', 'year'], inplace=True)
df_team_attributes_first_year.head()
```

    /opt/conda/lib/python3.6/site-packages/pandas/core/frame.py:3697: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      errors=errors)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>team_fifa_api_id</th>
      <th>team_api_id</th>
      <th>buildUpPlaySpeed</th>
      <th>buildUpPlaySpeedClass</th>
      <th>buildUpPlayDribblingClass</th>
      <th>buildUpPlayPassing</th>
      <th>buildUpPlayPassingClass</th>
      <th>buildUpPlayPositioningClass</th>
      <th>chanceCreationPassing</th>
      <th>chanceCreationPassingClass</th>
      <th>...</th>
      <th>defencePressure</th>
      <th>defencePressureClass</th>
      <th>defenceAggression</th>
      <th>defenceAggressionClass</th>
      <th>defenceTeamWidth</th>
      <th>defenceTeamWidthClass</th>
      <th>defenceDefenderLineClass</th>
      <th>total_wins</th>
      <th>total_played</th>
      <th>win_percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>434</td>
      <td>9930</td>
      <td>60</td>
      <td>Balanced</td>
      <td>Little</td>
      <td>50</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>60</td>
      <td>Normal</td>
      <td>...</td>
      <td>50</td>
      <td>Medium</td>
      <td>55</td>
      <td>Press</td>
      <td>45</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>5.0</td>
      <td>18.0</td>
      <td>0.277778</td>
    </tr>
    <tr>
      <th>3</th>
      <td>77</td>
      <td>8485</td>
      <td>70</td>
      <td>Fast</td>
      <td>Little</td>
      <td>70</td>
      <td>Long</td>
      <td>Organised</td>
      <td>70</td>
      <td>Risky</td>
      <td>...</td>
      <td>60</td>
      <td>Medium</td>
      <td>70</td>
      <td>Double</td>
      <td>70</td>
      <td>Wide</td>
      <td>Cover</td>
      <td>11.0</td>
      <td>40.0</td>
      <td>0.275000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>47</td>
      <td>8564</td>
      <td>45</td>
      <td>Balanced</td>
      <td>Little</td>
      <td>30</td>
      <td>Short</td>
      <td>Free Form</td>
      <td>55</td>
      <td>Normal</td>
      <td>...</td>
      <td>30</td>
      <td>Deep</td>
      <td>35</td>
      <td>Press</td>
      <td>60</td>
      <td>Normal</td>
      <td>Offside Trap</td>
      <td>22.0</td>
      <td>39.0</td>
      <td>0.564103</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1901</td>
      <td>10215</td>
      <td>30</td>
      <td>Slow</td>
      <td>Little</td>
      <td>30</td>
      <td>Short</td>
      <td>Organised</td>
      <td>50</td>
      <td>Normal</td>
      <td>...</td>
      <td>30</td>
      <td>Deep</td>
      <td>30</td>
      <td>Contain</td>
      <td>30</td>
      <td>Narrow</td>
      <td>Offside Trap</td>
      <td>10.0</td>
      <td>30.0</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>25</th>
      <td>650</td>
      <td>10217</td>
      <td>30</td>
      <td>Slow</td>
      <td>Little</td>
      <td>35</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>30</td>
      <td>Safe</td>
      <td>...</td>
      <td>30</td>
      <td>Deep</td>
      <td>30</td>
      <td>Contain</td>
      <td>30</td>
      <td>Narrow</td>
      <td>Cover</td>
      <td>12.0</td>
      <td>35.0</td>
      <td>0.342857</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>




```python
df_team_attributes_last_year=df_team_attributes_plus_win_stats[df_team_attributes_plus_win_stats.year==last_year]
df_team_attributes_last_year.drop(columns=['date', 'year'], inplace=True)
df_team_attributes_last_year.head()
```

    /opt/conda/lib/python3.6/site-packages/pandas/core/frame.py:3697: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      errors=errors)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>team_fifa_api_id</th>
      <th>team_api_id</th>
      <th>buildUpPlaySpeed</th>
      <th>buildUpPlaySpeedClass</th>
      <th>buildUpPlayDribblingClass</th>
      <th>buildUpPlayPassing</th>
      <th>buildUpPlayPassingClass</th>
      <th>buildUpPlayPositioningClass</th>
      <th>chanceCreationPassing</th>
      <th>chanceCreationPassingClass</th>
      <th>...</th>
      <th>defencePressure</th>
      <th>defencePressureClass</th>
      <th>defenceAggression</th>
      <th>defenceAggressionClass</th>
      <th>defenceTeamWidth</th>
      <th>defenceTeamWidthClass</th>
      <th>defenceDefenderLineClass</th>
      <th>total_wins</th>
      <th>total_played</th>
      <th>win_percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>434</td>
      <td>9930</td>
      <td>47</td>
      <td>Balanced</td>
      <td>Normal</td>
      <td>54</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>54</td>
      <td>Normal</td>
      <td>...</td>
      <td>47</td>
      <td>Medium</td>
      <td>44</td>
      <td>Press</td>
      <td>54</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>3.0</td>
      <td>18.0</td>
      <td>0.166667</td>
    </tr>
    <tr>
      <th>8</th>
      <td>77</td>
      <td>8485</td>
      <td>59</td>
      <td>Balanced</td>
      <td>Normal</td>
      <td>53</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>51</td>
      <td>Normal</td>
      <td>...</td>
      <td>49</td>
      <td>Medium</td>
      <td>45</td>
      <td>Press</td>
      <td>63</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>24.0</td>
      <td>41.0</td>
      <td>0.585366</td>
    </tr>
    <tr>
      <th>18</th>
      <td>47</td>
      <td>8564</td>
      <td>48</td>
      <td>Balanced</td>
      <td>Lots</td>
      <td>52</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>66</td>
      <td>Normal</td>
      <td>...</td>
      <td>58</td>
      <td>Medium</td>
      <td>57</td>
      <td>Press</td>
      <td>49</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>15.0</td>
      <td>39.0</td>
      <td>0.384615</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1901</td>
      <td>10215</td>
      <td>53</td>
      <td>Balanced</td>
      <td>Normal</td>
      <td>44</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>55</td>
      <td>Normal</td>
      <td>...</td>
      <td>39</td>
      <td>Medium</td>
      <td>38</td>
      <td>Press</td>
      <td>61</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>5.0</td>
      <td>34.0</td>
      <td>0.147059</td>
    </tr>
    <tr>
      <th>30</th>
      <td>650</td>
      <td>10217</td>
      <td>56</td>
      <td>Balanced</td>
      <td>Normal</td>
      <td>66</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>55</td>
      <td>Normal</td>
      <td>...</td>
      <td>40</td>
      <td>Medium</td>
      <td>50</td>
      <td>Press</td>
      <td>52</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>10.0</td>
      <td>34.0</td>
      <td>0.294118</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>




```python
# rename first year columns
df_team_attributes_first_year.rename(columns=lambda x: x[:30] + "_first", inplace=True)
df_team_attributes_first_year.head()
```

    /opt/conda/lib/python3.6/site-packages/pandas/core/frame.py:3781: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      return super(DataFrame, self).rename(**kwargs)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>team_fifa_api_id_first</th>
      <th>team_api_id_first</th>
      <th>buildUpPlaySpeed_first</th>
      <th>buildUpPlaySpeedClass_first</th>
      <th>buildUpPlayDribblingClass_first</th>
      <th>buildUpPlayPassing_first</th>
      <th>buildUpPlayPassingClass_first</th>
      <th>buildUpPlayPositioningClass_first</th>
      <th>chanceCreationPassing_first</th>
      <th>chanceCreationPassingClass_first</th>
      <th>...</th>
      <th>defencePressure_first</th>
      <th>defencePressureClass_first</th>
      <th>defenceAggression_first</th>
      <th>defenceAggressionClass_first</th>
      <th>defenceTeamWidth_first</th>
      <th>defenceTeamWidthClass_first</th>
      <th>defenceDefenderLineClass_first</th>
      <th>total_wins_first</th>
      <th>total_played_first</th>
      <th>win_percent_first</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>434</td>
      <td>9930</td>
      <td>60</td>
      <td>Balanced</td>
      <td>Little</td>
      <td>50</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>60</td>
      <td>Normal</td>
      <td>...</td>
      <td>50</td>
      <td>Medium</td>
      <td>55</td>
      <td>Press</td>
      <td>45</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>5.0</td>
      <td>18.0</td>
      <td>0.277778</td>
    </tr>
    <tr>
      <th>3</th>
      <td>77</td>
      <td>8485</td>
      <td>70</td>
      <td>Fast</td>
      <td>Little</td>
      <td>70</td>
      <td>Long</td>
      <td>Organised</td>
      <td>70</td>
      <td>Risky</td>
      <td>...</td>
      <td>60</td>
      <td>Medium</td>
      <td>70</td>
      <td>Double</td>
      <td>70</td>
      <td>Wide</td>
      <td>Cover</td>
      <td>11.0</td>
      <td>40.0</td>
      <td>0.275000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>47</td>
      <td>8564</td>
      <td>45</td>
      <td>Balanced</td>
      <td>Little</td>
      <td>30</td>
      <td>Short</td>
      <td>Free Form</td>
      <td>55</td>
      <td>Normal</td>
      <td>...</td>
      <td>30</td>
      <td>Deep</td>
      <td>35</td>
      <td>Press</td>
      <td>60</td>
      <td>Normal</td>
      <td>Offside Trap</td>
      <td>22.0</td>
      <td>39.0</td>
      <td>0.564103</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1901</td>
      <td>10215</td>
      <td>30</td>
      <td>Slow</td>
      <td>Little</td>
      <td>30</td>
      <td>Short</td>
      <td>Organised</td>
      <td>50</td>
      <td>Normal</td>
      <td>...</td>
      <td>30</td>
      <td>Deep</td>
      <td>30</td>
      <td>Contain</td>
      <td>30</td>
      <td>Narrow</td>
      <td>Offside Trap</td>
      <td>10.0</td>
      <td>30.0</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>25</th>
      <td>650</td>
      <td>10217</td>
      <td>30</td>
      <td>Slow</td>
      <td>Little</td>
      <td>35</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>30</td>
      <td>Safe</td>
      <td>...</td>
      <td>30</td>
      <td>Deep</td>
      <td>30</td>
      <td>Contain</td>
      <td>30</td>
      <td>Narrow</td>
      <td>Cover</td>
      <td>12.0</td>
      <td>35.0</td>
      <td>0.342857</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>




```python
# rename last year columns
df_team_attributes_last_year.rename(columns=lambda x: x[:30] + "_last", inplace=True)
df_team_attributes_last_year.head()
```

    /opt/conda/lib/python3.6/site-packages/pandas/core/frame.py:3781: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      return super(DataFrame, self).rename(**kwargs)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>team_fifa_api_id_last</th>
      <th>team_api_id_last</th>
      <th>buildUpPlaySpeed_last</th>
      <th>buildUpPlaySpeedClass_last</th>
      <th>buildUpPlayDribblingClass_last</th>
      <th>buildUpPlayPassing_last</th>
      <th>buildUpPlayPassingClass_last</th>
      <th>buildUpPlayPositioningClass_last</th>
      <th>chanceCreationPassing_last</th>
      <th>chanceCreationPassingClass_last</th>
      <th>...</th>
      <th>defencePressure_last</th>
      <th>defencePressureClass_last</th>
      <th>defenceAggression_last</th>
      <th>defenceAggressionClass_last</th>
      <th>defenceTeamWidth_last</th>
      <th>defenceTeamWidthClass_last</th>
      <th>defenceDefenderLineClass_last</th>
      <th>total_wins_last</th>
      <th>total_played_last</th>
      <th>win_percent_last</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>434</td>
      <td>9930</td>
      <td>47</td>
      <td>Balanced</td>
      <td>Normal</td>
      <td>54</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>54</td>
      <td>Normal</td>
      <td>...</td>
      <td>47</td>
      <td>Medium</td>
      <td>44</td>
      <td>Press</td>
      <td>54</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>3.0</td>
      <td>18.0</td>
      <td>0.166667</td>
    </tr>
    <tr>
      <th>8</th>
      <td>77</td>
      <td>8485</td>
      <td>59</td>
      <td>Balanced</td>
      <td>Normal</td>
      <td>53</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>51</td>
      <td>Normal</td>
      <td>...</td>
      <td>49</td>
      <td>Medium</td>
      <td>45</td>
      <td>Press</td>
      <td>63</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>24.0</td>
      <td>41.0</td>
      <td>0.585366</td>
    </tr>
    <tr>
      <th>18</th>
      <td>47</td>
      <td>8564</td>
      <td>48</td>
      <td>Balanced</td>
      <td>Lots</td>
      <td>52</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>66</td>
      <td>Normal</td>
      <td>...</td>
      <td>58</td>
      <td>Medium</td>
      <td>57</td>
      <td>Press</td>
      <td>49</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>15.0</td>
      <td>39.0</td>
      <td>0.384615</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1901</td>
      <td>10215</td>
      <td>53</td>
      <td>Balanced</td>
      <td>Normal</td>
      <td>44</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>55</td>
      <td>Normal</td>
      <td>...</td>
      <td>39</td>
      <td>Medium</td>
      <td>38</td>
      <td>Press</td>
      <td>61</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>5.0</td>
      <td>34.0</td>
      <td>0.147059</td>
    </tr>
    <tr>
      <th>30</th>
      <td>650</td>
      <td>10217</td>
      <td>56</td>
      <td>Balanced</td>
      <td>Normal</td>
      <td>66</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>55</td>
      <td>Normal</td>
      <td>...</td>
      <td>40</td>
      <td>Medium</td>
      <td>50</td>
      <td>Press</td>
      <td>52</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>10.0</td>
      <td>34.0</td>
      <td>0.294118</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>




```python
# merge datasets
df_combined_first_last =df_team_attributes_first_year.merge(df_team_attributes_last_year,left_on='team_api_id_first', right_on='team_api_id_last',how='inner')
```


```python
df_combined_first_last['win_percent_comparison']=df_combined_first_last['win_percent_last']/df_combined_first_last['win_percent_first']
df_combined_first_last.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>team_fifa_api_id_first</th>
      <th>team_api_id_first</th>
      <th>buildUpPlaySpeed_first</th>
      <th>buildUpPlayPassing_first</th>
      <th>chanceCreationPassing_first</th>
      <th>chanceCreationCrossing_first</th>
      <th>chanceCreationShooting_first</th>
      <th>defencePressure_first</th>
      <th>defenceAggression_first</th>
      <th>defenceTeamWidth_first</th>
      <th>...</th>
      <th>chanceCreationPassing_last</th>
      <th>chanceCreationCrossing_last</th>
      <th>chanceCreationShooting_last</th>
      <th>defencePressure_last</th>
      <th>defenceAggression_last</th>
      <th>defenceTeamWidth_last</th>
      <th>total_wins_last</th>
      <th>total_played_last</th>
      <th>win_percent_last</th>
      <th>win_percent_comparison</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>141.000000</td>
      <td>141.000000</td>
      <td>141.000000</td>
      <td>141.000000</td>
      <td>141.000000</td>
      <td>141.000000</td>
      <td>141.000000</td>
      <td>141.000000</td>
      <td>141.000000</td>
      <td>141.000000</td>
      <td>...</td>
      <td>141.000000</td>
      <td>141.000000</td>
      <td>141.000000</td>
      <td>141.000000</td>
      <td>141.000000</td>
      <td>141.000000</td>
      <td>141.000000</td>
      <td>141.000000</td>
      <td>141.000000</td>
      <td>141.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>13038.361702</td>
      <td>9070.092199</td>
      <td>50.120567</td>
      <td>44.546099</td>
      <td>51.794326</td>
      <td>53.120567</td>
      <td>62.177305</td>
      <td>50.595745</td>
      <td>52.014184</td>
      <td>52.822695</td>
      <td>...</td>
      <td>53.354610</td>
      <td>53.241135</td>
      <td>49.553191</td>
      <td>47.312057</td>
      <td>50.000000</td>
      <td>51.886525</td>
      <td>13.716312</td>
      <td>33.907801</td>
      <td>0.394164</td>
      <td>1.110834</td>
    </tr>
    <tr>
      <th>std</th>
      <td>34261.237880</td>
      <td>1464.864011</td>
      <td>15.787190</td>
      <td>13.851496</td>
      <td>12.857192</td>
      <td>12.768530</td>
      <td>8.449247</td>
      <td>15.656116</td>
      <td>16.740450</td>
      <td>15.482840</td>
      <td>...</td>
      <td>10.574994</td>
      <td>10.390174</td>
      <td>12.169884</td>
      <td>9.102696</td>
      <td>8.180988</td>
      <td>7.517686</td>
      <td>6.402373</td>
      <td>6.330088</td>
      <td>0.161821</td>
      <td>0.679384</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1601.000000</td>
      <td>30.000000</td>
      <td>30.000000</td>
      <td>30.000000</td>
      <td>30.000000</td>
      <td>35.000000</td>
      <td>30.000000</td>
      <td>30.000000</td>
      <td>30.000000</td>
      <td>...</td>
      <td>28.000000</td>
      <td>26.000000</td>
      <td>22.000000</td>
      <td>25.000000</td>
      <td>33.000000</td>
      <td>29.000000</td>
      <td>1.000000</td>
      <td>9.000000</td>
      <td>0.090909</td>
      <td>0.207407</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>72.000000</td>
      <td>8543.000000</td>
      <td>30.000000</td>
      <td>30.000000</td>
      <td>45.000000</td>
      <td>43.000000</td>
      <td>55.000000</td>
      <td>30.000000</td>
      <td>30.000000</td>
      <td>35.000000</td>
      <td>...</td>
      <td>48.000000</td>
      <td>48.000000</td>
      <td>40.000000</td>
      <td>41.000000</td>
      <td>44.000000</td>
      <td>48.000000</td>
      <td>10.000000</td>
      <td>32.000000</td>
      <td>0.275000</td>
      <td>0.740038</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>452.000000</td>
      <td>9772.000000</td>
      <td>50.000000</td>
      <td>45.000000</td>
      <td>50.000000</td>
      <td>50.000000</td>
      <td>65.000000</td>
      <td>50.000000</td>
      <td>55.000000</td>
      <td>55.000000</td>
      <td>...</td>
      <td>53.000000</td>
      <td>53.000000</td>
      <td>50.000000</td>
      <td>47.000000</td>
      <td>49.000000</td>
      <td>52.000000</td>
      <td>13.000000</td>
      <td>34.000000</td>
      <td>0.368421</td>
      <td>0.950000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1873.000000</td>
      <td>9941.000000</td>
      <td>65.000000</td>
      <td>55.000000</td>
      <td>65.000000</td>
      <td>65.000000</td>
      <td>70.000000</td>
      <td>65.000000</td>
      <td>70.000000</td>
      <td>70.000000</td>
      <td>...</td>
      <td>62.000000</td>
      <td>61.000000</td>
      <td>58.000000</td>
      <td>53.000000</td>
      <td>57.000000</td>
      <td>57.000000</td>
      <td>18.000000</td>
      <td>38.000000</td>
      <td>0.473684</td>
      <td>1.250000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>111092.000000</td>
      <td>10269.000000</td>
      <td>70.000000</td>
      <td>70.000000</td>
      <td>70.000000</td>
      <td>70.000000</td>
      <td>70.000000</td>
      <td>70.000000</td>
      <td>70.000000</td>
      <td>70.000000</td>
      <td>...</td>
      <td>77.000000</td>
      <td>80.000000</td>
      <td>80.000000</td>
      <td>72.000000</td>
      <td>72.000000</td>
      <td>73.000000</td>
      <td>30.000000</td>
      <td>41.000000</td>
      <td>0.794118</td>
      <td>5.800000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 27 columns</p>
</div>




```python
df_most_improved_2010_2015=df_combined_first_last[df_combined_first_last['win_percent_comparison']>1.25]
```


```python
df_most_improved_teams_2010_2015=df_most_improved_2010_2015.merge(df_team, left_on='team_api_id_first', right_on='team_api_id', how='inner')
df_most_improved_teams_2010_2015_w_attrib=df_most_improved_teams_2010_2015[['team_long_name','win_percent_comparison', 'buildUpPlaySpeedClass_last', 'buildUpPlayPassingClass_last','buildUpPlayPassingClass_last',
 'buildUpPlayPositioningClass_last',
 'chanceCreationPassingClass_last',
 'chanceCreationCrossingClass_last',
 'chanceCreationShootingClass_last',
 'chanceCreationPositioningClass_last',
 'defencePressureClass_last',
 'defenceAggressionClass_last',
 'defenceTeamWidthClass_last',
 'defenceDefenderLineClass_last']].sort_values('win_percent_comparison', ascending=False)
df_most_improved_teams_2010_2015_w_attrib
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>team_long_name</th>
      <th>win_percent_comparison</th>
      <th>buildUpPlaySpeedClass_last</th>
      <th>buildUpPlayPassingClass_last</th>
      <th>buildUpPlayPassingClass_last</th>
      <th>buildUpPlayPositioningClass_last</th>
      <th>chanceCreationPassingClass_last</th>
      <th>chanceCreationCrossingClass_last</th>
      <th>chanceCreationShootingClass_last</th>
      <th>chanceCreationPositioningClass_last</th>
      <th>defencePressureClass_last</th>
      <th>defenceAggressionClass_last</th>
      <th>defenceTeamWidthClass_last</th>
      <th>defenceDefenderLineClass_last</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>25</th>
      <td>Sporting Charleroi</td>
      <td>5.800000</td>
      <td>Balanced</td>
      <td>Mixed</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Willem II</td>
      <td>3.774510</td>
      <td>Balanced</td>
      <td>Mixed</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Borussia Mönchengladbach</td>
      <td>3.500000</td>
      <td>Slow</td>
      <td>Short</td>
      <td>Short</td>
      <td>Organised</td>
      <td>Normal</td>
      <td>Little</td>
      <td>Normal</td>
      <td>Free Form</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Vitesse</td>
      <td>2.873950</td>
      <td>Balanced</td>
      <td>Mixed</td>
      <td>Mixed</td>
      <td>Free Form</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Lots</td>
      <td>Free Form</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Paris Saint-Germain</td>
      <td>2.500000</td>
      <td>Balanced</td>
      <td>Mixed</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Piast Gliwice</td>
      <td>2.437500</td>
      <td>Balanced</td>
      <td>Long</td>
      <td>Long</td>
      <td>Organised</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Hull City</td>
      <td>2.368421</td>
      <td>Balanced</td>
      <td>Mixed</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>7</th>
      <td>SM Caen</td>
      <td>2.131579</td>
      <td>Balanced</td>
      <td>Mixed</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Aberdeen</td>
      <td>2.128603</td>
      <td>Balanced</td>
      <td>Mixed</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>Normal</td>
      <td>Lots</td>
      <td>Normal</td>
      <td>Free Form</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Cracovia</td>
      <td>2.041667</td>
      <td>Balanced</td>
      <td>Mixed</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Little</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Fiorentina</td>
      <td>2.037296</td>
      <td>Balanced</td>
      <td>Short</td>
      <td>Short</td>
      <td>Organised</td>
      <td>Risky</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>15</th>
      <td>AS Monaco</td>
      <td>1.950000</td>
      <td>Balanced</td>
      <td>Mixed</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Little</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Rio Ave FC</td>
      <td>1.941176</td>
      <td>Balanced</td>
      <td>Mixed</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Hertha BSC Berlin</td>
      <td>1.750000</td>
      <td>Balanced</td>
      <td>Mixed</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>Normal</td>
      <td>Lots</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Feyenoord</td>
      <td>1.727273</td>
      <td>Balanced</td>
      <td>Mixed</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>Normal</td>
      <td>Lots</td>
      <td>Little</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Real Sporting de Gijón</td>
      <td>1.625000</td>
      <td>Balanced</td>
      <td>Mixed</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Vitória Setúbal</td>
      <td>1.588235</td>
      <td>Balanced</td>
      <td>Mixed</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CF Os Belenenses</td>
      <td>1.568627</td>
      <td>Balanced</td>
      <td>Mixed</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Juventus</td>
      <td>1.558974</td>
      <td>Balanced</td>
      <td>Short</td>
      <td>Short</td>
      <td>Organised</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Lots</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Sporting CP</td>
      <td>1.529412</td>
      <td>Balanced</td>
      <td>Mixed</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>17</th>
      <td>OGC Nice</td>
      <td>1.500000</td>
      <td>Balanced</td>
      <td>Mixed</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>28</th>
      <td>St. Mirren</td>
      <td>1.500000</td>
      <td>Balanced</td>
      <td>Short</td>
      <td>Short</td>
      <td>Organised</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>32</th>
      <td>West Ham United</td>
      <td>1.428571</td>
      <td>Fast</td>
      <td>Mixed</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>Risky</td>
      <td>Lots</td>
      <td>Little</td>
      <td>Organised</td>
      <td>Deep</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>34</th>
      <td>VfL Wolfsburg</td>
      <td>1.416667</td>
      <td>Fast</td>
      <td>Mixed</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>Risky</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>23</th>
      <td>AS Saint-Étienne</td>
      <td>1.384615</td>
      <td>Balanced</td>
      <td>Mixed</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>16</th>
      <td>CD Nacional</td>
      <td>1.372549</td>
      <td>Balanced</td>
      <td>Mixed</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>2</th>
      <td>FC Bayern Munich</td>
      <td>1.368421</td>
      <td>Balanced</td>
      <td>Short</td>
      <td>Short</td>
      <td>Free Form</td>
      <td>Normal</td>
      <td>Little</td>
      <td>Little</td>
      <td>Free Form</td>
      <td>High</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Atlético Madrid</td>
      <td>1.333333</td>
      <td>Balanced</td>
      <td>Mixed</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>18</th>
      <td>FC Paços de Ferreira</td>
      <td>1.323529</td>
      <td>Balanced</td>
      <td>Mixed</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>11</th>
      <td>KAA Gent</td>
      <td>1.311905</td>
      <td>Balanced</td>
      <td>Mixed</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>21</th>
      <td>PSV</td>
      <td>1.310924</td>
      <td>Fast</td>
      <td>Mixed</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Little</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Sevilla FC</td>
      <td>1.279688</td>
      <td>Balanced</td>
      <td>Mixed</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>29</th>
      <td>St. Johnstone FC</td>
      <td>1.266667</td>
      <td>Balanced</td>
      <td>Mixed</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Burnley</td>
      <td>1.263158</td>
      <td>Balanced</td>
      <td>Long</td>
      <td>Long</td>
      <td>Organised</td>
      <td>Risky</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bologna</td>
      <td>1.251337</td>
      <td>Balanced</td>
      <td>Mixed</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>Safe</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>Deep</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
  </tbody>
</table>
</div>



<a id='conclusions'></a>
## Conclusions

> **Tip**: Finally, summarize your findings and the results that have been performed in relation to the question(s) provided at the beginning of the analysis. Summarize the results accurately, and point out where additional research can be done or where additional information could be useful.


> **Tip**: If you haven't done any statistical tests, do not imply any statistical conclusions. And make sure you avoid implying causation from correlation!

### Research Question 1 (Which players scored penalties the most (penalty top-scorers)?)


```python
df_top_penalty_scorers_player_details
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>player_name</th>
      <th>scored_penalties</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>Cristiano Ronaldo</td>
      <td>55</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Zlatan Ibrahimovic</td>
      <td>41</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Lionel Messi</td>
      <td>36</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Francesco Totti</td>
      <td>29</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Antonio Di Natale</td>
      <td>28</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Sejad Salihovic</td>
      <td>27</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Steven Gerrard</td>
      <td>25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Frank Lampard</td>
      <td>25</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Diego Milito</td>
      <td>23</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Edinson Cavani</td>
      <td>23</td>
    </tr>
  </tbody>
</table>
</div>



>**Cristiano Ronaldo is the penalties top-scorer from 2008 until 2016.**

### Research Question 2 (Which players had the most penalties (the fouled players who were granted the most penalties)?)


```python
df_top_penalty_getters_player_details
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>player_name</th>
      <th>penalties_granted</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6377</th>
      <td>Luis Suarez</td>
      <td>18</td>
    </tr>
    <tr>
      <th>1995</th>
      <td>Cristiano Ronaldo</td>
      <td>14</td>
    </tr>
    <tr>
      <th>7867</th>
      <td>Neymar</td>
      <td>12</td>
    </tr>
    <tr>
      <th>11057</th>
      <td>Zlatan Ibrahimovic</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2864</th>
      <td>Edinson Cavani</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>



>**Luis Suarez is the penalties top-getter from 2008 until 2016.**

### Research Question 3 (What team attributes lead to the most victories?)



```python
print("Based on the above pie charts, the most of the 'High Vectory Teams' apply the following attribute Classes:\n")
for i in range(len(df_list)):
    print(attrib_class_list[i], " -> ", df_list[i][df_list[i]==df_list[i].max()].index[0])
```

    Based on the above pie charts, the most of the 'High Vectory Teams' apply the following attribute Classes:
    
    buildUpPlaySpeedClass  ->  Balanced
    buildUpPlayDribblingClass  ->  Little
    buildUpPlayPassingClass  ->  Mixed
    buildUpPlayPositioningClass  ->  Organised
    chanceCreationPassingClass  ->  Normal
    chanceCreationCrossingClass  ->  Normal
    chanceCreationShootingClass  ->  Normal
    chanceCreationPositioningClass  ->  Organised
    defencePressureClass  ->  Medium
    defenceAggressionClass  ->  Press
    defenceTeamWidthClass  ->  Normal
    defenceDefenderLineClass  ->  Cover


### Research Question 4 (What teams improved the most over the time period?)
#### The following Table shows the teams who have improved their winning records by more than 125% when comparing the results of years 2010 and 2015. Most of them share the same attributes highlighted in the above conclusion.


```python
df_most_improved_teams_2010_2015_w_attrib
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>team_long_name</th>
      <th>win_percent_comparison</th>
      <th>buildUpPlaySpeedClass_last</th>
      <th>buildUpPlayPassingClass_last</th>
      <th>buildUpPlayPassingClass_last</th>
      <th>buildUpPlayPositioningClass_last</th>
      <th>chanceCreationPassingClass_last</th>
      <th>chanceCreationCrossingClass_last</th>
      <th>chanceCreationShootingClass_last</th>
      <th>chanceCreationPositioningClass_last</th>
      <th>defencePressureClass_last</th>
      <th>defenceAggressionClass_last</th>
      <th>defenceTeamWidthClass_last</th>
      <th>defenceDefenderLineClass_last</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>25</th>
      <td>Sporting Charleroi</td>
      <td>5.800000</td>
      <td>Balanced</td>
      <td>Mixed</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Willem II</td>
      <td>3.774510</td>
      <td>Balanced</td>
      <td>Mixed</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Borussia Mönchengladbach</td>
      <td>3.500000</td>
      <td>Slow</td>
      <td>Short</td>
      <td>Short</td>
      <td>Organised</td>
      <td>Normal</td>
      <td>Little</td>
      <td>Normal</td>
      <td>Free Form</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Vitesse</td>
      <td>2.873950</td>
      <td>Balanced</td>
      <td>Mixed</td>
      <td>Mixed</td>
      <td>Free Form</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Lots</td>
      <td>Free Form</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Paris Saint-Germain</td>
      <td>2.500000</td>
      <td>Balanced</td>
      <td>Mixed</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Piast Gliwice</td>
      <td>2.437500</td>
      <td>Balanced</td>
      <td>Long</td>
      <td>Long</td>
      <td>Organised</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Hull City</td>
      <td>2.368421</td>
      <td>Balanced</td>
      <td>Mixed</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>7</th>
      <td>SM Caen</td>
      <td>2.131579</td>
      <td>Balanced</td>
      <td>Mixed</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Aberdeen</td>
      <td>2.128603</td>
      <td>Balanced</td>
      <td>Mixed</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>Normal</td>
      <td>Lots</td>
      <td>Normal</td>
      <td>Free Form</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Cracovia</td>
      <td>2.041667</td>
      <td>Balanced</td>
      <td>Mixed</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Little</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Fiorentina</td>
      <td>2.037296</td>
      <td>Balanced</td>
      <td>Short</td>
      <td>Short</td>
      <td>Organised</td>
      <td>Risky</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>15</th>
      <td>AS Monaco</td>
      <td>1.950000</td>
      <td>Balanced</td>
      <td>Mixed</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Little</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Rio Ave FC</td>
      <td>1.941176</td>
      <td>Balanced</td>
      <td>Mixed</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Hertha BSC Berlin</td>
      <td>1.750000</td>
      <td>Balanced</td>
      <td>Mixed</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>Normal</td>
      <td>Lots</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Feyenoord</td>
      <td>1.727273</td>
      <td>Balanced</td>
      <td>Mixed</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>Normal</td>
      <td>Lots</td>
      <td>Little</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Real Sporting de Gijón</td>
      <td>1.625000</td>
      <td>Balanced</td>
      <td>Mixed</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Vitória Setúbal</td>
      <td>1.588235</td>
      <td>Balanced</td>
      <td>Mixed</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CF Os Belenenses</td>
      <td>1.568627</td>
      <td>Balanced</td>
      <td>Mixed</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Juventus</td>
      <td>1.558974</td>
      <td>Balanced</td>
      <td>Short</td>
      <td>Short</td>
      <td>Organised</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Lots</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Sporting CP</td>
      <td>1.529412</td>
      <td>Balanced</td>
      <td>Mixed</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>17</th>
      <td>OGC Nice</td>
      <td>1.500000</td>
      <td>Balanced</td>
      <td>Mixed</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>28</th>
      <td>St. Mirren</td>
      <td>1.500000</td>
      <td>Balanced</td>
      <td>Short</td>
      <td>Short</td>
      <td>Organised</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>32</th>
      <td>West Ham United</td>
      <td>1.428571</td>
      <td>Fast</td>
      <td>Mixed</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>Risky</td>
      <td>Lots</td>
      <td>Little</td>
      <td>Organised</td>
      <td>Deep</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>34</th>
      <td>VfL Wolfsburg</td>
      <td>1.416667</td>
      <td>Fast</td>
      <td>Mixed</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>Risky</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>23</th>
      <td>AS Saint-Étienne</td>
      <td>1.384615</td>
      <td>Balanced</td>
      <td>Mixed</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>16</th>
      <td>CD Nacional</td>
      <td>1.372549</td>
      <td>Balanced</td>
      <td>Mixed</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>2</th>
      <td>FC Bayern Munich</td>
      <td>1.368421</td>
      <td>Balanced</td>
      <td>Short</td>
      <td>Short</td>
      <td>Free Form</td>
      <td>Normal</td>
      <td>Little</td>
      <td>Little</td>
      <td>Free Form</td>
      <td>High</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Atlético Madrid</td>
      <td>1.333333</td>
      <td>Balanced</td>
      <td>Mixed</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>18</th>
      <td>FC Paços de Ferreira</td>
      <td>1.323529</td>
      <td>Balanced</td>
      <td>Mixed</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>11</th>
      <td>KAA Gent</td>
      <td>1.311905</td>
      <td>Balanced</td>
      <td>Mixed</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>21</th>
      <td>PSV</td>
      <td>1.310924</td>
      <td>Fast</td>
      <td>Mixed</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Little</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Sevilla FC</td>
      <td>1.279688</td>
      <td>Balanced</td>
      <td>Mixed</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>29</th>
      <td>St. Johnstone FC</td>
      <td>1.266667</td>
      <td>Balanced</td>
      <td>Mixed</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Burnley</td>
      <td>1.263158</td>
      <td>Balanced</td>
      <td>Long</td>
      <td>Long</td>
      <td>Organised</td>
      <td>Risky</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>Medium</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bologna</td>
      <td>1.251337</td>
      <td>Balanced</td>
      <td>Mixed</td>
      <td>Mixed</td>
      <td>Organised</td>
      <td>Safe</td>
      <td>Normal</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>Deep</td>
      <td>Press</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
  </tbody>
</table>
</div>







### Limitations

## **** The limitation I found was that whereas the 'Match' table has records for matches from 2008 until 2016, the 'Team Attributes' table has records from 2010 until 2015 only. This didn't allow me to explore how the attributes of the improved teams changed in 2016.


## Submitting your Project 



```python
from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])
```




    0




```python

```
