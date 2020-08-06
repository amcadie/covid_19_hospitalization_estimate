import requests
import re
import zipfile
import sqlite3
import datetime
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import altair as alt

# get NYT case data
cases = pd.read_csv('https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv')
cases = cases.assign(
    date = pd.to_datetime(cases['date']),
    fips = cases['fips'].astype(pd.Int32Dtype())
)

# Cases are cumulative, we want new cases and deaths each day to estimate the hospital case load.
cases['cases_shifted'] = (
    cases.groupby(['county', 'state'])
    .cases
    .shift(1)
    .fillna(0)
    .astype(int)
)

cases['deaths_shifted'] = (
    cases.groupby(['county', 'state'])
    .deaths
    .shift(1)
    .fillna(0)
    .astype(int)
)

cases['cases_new'] = cases['cases'] - cases['cases_shifted']

cases['deaths_new'] = cases['deaths'] - cases['deaths_shifted']

# 2017 county census estimate from US Census Bureau.  Written locally to SQLite database
# to avoid having to re-download every time I run the notebook
# see census_etl.py for details

conn = sqlite3.connect('/Users/amcadi/Documents/opensource/covid_19_hospitalization_estimate/US_county_census.db')

# start with 2017 census and total population
# might as well resort to SQL yelling bc names are all in caps
pop17 = pd.read_sql('SELECT STNAME, CTYNAME, COUNTY_KEY, TOT_POP, REGION FROM CENSUS WHERE YEAR = 10 AND AGEGRP = 0;',
                 con = conn)

conn.close()

# All cases for the five boroughs of New York City
# (New York, Kings, Queens, Bronx and Richmond counties)
# are assigned to a single area called New York City.
pop17 = (pop17
       .groupby(['STNAME', 'COUNTY_KEY', 'region'])
       .aggregate({'TOT_POP': 'sum'})
       .reset_index()
      )

cases['county_key'] = cases.county.str.lower()

# in NYT data - "Cities like St. Louis and Baltimore that are administered separately from an adjacent
# county of the same name are counted separately."
# strip off the -city and combine with the surrounding county to join correctly to the county-level census data
def rm_city(county_name):
    if re.search(r'(?<!new york)\scity', county_name):
        return re.sub(r'([A-Za-z\s\.\-\']*)(\scity)', r'\1', county_name)
    else:
        return(county_name)

cases['county_key'] = cases.county_key.apply(rm_city)

# aggregate across new key with combined cities and counties
cases = (cases
         .groupby(['county_key', 'county', 'state', 'date'])
         .aggregate({'cases': 'sum', 'deaths': 'sum', 'deaths_new': 'sum', 'cases_new': 'sum'})
         .reset_index()
        )

cases_percap = cases.merge(pop17, how = 'left', left_on = ['county_key', 'state'],
                           right_on = ['COUNTY_KEY', 'STNAME'])

# calculate percap rates
cases_percap = (cases_percap
                .dropna()
                .assign(cases_per1k = lambda x: x.cases / x.TOT_POP * 1000,
                        cases_new_per1k = lambda x: x.cases_new / x.TOT_POP * 1000,
                        deaths_per1k = lambda x: x.deaths / x.TOT_POP * 1000,
                        deaths_new_per1k = lambda x: x.deaths_new / x.TOT_POP * 1000,
                        county_label = lambda x: x.county + ', ' + x.STNAME)
                .sort_values(['county_label', 'date'])
               )

# create 14-day rolling sum of per-capita new cases, which should be roughly proportional to
# the current case load
cases_percap['cases_new_per1k_sum14'] = (cases_percap
                                        .groupby('county_label')['cases_new_per1k']
                                        .transform(lambda x: x.rolling(14).sum())
                                       )

# create 14-day rolling sum in raw count for tooltip
cases_percap['cases_new_sum14'] = (cases_percap
                                   .groupby('county_label')['cases_new']
                                   .transform(lambda x: x.rolling(14).sum())
                                  )

# filter for 3 days ago, giving some time for the latest data to be populated
three_days_back = cases_percap.date.max() - datetime.timedelta(days = 3)

# make copy explicit to avoid subsequent SettingWithCopy warning
cases_percap_recent = cases_percap.query('TOT_POP > 100000 and date == @three_days_back').copy()

# calculate week-over-week proportional change
# generate limits for weekly change
last_wk_max = cases_percap.date.max() - datetime.timedelta(days = 3)
last_wk_min = last_wk_max - datetime.timedelta(days = 7)
two_wks_min = last_wk_max - datetime.timedelta(days = 14)

last_wk_total_cases = (cases_percap
                       .query('date >= @last_wk_min and date < @last_wk_max')
                       .groupby('county_label')
                       .aggregate({'cases_new': 'sum'})
                       .rename(columns = {'cases_new': 'last_wk_cases'})
                      )

two_wks_total_cases = (cases_percap
                       .query('date >= @two_wks_min and date < @last_wk_min')
                       .groupby('county_label')
                       .aggregate({'cases_new': 'sum'})
                       .rename(columns = {'cases_new': 'two_wks_ago_cases'})
                      )

wkly_diff = (last_wk_total_cases
             .merge(two_wks_total_cases, how = 'inner', on = 'county_label')
             .assign(wkly_diff = lambda x: (x['last_wk_cases'] - x['two_wks_ago_cases']) / x['two_wks_ago_cases'])
            )

cases_percap_recent = (cases_percap_recent
                       .merge(wkly_diff, how = 'left', on = 'county_label')
                       .rename(columns = {'county_label': 'County', 'TOT_POP': 'Population', 'cases_new_sum14': 'Current Cases'})
                      )
