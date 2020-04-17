import requests
import re
import sqlite3
import datetime
import pandas as pd
import numpy as np

# get NYT data
cases = pd.read_csv('https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv')

cases = cases.assign(
    date = pd.to_datetime(cases['date']),
    fips = cases['fips'].astype(pd.Int32Dtype())
)

# calculate new cases per day
cases['cases_shifted'] = (
    cases.groupby(['county', 'state'])
    .cases
    .shift(1)
    .fillna(0)
    .astype(int)
)

cases['cases_new'] = cases['cases'] - cases['cases_shifted']

# 2017 county census estimate from US Census Bureau.  Written locally to SQLite database
# to avoid having to re-download every time I run the notebook
# see census_etl.py for details

conn = sqlite3.connect('US_county_census.db')

# start with 2017 census and total population
# might as well resort to SQL yelling bc names are all in caps
pop17 = pd.read_sql('SELECT STNAME, CTYNAME, COUNTY_KEY, TOT_POP FROM CENSUS WHERE YEAR = 10 AND AGEGRP = 0;',
                 con = conn)

conn.close()

# All cases for the five boroughs of New York City 
# (New York, Kings, Queens, Bronx and Richmond counties) 
# are assigned to a single area called New York City.
pop17 = (pop17
       .groupby(['STNAME', 'COUNTY_KEY'])
       .aggregate({'TOT_POP': 'sum'})
       .reset_index()
      )

# make key case insensitive
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
cases_clean = (cases
               .groupby(['county_key', 'county', 'state', 'date'])
               .aggregate({'cases': 'sum', 'deaths': 'sum', 'cases_shifted': 'sum', 'cases_new': 'sum'})
               .reset_index()
              )

# join cases and census
cases_percap = (cases_clean.merge(pop17, how = 'left', left_on = ['county_key', 'state'], 
                                 right_on = ['COUNTY_KEY', 'STNAME'])
                .dropna()
                .assign(cases_per10k = lambda x: x.cases / x.TOT_POP * 10000,
                        cases_new_per1k = lambda x: x.cases_new / x.TOT_POP * 1000,
                        county_label = lambda x: x.county + ', ' + x.STNAME)
                .sort_values(['county_label', 'date'])
               )

# create 5-day rolling average of new cases
# can't figure out how to accomplish this in the pipeline above
cases_percap['cases_new_per1k_MA'] = (cases_percap
                                      .groupby('county_label')['cases_new_per1k']
                                      .transform(lambda x: x.rolling(5).mean())
                                      )

# write data for counties with >1M to a JSON string
filename = f"cases_percap_1M+_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.json"
cases_percap[cases_percap.TOT_POP > 1000000].to_json(filename, orient = 'records')