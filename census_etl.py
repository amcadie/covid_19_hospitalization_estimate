# Small ETL job to pull 2017 census from the Census Bureau, build the join key, and write to a sqlite database

import pandas as pd
import sqlite3

print('Fetching data...')
# get data
pop = pd.read_csv('https://www2.census.gov/programs-surveys/popest/datasets/2010-2017/counties/asrh/cc-est2017-alldata.csv',
                 encoding='ISO-8859-1')


print('Processing...')
    
# create join key, add description for keyed fields
# data dictionary: https://www2.census.gov/programs-surveys/popest/technical-documentation/file-layouts/2010-2017/cc-est2017-alldata.pdf?#

year_map = {1: '4/1/2010 census', 2: '4/1/2010 est. base', 3: '7/1/2010 est.', 4: '7/1/2011 est.', 5: '7/1/2012 est.',
            6: '7/1/2013 est.', 7: '7/1/2014 est.', 8: '7/1/2015 est.', 9: '7/1/2016 est.', 10: '7/1/17 est.'}
agegrp_map = {0: 'Total', 1: '0-4', 2: '5-9', 3: '10-14', 4: '15-19', 5: '20-24', 6: '25-29', 7: '30-34', 8: '35-38',
              9: '40-44', 10: '45-49', 11: '50-54', 12: '55-59', 13: '60-64', 14: '65-69', 15: '70-74',
              16: '75-79', 17: '80-84', 18: '85+'}

pop = pop.assign(COUNTY_KEY = lambda x: x['CTYNAME'].str.extract(r'([A-Za-z\s\.\-\']*)(?=\sCounty|\sParish)'),
                 YEAR_DESCR = lambda x: x['YEAR'].map(year_map),
                 AGEGRP_DESCR = lambda x: x['AGEGRP'].map(agegrp_map)
                )

       
# Manually fix a few County/district names so they join to the NYT cases dataset

# All cases for the five boroughs of New York City 
# (New York, Kings, Queens, Bronx and Richmond counties) 
# are assigned to a single area called New York City.
nyc_idx = (pop.CTYNAME.str.contains(r'New York|Kings|Queens|Bronx|Richmond')) \
    & (pop.STNAME == 'New York')
    
pop.loc[nyc_idx, 'COUNTY_KEY'] = 'New York City'

# D.C
pop.loc[pop.CTYNAME == 'District of Columbia', 'COUNTY_KEY'] = 'District of Columbia'

# make case insensitive
pop['COUNTY_KEY'] = pop.COUNTY_KEY.str.lower()

print('Writing to DB...')

# create DB
conn = sqlite3.connect('US_county_census.db')

pop.to_sql(name = 'census', con = conn, if_exists = 'replace')

conn.close()

print('ETL Complete.')