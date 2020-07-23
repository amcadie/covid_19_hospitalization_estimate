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

cases['cases_new_shifted'] = (
    cases.groupby(['county', 'state'])
    .cases_new
    .shift(1)
    .fillna(0)
    .astype(int)
)

cases['deaths_new_shifted'] = (
    cases.groupby(['county', 'state'])
    .deaths_new
    .shift(1)
    .fillna(0)
    .astype(int)
)

cases['cases_new_delta'] = (cases['cases_new'] - cases['cases_new_shifted']) / cases['cases_new_shifted']
cases['deaths_new_delta'] = (cases['deaths_new'] - cases['deaths_new_shifted']) / cases['deaths_new_shifted']

# 2017 county census estimate from US Census Bureau.  Written locally to SQLite database
# to avoid having to re-download every time I run the notebook
# see census_etl.py for details

conn = sqlite3.connect('/Users/amcadi/Documents/opensource/covid_19_hospitalization_estimate/US_county_census.db')

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

# shift new cases and deaths to get day-over-day change
cases_percap['cases_new_shifted'] = (
   cases_percap.groupby(['county', 'state'])
   .cases_new
   .shift(1)
   .fillna(0)
   .astype(int)
)

cases_percap['deaths_new_shifted'] = (
   cases_percap.groupby(['county', 'state'])
   .deaths_new
   .shift(1)
   .fillna(0)
   .astype(int)
)

cases_percap['cases_new_delta'] = (cases_percap['cases_new'] - cases_percap['cases_new_shifted']) / cases_percap['cases_new_shifted']
cases_percap['deaths_new_delta'] = (cases_percap['deaths_new'] - cases_percap['deaths_new_shifted']) / cases_percap['deaths_new_shifted']

# create 5-day rolling average of new cases and deaths
# can't figure out how to accomplish this in the pipeline above
cases_percap['cases_new_MA'] = (cases_percap
                                .groupby('county_label')['cases_new']
                                .transform(lambda x: x.rolling(5).mean())
                               )

cases_percap['cases_new_per1k_MA'] = (cases_percap
                                      .groupby('county_label')['cases_new_per1k']
                                      .transform(lambda x: x.rolling(5).mean())
                                      )

cases_percap['deaths_new_MA'] = (cases_percap
                                 .groupby('county_label')['deaths_new']
                                 .transform(lambda x: x.rolling(5).mean())
                                )

cases_percap['deaths_new_per1k_MA'] = (cases_percap
                                       .groupby('county_label')['deaths_new_per1k']
                                       .transform(lambda x: x.rolling(5).mean())
                                      )

# also create 7-day rolling sum of per-capita new cases, which should be roughly proportional to
# the currently hospitalized population, assuming a 7-day length of stay
cases_percap['cases_new_per1k_sum7'] = (cases_percap
                                        .groupby('county_label')['cases_new_per1k']
                                        .transform(lambda x: x.rolling(7).sum())
                                       )

# non-normalized 12-day rolling sum of new cases, using 12-day avg LOS that Harvard used
cases_percap['cases_new_sum12'] = (cases_percap
                                   .groupby('county_label')['cases_new']
                                   .transform(lambda x: x.rolling(12).sum())
                                  )

# and finally a 7-day rolling mean of daily proportional change in new cases
cases_percap['cases_new_delta_MA7'] = (cases_percap
                                       .groupby('county_label')['cases_new_delta']
                                       .transform(lambda x: x.rolling(7).mean())
                                      )

# and a 14-day to try to get rid of some of the noise
cases_percap['cases_new_delta_MA14'] = (cases_percap
                                       .groupby('county_label')['cases_new_delta']
                                       .transform(lambda x: x.rolling(14).mean())
                                      )

# filter for 3 days ago, giving some time for the latest data to be populated
three_days_back = cases_percap.date.max() - datetime.timedelta(days = 3)

cases_percap_recent = cases_percap.query('TOT_POP > 100000 and date == @three_days_back')

# pull in census bureau regions to add a geographic dimension
regions_html = requests.get('https://simple.wikipedia.org/wiki/List_of_regions_of_the_United_States')
regions_soup = BeautifulSoup(regions_html.content)

def is_division(tag):
    return tag.name == 'li' and re.search('Division', tag.contents[0])

regions_list = regions_soup.find('ul')
regions = regions_list.find_all('li')
regions_dict = dict()
for region in regions:
    for division in region.find_all(is_division):
        div_name = re.sub(r'([/w]*)(\s\(not yet started\))', r'\1', division.a['title'])
        states = [item.string for item in division.find('ul').find_all('li')]
        regions_dict[div_name] = states

# reshape into a state:region dictionary
regions_dict_inverted = {}
for k, v in regions_dict.items():
    for state in v:
        regions_dict_inverted[state] = k

# map to dataset
cases_percap_recent['region'] = cases_percap_recent['STNAME'].map(regions_dict_inverted)

# try the week-over-week approach
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

cases_percap_recent = cases_percap_recent.merge(wkly_diff, how = 'left', on = 'county_label')

# get hospital bed data
hosp_req = requests.get('https://opendata.arcgis.com/datasets/6ac5e325468c4cb9b905f1728d6fbf0f_0.geojson')
hosp_json = hosp_req.json()
hosp_data = [x['properties'] for x in hosp_json['features']]

hosp_df = pd.DataFrame(hosp_data)

# get total acute/ICU beds per county
county_beds = (hosp_df.query("TYPE in ['GENERAL ACUTE CARE', 'CRITICAL ACCESS'] and BEDS != -999")
 .groupby(['STATE', 'COUNTY'])
 .aggregate({'BEDS': 'sum'})
 .reset_index()
 .assign(county_key = lambda x: x['COUNTY'].str.lower())
)

# map state abbreviations to full name to join to case data
state_abbrev_html = requests.get('https://docs.omnisci.com/v3.6.1/immerse-user-guide/state-abbreviations/')
state_abbrev_soup = BeautifulSoup(state_abbrev_html.content)
state_abbrev = state_abbrev_soup.find('tbody')

state_abbrev_dict = {}

for row in state_abbrev.find_all('tr'):
    cols = row.find_all('td')
    full = cols[0].string
    abbrev = cols[1].string
    state_abbrev_dict[abbrev] = full

county_beds['state_full'] = county_beds['STATE'].map(state_abbrev_dict)

# have to also do the NYC correction
ny_idx = county_beds['county_key'].str.contains('new york|kings|queens|bronx|richmond') & \
    county_beds['state_full'].eq('New York')

county_beds.loc[ny_idx, 'county_key'] = 'new york city'

county_beds = (county_beds
               .groupby(['county_key', 'state_full', 'STATE'])
               .aggregate({'BEDS': 'sum'})
               .reset_index()
              )
# join back to cases
cases_percap_recent = cases_percap_recent.merge(county_beds,
                                                how = 'left',
                                                left_on = ['county_key', 'state'],
                                                right_on = ['county_key', 'state_full']
                                               )

# estimate of free bed percentage assuming 20% hospitalization rate and 39% of beds available for covid patients
cases_percap_recent['free_bed_perc'] = (cases_percap_recent['BEDS'] * 0.39 - cases_percap_recent['cases_new_sum12'] * 0.2) / (cases_percap_recent['BEDS'] * 0.39)

# create chart and save it
capacity_vs_case_delta = (alt.Chart(cases_percap_recent.query('wkly_diff < 3 and wkly_diff > -3 and free_bed_perc > -10')) # dangerous, but seems like >3x is reporting issue
 .mark_point()
 .encode(x = alt.X('wkly_diff', axis = alt.Axis(title = 'Week-over-week New Case Delta')),
         y = alt.Y('free_bed_perc', axis = alt.Axis(title = 'Open Beds / Available Beds')),
         color = alt.Color('region', legend = alt.Legend(title = 'Region')),
         size = alt.Size('TOT_POP', legend = alt.Legend(title = 'Population')),
         tooltip = ['county_label', 'TOT_POP', 'free_bed_perc'])
 .interactive()
 .properties(height = 400,
             width = 750,
             title = ['Estimated Bed Capacity vs Weekly Change in New Cases',
                      'US counties > 100,000 residents'])
)

capacity_vs_case_delta.save('capacity_vs_case_delta.html')
