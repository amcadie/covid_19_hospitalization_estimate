FROM python:3.7.1-stretch

LABEL name='covid_county_viz' version='0.1'

RUN pip3 install --upgrade pip \
    jupyter \
    pandas \
    bs4 \
    numpy \
    altair

RUN mkdir /usr/covid_viz/

WORKDIR /usr/covid_viz/

ENTRYPOINT ["/bin/bash", "nbconvert.sh"]
