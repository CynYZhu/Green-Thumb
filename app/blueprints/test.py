# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 14:42:28 2022

@author: turna
"""
import pandas as pd

url = "http://20.127.87.137/plants"
df = pd.read_json(url)

df.columns

variables = [ 
'attracts'
, 'common'
, 'flower'
, 'fruit'
, 'leaf'
, 'water'
, 'genus'
, 'height'
, 'width'
, 'special_feature'
, 'scientific_name'
, 'description'
, 'soil_conditions'
, 'texture'
, 'temperature'
, 'emergence_days'
, 'maintenance'
, 'native_region'
, 'propagation'
, 'bloom_time'
, 'sun'
, 'problem_solvers'
, 'type'
, 'zones'
, 'family'
, 'color'
, 'flowering'
, 'smell'
, 'annual'
, 'perennial'
, 'biennial'
, 'drought'
, 'air_pollution'
, 'dry_soil'
, 'wet_soil'
, 'clay_soil'
, 'rain_garden'
, 'hedge'
, 'herb'
, 'good_for_containers'
, 'vegetable'
, 'water_plant'
, 'bulb'
]


def build_filters():
    filters = {}
    for v in variables:
        # list the unique values for df column   
        f = set([item for sublist in df[v].unique() for item in sublist.strip('][\'').replace("', '", ', ').split(', ')])
        
        dk = '...'
        if dk in f:
            f.remove(dk)
        
        filters[v] = f
        
    return filters


# max number in list
l = [int(i) for i in filters[list(filters.keys())[7]]]

for n in l:
    if (m is None or n > m):
        m = n