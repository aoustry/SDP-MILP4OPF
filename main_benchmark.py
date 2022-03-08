# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 15:57:17 2021

@author: aoust

Special purpose file to test the Bukhsh instances.

Comparison with H. Godard's PhD thesis results => same total time limit (1h) and same accuracy (0.1\%)
"""


from master import global_algo
import sys

#Instance parameters
lineconstraints = False

#Overriding Main algo parameters
BTtimeLimit = 1800
MILPtimeLimit = 3600
reltol = 1E-3 
     
        
instances = [
        'WB2',
        'WB3',
        'WB5',
    'WB5mod',
    'case9mod',
  'case22loop',
  'case30loopmod',
'case39mod1',
  'case39mod2',
  'case118mod',
  'case5',
  'case6ww',
  'case9',
  'case14',
  'case30',
  'case39',
  'case57',
  'case89pegase',
  'case118',
    'case300',
  'case300mod',
 'pglib_opf_case118_ieee',
 'pglib_opf_case118_ieee__api',
 'pglib_opf_case118_ieee__sad',
 'pglib_opf_case162_ieee_dtc',
 'pglib_opf_case162_ieee_dtc__api',
 'pglib_opf_case162_ieee_dtc__sad',
 'pglib_opf_case300_ieee',
 'pglib_opf_case300_ieee__api',
 'pglib_opf_case300_ieee__sad'
 ]


for name_instance in instances:
    try:
        global_algo(name_instance.replace('.m',''),lineconstraints,'data/benchmark_godard',BTtimeLimit,MILPtimeLimit,reltol)
    except:
        print("Unexpected error:", sys.exc_info()[0])