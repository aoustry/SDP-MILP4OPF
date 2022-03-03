# -*- coding: utf-8 -*-
"""
@author: aoust

Special purpose file to test some variants of instance MATPOWER case57.
"""


from master import *

#Global algo parameters
BTtimeLimit = 36000
MILPtimeLimit = 5*3600
reltol = 1E-4


#Instance parameters
lineconstraints = 'I'


        
instances = [ 
 'case57',
'case57_84',
'case57_260',
'case57_267',
'case57_299',
'case57_628',
'case57_683',
'case57_829',
'case57_868',
'case57_974'
 ]


for name_instance in instances:
    global_algo(name_instance.replace('.m',''),lineconstraints,'data/case57',BTtimeLimit,MILPtimeLimit,reltol)