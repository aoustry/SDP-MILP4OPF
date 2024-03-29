from master import global_algo,basicsdp_relaxation_value
import sys

#Global algo parameters
BTtimeLimit = 10*3600
MILPtimeLimit = 5*3600
reltol = 1E-4


#Instance parameter
lineconstraints = 'S'

#Instances list
instances = [         'pglib_opf_case3_lmbd.m',
                            'pglib_opf_case5_pjm.m', 
                          'pglib_opf_case14_ieee.m',
                            'pglib_opf_case24_ieee_rts.m',
                            'pglib_opf_case30_as.m',
                    'pglib_opf_case30_ieee.m',
                      'pglib_opf_case39_epri.m',
                    'pglib_opf_case57_ieee.m',
                     'pglib_opf_case73_ieee_rts.m',
           'pglib_opf_case89_pegase.m',    
           'pglib_opf_case118_ieee.m',  
       'pglib_opf_case162_ieee_dtc.m', 
      'pglib_opf_case179_goc.m', 
        'pglib_opf_case200_activ.m', 
      'pglib_opf_case240_pserc.m', 
    'pglib_opf_case300_ieee.m',
    'pglib_opf_case500_goc.m'

]


for name_instance in instances:
    basicsdp_relaxation_value(name_instance.replace('.m','')+"__api",lineconstraints,'data/pglib-opf/api')
    global_algo(name_instance.replace('.m','')+"__api",lineconstraints,'data/pglib-opf/api',BTtimeLimit,MILPtimeLimit,reltol)
    
    
    