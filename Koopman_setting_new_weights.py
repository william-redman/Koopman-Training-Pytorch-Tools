#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 10:36:49 2021

@author: wtredman
"""


def map_fixed_pt(params, fixed_pt):
    new_params = params
    counter = 0
    
    for key in params.keys():
        n_weights = np.prod(params[key]['weight'].shape)
        new_params[key]['weight'] = np.reshape(fixed_pt[counter:(counter + n_weights)], params[key]['weight'].shape)
        counter = counter + n_weights
        
        n_bias = params[key]['bias'].shape[0]
        new_params[key]['bias'] = np.reshape(fixed_pt[counter:(counter + n_bias)], params[key]['bias'].shape)
        counter = counter + n_bias
        
        return new_params