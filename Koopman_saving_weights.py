#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 10:38:15 2021

@author: wtredman
"""


def flat_params_as_numpy(self):
    weights = []
    for p in self.model.parameters():
        weights.append(p.view(-1))
        
    return torch.cat(weights,0).cpu().detach().numpy()
    
def save_weights(self):
    self.model.W.append(self.flat_params_as_numpy())