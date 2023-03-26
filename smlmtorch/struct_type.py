# -*- coding: utf-8 -*-
"""
Dictionary that allows indexing like struct.member

Created on Thu May 26 21:03:08 2022

@author: jelmer
"""
from collections.abc import Mapping

class struct(Mapping):
    def __init__(self,  d=None, **kwargs):
        if d is not None:
            self.__dict__.update(d)        
        self.__dict__.update(kwargs)
        
    def __getitem__(self, key):
        return self.__dict__[key]
    
    def __setitem__(self, key, val):
        self.__dict__[key] = val
        
    def __repr__(self):
        return str(self.__dict__)

    def __len__(self):
        return len(self.__dict__)
    
    def __contains__(self, key):
        return key in self.__dict__
    
    def __iter__(self):
        return self.__dict__.__iter__()
        
    
    @staticmethod
    def from_dict(v):
        s = struct(v)
        for k,v in s.items():
            if type(v) == dict:
                s[k] = struct(v)
        return s        
        
if __name__ == '__main__':
    b = struct(x=1)

    print(b)
    
    print(struct.from_dict({ 'x':{'y':2} }))
    