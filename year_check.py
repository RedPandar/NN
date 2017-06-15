# -*- coding: utf-8 -*-
"""
Created on Wed May  3 11:14:26 2017

@author: Alexsandr
"""

import datetime 

def check_Year(a):
    now = datetime().now
    if ((datetime(now.year)-a)<(datetime-4)):
        return 0
    else:
        return 1