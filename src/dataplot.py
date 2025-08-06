# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 22:13:40 2024

@author: kyleq
"""

import numpy as np
import matplotlib.pyplot as plt
a=np.loadtxt("0TK0.2.txt")
for i in range(len(a)):
    for j in range(len(a[0])):
        if a[i][j] < 0:
            a[i][j] = np.nan
plt.imshow(a[0:51, :],vmin = 0, vmax=0.008, origin='lower', interpolation = 'antialiased',cmap ="jet")
