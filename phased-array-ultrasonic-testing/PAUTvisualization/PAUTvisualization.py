import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import math


pautcolorscale = [
    [0, 'rgb(255,255,255)'],
    [0.066, 'rgb(184,212,244)'],
    [0.133, 'rgb(113,170,233)'],
    [0.2, 'rgb(62,105,190)'],
    [0.266, 'rgb(14,37,143)'],
    [0.333, 'rgb(27,72,129)'],
    [0.4, 'rgb(59,140,127)'],
    [0.466, 'rgb(126,187,94)'],
    [0.533, 'rgb(211,223,45)'],
    [0.6, 'rgb(241,211,43)'],
    [0.666, 'rgb(222,156,80)'],
    [0.733, 'rgb(209,121,87)'],
    [0.8, 'rgb(205,116,49)'],
    [0.866, 'rgb(194,98,23)'],
    [0.933, 'rgb(167,50,26)'],
    [1, 'rgb(145,12,29)']
]


def PAUTvisualization(file):
    df = pd.read_excel(file, header = None)
    print('a')

if __name__ == "__main__":
    
    for a in sys.argv:
        if a == '-i':
            file = sys.argv[2]
        
    PAUTvisualization(file)