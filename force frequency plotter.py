# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 14:25:56 2023

@author: jacob

Force frequency analysis script

Designed to find peak force for each frequency within the trace, extract these
values, plot a dose response fit for this data and estimate an Freq50 value
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import pandas as pd
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import  numpy as np
#%% define file and well to analyse and frequencies used in the analysis
input_path = r"G:\Shared drives\Science Team\Science R&D\NMJ\2_Data\Lonza-UC2 collagen co-culture\MA force\20220719 - Day 23\ML2022071903__2022_07_19_Force-freq.xlsx"

well = 'D1'

frequencies = [1,2,3,5,10,20,30,40,50]

#%% formating arguments
plt.rcParams['font.size'] = 22
plt.rcParams['font.weight'] = 'bold'

#%% load in raw MA trace data
df = pd.read_excel(input_path, 'continuous-waveforms')

#%% find peaks in data
peak_prom = df[f'{well} - Active Twitch Force (μN)'][:300].max() - df[f'{well} - Active Twitch Force (μN)'][:300].min()

# find peaks within the data, the distance assumes a pusle chain of 2 seconds duration, 
# increase this to match number of samples per pulse chain if longer pulses are used
peaks,_ = find_peaks(df[f'{well} - Active Twitch Force (μN)'],
                     prominence=5 * peak_prom,
                     distance=200)

#%% this is an attempt to remove extranious peaks or add in peaks which are smaller than the initial sd estimate -  it is rather crude
sd = 3
while len(peaks) != len(frequencies):
    peaks,_ = find_peaks(df[f'{well} - Active Twitch Force (μN)'],
                         prominence=sd * peak_prom,
                         distance=200)
    
    sd+=1


#%% plot waveform
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(25,7),
                        gridspec_kw={'width_ratios':(3,1)})

fig.suptitle(f'Force frequency relationship for well {well}')

X = df['Time (seconds)']
Y = df[f'{well} - Active Twitch Force (μN)']
axs[0].scatter(X[peaks], Y[peaks], s=150, marker = 'o', facecolors='none', edgecolors='r')
axs[0].plot(X,Y, c='g')

axs[0].set(xlabel='Time (s)', ylabel=u'Force (\u03bcN)')

#%% extrace peak heights and fit frequency curve
max_peak = df[f'{well} - Active Twitch Force (μN)'][peaks].max()
min_peak = df[f'{well} - Active Twitch Force (μN)'][peaks].min()

peak_heights = [100*(i-min_peak)/(max_peak-min_peak) for i in df[f'{well} - Active Twitch Force (μN)'][peaks]]

def logistic4(x, A, B, C, D):
    """4PL lgoistic equation."""
    return ((A-D)/(1.0+((x/C)**B))) + D

popt, pcov = curve_fit(logistic4, frequencies, peak_heights)
fitted_x = np.linspace(min(frequencies), max(frequencies), 100)
fitted_y = logistic4(fitted_x, *popt)

Freq50 = popt[2]

axs[1].semilogx(fitted_x,fitted_y, c='g')
axs[1].scatter(frequencies, peak_heights, c='g')
axs[1].hlines(logistic4(Freq50, *popt),0,Freq50, linestyles = 'dashed', color='k')
axs[1].vlines(Freq50, 0, logistic4(Freq50, *popt), linestyles = 'dashed', color='k')
axs[1].annotate(f'Freq50 : {round(Freq50, 2)}Hz', 
                (frequencies[4],logistic4(Freq50, *popt)),
                size=14)

axs[1].set(xlabel='Frequency (Hz)', ylabel='Response range (%)')
axs[1].get_xaxis().set_major_formatter(ScalarFormatter())
axs[1].set_xticks([*frequencies[0:5],frequencies[-1]])
