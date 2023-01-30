# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 15:23:18 2022

@author: jacob

Nerve-on-a-chip signal processing for automated processing of multiple channels and wells.

v1 - Peak finding in associated utils uses two stage std() based approach
"""

# %%
### Set parameters for peak finding code

# self-explanatory -- default 300
MAX_TWITCH_FREQUENCY=300

# scaling factor for minimum twitch prominence -- larger values make peak-finding more sensitive - default 6
PROMINENCE_FACTOR=6

# scaling factor for minimum twitch width -- larger values make peak-finding more sensitive - default 15
WIDTH_FACTOR=15

# scaling factor for minimum twitch height -- larger values make peak-finding less sensitive - default 5
HEIGHT_FACTOR=5

# direction of channels - 'down' type str() should be used for channels from row8 --> row 1, any other value will be row 1 --> row 8
direction = 'up'

# %%load packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from random import randint
import os

import utils

# %%empty output directories
# empty output directories
dir_list = ['./data/results/','./data/results/plots/','./data/results/traces/']

# search directories and only remove .png images other files should be overwritten
for d in dir_list:
    l = [file for file in os.listdir(d) if '.png' in file]
    for file in l:
        os.remove(f'{d}/{file}')
        
# %%load in CSV
# read in recording file
recording_file=r"G:\Shared drives\Science Team\Science R&D\NoC\2_Data\Chemotherapy drug study\MEA data\15.7.22 - 22.7.22 Chemo study\20220722 - Drug Study - Plate 4(000).csv"
df = pd.read_csv(recording_file, engine="pyarrow")


# look at data frame structure
df.head()

# %%generate electrode names and columns

if direction == 'down':
    first_row = 'Row 8'
else:
    first_row = 'Row 1'

# list all electrodes in recording
electrodes_total = [column for column in df.columns if column != 'Time (s)']

# number of electrodes on one side of electrode array
electrode_shape = 8

# generate electrode bases and intergers
electrode_bases = [10*(i+1) for i in range(electrode_shape)]
electrode_int = [(i+1) for i in range(electrode_shape)]

# generate electrode numbers as rows
rows = [f'Row {i+1}' for i in range(electrode_shape)]
electrode_rows = [[x+electrode_int[i] for x in electrode_bases] for i in range(electrode_shape)]

# create dictionary containing rows
mea_rows = {}
for key, value in zip(rows[::-1], electrode_rows[::-1]):
    mea_rows[key] = value

# generate electrode numbers as columns
columns = [f'Channel {i+1}' for i in range(electrode_shape)]

if  direction == 'down':
    electrode_columns = [[x+electrode_bases[i] for x in electrode_int[::-1]] for i in range(electrode_shape)]
else:
    electrode_columns = [[x+electrode_bases[i] for x in electrode_int] for i in range(electrode_shape)]

# create dictionary containing rows
mea_columns = {}
for key, value in zip(columns, electrode_columns):
    mea_columns[key] = value
    
print(mea_columns)

# %%define well names

# define well names with plate dimensions 
plate_row = [1,2,3]
plate_col = ['A','B']
wells = [f'{col}{row}' for col in plate_col for row in plate_row]

all_electrode_channels={f'{well} {key}':[f'{well} {electrode}' for electrode in mea_columns[key]] for well in wells for key in mea_columns}

# %%Plot all wells and all electrodes of a plate
# # title is used for name of saved file
# title = recording_file.split('.')[0].split('\\')[-1]

# # define well names and shape of the well plate
# wells_shape = [2,3]

# # run plotting function
# F = utils.plot_well_plate(df, wells, wells_shape, electrode_shape)

# # save figure
# F.savefig(f'./data/results/{title}.png')

# # show figure
# plt.show()

# %%Find peaks on first electrode of every channel on plate 
# define well names and shape of the well plate
w_row = [1,2,3]
w_col = ['A','B']
wells = [f'{col}{row}' for col in w_col for row in w_row]

# define first electrodes of each channel, use the mea_rows key to define top or bottom of channels as start
first_electrodes = [f'{well} {electrode}' for well in wells for electrode in mea_rows[first_row]]

# get sampling frequency
dt = df['Time (s)'][1] - df['Time (s)'][0]

# map all electrodes to color for plotting
colors = ['#%06X' % randint(0, 0xFFFFFF) for i in range(len(electrodes_total))]
cmap = dict(zip(electrodes_total, colors))

# define electrodes of interest as first electrodes of channel
electrodes_of_interest = [electrode for electrode in first_electrodes if electrode in df.columns]

# %%Find initial peaks
# intiailize peak-finding algoithm
PeakFinder = utils.Peaks(max_twitch_frequency=MAX_TWITCH_FREQUENCY, 
                         prominence_factor=PROMINENCE_FACTOR, 
                         width_factor=WIDTH_FACTOR, 
                         height_factor=HEIGHT_FACTOR)

# fit peak finding algorithm
electrode_peaks = PeakFinder.fit(recordings=df,
                                 electrodes=electrodes_of_interest)

# %%Filter peaks for active channels
# identify any electrodes at channel start with more than 75 peaks identified by peak finding
active_electrodes = [key for key in electrode_peaks if len(electrode_peaks[key])>75]


# filter electrode channels to select only channels with active electrodes at start
filtered_channels = {key:all_electrode_channels[key] for key in all_electrode_channels\
                     if all_electrode_channels[key][0] in active_electrodes}

# flatten dictionary to give all electrodes within filtered channels
filtered_electrodes=[electrode for channel in filtered_channels.values() for electrode in channel]

# %%refind peaks on all electrodes within active channels
# intiailize peak-finding algoithm
PeakFinder = utils.Peaks(max_twitch_frequency=MAX_TWITCH_FREQUENCY, 
                         prominence_factor=PROMINENCE_FACTOR, 
                         width_factor=WIDTH_FACTOR, 
                         height_factor=HEIGHT_FACTOR)

# fit peak finding algorithm
electrode_peaks = PeakFinder.fit(recordings=df,
                                 electrodes=filtered_electrodes)


# remove any electrodes from channels if number of peaks is less than 25% of starting electrode and remove channels with
# less than 3 active electrodes
short_channels=[]

for key in filtered_channels:
    for electrode in filtered_channels[key]:
        if len(electrode_peaks[electrode]) < len(electrode_peaks[filtered_channels[key][0]]) * 0.25:
            del filtered_channels[key][filtered_channels[key].index(electrode):]
        
        if len(filtered_channels[key]) < 3:
            short_channels.append(key)
            
        if df[filtered_channels[key][0]].max() < 0.00008:
            short_channels.append(key)
            

            
filtered_channels = {key:filtered_channels[key] for key in filtered_channels\
                    if key not in short_channels}

print(filtered_channels.keys())

# %%Find peaklets
# find minumum time between peaks to define search window for peaklets
# peaklets cannot occur with a spacing greater then inter peak interval as this will allow independent peaks 
# to be linked as peaklets
minimum_sample_interval = {}

for key in filtered_channels:
    for electrode in filtered_channels[key]:
        inter_peak_interval = [x-y \
                           for x,y \
                           in zip(electrode_peaks[electrode][1:],electrode_peaks[electrode][:-1])]
        
        inter_peak_interval = np.concatenate(inter_peak_interval, axis=0)
        
        minimum_sample_interval[key] = inter_peak_interval.min()
    
minimum_sample_interval

filtered_peaklets = {}
peaklets_found = []
peaks_found = []

for key in filtered_channels:
    # generate peaks dictionary for channel in question
    channel_peaks = {e:electrode_peaks[e] for e in electrode_peaks if e in filtered_channels[key]}
    
    # maximum amount of time between any pair of peaks in a peaklet
    # increasing `n_time_samples` will make peaklet finding more sensitive (e.g. will identify more peaklets)
    n_time_samples=minimum_sample_interval[key]-1
    max_peaklet_distance = dt*n_time_samples
    
    # empty peaklets dictionary
    peaklets = {}
        
    # intialize peaklet finding
    PeakletFinder = utils.Peaklets(max_peaklet_distance=max_peaklet_distance, 
                                   traversal_order=filtered_channels[key])
    
    # fit peaklet finding
    peaklets = PeakletFinder.fit(peaks=channel_peaks, 
                                 recordings=df)
    
    print(f'Total of {peaklets["indices"].shape[0]} peaklets identified in {key}')
    peaklets_found.append(peaklets["indices"].shape[0])

    # print number of peaks found of the final electrode of the channel
    print(f'Total of {len(channel_peaks[filtered_channels[key][-1]])} peaks found on final electrode in channel')
    peaks_found.append(len(channel_peaks[filtered_channels[key][-1]]))
    
    # check if sufficient peaks have been identified as peaklets. Low peaklet identification ratios suggests poor quality data
    if peaklets["indices"].shape[0]/len(channel_peaks[filtered_channels[key][-1]]) >= 0.4:
        filtered_peaklets[key] = peaklets     
        
# save peaklets to peak ratio
peaklet_peaks_ratio=np.asarray(peaklets_found)/np.asarray(peaks_found)
pd.DataFrame(peaklet_peaks_ratio).to_csv('./data/results/xlsx/peaklets_to_peak_ratios.csv')

# remove any channels which fail to give sufficient transmitted events from channel dictionary
filtered_channels = {k:filtered_channels[k] for k in filtered_channels if k in filtered_peaklets.keys()}

# display channels which have passed all QC checks
print([key for key in filtered_channels.keys()])

# %%Compute conduction velocities
# distance between electrodes on the multi-electrode array (in micrometers)
ELECTRODE_SPATIAL_FREQUENCY = 300
# conversion factor for micro to meters
MICRO_TO_BASE_CONVERSION=1e6

# generate dictionary of all possible electrode pairs for each channel
electrode_pairs = {}

for key in filtered_channels:
    electrode_pairs[key] = [f'{ie}->{je}' \
                            for i,ie in enumerate(filtered_channels[key][:-1]) \
                            for j,je in enumerate(filtered_channels[key][(i+1):])]
        
filtered_metrics = {}
filtered_summary = {}

# for each channel which has passed QC generate conduction velocity metrics
for key in filtered_channels:
    # extract channel peaklets
    peaklets = filtered_peaklets[key]
    
    # extract channel electrode names
    electrodes_of_interest = filtered_channels[key]
    
    # build empty dataframe to store time-delay and conduction velocities
    columns = pd.MultiIndex.from_product(
        [electrode_pairs[key], ['Time Delay (s)', 'Velocity', 'Distance (um)']],
        names=["Electrode Pair", "Statistic"])
    
    metrics = pd.DataFrame(index=np.arange(peaklets['indices'].shape[0]), columns=columns)
    metrics.columns = metrics.sort_index(
        axis=1, level=[0, 1], ascending=[True, True]
    ).columns
    
    # fill dataframe with time-delay and conduction velocity estimates for each electrode pair, for all peaklets
    for i in range(0,len(electrodes_of_interest)-1):
        for j in range((i+1),len(electrodes_of_interest)):
    
            # get electrode names
            electrode_a=electrodes_of_interest[i]
            electrode_b=electrodes_of_interest[j]
            
            # build string for filling dataframe
            electrode_pair=f'{electrode_a}->{electrode_b}'
            
            # get time-delay between peaklet indices of each electrode
            time_difference = peaklets['time'][electrode_b] - peaklets['time'][electrode_a]
            # compute conduction velocity between peaklet indices of each electrode
            velocity = (ELECTRODE_SPATIAL_FREQUENCY*(j-i))/time_difference/MICRO_TO_BASE_CONVERSION
            
            # fill dataframe
            metrics[electrode_pair, 'Time Delay (s)'] = time_difference
            metrics[electrode_pair, 'Velocity'] = velocity
            metrics[electrode_pair, 'Distance (um)'] = ELECTRODE_SPATIAL_FREQUENCY*(j-i)
    
    # for some reason, data isn't numeric, so we convert it to numeric here
    for pair in electrode_pairs[key]:
        metrics[pair, 'Time Delay (s)'] = pd.to_numeric(metrics[pair, 'Time Delay (s)'])
        metrics[pair, 'Velocity'] = pd.to_numeric(metrics[pair, 'Velocity'])
        
    # commit df to metrics dictionary
    filtered_metrics[key] = metrics
    
    # convert metrics dataframe to long-form
    M = metrics.T.unstack(level=0).T.reset_index().drop(columns=['level_0'])
    M.head()
   
    # generate summary statistics for time-delays and conduction velocities and save to csv. 
    # infinite speeds in velocity throw a warning, only describe Distance (um) and Time Delay (s)
    summary = pd.DataFrame(M.loc[:,:'Time Delay (s)'].groupby(by=['Electrode Pair']).describe()) 
    
    ## delete velocity summary and recreate using time difference means to account for error distribution
    #del summary['Velocity']
    summary['Velocity', 'mean'] = summary['Distance (um)', 'mean']/summary['Time Delay (s)', 'mean']\
                                   /MICRO_TO_BASE_CONVERSION
    
    summary['Velocity', 'upper confidence'] = summary['Distance (um)', 'mean']/\
                                             (summary['Time Delay (s)', 'mean']-1.96*(summary['Time Delay (s)', 'std']/peaklets["indices"].shape[0]**.5))\
                                               /MICRO_TO_BASE_CONVERSION
    
    summary['Velocity', 'lower confidence'] = summary['Distance (um)', 'mean']/\
                                             (summary['Time Delay (s)', 'mean']+1.96*(summary['Time Delay (s)', 'std']/peaklets["indices"].shape[0]**.5))\
                                               /MICRO_TO_BASE_CONVERSION
    
    # commit summary df to summary dictionary
    filtered_summary[key] = summary
    
for frame in filtered_summary.values():
    print(frame)
    
# save summary conduction metrics dataframes to xlsx
with pd.ExcelWriter('./data/results/xlsx/Conduction_Velocity_Summary.xlsx') as writer:
        for key, value in filtered_summary.items():
            value.to_excel(writer, key)

# save all metrics value dataframe to xlsx
with pd.ExcelWriter('./data/results/xlsx/Conduction_Velocities.xlsx') as writer:
        for key, value in filtered_metrics.items():
            value.to_excel(writer, key)
            
# %%Plot filtered channel waveforms with identified peaks

for key in filtered_channels:
        
    height = len(filtered_channels[key])
    
    fig = plt.figure(figsize=(15, 7*height))
    outer = gridspec.GridSpec(height, 1, wspace=0.2, hspace=0.2)
    
    for i, electrode in enumerate(filtered_channels[key]):
        peaks = electrode_peaks[electrode].squeeze()
    
        X = df['Time (s)']
        Y = df[electrode]
    
        ax=fig.add_subplot(outer[i])
        
        # plot waveform
        ax.plot(X, Y, c=cmap[electrode], label='Waveform')
        # plot peaks
        ax.scatter(df['Time (s)'].loc[peaks], df[electrode].loc[peaks], c='k', label='Peaks')
    
        # set plot parameters
        ax.set_ylabel('Voltage', fontsize=15)
        ax.set_xlabel('Time (s)', fontsize=15)
        ax.tick_params(labelsize=12)
        ax.ticklabel_format(useOffset=False, style='plain')
        ax.set_ylim(df[filtered_channels[key]].min().min(), df[filtered_channels[key]].max().max())
        ax.set_title(f'Recording for electrode {electrode} with identified peaks', fontsize=15)
    
    
    plt.savefig(f'./data/results/traces/Peak Traces for {key}.png', bbox_inches='tight')
    
# %%Burst detection
filtered_bursts_metrics = {}
bursts_dict = {}

for key in filtered_channels:
    
    # create data frame to hold statistics
    columns = pd.MultiIndex.from_product(
        [filtered_channels[key], ['Burst Intervals (s)', 'Burst Frequency (Hz)', 'Burst Length (s)']],
        names=['Electrode', 'Statistic'])
    
    burst_metrics = pd.DataFrame(columns=columns)
    
    # for every electrode in the channel find bursts
    for electrode in filtered_channels[key]:
        # extract only electrode of interest from electrode peaks data
        burst_data = electrode_peaks[electrode]
        
        # maximum time between peaks within a burst (ms) and the minimum number of spikes required to define a burst
        MAX_BURST_TIME=200
        MIN_BURST_NUMBER=5
        
        # run burst finding function
        bursts = utils.find_bursts(burst_data,
                                   MAX_BURST_TIME,
                                   MIN_BURST_NUMBER,
                                   dt)
        
        # add bursts to bursts dictionary
        bursts_dict[electrode] = bursts
        
        # print number of bursts found for each electrode
        bursts_found = len(bursts['start times'])
        
        # check if any bursts were found
        if bursts_found > 0:
            # burst statistics
        
            # inter burst interval in seconds
            burst_intervals = [None]
            
            for burst in range(1,bursts_found):
                interval = df['Time (s)'].loc[bursts['start times'][burst]]-df['Time (s)'].loc[bursts['start times'][burst-1]]
                burst_intervals.append(interval)
            
            # average frequncy of bursts in Hz
            burst_frequency = []
            burst_length = []
            
            for burst in range(bursts_found):
                burst_time = df['Time (s)'].loc[bursts['end times'][burst]]-df['Time (s)'].loc[bursts['start times'][burst]]
                frequency = bursts['number of peaks'][burst]/burst_time
                    
                burst_frequency.append(frequency)
                burst_length.append(burst_time)
            
            # populate data frame
            # if more bursts were found than first electrode pd.Dataframe is too short, reindex
            if len(burst_intervals) > burst_metrics.shape[0]:
                burst_metrics = burst_metrics.reindex(range(len(burst_intervals)))
            # add values
            burst_metrics[electrode, 'Burst Intervals (s)']=pd.Series(burst_intervals)
            burst_metrics[electrode, 'Burst Frequency (Hz)']=pd.Series(burst_frequency)
            burst_metrics[electrode, 'Burst Length (s)']=pd.Series(burst_length)
            
            # add burst metrics to dictionary          
            filtered_bursts_metrics[key] = burst_metrics
 
        
for frame in filtered_bursts_metrics.values():
    print(frame)
    
# save burst metrics dataframes to xlsx
with pd.ExcelWriter('./data/results/xlsx/Burst_Metrics.xlsx') as writer:
        for key, value in filtered_bursts_metrics.items():
            value.to_excel(writer, key)
            
for key in filtered_channels:
        
    height = len(filtered_channels[key])
    main_figure = plt.figure(figsize=(15,10*height))
    
    outer = main_figure.add_gridspec(height, 1, wspace=0.2, hspace=0.2)
    
    for i, electrode in enumerate(filtered_channels[key]):
        inner = outer[i].subgridspec(2, 1, wspace=0.2, hspace=0.2, height_ratios=[8,1])
        
        # plot these bursts as overlays of the waveform graphs
        peaks = electrode_peaks[electrode].squeeze()
        
        X = df['Time (s)']
        Y = df[electrode]
        
        ax0=main_figure.add_subplot(inner[0])
        ax1=main_figure.add_subplot(inner[1])
        
        # plot waveform
        ax0.plot(X, Y, c='#19AB8A', linewidth=.2, label='Waveform')
        
        # plot peaks
        ax0.scatter(df['Time (s)'].loc[peaks], df[electrode].loc[peaks],
                      facecolors = 'None', edgecolors = '#A7A7A8',
                      label='Peaks')
        
        
        # plot peaks identified as belonging to a peaklet   
        peaklet = filtered_peaklets[key]['indices'][electrode]
        ax0.scatter(df['Time (s)'].loc[peaklet], df[electrode].loc[peaklet], c = '#A7A7A8',
                      label='Transmitted peaks')
        
        # plot overlay shading for bursts
        bursts =  bursts_dict[electrode]
        for i in range(len(bursts['start times'])):
            ax1.axvspan(df['Time (s)'].loc[bursts['start times'][i]],
                          df['Time (s)'].loc[bursts['end times'][i]],
                          color='#00263E', alpha=0.5)
        
        # set plot parameters
        ax0.set_ylabel('Voltage', fontsize=15)
        ax0.set(xticklabels=[])
        ax0.tick_params(labelsize=12, bottom = False)
        ax0.set_title(f'Recording for electrode {electrode} with identified peaks', fontsize=15);
        
        ax1.tick_params(labelsize=12)
        ax1.set(yticklabels=[])
        ax1.set_xlabel('Time (s)', fontsize=15)
        ax1.set_title('Bursts', fontsize=15)
            
        # set x limits so plots aline
        ax0.set_xlim(-X.max()*.01,X.max()*1.01)
        ax1.set_xlim(-X.max()*.01,X.max()*1.01)
        
        # show legend
        ax0.legend(fontsize=12, loc='lower center', ncol=3)
        
        
        
        # save figure as .png
        plt.savefig(f'./data/results/traces/Bursts and peaks for {key}.png',
                    bbox_inches='tight')
        
# %%Plot highest voltage peaklet from each channel

# find index of highest voltage peaklet for each channel
max_peak_index = {}

for key in filtered_peaklets:
    index = filtered_peaklets[key]['voltage'][filtered_channels[key][0]].to_list()\
            .index(filtered_peaklets[key]['voltage'][filtered_channels[key][0]].max())
  
    max_peak_index[key] = index
    
max_peak_index

# look at individual waveforms of each peaklet
# `index` corresponds to a row in peaklets['indices'], peaklets['time'], and peaklets['voltage']
for key in max_peak_index:
    
    F = utils.plot_peaklet(peaklets=filtered_peaklets[key], recordings=df,index=max_peak_index[key], window=50, cmap=cmap)
    plt.tight_layout()

    # save waveform as .png
    F.savefig(f'./data/results/traces/Peaklet from {key} - Peaklet no. {max_peak_index[key]}.png', bbox_inches='tight')
    plt.show()
    
# %%Plot voltage vs time delay
# extract only sequential pairs of electrodes from electrode pairs list
sequential_pairs_dict={}

for key in filtered_channels:
    electrode_channel=filtered_channels[key]
    
    # generate index list of pairs within list of length number of electrodes
    sequential_pairs = []
    x=0
    for i in range(len(electrode_channel), 1, -1):
        sequential_pairs.append(x)
        x = x+(i-1)
    
    # extract pairs from electrode pairs list
    sequential_pairs = [electrode_pairs[key][pair] for pair in sequential_pairs]
    
    # append longest possible pair from pairs list
    sequential_pairs.append(electrode_pairs[key][len(electrode_channel)-2])
    
    # write to sequential pairs dictionary
    sequential_pairs_dict[key]=sequential_pairs
    
# plot velocity of peaklet identified against the time delay of the peaks within the signal

for key in sequential_pairs_dict:
    # choose which electrode pair to examine
    electrode_pair_velocity = sequential_pairs_dict[key][-1]
    
    # create dataframe for sorting data
    velocity_plot = pd.DataFrame({
        'Delay':[],
        'Voltage':[]})
    
    # collect peaklet voltages
    velocity_plot['Voltage'] = filtered_peaklets[key]['voltage'][filtered_channels[key][0]]
    
    # collect velocity of peaklets
    velocity_plot['Delay'] = filtered_metrics[key][electrode_pair_velocity, 'Time Delay (s)']
    
    # sort frame in order of ascending velocity
    velocity_plot = velocity_plot.sort_values(by=['Delay'])
    
    # set colmap if delay is 0s colour red
    colmap = np.where(velocity_plot['Delay']==0, 'r', 'k')
    
    # plot
    fig, ax = plt.subplots(1,1, figsize=(10, 7))
    ax.scatter(velocity_plot['Delay'],
               velocity_plot['Voltage'],
               color = colmap)
    ax.set_xlim(0-velocity_plot['Delay'].max()*0.03,
                velocity_plot['Delay'].max()+0.2*velocity_plot['Delay'].max())
    ax.set_xlabel('Time Delay (s)', fontsize=20)
    ax.set_ylabel('Voltage', fontsize=20)
    ax.set_title(f'Voltage Velocity relationship for {key}' 
                 '\n' 
                 f'Total transmittion events = {len(velocity_plot.index)}', fontsize=20)
    
    plt.savefig(f'./data/results/plots/Velocity Delay for {key}.png', bbox_inches='tight')