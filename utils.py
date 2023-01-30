import itertools
import pandas as pd
import numpy as np
from scipy import signal
from typing import List
from nptyping import NDArray
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class Peaklets(object):
    """
    Find peaklets (N-tuples) of peaks corresponding to the same action potential across electrode recordings.
    
    Args:
        max_peaklet_distance (float): maximum allowed time between peaks within a peaklet
        traversal_order (list, str): traversal order of electrodes
        
        Notes:
        We can think of ```max_peaklet_distance``` as the sampling frequency (dt) * max number of samples btw. peaklet peaks
    """
    
    def __init__(self, max_peaklet_distance: float, traversal_order: list):
        self.max_peaklet_distance = max_peaklet_distance
        self.electrode_order = traversal_order
    
    def fit(self, peaks: dict, recordings: pd.DataFrame):
        """
        Args:
        peaks (dict): identified peak indices of each electrode.
        recording (Pandas DataFrame): raw voltage recordings
        """

        time = recordings['Time (s)']
        voltage = recordings[self.electrode_order]

        # find all peaklets that meet temporal filtering criteria
        # we iteratively build up 1-tuple, 2-tuple, ... N-tuple, performing filtering at each at traversal level
        # this is faster and more RAM-efficient than building all possible N-tuples and then filtering
        for i, electrode in enumerate(self.electrode_order):
            se_peaks = peaks[electrode]
            if i == 0:
                peaklets = self.__find_peaklets([se_peaks])
            else:
                peaklets = self.__find_peaklets([peaklets, se_peaks])

            peaklets = self.__filter_peaklets(peaklets, np.asarray(time), self.max_peaklet_distance)

        peaklets = pd.DataFrame(peaklets, columns=self.electrode_order)
        
        # get time of each peaklet peak
        peaklet_time = pd.DataFrame(np.asarray(time)[peaklets], columns=self.electrode_order)
        
        # get voltage of each peaklet peak
        peaklet_volt = pd.DataFrame(np.column_stack([voltage.loc[peaklets[elec], elec] \
                                                     for elec in self.electrode_order]), 
                                    columns=self.electrode_order)

        peaklets =  {'indices': peaklets,
                     'time': peaklet_time,
                     'voltage': peaklet_volt}
        
        return peaklets
    
    def __find_peaklets(self, peaks: List):
        """
        Compute all combinations of peak indices.

        Args:
        peaks (list): lists of lists of peak indices for each electrode recording.

        Returns:
        iterator over combinations of indices
        """
        if len(peaks) == 1:
            return peaks[0]

        peaklets = []
        # iterate over all pairs of peaks
        for i, (peaklet_a, peaklet_b) in enumerate(itertools.product(*peaks)):

            if isinstance(peaklet_a, int):
                peaklet_a = [peaklet_a]
            elif isinstance(peaklet_a, np.ndarray):
                peaklet_a = list(peaklet_a)
            else:
                raise(TypeError)

            if isinstance(peaklet_b, int):
                peaklet_b = [peaklet_b]
            elif isinstance(peaklet_b, np.ndarray):
                peaklet_b = list(peaklet_b)
            else:
                raise(TypeError)

            p = np.concatenate([peaklet_a, peaklet_b])
            peaklets.append(p)

        peaklets = np.row_stack(peaklets)

        return peaklets

    def __filter_peaklets(self, peaklets: NDArray, time: NDArray, max_peaklet_distance: float):
        """Find valid peaklets by minimum distance between peaks, and by action potential timing.

        Args:
        peaklets (iterator): list of identified peaklets
        time (NDArray): time points of all samples
        dt (float): sampling period
        window (int): number of time samples between peaks
        """
        
        # temporal filtering
        # make sure minimum distance between peaklet peaks is less than max distance
        T = np.asarray(time)[peaklets]
        valid_peaklets_temporal = (T.max(1) - T.min(1)) <= (max_peaklet_distance)
        peaklets = peaklets[valid_peaklets_temporal]
                
        # traversal filtering
        # make sure time of indicence of each action potential on each electrode follows expected order
        T = np.asarray(time)[peaklets]
        valid_peaklets_traversal = np.all(np.diff(T, axis=1)>=0, axis=1)
        peaklets = peaklets[valid_peaklets_traversal]
                
        return peaklets
    
    
class Peaks(object):
    """
    
    """
    
    def __init__(self, max_twitch_frequency:float=200, prominence_factor:float=10, width_factor:float=10, height_factor:float=0):
        self.max_twitch_frequency = max_twitch_frequency
        self.prominence_factor=prominence_factor
        self.width_factor=width_factor
        self.height_factor=height_factor
    
    def fit(self, recordings:pd.DataFrame, electrodes:list=None):
        electrode_peaks = {}
        # iterate over each electrode, find peaks
        
        if electrodes is None:
            electrodes = [column for column in recordings.columns if column != 'Time (s)']
            
        for electrode in electrodes:
            filtered_data = np.column_stack([recordings['Time (s)'], recordings[electrode]]).T
            peaks = self.__find_peaks(filtered_data,
                                      max_twitch_frequency=self.max_twitch_frequency,
                                      prominence_factor=self.prominence_factor,
                                      width_factor=self.width_factor,
                                      height_factor=self.height_factor)

            electrode_peaks[electrode] = peaks
        
        return electrode_peaks

    def __find_peaks(self, 
                     recording:float,
                     max_twitch_frequency:int,
                     prominence_factor:float,
                     width_factor:float,
                     height_factor:float):
        """
        Identify peaks in a time-series (based on Pulse3D peak finder).

        Args:
        recording (NDArray, 2D): Elecrode ime-series data (x: time, y: voltage)
        max_twitch_frequency (float, >0): Twitches cannot occur more frequently than this value.
        prominence_factor (float): Scaling factor for peak prominence.  Larger values make the peak-finding more sensitive
        width_factor (float): Scaling factor for minimum peak width.  Larger values make the peak-finding more sensitive.
        height_factor (float, >=0): Scaling factor for minimum peak height.  Larger values make peak-finding less sensitive.

        Returns:
        peak_indices (list): indices of peaks in electrode recording

        Raises:
        ValueError: when scaling factor values are out of range

        Notes:
        ```max_twitch_frequency``` can likely be informed biologically, based on known neuron firing rates.
        """

        try:
            assert prominence_factor>0
            assert width_factor>0
            assert height_factor>=0
        except:
            raise(ValueError)

        X = recording[0,:]
        Y = recording[1,:]

        # get sampling frequency
        dt = X[1]-X[0]

        # set minimum number of required time-points between peaks
        min_samples_between_twitches = int(round((1/max_twitch_frequency)/dt,0))

        # set max possible prominence
        max_prominence = abs(np.max(Y) - np.min(Y))

        # set minimum required peak height
        min_height = Y.mean() + height_factor*Y.std()

        peak_indices, _ = signal.find_peaks(Y,
                                            height=min_height,
                                            width=min_samples_between_twitches/width_factor,
                                            distance=min_samples_between_twitches,
                                            prominence=max_prominence/prominence_factor)


        return np.asarray(peak_indices)[:,None]
    
    
def plot_peaklet(peaklets: dict,
                 recordings: pd.DataFrame, 
                 index: int = 0,
                 window:float = 15, 
                 cmap:dict = None):
    
    """
    Plot waveform around peaklet.
    
    Args:
    peaklets (dict): Previously identified peaklet indices, time, and voltage.
    recordings (pd.DataFrame, required): Electrode recordings.
    index (int, optional): Peaklet index to plot. Default=15.
    window (int, required): Number of time-samples before and after peaklet indices.
    cmap (dict, optional): Colormap mapping each electrode to a color.
    """
    
    peaklet = peaklets['indices'].loc[index]
    peaklet_time = peaklets['time'].loc[index]
    peaklet_voltage = peaklets['voltage'].loc[index]
    
    electrodes = peaklets['indices'].columns
    max_volt = peaklets['voltage'].max().max()*1.1
    min_volt = recordings[electrodes].min().min()
    
    # get upper and lower bound of peaklet window
    lower=peaklet[0] - window
    upper=peaklet[-1] + window

    voltage = recordings
    
    fig, ax = plt.subplots(1,1, figsize=(10, 7))
    for electrode in peaklet.index:
        if cmap:
            color=cmap[electrode]
        else:
            color=None
        ax.plot(recordings.loc[lower:upper]['Time (s)'], 
                recordings.loc[lower:upper][electrode], 
                linewidth=3, label=electrode, c=color)
        
        ax.scatter(peaklet_time[electrode],
                   peaklet_voltage[electrode],
                   s=100, 
                   c=color)

    ax.set_ylim([min_volt, max_volt])
    ax.ticklabel_format(useOffset=False)
    ax.set_xlabel('Time (seconds)', fontsize=20)
    ax.set_ylabel('Voltage', fontsize=20)
    ax.tick_params(labelsize=15)
    ax.set_title(f'Waveforms for Peaklet {index}', fontsize=20)
    
    plt.legend(loc=(1.01, 0), fontsize=15, title='Electrode', title_fontsize=15)
    return fig

def plot_peaklet_scatter(peaklets: dict,
                         recordings: pd.DataFrame, 
                         index: int = 0,
                         window:float = 15, 
                         cmap:dict = None):
    
    """
    Plot waveform around peaklet as individual points not as curve.
    
    Args:
    peaklets (dict): Previously identified peaklet indices, time, and voltage.
    recordings (pd.DataFrame, required): Electrode recordings.
    index (int, optional): Peaklet index to plot. Default=15.
    window (int, required): Number of time-samples before and after peaklet indices.
    cmap (dict, optional): Colormap mapping each electrode to a color.
    """
    
    peaklet = peaklets['indices'].loc[index]
    peaklet_time = peaklets['time'].loc[index]
    peaklet_voltage = peaklets['voltage'].loc[index]
    
    electrodes = peaklets['indices'].columns
    max_volt = peaklets['voltage'].max().max()*1.1
    min_volt = recordings[electrodes].min().min()
    
    # get upper and lower bound of peaklet window
    lower=peaklet[0] - window
    upper=peaklet[-1] + window

    voltage = recordings
    
    fig, ax = plt.subplots(1,1, figsize=(10, 7))
    for electrode in peaklet.index:
        if cmap:
            color=cmap[electrode]
        else:
            color=None
        ax.scatter(recordings.loc[lower:upper]['Time (s)'], 
                   recordings.loc[lower:upper][electrode], 
                   linewidth=3, label=electrode, c=color)
        
        ax.scatter(peaklet_time[electrode],
                   peaklet_voltage[electrode],
                   s=200,
                   marker="p",
                   c=color)

    ax.set_ylim([min_volt, max_volt])
    ax.ticklabel_format(useOffset=False)
    ax.set_xlabel('Time (seconds)', fontsize=20)
    ax.set_ylabel('Voltage', fontsize=20)
    ax.tick_params(labelsize=15)
    ax.set_title(f'Waveforms for Peaklet {index}', fontsize=20)
    
    plt.legend(loc=(1.01, 0), fontsize=15, title='Electrode', title_fontsize=15)
    return fig

def find_bursts(electrode_peaks: dict,
                max_burst_time: int = 200,
                min_burst_number: int = 5,
                delay_time: int = 8e-05):
    """
    Find peaks within an electrode recording which can be defined as a burst
    
    Args:
    electrode_peaks (dict): Peaks dictionary containing for a given electrode detected peaks (dict)
    max_burst_time (int): Time in (ms) within which two peaks must occur to be considered within a single burst
    min_burst_number (int): Number of consequtive peaks falling within the max_burst_time needed to consider the group a burst
    delay_time (int): Delay time between samples in the raw data, the inverse of the sample rate
    """
    
    #extract turple from dictionary
    burst_data = electrode_peaks
    
    # define max burst interval in terms of df index 
    max_burst_interval=round((max_burst_time/1000)/delay_time)
        
    bursts_found=0
    n=0
    peak_number=0
    
    #create empty lists for outputs
    burst_start_times=[]
    burst_end_times=[]
    burst_peak_number=[]
    burst_peak_indecies=[]
    
    # iterate through full list of peaks
    while n<len(burst_data) and n+peak_number+1<len(burst_data)-1:
        peak_number=0
        
        # if difference in index is smaller than defined distance check next pair
        if burst_data[n+peak_number] >= burst_data[n+peak_number+1]-max_burst_interval:
            while burst_data[n+peak_number] >= burst_data[n+peak_number+1]-max_burst_interval and n+peak_number+1<len(burst_data)-1:
                peak_number+=1
            
            # filter detected peak sequences by minumum peak number
            if peak_number>=min_burst_number:
                burst_start_times.extend(burst_data[n])
                burst_end_times.extend(burst_data[n+peak_number])
                burst_peak_number.append(peak_number)
                burst_peak_indecies.append(burst_data[n:n+peak_number])
                
                bursts_found=bursts_found+1
            
                # bursts should not overlap so after detecing a burst move to end of burst
                n=n+peak_number
                
            n+=1   
        
        # no peak within the minimum distance so move on to evaluate the next identified peak
        else:
            n+=1
        
    # compile the bursts dictionary    
    bursts = {'start times':burst_start_times, 
              'end times':burst_end_times,
              'number of peaks':burst_peak_number,
              'peak indicies':burst_peak_indecies}
    
    return bursts

def plot_well_plate(df: pd.DataFrame,
                    wells: list,
                    wells_shape: list,
                    electrode_shape: int,
                    show_electrode_titles: bool = False):
    """
    Plot all electrodes for all wells of a 6-well plate
    
    Args:
    df (pd.Dataframe): Dataframe created from CSV file containing data for all electrodes
    wells (list): list of wells expected within the plate
    wells_shape (list): The length of the short side of the well plate, followed by long side as a list
    electrode_shape (int): The number of electrodes in one side of the electrode array
    show_electrode_titles (bool): If True every electrode within the plot with be titled with the electrode designation 
    """
    
    # get all electrodes
    electrodes_total = [column for column in df.columns if column != 'Time (s)' and column.startswith(tuple(wells))]
    
    # create list to define order to allow plotting in correct positions
    electrode_order = [(row*electrode_shape)-column for row in range(electrode_shape, 0 , -1)\
                       for column in range(electrode_shape, 0, -1)]
    
    # figure size parameters
    width = wells_shape[0]*10
    height = wells_shape[1]*10
    
    # define figure size and main gridshape
    fig = plt.figure(figsize=(height, width))
    outer = gridspec.GridSpec(wells_shape[0], wells_shape[1], wspace=0.2, hspace=0.2)
    
    # find min and max values for electrode traces for y_lims
    y_upper = df[electrodes_total].max().max()
    y_lower = df[electrodes_total].min().min()
    
    # plot all wells and all electrodes
    for i, well in enumerate(wells):
        # define inner grid size and shape
        inner = gridspec.GridSpecFromSubplotSpec(electrode_shape, electrode_shape,
                        subplot_spec=outer[i], wspace=0.1, hspace=0.1)
    
        # select only electrodes in well of interest
        electrode_array = [electrode for electrode in electrodes_total if electrode.startswith(well)]
        
        # change list order to match electrode array layout
        electrode_array = [electrode_array[i] for i in electrode_order]
        
        # set title for each well plot
        ax = plt.Subplot(fig, outer[i])
        ax.set_title(f'Well {well}', fontsize = 15, pad=25)
        ax.axis('off')
        fig.add_subplot(ax)
            
        # create each subplot
        for j, electrode in enumerate(electrode_array):
            ax = plt.Subplot(fig, inner[j])   
            
            X = df['Time (s)']
            Y = df[electrode]
        
            # plot waveform
            ax.plot(X, Y)
               
            # hide all axis ticks
            ax.axis('off')
            
            # set y_axis limits so all plots share same y_axis
            ax.set_ylim(y_lower, y_upper)
            
            # show electrode titles
            if show_electrode_titles == True:
                ax.set_title(f'{electrode}', fontsize = 10)
            
            fig.add_subplot(ax)
            
    return fig