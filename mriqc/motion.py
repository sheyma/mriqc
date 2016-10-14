import numpy as np
import pylab as plt
import seaborn as sns
import sys, os
sys.path.append(os.path.expanduser('/home/raid/bayrak/devel/mriqc/mriqc'))
from misc import plot_vline
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import FigureCanvasPdf as FigureCanvas
from matplotlib.gridspec import GridSpec

# QUALITY CONTROL FOR MOTION CORRECTION 

# framewise displacement (FD) for a subject's motion parameters
def calc_frame_dispalcement(realignment_parameters_file):
    lines = open(realignment_parameters_file, 'r').readlines()
    rows = [[float(x) for x in line.split()] for line in lines]
    cols = np.array([list(col) for col in zip(*rows)])

    translations = np.transpose(np.abs(np.diff(cols[0:3, :])))
    rotations = np.transpose(np.abs(np.diff(cols[3:6, :])))

    FD_power = np.sum(translations, axis = 1) + (50*3.141/180)*np.sum(rotations, axis =1)

    #FD is zero for the first time point
    FD_power = np.insert(FD_power, 0, 0)
    
    return FD_power

# get mean and max FD for all subjects iteratively 
def get_mean_frame_displacement_disttribution(realignment_parameters_files):
    mean_FDs = []
    max_FDs = []
    for realignment_parameters_file in realignment_parameters_files:
        FD_power = calc_frame_dispalcement(realignment_parameters_file)
        mean_FDs.append(FD_power.mean())
        max_FDs.append(FD_power.max())
        
    return mean_FDs, max_FDs

# plotting a subject's FD distribution and optionally mean_FD line
def plot_frame_displacement(realignment_parameters_file, mean_FD_distribution=None, figsize=(11.7,8.3)):

    FD_power = calc_frame_dispalcement(realignment_parameters_file)

    fig = Figure(figsize=figsize)
    FigureCanvas(fig)
    
    if mean_FD_distribution:
        grid = GridSpec(2, 4)
    else:
        grid = GridSpec(1, 4)
    
    ax = fig.add_subplot(grid[0,:-1])
    ax.plot(FD_power)
    ax.set_xlim((0, len(FD_power)))
    ax.set_ylabel("Frame Displacement [mm]")
    ax.set_xlabel("Frame number")
    ylim = ax.get_ylim()
    
    ax = fig.add_subplot(grid[0,-1])
    sns.distplot(FD_power, vertical=True, ax=ax)
    ax.set_ylim(ylim)
    
    if mean_FD_distribution:
        ax = fig.add_subplot(grid[1,:])
        sns.distplot(mean_FD_distribution, ax=ax)
        ax.set_xlabel("Mean Frame Dispalcement (over all subjects) [mm]")
        MeanFD = FD_power.mean()
        label = "MeanFD = %g"%MeanFD
        plot_vline(MeanFD, label, ax=ax)
        
    return fig

infiles = ['/nobackup/ilz2/bayrak/subjects/hc01_d00/preprocessed/func/realign/rest_roi.nii.gz.par',
	   '/nobackup/ilz2/bayrak/subjects/hc01_d00/preprocessed/func/realign/rest_roi.nii.gz.par' ]

infile = infiles[0]

A = calc_frame_dispalcement(infile) 

mean_FD_dist, max_FD_dist = get_mean_frame_displacement_disttribution(infiles)

Figure = plot_frame_displacement(infile, mean_FD_distribution = mean_FD_dist, 
	                         figsize=(11.7,8.3))

Figure.savefig('C.pdf', format='pdf')




