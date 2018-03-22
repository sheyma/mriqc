import matplotlib as mpl
mpl.use('Agg')
import sys, os
import numpy as np
import pylab as plt
import seaborn as sns
from misc import plot_vline
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import FigureCanvasPdf as FigureCanvas
from matplotlib.gridspec import GridSpec
import nibabel as nb
import matplotlib.pyplot as pyplt
import matplotlib.patches as mpatches
from motion import calc_frame_dispalcement



def calculate_DVARS(rest, mask):
    import numpy as np
    import nibabel as nib
    import os
    
    rest_data = nib.load(rest).get_data().astype(np.float32)
    mask_data = nib.load(mask).get_data().astype('bool')
    
    #square of relative intensity value for each voxel across
    #every timepoint
    data = np.square(np.diff(rest_data, axis = 3))
    #applying mask, getting the data in the brain only
    data = data[mask_data]
    #square root and mean across all timepoints inside mask
    DVARS = np.sqrt(np.mean(data, axis=0))
    
    return DVARS

def get_GS(image, image_mask):

    mask = nb.load(image_mask).get_data().astype('int')    
    data = nb.load(image).get_data().astype('float64')

    GS = np.mean(data[mask==1], axis=0)

    return GS


def timeseries(rest, grey):
    import numpy as np
    import nibabel as nib
    import os
    
    rest_data = nib.load(rest).get_data().astype(np.float32)
    gm_mask = nib.load(grey).get_data().astype('bool')
    
    #applying GM mask, getting GM voxel intensities
    rest_gm = rest_data[gm_mask]
    #applying WM mask, getting WM voxel intensities
    
    return rest_gm

def plot_power(fd_file,
               im_pre_ddf,
               im_post,
	           mask_func,
	           GM_msk_file,
               figsize=(7,8)):
    
    #sns.set_context("paper", rc={"font.size":8,
    #                         "axes.titlesize":8,
    #                         "axes.labelsize":8}) 
    
    
    FD_file = calc_frame_dispalcement(fd_file)
    FD      = np.loadtxt(FD_file)
    nvol    = len(FD)


    DVARSpre  = calculate_DVARS(im_pre_ddf, mask_func)
    DVARSpost = calculate_DVARS(im_post, mask_func) 

    GSpre     = get_GS(im_pre_ddf, mask_func)
    GSpost    = get_GS(im_post, mask_func)

    timeseries_pre   = timeseries(im_pre_ddf, GM_msk_file)
    timeseries_post  = timeseries(im_post, GM_msk_file)


    sns.set_style('white') 
    sns.set(font_scale=.8)
    
    fig = pyplt.figure(1)
    fig.suptitle('(Power et al., 2014)')
    fig.set_size_inches=figsize
    fig.subplots_adjust(hspace=.25)
    
    ax1 = pyplt.subplot2grid((5,3), (0,0), colspan=3)
    sns.tsplot(FD, range(1,nvol+1), color='b', ax=ax1, linewidth=1)

    ax1.set_xticks([])
    ax1.set_ylabel('FD (mm)')
    #ax1.set_title('Head motion')
    ax1.set_xlim([1,nvol])

    ax2 = pyplt.subplot2grid((5,3), (1,0), colspan=3)
    sns.tsplot(DVARSpre, range(2,nvol+1), color='k', ax=ax2, linewidth=1)
    sns.tsplot(DVARSpost, range(2,nvol+1), color='r', ax=ax2, linewidth=1)
    ax2.set_ylabel('DVARS (std)')
    ax2.yaxis.tick_left()
    ax2.set_xticks([])
    ax2.legend(['Pre', 'Post'], bbox_to_anchor=(1, 1), loc=2)

    ax3 = pyplt.subplot2grid((5,3), (2,0), colspan=3)
    sns.tsplot(GSpre, range(1,nvol+1), color='g', ax=ax3, linewidth=1)
    sns.tsplot(GSpost, range(1,nvol+1), color='y', ax=ax3, linewidth=1)
    ax3.set_ylabel('BOLD (GS)')
    ax3.yaxis.tick_left()
    ax3.set_xticks([])
    ax3.set_xlim([1,nvol])
    ax3.legend(['Pre', 'Post'], bbox_to_anchor=(1, 1), loc=2)


    ax4 = pyplt.subplot2grid((5,3), (3,0), colspan=3)
    ax4.figure.set_size_inches=(12,8)
    ax4.imshow(timeseries_pre, 
                interpolation='nearest', 
                aspect = 'auto', 
                #vmin=m-3*std, vmax=m+3*std,
                cmap='gray')
    ax4.set_yticks([])
    ax4.set_xticks([])
    ax4.grid(True)
    ax4.set_xlim([1,nvol])
    ax4.set_ylabel('t-series, pre')

    ax5 = pyplt.subplot2grid((5,3), (4,0), colspan=3)
    ax5.figure.set_size_inches=(12,8)
    ax5.imshow(timeseries_post, 
               interpolation='nearest', 
               aspect = 'auto', 
               #vmin=m-3*std, vmax=m+3*std,
               cmap='gray')
    ax5.set_yticks([])
    ax5.set_xticks([20, 40, 60, 80, 100, 120, 140])
    ax5.set_ylabel('t-series, post')
    ax5.set_xlabel('Gray Matter Volume # (TR=2.3s)')
    ax5.grid(False)
    ax5.set_xlim([1,nvol])

    
    return fig


## single run
#fd_file = '/data/pt_mar006/subjects/sd14_d05/preprocessed/func/realign/rest_roi.nii.gz.par'

#im_post = '/data/pt_mar006/subjects/sd14_d05/preprocessed/func/rest_preprocessed.nii.gz'

#mask_func = '/data/pt_mar006/subjects/sd14_d05/preprocessed/func/denoise/mask/brain_mask_func.nii.gz'

#GM_msk_file = '/data/pt_mar006/subjects/sd14_d05/preprocessed/func/connectivity/gm_func.nii.gz'

#im_pre_ddf = '/data/pt_mar006/subjects/sd14_d05/preprocessed/func/realign/corr_rest_roi_ddf.nii.gz'


#fig = plot_power(fd_file,
#               im_pre_ddf,
#               im_post,
#	       mask_func,
#	       GM_msk_file,
#               figsize=(7,8))
#
#fig.savefig('BBB.pdf')
