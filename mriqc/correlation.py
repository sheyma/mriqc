import os, sys
sys.path.append(os.path.expanduser('/home/raid/bayrak/devel/mriqc/mriqc'))
import nibabel as nb
import numpy as np
import seaborn as sns
from pylab import cm
from nipype.interfaces.freesurfer.preprocess import ApplyVolTransform
from nipy.labs import viz
from misc import plot_vline
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import FigureCanvasPdf as FigureCanvas
from matplotlib.gridspec import GridSpec
import pylab as plt

def get_similarity_distribution(mincost_files):
    similarities = []
    for mincost_file in mincost_files:
        similarity = float(open(mincost_file, 'r').readlines()[0].split()[0])
        similarities.append(similarity)
    return similarities
    
    
def plot_epi_T1_corregistration(mean_epi_to_anat, wm_file, fssubjects_dir, 
				subject_id, similarity_distribution=None, 
				figsize=(11.7,8.3),):
       
    fig = plt.figure(figsize=figsize)
    
    if similarity_distribution:
        ax = plt.subplot(2,1,1)
        sns.distplot(similarity_distribution.values(), ax=ax)
        ax.set_xlabel("EPI-T1 similarity after coregistration (over all subjects)")
        cur_similarity = similarity_distribution[subject_id]
        label = "similarity = %g"%cur_similarity
        plot_vline(cur_similarity, label, ax=ax)
        
        ax = plt.subplot(2,1,1)
    else:
        ax = plt.subplot(1,1,1)
	
    func = nb.load(mean_epi_to_anat).get_data()
    func_affine = nb.load(mean_epi_to_anat).get_affine()
    
    #ribbon_file = "%s/%s/mri/ribbon.mgz"%(fssubjects_dir, subject_id)
    #ribbon_nii = nb.load(ribbon_file)
    #ribbon_data = ribbon_nii.get_data()
    #ribbon_data[ribbon_data > 1] = 1
    #ribbon_affine = ribbon_nii.get_affine()
    
    wm_nii = nb.load(wm_file)
    wm_data = wm_nii.get_data()
    wm_data[wm_data > 1] = 1
    wm_affine = wm_nii.get_affine()
    
    slicer = viz.plot_anat(np.asarray(func), np.asarray(func_affine), black_bg=True,
                           cmap = cm.Greys_r,  # @UndefinedVariable
                           figure = fig,
                           axes = ax,
                           draw_cross = False)
    slicer.contour_map(np.asarray(wm_data), np.asarray(wm_affine), 
    		       linewidths=[0.1], colors=['r',])
    
    fig.suptitle('coregistration', fontsize='14')

    return fig

## simple run
#subject_id = 'hc02_d00'

#infiles = ['/nobackup/ilz2/bayrak/subjects/hc01_d00/preprocessed/func/coregister/transforms2anat/rest2anat.dat.mincost',
#           '/nobackup/ilz2/bayrak/subjects/hc02_d00/preprocessed/func/coregister/transforms2anat/rest2anat.dat.mincost']

#similarities = get_similarity_distribution(infiles)
#print similarities

#mean_epi_to_anat = "/nobackup/ilz2/bayrak/subjects/%s/preprocessed/func/coregister/rest2anat_highRes.nii.gz"%(subject_id)
#wm_file = '/nobackup/ilz2/bayrak/subjects/%s/preprocessed/anat/brain_wmedge.nii.gz'%(subject_id)
#fssubjects_dir = '/nobackup/ilz2/bayrak/freesurfer'


#Figure = plot_epi_T1_corregistration(mean_epi_to_anat,  wm_file, fssubjects_dir, subject_id, 
#					similarity_distribution=None, figsize=(11.7,8.3))
plt.show()
#Figure.savefig('A_%s_shortway2.pdf'%(subject_id), format='pdf')
