def create_report(subject_id, tsnr_file, realignment_parameters_file, 
		  mean_epi_file, wm_file, mask_file, reg_file, 
		  fssubjects_dir, similarity_distribution, 
 		  mean_FD_distribution, tsnr_distributions, output_file):

    import gc
    import pylab as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from volumes import plot_mosaic, plot_distrbution_of_values
    from correlation import plot_epi_T1_corregistration
    from motion import plot_frame_displacement
    
    report = PdfPages(output_file)
    
    fig = plot_mosaic(mean_epi_file, title="Mean EPI", figsize=(8.3, 11.7))
    report.savefig(fig, dpi=300)
    fig.clf()
    
    fig = plot_mosaic(mean_epi_file, "Brain mask", mask_file, figsize=(8.3, 11.7))
    report.savefig(fig, dpi=600)
    fig.clf()
    
    fig = plot_mosaic(tsnr_file, title="tSNR", figsize=(8.3, 11.7))
    report.savefig(fig, dpi=300)
    fig.clf()
    
    fig = plot_distrbution_of_values(tsnr_file, mask_file, 
        "Subject %s tSNR inside the mask" % subject_id, 
        tsnr_distributions, 
        "Median tSNR (over all subjects)", 
        figsize=(8.3, 8.3))
    report.savefig(fig, dpi=300)
    fig.clf()
    plt.close()
    
    fig = plot_frame_displacement(realignment_parameters_file, mean_FD_distribution, 
				  figsize=(8.3, 8.3))
    report.savefig(fig, dpi=300)
    fig.clf()
    plt.close()
        
    fig = plot_epi_T1_corregistration(mean_epi_file, wm_file, 
                                      reg_file, fssubjects_dir, subject_id, 
                                      similarity_distribution, figsize=(8.3, 8.3))
    report.savefig(fig, dpi=300)
    fig.clf()
    plt.close()
    
    report.close()
    gc.collect()
    plt.close()
    
    return output_file, subject_id



def read_dists(csv_file):
    
    import pandas as pd
    import numpy as np
    df = pd.read_csv(csv_file, dtype=object)
    sim = dict(zip(df['subject_id'], list(np.asarray(df['coregistration quality'], dtype='float64'))))
    mfd = list(np.asarray(df['Mean FD'], dtype='float64'))
    tsnr = list(np.asarray(df['Median tSNR'], dtype='float64'))
    
    return sim, mfd, tsnr


def check(subject_id, scan_id, checklist):
    
    with open(checklist%(scan_id), 'a') as f:
        f.write(subject_id+'\n')
    return checklist

## single run
#data_dir   = "/nobackup/ilz2/bayrak/subjects/"

#subject_id                  = 'hc01_d00'
#tsnr_file  		    = data_dir + "%s/preprocessed/func/realign/corr_rest_roi_tsnr.nii.gz"%(subject_id) 
#realignment_parameters_file = data_dir + "%s/preprocessed/func/realign/rest_roi.nii.gz.par"%(subject_id)
#mean_epi_file 		    = data_dir + "%s/preprocessed/func/realign/mean_corr_rest_roi.nii.gz"%(subject_id)
#wm_file 		    = data_dir + "%s/preprocessed/func/denoise/mask/aparc_aseg.nii.gz"%(subject_id)
#mask_file		    = data_dir + "%s/preprocessed/func/denoise/mask/brain_mask_func.nii.gz"%(subject_id)
#reg_file 		    = data_dir + "%s/preprocessed/func/coregister/transforms2anat/rest2anat.dat"%(subject_id)
#fssubjects_dir 		    = "/nobackup/ilz2/bayrak/freesurfer/"
#similarity_distribution     = {'hc01_d00': 0.45303, 'hc02_d00' : 0.4000}
#mean_FD_distribution        = [0.07, 0.08]
#tsnr_distributions	    = [65, 79]
#output_file                 = "bla.pdf"

#create_report(subject_id, tsnr_file, realignment_parameters_file, 
#              mean_epi_file, wm_file, mask_file, reg_file, 
#	      fssubjects_dir, similarity_distribution, 
#              mean_FD_distribution, tsnr_distributions, output_file)


