from glob import glob
from correlation import get_similarity_distribution
from motion import get_mean_frame_displacement_disttribution
from volumes import get_median_distribution
from reports import create_report
import pandas as pd
from nipype.pipeline.engine import Workflow, Node
from nipype.interfaces.utility import Function


if __name__ == '__main__':
    data_dir = "/nobackup/ilz2/bayrak/subjects/"
    out_dir= "/nobackup/ilz2/bayrak/results/reports/hc/"
    freesurfer_dir="/nobackup/ilz2/bayrak/freesurfer/"
    
    wf = Workflow("reports")
    wf.base_dir = '/nobackup/ilz2/bayrak/results_workdir'
    with open('/nobackup/ilz2/bayrak/documents/all_hc.txt', 'r') as f:
        subjects = [line.strip() for line in f]
    
    subjects.sort()

    #generating distributions
    mincost_files = [data_dir + "%s/preprocessed/func/coregister/transforms2anat/rest2anat.dat.mincost"%(subject) for subject in subjects]
    similarity_distribution = get_similarity_distribution(mincost_files)
    
      
    realignment_parameters_files = [data_dir + "%s/preprocessed/func/realign/rest_roi.nii.gz.par"%(subject) for subject in subjects]
    mean_FD_distribution, max_FD_distribution = get_mean_frame_displacement_disttribution(realignment_parameters_files)
    
    tsnr_files = [data_dir + "%s/preprocessed/func/realign/corr_rest_roi_tsnr.nii.gz"%(subject) for subject in subjects]
    mask_files = [data_dir + "%s/preprocessed/func/denoise/mask/brain_mask_func.nii.gz"%(subject) for subject in subjects]
    tsnr_distributions = get_median_distribution(tsnr_files, mask_files)
    
    df = pd.DataFrame(zip(subjects, similarity_distribution, mean_FD_distribution, 
			  max_FD_distribution, tsnr_distributions), columns = ["subject_id", 
                          "coregistration quality", "Mean FD", "Max FD", "Median tSNR"])
    df.to_csv(out_dir+"rest_summary.csv")
    
    similarity_distribution = dict(zip(subjects, similarity_distribution))

    for subject_id in subjects:
	print subject_id
        # WTF, sort that out!

       

        #setting paths for this subject
        tsnr_file                   = data_dir + "%s/preprocessed/func/realign/corr_rest_roi_tsnr.nii.gz"%(subject_id) 
        realignment_parameters_file = data_dir + "%s/preprocessed/func/realign/rest_roi.nii.gz.par"%(subject_id)
        mean_epi_file               = data_dir + "%s/preprocessed/func/realign/mean_corr_rest_roi.nii.gz"%(subject_id)
        wm_file 		    = data_dir + "%s/preprocessed/anat/brain_wmedge.nii.gz"%(subject_id)
        mask_file                   = data_dir + "%s/preprocessed/func/denoise/mask/brain_mask_func.nii.gz"%(subject_id)
        mean_epi_to_anat            = data_dir + "%s/preprocessed/func/coregister/rest2anat_highRes.nii.gz"%(subject_id) 
        fssubjects_dir              = "/nobackup/ilz2/bayrak/freesurfer"
        output_file                 = out_dir + "%s_rest_report.pdf"%(subject_id)


	report = Node(Function(input_names=['subject_id', 
		                             'tsnr_file', 
		                             'realignment_parameters_file', 
		                             'mean_epi_file', 
					     'wm_file',
		                             'mask_file', 
		                             'mean_epi_to_anat', 
		                             'fssubjects_dir', 
		                             'similarity_distribution', 
		                             'mean_FD_distribution', 
		                             'tsnr_distributions', 
		                             'output_file'], 
	                        output_names=['out'],
	                        function = create_report), 
				name="report_%s"%(subject_id).replace(".", "_"))

        report.inputs.subject_id                  = subject_id
        report.inputs.tsnr_file                   = tsnr_file
        report.inputs.realignment_parameters_file = realignment_parameters_file
        report.inputs.mean_epi_file               = mean_epi_file
	report.inputs.wm_file 			  = wm_file
        report.inputs.mask_file                   = mask_file
        report.inputs.mean_epi_to_anat            = mean_epi_to_anat
        report.inputs.fssubjects_dir 		  = fssubjects_dir
        report.inputs.similarity_distribution     = similarity_distribution
        report.inputs.mean_FD_distribution        = mean_FD_distribution
        report.inputs.tsnr_distributions          = tsnr_distributions
        report.inputs.output_file                 = output_file
        report.plugin_args			  ={'override_specs': 'request_memory = 4000'}
	wf.add_nodes([report])
wf.run()	
#wf.run(plugin='MultiProc', plugin_args={'n_procs' : 30})
