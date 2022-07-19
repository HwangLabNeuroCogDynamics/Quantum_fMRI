
#smooth not sure we will do this for decoding...?
# for run in 1 2 3 4 5 6; do
#     3dBlurToFWHM -FWHM 5 -automask -prefix /data/backed_up/shared/Quantum_data/MRI_data/fmriprep/fmriprep/sub-JHtest/func/sub-JHtest_task-Quantum_run-${run}_space-MNI152NLin2009cAsym_desc-preproc_bold_s.nii.gz \
#     -input /data/backed_up/shared/Quantum_data/MRI_data/fmriprep/fmriprep/sub-JHtest/func/sub-JHtest_task-Quantum_run-${run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz
# done

# models: 'TWOGAM(8.6,0.547,0.2,8.6,0.547)' or 'TENT(0, 12.222, 7)' ?

3dDeconvolve -input $(/bin/ls /data/backed_up/shared/Quantum_data/MRI_data/fmriprep/fmriprep/sub-JHtest/func/sub-JHtest_task-Quantum_run-*_space-MNI152NLin2009cAsym_desc-preproc_bold_s.nii.gz | sort -V) \
-mask "/data/backed_up/shared/Quantum_data/MRI_data/fmriprep/fmriprep/sub-JHtest/func/sub-JHtest_task-Quantum_run-6_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz" \
-polort A \
-num_stimts 2 \
-CENSORTR 1:0..2 \
-CENSORTR 2:0..2 \
-CENSORTR 3:0..2 \
-CENSORTR 4:0..2 \
-CENSORTR 5:0..2 \
-CENSORTR 6:0..2 \
-ortvec /data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/noise.1D noise \
-stim_times 1 /data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/cue_stimtime.1D 'TWOGAM(8.6,0.547,0.2,8.6,0.547)' -stim_label 1 cue \
-stim_times 2 /data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/probe_stimtime.1D 'TWOGAM(8.6,0.547,0.2,8.6,0.547)' -stim_label 2 probe \
-iresp 1 /data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/cue_FIR.nii.gz \
-iresp 2 /data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/probe_FIR.nii.gz \
-gltsym 'SYM: +1*cue' -glt_label 1 cue \
-gltsym 'SYM: +1*probe' -glt_label 2 probe \
-fout \
-rout \
-tout \
-bucket /data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/cue_probe_FIRmodel_stats.nii.gz \
-GOFORIT 100 \
-noFDR \
-nocout \
-jobs 40



#### amplitude modulation models see: https://afni.nimh.nih.gov/pub/dist/doc/misc/Decon/AMregression.pdf
1dMarry /data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/cue_stimtime.1D /data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/cue_state_am.1D > /data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/cue_state_stimtime.1D
1dMarry /data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/probe_stimtime.1D /data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/probe_state_am.1D > /data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/probe_state_stimtime.1D
1dMarry /data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/cue_stimtime.1D /data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/cue_task_am.1D > /data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/cue_task_stimtime.1D
1dMarry /data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/probe_stimtime.1D /data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/probe_task_am.1D > /data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/probe_task_stimtime.1D
1dMarry /data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/cue_stimtime.1D /data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/cue_entropy_am.1D > /data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/cue_entropy_stimtime.1D
1dMarry /data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/probe_stimtime.1D /data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/probe_entropy_am.1D > /data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/probe_entropy_stimtime.1D
1dMarry /data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/cue_stimtime.1D /data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/cue_entropy_change_am.1D > /data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/cue_entropy_change_stimtime.1D
1dMarry /data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/probe_stimtime.1D /data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/probe_entropy_change_am.1D > /data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/probe_entropy_change_stimtime.1D


#state
3dDeconvolve -input $(/bin/ls /data/backed_up/shared/Quantum_data/MRI_data/fmriprep/fmriprep/sub-JHtest/func/sub-JHtest_task-Quantum_run-*_space-MNI152NLin2009cAsym_desc-preproc_bold_s.nii.gz | sort -V) \
-mask /data/backed_up/shared/Quantum_data/MRI_data/fmriprep/fmriprep/sub-JHtest/func/sub-JHtest_task-Quantum_run-6_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz \
-polort A \
-num_stimts 2 \
-CENSORTR 1:0..2 \
-CENSORTR 2:0..2 \
-CENSORTR 3:0..2 \
-CENSORTR 4:0..2 \
-CENSORTR 5:0..2 \
-CENSORTR 6:0..2 \
-ortvec /data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/noise.1D noise \
-stim_times_AM2 1 /data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/cue_state_stimtime.1D 'TWOGAM(8.6,0.547,0.2,8.6,0.547)' -stim_label 1 cue_state \
-stim_times_AM2 2 /data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest//probe_state_stimtime.1D 'TWOGAM(8.6,0.547,0.2,8.6,0.547)' -stim_label 2 probe_state \
-iresp 1 /data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/cue_FIR_2gamma.nii.gz \
-iresp 2 /data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/probe_FIR_2gamma.nii.gz \
-gltsym 'SYM: +1*cue_state[0]' -glt_label 1 cue \
-gltsym 'SYM: +1*probe_state[0]' -glt_label 2 probe \
-gltsym 'SYM: +1*cue_state[1]' -glt_label 3 cue_state \
-gltsym 'SYM: +1*probe_state[1]' -glt_label 4 probe_state \
-fout \
-rout \
-tout \
-bucket /data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/cue_probe_state_FIRmodel_stats.nii.gz \
-GOFORIT 100 \
-noFDR \
-jobs 40

3dREMLfit \
-matrix /data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/cue_probe_state_FIRmodel_stats.xmat.1D \
-input "/data/backed_up/shared/Quantum_data/MRI_data/fmriprep/fmriprep/sub-JHtest/func/sub-JHtest_task-Quantum_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold_s.nii.gz 
/data/backed_up/shared/Quantum_data/MRI_data/fmriprep/fmriprep/sub-JHtest/func/sub-JHtest_task-Quantum_run-2_space-MNI152NLin2009cAsym_desc-preproc_bold_s.nii.gz 
/data/backed_up/shared/Quantum_data/MRI_data/fmriprep/fmriprep/sub-JHtest/func/sub-JHtest_task-Quantum_run-3_space-MNI152NLin2009cAsym_desc-preproc_bold_s.nii.gz 
/data/backed_up/shared/Quantum_data/MRI_data/fmriprep/fmriprep/sub-JHtest/func/sub-JHtest_task-Quantum_run-4_space-MNI152NLin2009cAsym_desc-preproc_bold_s.nii.gz 
/data/backed_up/shared/Quantum_data/MRI_data/fmriprep/fmriprep/sub-JHtest/func/sub-JHtest_task-Quantum_run-5_space-MNI152NLin2009cAsym_desc-preproc_bold_s.nii.gz 
/data/backed_up/shared/Quantum_data/MRI_data/fmriprep/fmriprep/sub-JHtest/func/sub-JHtest_task-Quantum_run-6_space-MNI152NLin2009cAsym_desc-preproc_bold_s.nii.gz" \
-mask /data/backed_up/shared/Quantum_data/MRI_data/fmriprep/fmriprep/sub-JHtest/func/sub-JHtest_task-Quantum_run-6_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz \
-fout -tout -rout -noFDR -Rbuck /data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/cue_probe_state_FIRmodel_stats_REML.nii.gz

# task
3dDeconvolve -input $(/bin/ls /data/backed_up/shared/Quantum_data/MRI_data/fmriprep/fmriprep/sub-JHtest/func/sub-JHtest_task-Quantum_run-*_space-MNI152NLin2009cAsym_desc-preproc_bold_s.nii.gz | sort -V) \
-mask "/data/backed_up/shared/Quantum_data/MRI_data/fmriprep/fmriprep/sub-JHtest/func/sub-JHtest_task-Quantum_run-6_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz" \
-polort A \
-num_stimts 2 \
-CENSORTR 1:0..2 \
-CENSORTR 2:0..2 \
-CENSORTR 3:0..2 \
-CENSORTR 4:0..2 \
-CENSORTR 5:0..2 \
-CENSORTR 6:0..2 \
-ortvec /data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/noise.1D noise \
-stim_times_AM2 1 /data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/cue_task_stimtime.1D 'TWOGAM(8.6,0.547,0.2,8.6,0.547)' -stim_label 1 cue_task \
-stim_times_AM2 2 /data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest//probe_task_stimtime.1D 'TWOGAM(8.6,0.547,0.2,8.6,0.547)' -stim_label 2 probe_task \
-gltsym 'SYM: +1*cue_task[0]' -glt_label 1 cue \
-gltsym 'SYM: +1*probe_task[0]' -glt_label 2 probe \
-gltsym 'SYM: +1*cue_task[1]' -glt_label 3 cue_task \
-gltsym 'SYM: +1*probe_task[1]' -glt_label 4 probe_task \
-fout \
-rout \
-tout \
-bucket /data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/cue_probe_task_FIRmodel_stats.nii.gz \
-GOFORIT 100 \
-noFDR \
-jobs 40

3dREMLfit \
-matrix /data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/cue_probe_task_FIRmodel_stats.xmat.1D \
-input "/data/backed_up/shared/Quantum_data/MRI_data/fmriprep/fmriprep/sub-JHtest/func/sub-JHtest_task-Quantum_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold_s.nii.gz 
/data/backed_up/shared/Quantum_data/MRI_data/fmriprep/fmriprep/sub-JHtest/func/sub-JHtest_task-Quantum_run-2_space-MNI152NLin2009cAsym_desc-preproc_bold_s.nii.gz 
/data/backed_up/shared/Quantum_data/MRI_data/fmriprep/fmriprep/sub-JHtest/func/sub-JHtest_task-Quantum_run-3_space-MNI152NLin2009cAsym_desc-preproc_bold_s.nii.gz 
/data/backed_up/shared/Quantum_data/MRI_data/fmriprep/fmriprep/sub-JHtest/func/sub-JHtest_task-Quantum_run-4_space-MNI152NLin2009cAsym_desc-preproc_bold_s.nii.gz 
/data/backed_up/shared/Quantum_data/MRI_data/fmriprep/fmriprep/sub-JHtest/func/sub-JHtest_task-Quantum_run-5_space-MNI152NLin2009cAsym_desc-preproc_bold_s.nii.gz 
/data/backed_up/shared/Quantum_data/MRI_data/fmriprep/fmriprep/sub-JHtest/func/sub-JHtest_task-Quantum_run-6_space-MNI152NLin2009cAsym_desc-preproc_bold_s.nii.gz" \
-mask /data/backed_up/shared/Quantum_data/MRI_data/fmriprep/fmriprep/sub-JHtest/func/sub-JHtest_task-Quantum_run-6_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz \
-fout -tout -rout -noFDR -Rbuck /data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/cue_probe_task_FIRmodel_stats_REML.nii.gz


### entropy change and entropy
3dDeconvolve -input $(/bin/ls /data/backed_up/shared/Quantum_data/MRI_data/fmriprep/fmriprep/sub-JHtest/func/sub-JHtest_task-Quantum_run-*_space-MNI152NLin2009cAsym_desc-preproc_bold_s.nii.gz | sort -V) \
-mask "/data/backed_up/shared/Quantum_data/MRI_data/fmriprep/fmriprep/sub-JHtest/func/sub-JHtest_task-Quantum_run-6_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz" \
-polort A \
-num_stimts 2 \
-CENSORTR 1:0..2 \
-CENSORTR 2:0..2 \
-CENSORTR 3:0..2 \
-CENSORTR 4:0..2 \
-CENSORTR 5:0..2 \
-CENSORTR 6:0..2 \
-ortvec /data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/noise.1D noise \
-stim_times_AM2 1 /data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/cue_entropy_stimtime.1D 'TWOGAM(8.6,0.547,0.2,8.6,0.547)' -stim_label 1 cue_entropy \
-stim_times_AM2 2 /data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest//probe_entropy_stimtime.1D 'TWOGAM(8.6,0.547,0.2,8.6,0.547)' -stim_label 2 probe_entropy \
-gltsym 'SYM: +1*cue_entropy[0]' -glt_label 1 cue \
-gltsym 'SYM: +1*probe_entropy[0]' -glt_label 2 probe \
-gltsym 'SYM: +1*cue_entropy[1]' -glt_label 3 cue_entropy \
-gltsym 'SYM: +1*probe_entropy[1]' -glt_label 4 probe_entropy \
-fout \
-rout \
-tout \
-bucket /data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/cue_probe_entropy_FIRmodel_stats.nii.gz \
-GOFORIT 100 \
-noFDR \
-jobs 40

3dREMLfit \
-matrix /data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/cue_probe_entropy_FIRmodel_stats.xmat.1D \
-input "/data/backed_up/shared/Quantum_data/MRI_data/fmriprep/fmriprep/sub-JHtest/func/sub-JHtest_task-Quantum_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold_s.nii.gz 
/data/backed_up/shared/Quantum_data/MRI_data/fmriprep/fmriprep/sub-JHtest/func/sub-JHtest_task-Quantum_run-2_space-MNI152NLin2009cAsym_desc-preproc_bold_s.nii.gz 
/data/backed_up/shared/Quantum_data/MRI_data/fmriprep/fmriprep/sub-JHtest/func/sub-JHtest_task-Quantum_run-3_space-MNI152NLin2009cAsym_desc-preproc_bold_s.nii.gz 
/data/backed_up/shared/Quantum_data/MRI_data/fmriprep/fmriprep/sub-JHtest/func/sub-JHtest_task-Quantum_run-4_space-MNI152NLin2009cAsym_desc-preproc_bold_s.nii.gz 
/data/backed_up/shared/Quantum_data/MRI_data/fmriprep/fmriprep/sub-JHtest/func/sub-JHtest_task-Quantum_run-5_space-MNI152NLin2009cAsym_desc-preproc_bold_s.nii.gz 
/data/backed_up/shared/Quantum_data/MRI_data/fmriprep/fmriprep/sub-JHtest/func/sub-JHtest_task-Quantum_run-6_space-MNI152NLin2009cAsym_desc-preproc_bold_s.nii.gz" \
-mask /data/backed_up/shared/Quantum_data/MRI_data/fmriprep/fmriprep/sub-JHtest/func/sub-JHtest_task-Quantum_run-6_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz \
-fout -tout -rout -noFDR -Rbuck /data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/cue_probe_entropy_FIRmodel_stats_REML.nii.gz

3dDeconvolve -input $(/bin/ls /data/backed_up/shared/Quantum_data/MRI_data/fmriprep/fmriprep/sub-JHtest/func/sub-JHtest_task-Quantum_run-*_space-MNI152NLin2009cAsym_desc-preproc_bold_s.nii.gz | sort -V) \
-mask "/data/backed_up/shared/Quantum_data/MRI_data/fmriprep/fmriprep/sub-JHtest/func/sub-JHtest_task-Quantum_run-6_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz" \
-polort A \
-num_stimts 2 \
-CENSORTR 1:0..2 \
-CENSORTR 2:0..2 \
-CENSORTR 3:0..2 \
-CENSORTR 4:0..2 \
-CENSORTR 5:0..2 \
-CENSORTR 6:0..2 \
-ortvec /data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/noise.1D noise \
-stim_times_AM2 1 /data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/cue_entropy_change_stimtime.1D 'TWOGAM(8.6,0.547,0.2,8.6,0.547)' -stim_label 1 cue_entropy_change \
-stim_times_AM2 2 /data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest//probe_entropy_change_stimtime.1D 'TWOGAM(8.6,0.547,0.2,8.6,0.547)' -stim_label 2 probe_entropy_change \
-gltsym 'SYM: +1*cue_entropy_change[0]' -glt_label 1 cue \
-gltsym 'SYM: +1*probe_entropy_change[0]' -glt_label 2 probe \
-gltsym 'SYM: +1*cue_entropy_change[1]' -glt_label 3 cue_entropy_change \
-gltsym 'SYM: +1*probe_entropy_change[1]' -glt_label 4 probe_entropy_change \
-rout \
-tout \
-bucket /data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/cue_probe_entropy_change_FIRmodel_stats.nii.gz \
-GOFORIT 100 \
-noFDR \
-jobs 40

3dREMLfit \
-matrix /data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/cue_probe_entropy_change_FIRmodel_stats.xmat.1D \
-input "/data/backed_up/shared/Quantum_data/MRI_data/fmriprep/fmriprep/sub-JHtest/func/sub-JHtest_task-Quantum_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold_s.nii.gz 
/data/backed_up/shared/Quantum_data/MRI_data/fmriprep/fmriprep/sub-JHtest/func/sub-JHtest_task-Quantum_run-2_space-MNI152NLin2009cAsym_desc-preproc_bold_s.nii.gz 
/data/backed_up/shared/Quantum_data/MRI_data/fmriprep/fmriprep/sub-JHtest/func/sub-JHtest_task-Quantum_run-3_space-MNI152NLin2009cAsym_desc-preproc_bold_s.nii.gz 
/data/backed_up/shared/Quantum_data/MRI_data/fmriprep/fmriprep/sub-JHtest/func/sub-JHtest_task-Quantum_run-4_space-MNI152NLin2009cAsym_desc-preproc_bold_s.nii.gz 
/data/backed_up/shared/Quantum_data/MRI_data/fmriprep/fmriprep/sub-JHtest/func/sub-JHtest_task-Quantum_run-5_space-MNI152NLin2009cAsym_desc-preproc_bold_s.nii.gz 
/data/backed_up/shared/Quantum_data/MRI_data/fmriprep/fmriprep/sub-JHtest/func/sub-JHtest_task-Quantum_run-6_space-MNI152NLin2009cAsym_desc-preproc_bold_s.nii.gz" \
-mask /data/backed_up/shared/Quantum_data/MRI_data/fmriprep/fmriprep/sub-JHtest/func/sub-JHtest_task-Quantum_run-6_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz \
-fout -tout -rout -noFDR -Rbuck /data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/cue_probe_entropy_change_FIRmodel_stats_REML.nii.gz

## we can check the design matrix by doing ExamineXmat -input cue_probe_entropy_change_FIRmodel_stats.xmat.1D


