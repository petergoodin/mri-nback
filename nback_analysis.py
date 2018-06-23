# -*- coding: utf-8 -*-
"""
SPM analysis script
To do
-----
Fix thresholding. 
"""
from __future__ import division

from nipype.interfaces import spm, afni, ants, dcmstack
from nipype.interfaces.base import Bunch
from nipype.algorithms import modelgen, rapidart, confounds
import nipype.interfaces.matlab as mlab
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.pipeline.engine import Workflow, Node, MapNode
import nibabel as nb
import numpy as np
import multiprocessing as mp
import glob
import os

mlab.MatlabCommand.set_default_matlab_cmd('/home/orcasha/MATLAB/R2015b/bin/matlab -nodesktop -nosplash')
mlab.MatlabCommand.set_default_paths('/home/orcasha/matlab/spm12')

data_dir = '/media/orcasha/Analysis_Drive/mri_data/subj/'
template_dir = '/media/orcasha/Analysis_Drive/mri_data/template/'


h_dir = '/mnt/data/nback/'
work_dir = os.path.join(h_dir, 'work')
out_dir = os.path.join(h_dir, 'output')
crash_dir = out_dir
n_cores = mp.cpu_count()

try:
    os.mkdir(h_dir)
except:
    print('Directory {} exists.'.format(h_dir))


#Set up study info
subject_list=[os.path.basename(x) for x in glob.glob(data_dir + '/*')] #drops path from folder names.

def metaread(nifti):
    """
    Combines metadata read from the header, populates the SPM slice timing
    correction inputs and outputs the time corrected epi image.
    Uses dcmstack.lookup to get TR and slice times, and NiftiWrapper to get
    image dimensions (number of slices is the z [2]).
    """
    from nipype.interfaces import dcmstack
    from dcmstack.dcmmeta import NiftiWrapper
    nii = NiftiWrapper.from_filename(nifti)
    imdims = nii.meta_ext.shape
    sliceno = imdims[2]
    mid_slice = int(sliceno / 2)
    lookup = dcmstack.LookupMeta()
    lookup.inputs.meta_keys = {'RepetitionTime': 'TR', 'CsaImage.MosaicRefAcqTimes': 'ST'}
    lookup.inputs.in_file = nifti
    lookup.run()
    slicetimes = [int(lookup.result['ST'][0][x]) for x in range(0,imdims[2])] #Converts slice times to ints. 
    tr = lookup.result['TR'] / 1000 #Converts tr to seconds.
    ta = tr - (tr / sliceno)
    return (sliceno, slicetimes, tr, ta, mid_slice)
           
metadata = Node(Function(function = metaread, input_names = ['nifti'], output_names = ['sliceno', 'slicetimes', 'tr', 'ta', 'mid_slice']), name = 'metadata')


#Custom functions
def make_epi_list(epi1, epi2):
    epis = [epi1, epi2]
    return(epis)

epi_list = Node(Function(function = make_epi_list, input_names = ['epi1', 'epi2'], output_names = ['epis']), name = 'epi_list')
	

def make_despike_list(despike_output):
    despike_s1 = despike_output[0]
    despike_s2 = despike_output[1]
    despike_list = [despike_s1, despike_s2]
    return(despike_s1, despike_s2, despike_list)

despike_list = Node(Function(function = make_despike_list, input_names = ['despike_output'], output_names = ['despike_s1', 'despike_s2', 'despike_list']), name = 'despike_list')


def make_tsnr_list(mean_file, stddev_file, tsnr_file):
    mean_s1 = mean_file[0]
    mean_s2 = mean_file[1]
    mean_list = [mean_s1, mean_s2]

    stddev_s1 = mean_file[0]
    stddev_s2 = mean_file[1]

    tsnr_s1 = mean_file[0]
    tsnr_s2 = mean_file[1]
    return(mean_s1, mean_s2, mean_list, stddev_s1, stddev_s2, tsnr_s1, tsnr_s2)

tsnr_list = Node(Function(function = make_tsnr_list, input_names = ['mean_file', 'stddev_file', 'tsnr_file'], output_names = ['mean_s1', 'mean_s2', 'mean_list', 'stddev_s1', 'stddev_s2', 'tsnr_s1', 'tsnr_s2']), name = 'tsnr_list')
	

def make_anat_mask(csf_seg, gm_seg, wm_seg, anat):
    import numpy as np
    import nibabel as nb
    from scipy.ndimage import binary_fill_holes as bfh
    import os    

    anat_info = nb.load(anat)
    anat_data = anat_info.get_data()
    csf_thresh = nb.load(csf_seg).get_data() > 0.90
    gm_thresh = nb.load(gm_seg).get_data() > 0.90
    wm_thresh = nb.load(wm_seg).get_data() > 0.90

    s = np.ones([9, 9, 9])

    mask_thresh = np.sum([csf_thresh, gm_thresh, wm_thresh], axis = 0).astype(int)
    mask_thresh[mask_thresh > 0] = 1 #Binarise
    mask_thresh = bfh(mask_thresh).astype(int) #Fill holes in mask
 
    anat_mask = anat_data * mask_thresh

    anat_mask_fn = os.path.join(os.getcwd(), 'anat_mask.nii')
    anat_im = nb.Nifti1Image(anat_mask, header = anat_info.header, affine = anat_info.affine)
    anat_im.to_filename(anat_mask_fn)

    mask_thresh_fn = os.path.join(os.getcwd(), 'mask_thresh.nii')
    mask_thresh_im = nb.Nifti1Image(mask_thresh, header = anat_info.header, affine = anat_info.affine)
    mask_thresh_im.to_filename(mask_thresh_fn)

    return(anat_mask_fn, mask_thresh_fn)

anat_mask = Node(Function(function = make_anat_mask, input_names = ['csf_seg', 'gm_seg', 'wm_seg', 'anat'], output_names = ['anat_mask', 'mask_thresh']), name = 'anat_mask')   


def make_smooth_list(normed_output):
    mni_s1 = normed_output[0]
    mni_s2 = normed_output[1]
    mni_list = [mni_s1, mni_s2]
    return(mni_s1, mni_s2, mni_list)

normed_list = Node(Function(function = make_smooth_list, input_names = ['normed_output'], output_names = ['mni_s1', 'mni_s2', 'mni_list']), name = 'mni_list')


#def z_score_im(image, mask):
#    mask = nb.load(mask).get_data()
#    im_info = nb.load(image)
#    im_data = im_info.get_data()
#    im_z_data = (im_data - np.mean(im_data, axis = 3)) / np.std(im_data, axis = 3)
    
#    return(im_z)

#im_z_score = Node(Function(function = z_score_im, input makes = ['image', 'mask'], output_names = ['im_z'], name = 'im_z_score'))


#Nipype below

#Set up iteration over subjects
infosource = Node(IdentityInterface(fields=['subject_id']), name = 'infosource')
infosource.iterables = [('subject_id',subject_list)]

#Select files
template = {'anat': data_dir + '{subject_id}/T1_MPRAGE_SAG_ISO_0_7*/*.IMA',
          'epi_s1': data_dir + '{subject_id}/*PACE_MOCO_P2-1*/*.IMA',
          'epi_s2': data_dir + '{subject_id}/*PACE_MOCO_P2-2*/*.IMA',
          'template': template_dir + 't1_3mm_brain.nii',
          'mask': template_dir + 't1_3mm_mask.nii'}

select_files = Node(SelectFiles(template),name = 'select_files')
select_files.inputs.base_directory = h_dir
select_files.inputs.sort_files = True

anat_stack = Node(dcmstack.DcmStack(),name = 'anat_stack')
anat_stack.inputs.embed_meta = True
anat_stack.inputs.out_format = 'anat'
anat_stack.inputs.out_ext = '.nii'

epi_s1_stack = Node(dcmstack.DcmStack(),name = 'epi_s1_stack')
epi_s1_stack.inputs.embed_meta = True
epi_s1_stack.inputs.out_format = 'epi1'
epi_s1_stack.inputs.out_ext = '.nii'

epi_s2_stack = Node(dcmstack.DcmStack(),name = 'epi_s2_stack')
epi_s2_stack.inputs.embed_meta = True
epi_s2_stack.inputs.out_format = 'epi2'
epi_s2_stack.inputs.out_ext = '.nii'

st_corr = Node(spm.SliceTiming(), name = 'slicetiming_correction')

realign = Node(spm.Realign(),name = 'realign')
realign.inputs.register_to_mean = True

tsnr = MapNode(confounds.TSNR(), iterfield = 'in_file', name = 'tsnr')
tsnr.inputs.mean_file = 'mean.nii'
tsnr.inputs.stddev_file = 'stddev.nii'
tsnr.inputs.tsnr_file = 'tsnr.nii'

despike = MapNode(afni.Despike(),iterfield = 'in_file', name = 'despike')
despike.inputs.outputtype = 'NIFTI'

seg = Node(spm.Segment(), name = 'seg')
seg.inputs.csf_output_type = [False, False, True] #Output native CSF seg
seg.inputs.gm_output_type = [False, False, True] #Output native gm seg
seg.inputs.wm_output_type = [False, False, True] #Output native wm seg

coreg2epi = MapNode(spm.Coregister(), iterfield = 'target', name = 'coreg2epi')

#Warps to MNI space using a 3mm template image
antsnorm = MapNode(ants.Registration(), iterfield = 'moving_image', name = 'antsnorm')
antsnorm.inputs.collapse_output_transforms = True
antsnorm.inputs.initial_moving_transform_com = True
antsnorm.inputs.num_threads= 1
antsnorm.inputs.output_inverse_warped_image = True
antsnorm.inputs.output_warped_image = True
antsnorm.inputs.sigma_units = ['vox'] * 3
antsnorm.inputs.transforms = ['Rigid', 'Affine', 'SyN']
antsnorm.inputs.terminal_output = 'file'
antsnorm.inputs.winsorize_lower_quantile = 0.005
antsnorm.inputs.winsorize_upper_quantile = 0.995
antsnorm.inputs.convergence_threshold = [1e-06]
antsnorm.inputs.convergence_window_size = [10]
antsnorm.inputs.metric = ['MI', 'MI', 'CC']
antsnorm.inputs.metric_weight = [1.0] * 3
antsnorm.inputs.number_of_iterations = [[1000, 500, 250, 100],[1000, 500, 250, 100],[100, 70, 50, 20]]
antsnorm.inputs.radius_or_number_of_bins = [32, 32, 4]
antsnorm.inputs.sampling_percentage = [0.25, 0.25, 1]
antsnorm.inputs.sampling_strategy = ['Regular', 'Regular', 'None']
antsnorm.inputs.shrink_factors = [[8, 4, 2, 1]] * 3
antsnorm.inputs.smoothing_sigmas = [[3, 2, 1, 0]] * 3
antsnorm.inputs.transform_parameters = [(0.1,), (0.1,), (0.1, 3.0, 0.0)]
antsnorm.inputs.use_histogram_matching = True
antsnorm.inputs.write_composite_transform = False

apply2epi = MapNode(ants.ApplyTransforms(), iterfield = ['input_image', 'transforms'], name = 'apply2epi')
apply2epi.inputs.default_value = 0
apply2epi.inputs.input_image_type = 3
apply2epi.inputs.interpolation = 'Linear'
apply2epi.inputs.num_threads = 1
apply2epi.inputs.terminal_output = 'file'

smooth = Node(spm.Smooth(),name = 'smooth')
fwhmlist = [6, 8]
smooth.iterables = ('fwhm', fwhmlist)

#ad = Node.ArtifactDetect()
#ad.inputs.parameter_source = 'SPM'
#ad.inputs.use_norm = True
#ad.inputs.norm_threshold = 1.96


#######################
#First level modelling#
#######################

#Session 1
condnames = ['hap','sad','neu','hor','ver','chk']
o1 = [np.arange(0,502.2,194.4).tolist(), 
np.arange(129.6,502.2,194.4).tolist(),
np.arange(64.8,502.2,194.4).tolist(),
np.arange(97.2,502.2,194.4).tolist(),
np.arange(32.4,502.2,194.4).tolist(),
np.arange(162,502.2,194.4).tolist()]
d1 = [[16.8] * 3, [16.8] * 2, [16.8] *3, [16.8] * 3, [16.8] * 3, [16.8] * 2]

#Session 2
condnames = ['hap','sad','neu','hor','ver','chk']
o2 = [np.arange(97.2,502.2,194.4).tolist(),
np.arange(32.4,502.2,194.4).tolist(),
np.arange(162,502.2,194.4).tolist(),
np.arange(0,502.2,194.4).tolist(),
np.arange(129.6,502.2,194.4).tolist(),
np.arange(64.8,502.2,194.4).tolist()]
d2 = [[16.8] * 3, [16.8] * 3, [16.8] * 2, [16.8] * 3, [16.8] * 2, [16.8] * 3]

#Create list of Bunch objects
design = [Bunch(conditions = condnames, onsets = o1, durations = d1), Bunch(conditions = condnames, onsets = o2, durations = d2)]

#Input model specifications
modelspec = Node(interface = modelgen.SpecifySPMModel(), name = 'modelspec')
modelspec.inputs.input_units = 'secs'
modelspec.inputs.high_pass_filter_cutoff = 100.0
modelspec.inputs.concatenate_runs = False
modelspec.inputs.subject_info = design

#Design first level model
level1design = Node(interface = spm.Level1Design(), name = 'level1design')
level1design.inputs.interscan_interval = 3.0
level1design.inputs.timing_units = 'secs'
level1design.inputs.model_serial_correlations = 'AR(1)'
level1design.inputs.bases = {'hrf':{'derivs': [0, 0]}}
level1design.inputs.mask_threshold = '-Inf'
#level1design.inputs.mask_image = 

#Estimate first level design
level1estimate = Node(interface=spm.EstimateModel(),name = "level1estimate")
level1estimate.inputs.estimation_method = {'Classical':1}

#Contrasts
con1 = ['hap_pos', 'T', condnames, [1, 0, 0, 0, 0, 0]]
con2 = ['sad_pos', 'T', condnames, [0, 1, 0, 0, 0, 0]]
con3 = ['neu_pos', 'T', condnames, [0, 0, 1, 0, 0, 0]]
con4 = ['hor_pos', 'T', condnames, [0, 0, 0, 1, 0, 0]]
con5 = ['ver_pos', 'T', condnames, [0, 0, 0, 0, 1, 0]]
con6 = ['chk_pos', 'T', condnames, [0, 0, 0, 0, 0, 1]]
con7 = ['face_pos', 'T' ,condnames, [1/3, 1/3, 1/3, 0, 0, 0]]
con8 = ['fill_pos', 'T', condnames, [0, 0, 0, 1/3, 1/3, 1/3]]
con9 = ['faceNfill_pos', 'T', condnames, [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]]
con10 = ['hap_neg', 'T', condnames, [-1, 0, 0, 0, 0, 0]]
con11 = ['sad_neg', 'T', condnames, [0, -1, 0, 0, 0, 0]]
con12 = ['neu_neg', 'T', condnames, [0, 0, -1, 0, 0, 0]]
con13 = ['hor_neg', 'T', condnames, [0, 0, 0, -1, 0, 0]]
con14 = ['ver_neg', 'T', condnames, [0, 0, 0, 0, -1, 0]]
con15 = ['chk_neg', 'T', condnames, [0, 0, 0, 0, 0, -1]]
con16 = ['faces_neg', 'T', condnames,[-1/3, -1/3, -1/3, 0, 0, 0]]
con17 = ['fill_neg','T',condnames,[0, 0, 0, -1/3, -1/3, -1/3]]
con18 = ['faceNfill_neg', 'T', condnames, [-1/6, -1/6, -1/6, -1/6, -1/6, -1/6]]

#contrast_list=[con1,con2,con3,con4,con5,con6,con7,con8,con9]
contrast_list = [con1, con2, con3, con4, con5, con6, con7, con8, con9, con10, con11, con12, con13, con14, con15, con16, con17, con18]

conestimate = Node(spm.EstimateContrast(),name = 'contrast_estimate')
conestimate.inputs.contrasts = contrast_list

substitutions=('_subject_id_', '')
sink=Node(DataSink(), name = 'sink')
sink.inputs.substitutions = substitutions
sink.inputs.base_directory = out_dir

preproc = Workflow(name = 'preproc')
preproc.base_dir = work_dir
preproc.connect([(infosource, select_files, [('subject_id','subject_id')]),

                 #Convert files to nii
                 (select_files, anat_stack, [('anat', 'dicom_files')]),
                 (select_files, epi_s1_stack, [('epi_s1', 'dicom_files')]),
                 (select_files, epi_s2_stack, [('epi_s2', 'dicom_files')]),  
                 
                 #Add epis to list
                 (epi_s1_stack, epi_list, [('out_file', 'epi1')]),
                 (epi_s2_stack, epi_list, [('out_file', 'epi2')]),

                 #Read meta-data
                 (epi_s1_stack, metadata, [('out_file', 'nifti')]),

                 #Preprocessing
                 (anat_stack, seg, [('out_file', 'data')]), #Segment anatomical for normalisation mask
                        
                 (anat_stack, anat_mask, [('out_file', 'anat')]), #Make anatomical mask
                 (seg, anat_mask, [('native_csf_image', 'csf_seg'),
                                   ('native_gm_image', 'gm_seg'),
                                   ('native_wm_image', 'wm_seg')]),


                 (epi_list, st_corr, [('epis', 'in_files')]), #Slice timing correction
                 (metadata, st_corr, [('sliceno', 'num_slices'),
                                     ('slicetimes', 'slice_order'),
                                     ('tr', 'time_repetition'),
                                     ('ta', 'time_acquisition'),
                                     ('mid_slice', 'ref_slice')]),

                 (st_corr, realign, [('timecorrected_files', 'in_files')]), #Realign

                 (realign, despike, [('realigned_files', 'in_file')]), #Despike
                 (despike, despike_list, [('out_file', 'despike_output')]), #Get outputs of despike process
                 (despike_list, tsnr, [('despike_list', 'in_file')]), #Generate tSNR  

                 (tsnr, tsnr_list, [('mean_file', 'mean_file'),
                                    ('stddev_file', 'stddev_file'),
                                    ('tsnr_file', 'tsnr_file')]),
                               
                 (tsnr_list, coreg2epi, [('mean_list', 'target')]), #Coregister anat to epi
                 (anat_mask, coreg2epi, [('anat_mask', 'source')]), #Coregister anat to epi


                 #Normalisation
                 (select_files, antsnorm, [('template', 'fixed_image')]),
                 (coreg2epi, antsnorm, [('coregistered_source', 'moving_image')]),

                 (select_files, apply2epi, [('template', 'reference_image')]), 
                 (despike_list, apply2epi, [('despike_list', 'input_image')]),
                 (antsnorm, apply2epi, [('forward_transforms', 'transforms')]), 

                 #Smoothing
                 (apply2epi, normed_list,[('output_image', 'normed_output')]),
                 (normed_list, smooth,[('mni_list','in_files')]),

                 ##Artifact detection
                 #(smooth, ad,[('smoothed_files', 'realigned_files')]),
                 #(realign, ad,[('realignment_parameters', 'realignment_parameters')]),

                 #First level modelling
                 (smooth, modelspec,[('smoothed_files', 'functional_runs')]),
                 (realign, modelspec, [('realignment_parameters', 'realignment_parameters')]),
                 (metadata, modelspec, [('tr', 'time_repetition')]),  
                 (modelspec, level1design,[('session_info', 'session_info')]),
                 #(select_files, level1design,[('template', 'mask_image')]),
                 (level1design, level1estimate,[('spm_mat_file', 'spm_mat_file')]),
                 (level1estimate, conestimate,[('beta_images', 'beta_images'),
                                              ('residual_image', 'residual_image'),
                                              ('spm_mat_file', 'spm_mat_file')]),

                 #Collect outputs
                 (realign,sink,[('realignment_parameters', 'QC')]),
                 (conestimate,sink,[('con_images', 'contrasts')]),
                  
                 ])

preproc.write_graph(graph2use = 'orig')
preproc.run(plugin = 'MultiProc', plugin_args = {'n_procs': n_cores - 1})
