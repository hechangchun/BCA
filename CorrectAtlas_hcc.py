# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 16:50:38 2018

@author: wolf
"""

  """
    When I transform the Monstreal Neurological Institute (MNI) space 
    to the individual-participant diffusion space, ROI will be reduced
    Function to correct atlas after resampling (looses some rois),
    this function recovers those lost rois
  """
import os
import os.path as op
from os.path import join as opj
import numpy as np
import nibabel as nib
import subprocess

def atlas_with_all_rois(atlas_old, atlas_new):
    """
    Function to correct atlas after resampling (looses some rois),
    this function recovers those lost rois
    """

#    atlas_old = opj(EXTERNAL, 'bha_' + atlas + '_1mm_mni09c.nii.gz')
#    atlas_new = opj(PROCESSED, sub, ses, 'func', sub + '_' + ses +
#                    '_' + atlas + '_bold_space.nii.gz')

    atlas_new_img = nib.load(atlas_new)
    m = atlas_new_img.affine[:3, :3]

    atlas_old_data = nib.load(atlas_old).get_data()
    where_are_nan = np.isnan(atlas_old_data)
    atlas_old_data[where_are_nan] = 0
    atlas_old_data_rois = np.unique(atlas_old_data)
    atlas_new_data = atlas_new_img.get_data()
    where_are_nan = np.isnan(atlas_new_data)
    atlas_new_data[where_are_nan] = 0
    atlas_new_data_rois = np.unique(atlas_new_data)

    diff_rois = np.setdiff1d(atlas_old_data_rois, atlas_new_data_rois)

    for roi in diff_rois:
        p = np.argwhere(atlas_old_data == roi)[0]
        x, y, z = (np.round(np.diag(np.divide(p, m)))).astype(int)
        atlas_new_data[x, y, z] = roi

    atlas_new_data_img_corrected = nib.Nifti1Image(atlas_new_data,
                                                   affine=atlas_new_img.affine)
    ## save correct atlas
    father_path = os.path.abspath(os.path.dirname(atlas_new)+os.path.sep+".")
    save_path = os.path.join('%s\\%s'% (father_path, 'wlevel_2514_corrected.nii'))
    nib.save(atlas_new_data_img_corrected,
             save_path)
#atlas_new = ('C:/Users/wolf/Desktop/test/test/DTIImg_1_eddy/29181/wlevel_2514.nii')
#atlas_new = ('F:/BrainAging/NYU/DTIImg_1_eddy/29181/wlevel_0020.nii')
atlas_old = ('F:/BrainAging/template/function/level_2514.nii')
#atlas_old = ('C:/Users/wolf/Desktop/test/test/level_2514_hcc.nii')


filePath = ('F:/BrainAging/NYU/DTIImg_1_eddy')
pathDir = os.listdir(filePath)
for allDir in pathDir:
    child = os.path.join('%s/%s' % (filePath,allDir))
    dirs = os.listdir(child)
    for i in dirs:        
        if os.path.splitext(i)[0] == "wlevel_2514":
           atlas_new = os.path.join('%s/%s' % (child, i))
           atlas_with_all_rois(atlas_old,atlas_new)




