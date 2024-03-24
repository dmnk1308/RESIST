import nibabel as nib
import pydicom
from nibabel import processing
import numpy as np
import os
from tqdm import tqdm
import shutil

def nifti2dicom(path_in, path_out, draft_path='template.dcm', modality='CT', y_flip=True, z_flip=True, x_flip=True, resample=False):
    ''' 
    Converts nifti images to dicom images.

    path_in: path to nifti data
    path_out: path to output dicom directory
    draft_path: path to draft dicom file
    y_flip: flip image in y direction
    z_flip: flip image in z direction
    '''

    os.makedirs(path_out, exist_ok=True)
    dcm_draft = pydicom.dcmread(draft_path)
    if modality == 'PT':
        dcm_draft.file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.128'
    nii = nib.load(path_in)

    if resample:
        height = (nii.header['dim'][3]-1)*nii.header['pixdim'][3]
        z_spacing = height/200
        x_spacing = nii.header['pixdim'][1]
        y_spacing = nii.header['pixdim'][2]
        nii = processing.conform(nii, out_shape=[512, 512, 200], voxel_size=(x_spacing, y_spacing, z_spacing))

    # process nifti images
    imgs = nii.get_fdata().astype(np.int16)
    imgs = np.moveaxis(imgs, -1, 0)
    imgs = np.moveaxis(imgs, -1, 1)
    if x_flip:
        imgs = np.flip(imgs, axis=2)
    if y_flip:
        imgs = np.flip(imgs, axis=1)
    if z_flip:
        imgs = np.flip(imgs, axis=0)

    scale, intercept = [*nii.header.get_slope_inter()]
    if scale == None:
        scale = 1
    if intercept == None:
        intercept = 0
    
    # set general dicom tags
    dcm_draft.RescaleIntercept = intercept
    dcm_draft.RescaleSlope = scale
    slice_thickness = nii.header['pixdim'][3]
    x_dim, y_dim = nii.header['pixdim'][1], nii.header['pixdim'][2]
    dcm_draft.BitsStored = nii.header['bitpix']
    dcm_draft.SliceThickness = slice_thickness
    dcm_draft.SamplesPerPixel = 1
    dcm_draft.BitsAllocated = 16
    dcm_draft.BitsStored = 12
    dcm_draft.HighBit = 11
    dcm_draft.PixelRepresentation = 1
    dcm_draft.PhotometricInterpretation = 'MONOCHROME2'
    dcm_draft.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    # dcm_draft.SOPClassUID = '1.2.840.10008.1.2'
    dcm_draft.Rows = imgs.shape[1]
    dcm_draft.Columns = imgs.shape[2]
    dcm_draft.Modality = modality
    x_offset = float(nii.header['qoffset_x'])
    y_offset = float(nii.header['qoffset_y'])
    z_offset = float(nii.header['qoffset_z'])
    dcm_draft.add_new([0x0028,0x0030], 'DS', [float(x_dim), float(y_dim)])
    # dcm_draft.add_new([0x0020, 0x0037], 'DS', [1, 0, 0, 0, 1, 0])


    for i, img in enumerate(imgs):
        dcm_draft.add_new([0x0020, 0x0032], 'DS', [x_offset, -1*float(y_dim*imgs.shape[1]+y_offset), float(z_offset+slice_thickness*i)])
        dcm_draft.PixelData = img.astype(np.int16).tobytes()
        # dcm_draft.add_new([0x0020, 0x0037], 'DS', [1, 0, 0, 0, 1, 0])
        dcm_draft.InstanceNumber = i+1
        pydicom.write_file(path_out + '/' + str(i+1).zfill(4) + '.dcm', dcm_draft)