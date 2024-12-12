from datetime import datetime
from pathlib import Path
import pydicom

import SimpleITK as sitk

def itk_to_dicom(img:sitk.SimpleITK.Image, fname:str|Path, patientname=None, patientid=0, age=None, studyname=None, studyid=0, seriesname=None, seriesid=0, patient_diameter=None, dose=None, kernel='D45', recon='fbp') -> list[Path]:
        """
            write ct data to DICOM file, returns list of written dicom file names

            :param img: it
            :param fname: filename to save image to (preferably with '.dcm` or related extension)
            :returns: list[Path]
        """
        fpath = pydicom.data.get_testdata_file("CT_small.dcm")
        ds = pydicom.dcmread(fpath)
        # update meta info
        ds.Manufacturer = 'Siemens (simulated)'
        ds.ManufacturerModelName = 'Definition AS+ (simulated)'
        time = datetime.now()
        ds.InstanceCreationDate = time.strftime('%Y%m%d')
        ds.InstanceCreationTime = time.strftime('%H%M%S')
        ds.InstitutionName = 'FDA/CDRH/OSEL/DIDSR'
        ds.StudyDate = ds.InstanceCreationDate
        ds.StudyTime = ds.InstanceCreationTime
        
        if patientname:
            ds.PatientName = patientname
        if seriesid:
            ds.SeriesNumber = seriesid
        if age:
            ds.PatientAge = f'{int(age):03d}Y'
        if patientid:
            ds.PatientID = f'{int(patientid):03d}'
        del(ds.PatientWeight)
        del(ds.ContrastBolusRoute)
        del(ds.ContrastBolusAgent)
        if patient_diameter:
            ds.ImageComments = f"effctive diameter [cm]: {patient_diameter/10}"
        ds.ScanOptions = 'AXIAL MODE'
        ds.ReconstructionDiameter = img.GetSpacing()[0]*img.GetSize()[0] #mm
        ds.ConvolutionKernel =f'{recon} {kernel}'
        if dose:
            ds.Exposure = dose
        
        # load image data
        if seriesname:
            ds.StudyDescription = f"{dose} photons " + seriesname + " " + ds.ConvolutionKernel
        
        vol = sitk.GetArrayFromImage(img)
        if vol.ndim == 2:
            vol = vol[None]
        
        nslices, ds.Rows, ds.Columns = vol.shape
        ds.SpacingBetweenSlices = ds.SliceThickness
        ds.DistanceSourceToDetector = 1085.6
        ds.DistanceSourceToPatient = 595
        
        ds.PixelSpacing = 2*[img.GetSpacing()[-1]]
        ds.SliceThickness =  img.GetSpacing()[0]

        ds.KVP = 120
        ds.StudyID = str(studyid)
        # series instance uid unique for each series
        end = ds.SeriesInstanceUID.split('.')[-1]
        new_end = str(int(end) + studyid)
        ds.SeriesInstanceUID = ds.SeriesInstanceUID.replace(end, new_end)
        ds.AcquisitionNumber = studyid
        end = ds.StudyInstanceUID.split('.')[-1]
        new_end = str(int(end) + studyid)
        ds.StudyInstanceUID = ds.StudyInstanceUID.replace(end, new_end)

        fname = Path(fname)
        fname.parent.mkdir(exist_ok=True, parents=True)
        # saveout slices as individual dicom files
        fnames = []
        if vol.ndim == 2: vol = vol[None]
        for slice_idx, array_slice in enumerate(vol):
            ds.InstanceNumber = slice_idx + 1 # image number
            # SOP instance UID changes every slice
            end = ds.SOPInstanceUID.split('.')[-1]
            new_end = str(int(end) + slice_idx + studyid + seriesid)
            ds.SOPInstanceUID = ds.SOPInstanceUID.replace(end, new_end)
            # MediaStorageSOPInstanceUID changes every slice
            end = ds.file_meta.MediaStorageSOPInstanceUID.split('.')[-1]
            new_end = str(int(end) + slice_idx + studyid + seriesid)
            ds.file_meta.MediaStorageSOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID.replace(end, new_end)
            # slice location and image position changes every slice
            ds.SliceLocation = nslices//2*ds.SliceThickness + slice_idx*ds.SliceThickness
            ds.ImagePositionPatient[-1] = ds.SliceLocation
            ds.ImagePositionPatient[0] = -ds.Rows//2*ds.PixelSpacing[0]
            ds.ImagePositionPatient[1] = -ds.Columns//2*ds.PixelSpacing[1]
            ds.ImagePositionPatient[2] = ds.SliceLocation
            ds.PixelData = array_slice.copy(order='C').astype('int16') - int(ds.RescaleIntercept)
            dcm_fname = fname.parent / f'{fname.stem}_{slice_idx:03d}{fname.suffix}' if nslices > 1 else fname
            fnames.append(dcm_fname)
            pydicom.write_file(dcm_fname, ds)
        return fnames
    
import SimpleITK as sitk

def convert_itk_dataset_to_dicom(meta):
    rows = []

    for idx, row in meta.iterrows():
        img = sitk.ReadImage(row.file)
        pixel_size = row['FOV [cm]']*10/img.GetSize()[0]
        img.SetSpacing(3*[pixel_size])
        diam_str = f"diameter_{int(row['effective diameter [cm]']*10):03d}mm"
        outdir = anthro_dir / row.Name.replace(' ', '_') / diam_str / f"dose_{row['Dose [%]']:03d}" / row['recon'] / f"{row.Name.replace(' ', '_')}.dcm"
        fnames = itk_to_dicom(img, outdir, patientname=row.Name, patientid=idx, age=row['age [year]'], studyname='pediatric CT noise texture augmentation', seriesname=diam_str, patient_diameter=row['effective diameter [cm]'], dose=row['Dose [%]'])
        for fname in fnames:
            row.file = fname.relative_to(anthro_dir)
            row['instance'] = int(fname.stem.split('_')[-1])
            row.kernel = 'D45'
            row.scanner = 'Siemens Definition AS+ (simulated)'
            rows.append(pd.DataFrame(row).T)

    dicom_meta = pd.concat(rows).reset_index(drop=True)
    dicom_meta.to_csv(anthro_dir / 'metadata.csv', index=False)
    return dicom_meta