import torch
from lightning.pytorch.callbacks import BasePredictionWriter


def convert_to_dicom(img_slice: np.ndarray, phantom_path: str,
                     spacings:tuple|None=None):
    '''
    :param img_slice: input 2D ndarray to be saved
    :param phantom_path: filename to save dicom file to
    :param spacings: tuple containing pixel spacings in mm
    '''
    # https://github.com/DIDSR/pediatricIQphantoms/blob/main/src/pediatricIQphantoms/make_phantoms.py#L144
    Path(phantom_path).parent.mkdir(exist_ok=True, parents=True)
    fpath = pydicom.data.get_testdata_file("CT_small.dcm")
    ds = pydicom.dcmread(fpath)
    img_slice = img_slice.squeeze()
    ds.Rows, ds.Columns = img_slice.shape
    if spacings:
        ds.SliceThickness = spacings[0]
        ds.PixelSpacing = [spacings[1], spacings[2]]
    ds.PixelData = img_slice.copy(order='C').astype('int16') -\
        int(ds.RescaleIntercept)
    pydicom.dcmwrite(phantom_path, ds)


class DicomWriter(BasePredictionWriter):

    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        convert_to_dicom(predictions, os.path.join(self.output_dir, "predictions.pt"))
