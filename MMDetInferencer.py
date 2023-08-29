from metadata import DatasetMeta
from mmdet.apis import DetInferencer

metadata = DatasetMeta()


class MMDetInferencerObj(object):
    def __init__(self, threshold=0.3, model_name='FASTERRCNN_LH_DETECTOR_V1', device='cpu'):
        self.threshold = threshold
        self.model_name = model_name
        self.inferencer = DetInferencer(metadata.det_inferencer[model_name]['CONFIG'],
                                        metadata.det_inferencer[model_name]['CHECKPOINT'],
                                        device=device,
                                        show_progress=False)
        '''if model_name == 'FASTERRCNN_LH_Detector_V1':
            self.inferencer = DetInferencer(metadata.det_inferencer[model_name]['CONFIG'],
                                            metadata.det_inferencer[model_name]['CHECKPOINT'])
        if model_name == 'FASTERRCNN_LH_Detector_V2':
            self.inferencer = DetInferencer(metadata.det_inferencer[model_name]['CONFIG'],
                                            metadata.det_inferencer[model_name]['CHECKPOINT'])
        '''
    def __call__(self, roiframe):
        r = self.inferencer(roiframe)
        return r
