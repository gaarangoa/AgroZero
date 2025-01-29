from sam2.build_sam import load_model
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np 
import pandas as pd 
import torch
from torchvision import models, transforms
from itertools import combinations
import pickle

from samecode.plot.pyplot import subplots
import seaborn as sns

class MaskGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, *, model='large', **kwargs):
        self.model = load_model("../models/{}".format(model), apply_postprocessing=False)
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=self.model,
            points_per_side=16, 
            points_per_batch=32,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.92,
            stability_score_offset=0.7,
            crop_n_layers=1,
            box_nms_thresh=0.7,
            crop_n_points_downscale_factor=2,
        )
    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        print('masking image ...')
        masks = self.mask_generator.generate(X)
        return [masks, X]


class MaskFeaturizer(BaseEstimator, TransformerMixin):
    def __init__(self, **kwargs):
        self.min_intensity = kwargs.get('min_intensity', 10)

    def fit(self, X, y=None):
        pass

    def object2coords(self, obj):
        coords = []
        M, N = obj.shape[:2]
        m_indices, n_indices = np.indices((M, N))
        m_indices = m_indices.flatten()
        n_indices = n_indices.flatten()
        
        pixels = obj.reshape(-1, 3)
        indices = np.stack((m_indices, n_indices), axis=-1)
        pixels = np.concatenate([pixels, indices], axis=1)
        
        pixels = pixels[pixels[:, :3].sum(axis=1) != 0, :]
        
        return pixels 

    def transform(self, X, y=None):
        print('Extract feature ...')
        objects = []
        masks, image = X
        for obj in range(len(masks)):
            mask = masks[obj]['segmentation']
            x, y, w, h = [int(i) for i in masks[obj]['bbox']]
            simg = np.array([image[y:y+h, x:x+w, i] * mask[y:y+h, x:x+w] for i in range(3)])
            simg = np.transpose(simg, (1, 2, 0))

            x_intensity = simg.mean(axis=0).mean(axis=1) > self.min_intensity
            y_intensity = simg.mean(axis=1).mean(axis=1) > self.min_intensity
            
            x += np.where(np.cumsum(x_intensity) == 1)[0].astype(int)[0]
            y += np.where(np.cumsum(y_intensity) == 1)[0].astype(int)[0]

            x_box_index = (x_intensity)
            simg = simg[:, x_box_index, :]
    
            y_box_index = (y_intensity)
            simg = simg[y_box_index, :, :]
    
            w = simg.shape[1]
            h = simg.shape[0]

            coords = self.object2coords(simg)

            objects.append({
                'segmented_image': simg,
                'bbox': [int(x), int(y), w, h],
                'area': masks[obj]['area'] / (simg.shape[0]*simg.shape[1]),
                'area_relative_image': masks[obj]['area'] / (image.shape[0] * image.shape[1]),
                'predicted_iou':  masks[obj]['predicted_iou'],
                'stability_score':  masks[obj]['stability_score'],
                'coords': coords,
                'mask': mask,
                'index': obj
            })
            
        return objects


class Embedder(BaseEstimator, TransformerMixin):
    def __init__(self, **kwargs):
        self.embedder = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        self.embedder.eval()

    def get_mask_embeddings(self, mask):
        embeddings = []
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        with torch.no_grad():
            mask_image = transform(mask).unsqueeze(0)  # Add batch dimension
            features = self.embedder(mask_image)
            embeddings = features.squeeze().numpy()  # Remove batch dimension and convert to numpy
                
        return embeddings
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print('getting embeddings ...')
        for obj in X:
            obj['embeddings'] = self.get_mask_embeddings(obj['segmented_image'])
        return X


class PixelPredictor(BaseEstimator, TransformerMixin):
    def __init__(self, **kwargs):
        file = kwargs.get('model', '../models/corn2pixel.pk')
        self.features, self.model = pickle.load(open(file, 'rb'))

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print('pixel prediction ...')
        for obj in X:
            obj['pixel_labels'] = self.model.predict(obj['coords'][:, :3])
            
            coords_labels_1 = obj['coords'][obj['pixel_labels'] == 1, :3]
            coords_labels_0 = obj['coords'][obj['pixel_labels'] == 0, :3]
            
            meanI = np.array([0, 0, 0])
            meanB = np.array([0, 0, 0])
            if coords_labels_1.size != 0:
                meanI = coords_labels_1.mean(axis=0)
    
            if coords_labels_0.size != 0:
                meanB = coords_labels_0.mean(axis=0)

            obj['mean_R'] = meanI[0] / 255
            obj['mean_G'] = meanI[1] / 255
            obj['mean_B'] = meanI[2] / 255
            
            obj['meanb_R'] = meanB[0] / 255
            obj['meanb_G'] = meanB[1] / 255
            obj['meanb_B'] = meanB[2] / 255

            obj['w_pr'] = np.sum(obj['pixel_labels']) / (obj['segmented_image'].shape[0]*obj['segmented_image'].shape[1])
            
        return X


class ObjectPredictor(BaseEstimator, TransformerMixin):
    def __init__(self, **kwargs):
        file = kwargs.get('model', '../models/predict_corn.pk')
        self.features, self.model = pickle.load(open(file, 'rb'))
        self.area_relative_image = kwargs.get('area_relative_image', 0.1)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print('Object prediction ...')
        # data_objects = [ [i[k] for k in self.features] for i in X]
        data_objects = np.array([i[self.features] for i in X])
        # data_objects = pd.DataFrame(data_objects, columns = self.features)
        
        labels = self.model.predict(data_objects)

        for ix, obj in enumerate(X):
            obj['label'] = 'corn' if labels[ix] == 1 and 100*obj['area_relative_image'] >= self.area_relative_image else 'other'
            obj['corn_disease_percent'] = obj['pixel_labels'].sum() / obj['pixel_labels'].shape[0]

        return X


class ObjectCorrector(BaseEstimator, TransformerMixin):
    def __init__(self, **kwargs):
        self.label = kwargs.get('class', 'corn')
        self.box_overlap_fraction = kwargs.get('box_overlap_fraction', 0.75)

    def overlap_boxes(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
    
        sx1 = set(range(x1, x1+w1))
        sx2 = set(range(x2, x2+w2))
    
        sy1 = set(range(y1, y1+h1))
        sy2 = set(range(y2, y2+h2))
    
        ax = len(sx1 & sx2) / np.max([len(sx1), len(sx2)])
        ay = len(sy1 & sy2)/ np.max([len(sy1), len(sy2)])
    
        area_box1 = len(sx1) * len(sy1)
        area_box2 = len(sx2) * len(sy2)
        
        return (ax+ay) / 2, [area_box1, area_box2, int(area_box1 > area_box2)]
    
    
    def box_overlap(self, boxes):
        overlaps = []
        for box1, box2 in list(combinations(boxes, 2)):
            s, areas = self.overlap_boxes(box1[0], box2[0])
            indices = [box1[1], box2[1]]
            overlaps.append({
                'box1': box1[1], 'box2': box2[1], 
                'overlap': s, 'box1_area': areas[0], 
                'box2_area': areas[1], 
                'min_area': indices[areas[2]]
            })
        
        return pd.DataFrame(overlaps).sort_values('overlap')

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print('object box correction and filtering')
        objects = [i for i in X if i['label'] == 'corn']
        
        boxes = [[i['bbox'], i['index']] for i in objects]
        overlaps = self.box_overlap(boxes).query('overlap > {}'.format(self.box_overlap_fraction))
        
        objects = [i for i in objects if not i['index'] in overlaps.min_area.values]

        return objects


def visualize(image, objects):
    all_masks = []
    for ix, object_ in enumerate(objects):
        all_masks.append(np.array([object_['mask']]*3))
        
    all_masks = (np.array(all_masks).transpose((0, 2, 3, 1)).sum(axis=0) > 0)

    f, axs = subplots(cols=4, rows=1, h=7, w=35, return_f=True)
    masked_image = all_masks * image
    masked_image[masked_image == 0] = 0
    
    axs[1].imshow( masked_image )
    axs[0].imshow(image);
    axs[2].imshow( masked_image )
    
    
    for obj in objects:
        coords = obj['coords'][obj['pixel_labels'] == 0]
        labels= obj['pixel_labels'][obj['pixel_labels'] == 0]
        sns.scatterplot(
            y=coords[:, 3]+obj['bbox'][1], 
            x=coords[:, 4]+obj['bbox'][0], 
            hue=labels, ax=axs[2], legend=False, s=3, 
            alpha=0.5, 
            palette=['lightblue']
        )
    
    for ax in axs[:3]:
        ax.set_aspect('equal')
    
    sns.histplot(x=[i['corn_disease_percent'] for i in objects], ax=axs[3]);
    axs[3].set_title('Proporcion general de maiz sano', fontsize=20)
    axs[3].set_xlabel('% maiz sano', fontsize=25)
    
    axs[0].set_title('Imagen original', fontsize=20)
    axs[1].set_title('Imagen Segmentada', fontsize=20)
    axs[2].set_title('Identificacion de regiones sanas / enfermas', fontsize=20)
    
    sns.despine()

    return f