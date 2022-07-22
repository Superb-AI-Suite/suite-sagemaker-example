import os, json
from urllib.request import urlopen, urlretrieve
from io import BytesIO
from zipfile import ZipFile
from pathlib import Path
from typing import List, Optional, Callable

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO

import spb.sdk
from suite_detection.utils import call_with_retry, open_image
from suite_detection.official.utils import collate_fn
from suite_detection.official import (
    transforms as T,
    coco_utils as C,
)


def load_image(client, label_id, cache_dir):
    if cache_dir is not None:
        cache_path = os.path.join(cache_dir, label_id)
        if not os.path.isfile(cache_path):
            image_url = call_with_retry(client.get_data, id=label_id).get_image_url()
            call_with_retry(urlretrieve, image_url, cache_path)
        image = open_image(image_path=cache_path)
    else:
        image_url = call_with_retry(client.get_data, id=label_id).get_image_url()
        image = call_with_retry(open_image, image_url=image_url)
    return image


def obj2bbox(obj):
    b = obj['coord']
    x, y, w, h = [b['x'], b['y'], b['width'], b['height']]

    return {
        'bbox': [x, y, w, h],
        'segmentation': [[x, y, x + w, y, x + w, y + h, x, y + h]],
    }


def obj2segmentation(obj):
    assert obj.get('multiple', False), f'Unsupported annotation format: {obj}'

    points = obj['coord']['points']
    polygons = [
        [xy for p in pts[0][:-1] for xy in [p['x'], p['y']]]
        for pts in points if len(pts[0]) > 3
    ]

    outer_pts = [p for pts in points for p in pts[0] if len(pts[0]) > 3]
    if len(outer_pts) == 0:
        return {}

    outer_pts = outer_pts[:-1]
    x1 = min([p['x'] for p in outer_pts])
    y1 = min([p['y'] for p in outer_pts])
    x2 = max([p['x'] for p in outer_pts])
    y2 = max([p['y'] for p in outer_pts])
    w, h = x2 - x1, y2 - y1

    return {
        'bbox': [x1, y1, w, h],
        'segmentation': polygons,
    }


def obj2keypoints(obj):
    points = obj['coord']['points']
    keypoints = [
        [p['x'], p['y'], p['state']['visible']]
        for p in points
    ]

    xs = [x for x, y, v in keypoints]
    ys = [y for x, y, v in keypoints]
    if len(xs) == 0 or len(ys) == 0:
        return {}

    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    w, h = x2 - x1, y2 - y1
    x, y, w, h = [x1 - 0.1 * w, y1 - 0.1 * h, 1.2 * w, 1.2 * h]  # adds 10% margin

    return {
        'bbox': [x, y, w, h],
        'segmentation': [[x, y, x + w, y, x + w, y + h, x, y + h]],
        'keypoints': keypoints,
        'num_keypoints': len([v for x, y, v in keypoints if v != 0]),
    }


SUITE_OBJ_TO_COCO_ANNO = {
    'box': obj2bbox,
    'polygon': obj2segmentation,
    #'keypoint': obj2keypoints,
}


def load_label(export, filename, category_id_map, image_id):
    label_id = Path(filename).stem.split('.')[0]
    content = json.load(export.open(filename))

    annotations = []
    for obj in content.get('objects', []):
        class_id = obj['class_id']
        if class_id not in category_id_map:
            print(f'Unknown class: {class_id}, {obj["class_name"]}')
            continue

        anno_type = obj['annotation_type']
        if anno_type not in SUITE_OBJ_TO_COCO_ANNO:
            print(f'Unknown annotation type: {anno_type}')
            continue

        anno = SUITE_OBJ_TO_COCO_ANNO[anno_type](obj['annotation'])
        if not anno:
            continue

        annotations.append({
            **anno,
            'image_id': image_id,
            'category_id': category_id_map[class_id],
            'area': anno['bbox'][2] * anno['bbox'][3],
            'iscrowd': 0,
        })

    return {
        'label_id': label_id,
        'annotations': annotations,
    }


def build_transforms(train, categories, transforms=None, target_category_names=None):
    if transforms is None:
        transforms = [C.ConvertCocoPolysToMask()]

    if target_category_names is not None:
        category_ids = [cat['id'] for cat in categories if cat['name'] in target_category_names]
        pre_transforms = [C.FilterAndRemapCocoCategories(category_ids)]
    else:
        pre_transforms = []

    if train:
        post_transforms = [T.ToTensor(), T.RandomHorizontalFlip(0.5)]
    else:
        post_transforms = [T.ToTensor()]

    return T.Compose(pre_transforms + transforms + post_transforms)


class SuiteDataset(Dataset):
    def __init__(
        self,
        team_name: str,
        access_key: str,
        project_name: str,
        export_name: str,
        train: bool,
        caching_image: bool = True,
        transforms: Optional[List[Callable]] = None,
        category_names: Optional[List[str]] = None,
    ):
        super().__init__()

        client = spb.sdk.Client(team_name=team_name, access_key=access_key, project_name=project_name)
        export_info = call_with_retry(client.get_export, name=export_name)
        export_data = call_with_retry(urlopen, export_info.download_url).read()
        with ZipFile(BytesIO(export_data), 'r') as export:
            label_files = [f for f in export.namelist() if f.startswith('labels/')]
            label_interface = json.loads(export.open('project.json', 'r').read())
            category_infos = label_interface.get('object_detection', {}).get('object_classes', [])

        cache_dir = None
        if caching_image:
            cache_dir = f'/tmp/{team_name}/{project_name}'
            os.makedirs(cache_dir, exist_ok=True)

        self.client = client
        self.export_data = export_data
        self.categories = [
            { 'id': i + 1, 'name': cat['name'], 'type': cat['annotation_type'] }
            for i, cat in enumerate(category_infos)
        ]
        self.category_id_map = { cat['id']: i + 1 for i, cat in enumerate(category_infos) }
        self.transforms = build_transforms(train, self.categories, transforms, category_names)
        self.cache_dir = cache_dir

        # to resolve copy-on-access issue: https://github.com/pytorch/pytorch/issues/13246
        self.label_files = np.array(label_files).astype(np.string_)

    def __len__(self):
        return len(self.label_files)

    def __getitem__(self, idx):
        idx = idx if idx >= 0 else len(self) + idx
        if idx < 0 or idx >= len(self):
            raise IndexError(f'index out of range')

        image_id = idx + 1
        label_file = self.label_files[idx].decode('ascii')
        with ZipFile(BytesIO(self.export_data), 'r') as export:
            label = load_label(export, label_file, self.category_id_map, image_id)

        try:
            image = load_image(self.client, label['label_id'], self.cache_dir)
        except Exception as e:
            print(f'Failed to load the {idx}-th image due to {repr(e)}, getting {idx+1}-th data instead')
            return self.__getitem__(idx + 1)

        target = {
            'image_id': image_id,
            'label_id': label['label_id'],
            'annotations': label['annotations'],
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target


def build_coco_dataset(dataset, num_workers):
    data_loader = DataLoader(
        dataset, batch_size=1,
        shuffle=False, drop_last=False, num_workers=num_workers,
        collate_fn=collate_fn,
    )

    coco_set = { 'images': [], 'annotations': [], 'categories': dataset.categories }
    ann_id = 1
    for images, targets in data_loader:
        image, target = images[0], targets[0]
        coco_set['images'].append({
            'id': target['image_id'],
            'label_id': target['label_id'],
            'height': image.shape[-2],
            'width': image.shape[-1]
        })
        coco_set['annotations'].extend([
            { **ann, 'id': ann_id + i }
            for i, ann in enumerate(target['annotations'])
        ])
        ann_id += len(target['annotations'])

    coco = COCO()
    coco.dataset = coco_set
    coco.createIndex()
    return coco


class SuiteCocoDataset(C.CocoDetection):
    def __init__(
        self,
        team_name: str,
        access_key: str,
        project_name: str,
        export_name: str,
        train: bool,
        caching_image: bool = True,
        transforms: Optional[List[Callable]] = None,
        category_names: Optional[List[str]] = None,
        num_init_workers: int = 20,
    ):
        super().__init__(img_folder='', ann_file=None, transforms=None)

        dataset = SuiteDataset(
            team_name, access_key, project_name, export_name,
            train=False, transforms=[],
            caching_image=caching_image, category_names=category_names,
        )
        self.client = dataset.client
        self.cache_dir = dataset.cache_dir

        self.coco = build_coco_dataset(dataset, num_init_workers)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self._transforms = build_transforms(train, dataset.categories, transforms, category_names)

    def _load_image(self, id:int):
        label_id = self.coco.loadImgs(id)[0]['label_id']
        image = load_image(self.client, label_id, self.cache_dir)
        return image

    def __getitem__(self, idx):
        try:
            return super().__getitem__(idx)
        except Exception as e:
            print(f'Failed to load the {idx}-th image due to {repr(e)}, getting {idx+1}-th data instead')
            return self.__getitem__(idx + 1)
