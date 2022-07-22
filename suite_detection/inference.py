import os, json, time
from typing import List, Dict, Callable, Any
from io import BytesIO

import torch
from torch import nn, Tensor
from torchvision.transforms import functional as F
import cv2

import spb.sdk
from suite_detection.models import build_object_detector
from suite_detection.utils import call_with_retry, open_image


def build_model(model_dir):
    model_info_path = os.path.join(model_dir, 'model.json')
    model_info = json.load(open(model_info_path))
    categories = { cat['id']: cat for cat in model_info['categories'] }

    model_path = os.path.join(model_dir, 'model.pth')
    model = build_object_detector(len(categories), model_path=model_path)
    model.categories = categories
    return model


def load_data(
        client: spb.sdk.Client,
        label_ids: List[str],
    ) -> List[spb.sdk.DataHandle]:

    datas = [
        call_with_retry(client.get_data, id=label_id)
        for label_id in label_ids
    ]
    return datas


class ToTensor(nn.Module):
    def forward(self, image: Tensor) -> Tensor:
        image = F.pil_to_tensor(image)
        image = F.convert_image_dtype(image)
        return image


def preprocess_image_url(
        image_url: str,
        transforms: nn.Module = ToTensor(),
    ) -> Tensor:

    return transforms(call_with_retry(open_image, image_url=image_url))


def preprocess_bin_image(
        bin_image: BytesIO,
        transforms: nn.Module = ToTensor(),
    ) -> Tensor:

    return transforms(open_image(bin_image=bin_image))


def preprocess_datas(
        datas: List[spb.sdk.DataHandle],
        transforms: nn.Module = ToTensor(),
    ) -> List[Tensor]:

    images = [
        preprocess_image_url(image_url=data.get_image_url())
        for data in datas
    ]
    return images


def inference(
        model: nn.Module,
        images: List[Tensor],
    ) -> List[Dict[str, Tensor]]:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    with torch.no_grad():
        images = [img.to('cuda:0') for img in images]

        model_time = time.time()
        outputs = model(images)
        outputs = [{ k: v.to('cpu') for k, v in t.items() } for t in outputs]
        model_time = time.time() - model_time

    print(f'Model time: {model_time:.3f}s')
    return outputs


def box2anno(box: Tensor) -> Dict[str, Any]:
    x1, y1, x2, y2 = box.tolist()
    return {
        'coord': {
            'x': x1, 'y': y1, 'width': x2 - x1, 'height': y2 - y1,
        },
    }


def mask2anno(mask: Tensor) -> Dict[str, Any]:
    mask = (mask > 0.5).numpy().astype('uint8')[0]
    polygons, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    polygons = [polygon.reshape((-1, 2)).tolist() for polygon in polygons]
    polygons = [polygon for polygon in polygons if len(polygon) >= 3]
    polygons = [polygon + [polygon[0]] for polygon in polygons]

    if len(polygons) == 0:
        return {}
    return {
        'multiple': True,
        'coord': {
            'points': [
                [[{ 'x': x, 'y': y } for (x, y) in polygon]]
                for polygon in polygons
            ],
        },
    }


def keypoint2anno(keypoint: Tensor) -> Dict[str, Any]:
    return {
        'coord': {
            'points': [
                { 'x': x, 'y': y, 'state': { 'visible': v } }
                for x, y, v in keypoint.tolist()
            ],
        },
    }


CONVERTER: Dict[str, Callable] = {
    'boxes': box2anno,
    'masks': mask2anno,
    'keypoints': keypoint2anno,
}


def to_suite_objects(
        output: Dict[str, Tensor],
        categories: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

    objs = []

    for i, label in enumerate(output['labels'].tolist()):
        cat = categories[label]
        if output['scores'][i] < cat['score_thres']:
            continue

        name, anno_type = cat['name'], cat['type']
        if anno_type not in output:
            raise ValueError(f'Missing {anno_type} output for Category {name}')

        anno = CONVERTER[anno_type](output[anno_type][i])
        if not anno:
            continue

        objs.append({
            'class_name': name,
            'annotation': anno,
            'properties': [],
        })

    return objs


def postprocess(
        outputs: List[Dict[str, Tensor]],
        datas: List[spb.sdk.DataHandle],
        categories: Dict[str, Dict[str, Any]],
    ) -> str:

    for data, output in zip(datas, outputs):
        objs = to_suite_objects(output, categories)
        data.set_object_labels(objs)
        data.update_data()
    result = 'OK'
    return result
