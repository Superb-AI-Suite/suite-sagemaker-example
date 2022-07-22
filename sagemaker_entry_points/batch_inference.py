import json
from io import BytesIO

import spb.sdk
from suite_detection.inference import (
    build_model,
    load_data,
    preprocess_datas,
    inference,
    postprocess,
)


def model_fn(model_dir):
    model = build_model(model_dir)
    return model


def input_fn(request_body, request_content_type):
    if request_content_type != 'application/json':
        print(f'Ignoring unsupported content type {request_content_type}')
    if isinstance(request_body, bytes):
        request_body = BytesIO(request_body).read()

    args = json.loads(request_body)
    client = spb.sdk.Client(**args['auth'])
    args = json.loads(request_body)
    datas = load_data(client, args['label_ids'])
    images = preprocess_datas(datas)
    input_data = {
        'datas': datas,
        'images': images,
    }
    return input_data


def predict_fn(input_data, model):
    outputs = inference(model, input_data['images'])
    prediction = {
        'outputs': outputs,
        'datas': input_data['datas'],
        'categories': model.categories,
    }
    return prediction


def output_fn(prediction, content_type):
    if content_type != 'application/json':
        print(f'Ignoring unsupported content type {content_type}')

    result = postprocess(**prediction)
    output = json.dumps({ 'result': result })
    return output
