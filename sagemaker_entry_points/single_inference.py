import json

from suite_detection.inference import (
    build_model,
    preprocess_image_url,
    preprocess_bin_image,
    inference,
    to_suite_objects,
)


def model_fn(model_dir):
    model = build_model(model_dir)
    return model


def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        image = preprocess_image_url(json.loads(request_body)['image_url'])
    elif request_content_type == 'application/x-image':
        image = preprocess_bin_image(request_body)
    else:
        raise ValueError(f'Unsupported content type: {request_content_type}')

    input_data = image
    return input_data


def predict_fn(input_data, model):
    outputs = inference(model, [input_data])
    prediction = {
        'output': outputs[0],
        'categories': model.categories,
    }
    return prediction


def output_fn(prediction, content_type):
    if content_type != 'application/json':
        raise ValueError(f'Unsupported content type: {content_type}')

    objs = to_suite_objects(**prediction)
    output = json.dumps({ 'objs': objs })
    return output
