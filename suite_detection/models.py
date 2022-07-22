import torch
import torchvision.models.detection as M


def build_object_detector(num_classes, model_path=None):
    # Based on COCO-pretrained model
    model = M.maskrcnn_resnet50_fpn(pretrained=True)

    # Replace COCO-pretrained box head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = M.faster_rcnn.FastRCNNPredictor(in_features, num_classes+1)

    # Replace COCO-pretrained mask head
    in_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = M.mask_rcnn.MaskRCNNPredictor(in_features, hidden_layer, num_classes+1)

    # Load trained weights for inference
    if model_path is not None:
        content = torch.load(open(model_path, 'rb'), map_location='cpu')
        model.load_state_dict(content)
    return model
