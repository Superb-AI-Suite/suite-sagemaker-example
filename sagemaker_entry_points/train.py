import os, json, argparse
from pathlib import Path

import torch
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader

from suite_detection.models import build_object_detector
from suite_detection.dataset import (
    SuiteDataset,
    SuiteCocoDataset,
    collate_fn,
)
from suite_detection.official import (
    group_by_aspect_ratio as G,
    engine as E,
)


def train(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_dataset = SuiteDataset(
        team_name=args.team_name,
        access_key=args.access_key,
        project_name=args.project_name,
        export_name=args.train_export_name,
        caching_image=args.caching_image,
        train=True,
    )
    test_dataset = SuiteCocoDataset(
        team_name=args.team_name,
        access_key=args.access_key,
        project_name=args.project_name,
        export_name=args.test_export_name,
        caching_image=args.caching_image,
        train=False,
        num_init_workers=args.workers,
    )

    train_loader = DataLoader(
        train_dataset, num_workers=args.workers,
        batch_sampler=G.GroupedBatchSampler(
            RandomSampler(train_dataset),
            G.create_aspect_ratio_groups(train_dataset, k=3),
            args.batch_size,
        ),
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset, num_workers=args.workers,
        sampler=SequentialSampler(test_dataset), batch_size=1,
        collate_fn=collate_fn,
    )

    model = build_object_detector(len(train_dataset.categories))
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=0.9, weight_decay=1e-4,
    )
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, gamma=0.1,
        milestones=[int(args.epochs * 0.7), int(args.epochs * 0.9)],
    )
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    print_freq = 10

    for epoch in range(args.epochs):
        E.train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq, scaler)
        lr_scheduler.step()
        E.evaluate(model, test_loader, device)

    if args.model_dir:
        os.makedirs(args.model_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))
        save_model_info(train_dataset.categories, os.path.join(args.model_dir, 'model.json'))
        save_inference_code(args.model_dir)


def save_model_info(categories, model_info_path):
    type_converter = {
        'box': 'boxes',
        'polygon': 'masks',
        'keypoint': 'keypoints',
    }
    model_info = {
        'categories': [
            {
                'id': cat['id'],
                'name': cat['name'],
                'type': type_converter[cat['type']],
                'score_thres': 0.5,
            }
            for cat in categories
        ],
    }
    json_content = json.dumps(model_info, indent=4)
    with open(model_info_path, 'w') as f:
        f.write(json_content)

    print(model_info_path)
    print(json_content)


def save_inference_code(model_dir):
    inference_srcs = [
        'single_inference.py',
        'batch_inference.py',
        'requirements.txt',
    ]

    src_dir = Path(__file__).resolve().parent
    dest_dir = os.path.join(model_dir, 'code')
    os.makedirs(dest_dir, exist_ok=True)
    for f in inference_srcs:
        f_src, f_dest = f if isinstance(f, tuple) else (f, f)
        os.system(f'cp -f \"{src_dir}/{f_src}\" \"{dest_dir}/{f_dest}\"')

    print(dest_dir)
    print(os.system(f'ls -la {dest_dir}'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', ''))
    parser.add_argument('--team-name', type=str)
    parser.add_argument('--access-key', type=str)
    parser.add_argument('--project-name', type=str)
    parser.add_argument('--train-export-name', type=str)
    parser.add_argument('--test-export-name', type=str)
    parser.add_argument("--caching-image", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--amp", action="store_true")

    args = parser.parse_args()
    print(args)

    train(args)
