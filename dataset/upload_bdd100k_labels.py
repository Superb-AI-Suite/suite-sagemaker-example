import os
import json
import glob
import random
import tqdm
import spb.sdk
import multiprocessing

def get_spb_data(client, page_size=10):
    num_data = client.get_num_data()
    num_page = (num_data + page_size - 1) // page_size
    def generator():
        for page_idx in range(num_page):
            for data_handler in client.get_data_page(page_idx=page_idx, page_size=page_size):
                yield data_handler
    return {'iterable': generator(), 'total': num_data}

def get_bdd100k_labels(path):
    labels = None
    with open(path, 'r') as f:
        labels = json.load(f)
    labels_by_image_name = {}
    for label in labels:
        labels_by_image_name[label['name']] = label
    return labels_by_image_name

def upload_label(data_list, train_labels, val_labels, start_num, end_num):
    for data_handler in data_list[start_num:end_num]:
        data_key = data_handler.get_key()

        annotations = None

        if data_key in train_labels:
            print("train", data_key)
            annotations = train_labels[data_key]['labels']

        if data_key in val_labels:
            print("val", data_key)
            annotations = train_labels[data_key]['labels']

        if annotations is None:
            print(f"Not Found {data_key}")
            continue

        for annotation in annotations:
            try:
                if "box2d" in annotation:
                    box2d = annotation["box2d"]

                    if annotation['category'] == 'person':
                        annotation['category'] = 'pedestrian'

                    if annotation['category'] == 'bike':
                        annotation['category'] = 'bicycle'

                    if annotation['category'] == 'motor':
                        annotation['category'] = 'motorcycle'

                    suite_label = {
                        'class_name': annotation['category'],
                        'annotation': {
                            'coord': {
                                'x': box2d["x1"],
                                'y': box2d["y1"],
                                'width': box2d["x2"] - box2d["x1"],
                                'height': box2d["y2"] - box2d["y1"]
                            }
                        }
                    }

                    data_handler.add_object_label(suite_label['class_name'], suite_label['annotation'])
            except Exception as e:
                print(annotation)
        data_handler.update_data()

def main():
    project_name = os.environ.get('PROJECT_NAME')
    team_name = os.environ.get('TEAM_NAME')
    access_key = os.environ.get('ACCESS_KEY')

    bdd100k_train_labels_path = os.environ.get('BDD100K_TRAIN_LABELS_PATH')
    bdd100k_val_labels_path = os.environ.get('BDD100K_VAL_LABELS_PATH')

    client = spb.sdk.Client(project_name=project_name, team_name=team_name, access_key=access_key)

    print('Project Name: {}'.format(client.get_project_name()))
    print('Total number of data: {}'.format(client.get_num_data()))

    data_list = []

    for data_handler in tqdm.tqdm(**get_spb_data(client)):
        data_key = data_handler.get_key()
        data_list.append(data_handler)

    train_labels = get_bdd100k_labels(bdd100k_train_labels_path)
    val_labels = get_bdd100k_labels(bdd100k_val_labels_path)

    jobs = []
    data_num = len(data_list)
    divide_num = data_num / 5
    for i in range(5):
        start_num = int(i * divide_num)
        if i == 4:
            end_num = int(data_num)
        else:
            end_num = int((i+1) * divide_num)
        p = multiprocessing.Process(target=upload_label, args=(data_list, train_labels, val_labels, start_num, end_num))
        jobs.append(p)
        p.start()
    for p in jobs:
        p.join()



if __name__ == "__main__":
	main()