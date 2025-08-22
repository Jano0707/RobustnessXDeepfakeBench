"""
eval pretained model.
"""
import os
import numpy as np
from os.path import join
import cv2
import random
import datetime
import time
import yaml
import pickle
from tqdm import tqdm
from copy import deepcopy
from PIL import Image as pil_image
from metrics.utils import get_test_metrics
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from dataset.ff_blend import FFBlendDataset
from dataset.fwa_blend import FWABlendDataset
from dataset.pair_dataset import pairDataset

from trainer.trainer import Trainer
from detectors import DETECTOR
from metrics.base_metrics_class import Recorder
from collections import defaultdict

import argparse
from logger import create_logger

# NEU:
import csv
import json

parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str, 
                    default='./training/config/detector/xception.yaml',
                    help='path to detector YAML file')
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument('--weights_path', type=str, 
                    default='./training/weights/xception_best.pth')
#parser.add_argument("--lmdb", action='store_true', default=False)

# NEU: argparse-Erweiterung für die Nutzung der Analyse-Tools
parser.add_argument('--metrics_outdir', type=str, default='analysis_outputs/metrics',
                    help='Wohin die JSON-Metriken + y_true/y_score (+feat) geschrieben werden')
parser.add_argument('--tag', type=str, default='baseline',
                    help='Label für diesen Run (z.B. baseline, grayscale, jpeg_comp, face_smoothing, text_overlay)')
                    
# können weg
parser.add_argument('--save_predictions', action='store_true', default=True,
                    help='Speichere y_true/y_score je Datensatz')
parser.add_argument('--save_features', action='store_true', default=True,
                    help='Speichere Feature-Vektoren je Datensatz (falls vom Modell geliefert)')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])


def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        # update the config dictionary with the specific testing dataset
        config = config.copy()  # create a copy of config to avoid altering the original one
        config['test_dataset'] = test_name  # specify the current test dataset
        test_set = DeepfakeAbstractBaseDataset(
                config=config,
                mode='test', 
            )
        test_data_loader = \
            torch.utils.data.DataLoader(
                dataset=test_set, 
                batch_size=config['test_batchSize'],
                shuffle=False, 
                num_workers=int(config['workers']),
                collate_fn=test_set.collate_fn,
                drop_last=False
            )
        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring


def test_one_dataset(model, data_loader):
    prediction_lists = []
    feature_lists = []
    label_lists = []
    for i, data_dict in tqdm(enumerate(data_loader), total=len(data_loader)):
        # get data
        data, label, mask, landmark = \
        data_dict['image'], data_dict['label'], data_dict['mask'], data_dict['landmark']
        label = torch.where(data_dict['label'] != 0, 1, 0)
        # move data to GPU
        data_dict['image'], data_dict['label'] = data.to(device), label.to(device)
        if mask is not None:
            data_dict['mask'] = mask.to(device)
        if landmark is not None:
            data_dict['landmark'] = landmark.to(device)

        # NEU: bricht nicht mehr ab, wenn feat fehlt
        # forward
        predictions = inference(model, data_dict)

        # y_true / y_score (prob)
        label_lists += list(data_dict['label'].cpu().detach().numpy())
        prob = predictions['prob'] if isinstance(predictions, dict) else predictions
        prediction_lists += list(prob.cpu().detach().numpy())

        # Features
        #feat = predictions.get('feat', None) if isinstance(predictions, dict) else None
        #if feat is not None:
         #   feature_lists += list(feat.cpu().detach().numpy())
        feat_vec = None
        if isinstance(predictions, dict):
            # Bevorzugt: vorletzter Vektor (Embedding) oder logits
            if 'emb' in predictions and predictions['emb'] is not None:
                f = predictions['emb']
            elif 'feat' in predictions and predictions['feat'] is not None:
                f = predictions['feat']
            elif 'logits' in predictions and predictions['logits'] is not None:
                f = predictions['logits']
            else:
                f = None

            if f is not None:
                # auf 2D bringen
                if f.dim() == 4:  # (B, C, H, W) -> global avg pool
                    f = torch.nn.functional.adaptive_avg_pool2d(f, 1).flatten(1)
                elif f.dim() > 2:
                    f = f.flatten(1)
                feat_vec = f

        if feat_vec is not None:
            feature_lists += list(feat_vec.cpu().detach().numpy())
    y_pred_np = np.array(prediction_lists)
    y_true_np = np.array(label_lists)
    feat_np   = np.array(feature_lists) if len(feature_lists) else None
    return y_pred_np, y_true_np, feat_np
    
def test_epoch(model, test_data_loaders):
    # set model to eval mode
    model.eval()

    # define test recorder
    metrics_all_datasets = {}
    
    # NEU: Detector-Name aus Config
    detector_name = getattr(model, 'config', {}).get('model_name', 'detector')
    
    # testing for all test data
    keys = test_data_loaders.keys()
    for key in keys:
        data_dict = test_data_loaders[key].dataset.data_dict        
        # compute loss for each dataset
        predictions_nps, label_nps,feat_nps = test_one_dataset(model, test_data_loaders[key])
        
        # compute metric for each dataset
        metric_one_dataset = get_test_metrics(y_pred=predictions_nps, y_true=label_nps,
                                              img_names=data_dict['image'])
        metrics_all_datasets[key] = metric_one_dataset
        
        # NEU: Testbild-Zähler
        num_used = int(len(label_nps))
        num_total = int(len(test_data_loaders[key].dataset.image_list))
        print(f"[{key}] Bilder im Test: verwendet={num_used} / gesamt={num_total}")
        
        # NEU: Speichern für Analysis
        os.makedirs(args.metrics_outdir, exist_ok=True)
        y_true_path, y_score_path, feat_path = _pred_paths(args.metrics_outdir, detector_name, key, args.tag)

        # NEU: y_true (1D)
        y_true = np.asarray(label_nps, dtype=np.int64).reshape(-1)
        np.save(y_true_path, y_true)

        # NEU: y_score robust (positiv-Klasse = 1 falls 2 Spalten)
        pred = np.asarray(predictions_nps)
        if pred.ndim == 1:
            y_score = pred.astype(np.float32)
        else:
            y_score = (pred[:, 1] if pred.shape[1] > 1 else pred[:, 0]).astype(np.float32)
        np.save(y_score_path, y_score)

        # NEU: Features (nur wenn vorhanden)
        saved_feat_path = None
        if feat_nps is not None and len(feat_nps):
            feats = np.asarray(feat_nps)
            np.save(feat_path, feats)
            saved_feat_path = feat_path

        # NEU: JSON mit Pfaden
        _dump_metrics_json(
            detector_name, key, metric_one_dataset, num_used, num_total,
            args.metrics_outdir, args.tag, mode="frame",
            y_true_path=y_true_path, y_score_path=y_score_path, feat_path=saved_feat_path
        )

        # (Optional) Logging der Metriken
        tqdm.write(f"dataset: {key}")
        for k, v in metric_one_dataset.items():
            tqdm.write(f"{k}: {v}")

    return metrics_all_datasets

@torch.no_grad()
def inference(model, data_dict):
    predictions = model(data_dict, inference=True)
    return predictions
    
# NEU
def _make_records_dir(detector_name, analysis_root, exp='exp1'):
    rec_dir = os.path.join(analysis_root, f'{exp}_records_{detector_name}')
    os.makedirs(rec_dir, exist_ok=True)
    return rec_dir

# NEU
def _pred_paths(outdir, detector_name, dataset_name, tag):
    base = f"{detector_name}__{dataset_name}__{tag}"
    y_true_path  = os.path.join(outdir, base + "_y_true.npy")
    y_score_path = os.path.join(outdir, base + "_y_score.npy")
    feat_path    = os.path.join(outdir, base + "_feat.npy")
    return y_true_path, y_score_path, feat_path

# NEU  
def _dump_metrics_json(detector_name, dataset_name, metrics, used, total, outdir, tag, mode="frame", y_true_path=None, y_score_path=None, feat_path=None):
    os.makedirs(outdir, exist_ok=True)
    payload = {
        "detector": detector_name,
        "dataset": dataset_name,
        "tag": tag,                 # z.B. "baseline" oder "grayscale"
        "mode": mode,               # hier "frame"
        "count_total": int(total),
        "count_used": int(used),
        "metrics": {
            "auc": float(metrics.get('auc', 0.0)),
            "acc": float(metrics.get('acc', 0.0)),
            "eer": float(metrics.get('eer', 0.0)),
            "ap":  float(metrics.get('ap',  0.0)),
        },
        "y_true_path": y_true_path,
        "y_score_path": y_score_path,
        "feat_path": feat_path,
    }
    out_path = os.path.join(outdir, f"{detector_name}__{dataset_name}__{tag}.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    print(f"[Metriken] gespeichert: {out_path}")



def main():
    # parse options and load config
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open('./training/config/test_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    config.update(config2)
    if 'label_dict' in config:
        config2['label_dict']=config['label_dict']
    weights_path = None
    # If arguments are provided, they will overwrite the yaml settings
    if args.test_dataset:
        config['test_dataset'] = args.test_dataset
    if args.weights_path:
        config['weights_path'] = args.weights_path
        weights_path = args.weights_path
    
    # init seed
    init_seed(config)

    # set cudnn benchmark if needed
    if config['cudnn']:
        cudnn.benchmark = True

    # prepare the testing data loader
    test_data_loaders = prepare_testing_data(config)
    
    # prepare the model (detector)
    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device)
    epoch = 0
    if weights_path:
        try:
            epoch = int(weights_path.split('/')[-1].split('.')[0].split('_')[2])
        except:
            epoch = 0
        ckpt = torch.load(weights_path, map_location=device)
        model.load_state_dict(ckpt, strict=True)
        print('===> Load checkpoint done!')
    else:
        print('Fail to load the pre-trained weights')
    
    # start testing
    best_metric = test_epoch(model, test_data_loaders)
    print('===> Test Done!')

if __name__ == '__main__':
    main()
