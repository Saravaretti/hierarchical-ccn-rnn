# Saves h_t, s_t, r_t from CORnet model using 5 random neuron subsets
import h5py, os, argparse
import numpy as np
import fire
import torch
import torch.nn as nn
import torchvision
from PIL import Image
import cornet

torch.backends.cudnn.benchmark = True

normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

parser = argparse.ArgumentParser(description='ImageNet Feature Extraction')
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--data_path', required=True)
parser.add_argument('-o', '--output_path', default=None)
parser.add_argument('--model', choices=['Z', 'R', 'RT', 'S', 'RT1'], default='Z')
parser.add_argument('--times', default=0, type=int)
parser.add_argument('--ngpus', default=0, type=int)
parser.add_argument('-j', '--workers', default=4, type=int)
parser.add_argument('--epochs', default=20, type=int)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--lr', '--learning_rate', default=.1, type=float)
parser.add_argument('--step_size', default=10, type=int)
parser.add_argument('--momentum', default=.9, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--layer', choices=['V1', 'V2', 'V4', 'IT'], default='V1')

FLAGS, FIRE_FLAGS = parser.parse_known_args()
sim_index = FLAGS.seed

def get_model(pretrained=False, **kwargs):
    model = getattr(cornet, f'cornet_{FLAGS.model.lower()}')
    model = model(pretrained=pretrained, map_location='cpu', **kwargs)
    return model.module

def test(layer=FLAGS.layer, sublayer='output', imsize=224):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(720),
        torchvision.transforms.Resize((imsize, imsize)),
        normalize,
    ])

    with h5py.File(os.path.join(FLAGS.data_path, 'manual slow 2.hdf5'), 'r') as f:
        data = f['im_matrix'][()]
    data = np.transpose(data, (2, 1, 0))
    data = np.stack((data, data, data), axis=1)
    data = data.astype(np.float32) / 255
    data = torch.from_numpy(data)
    print("Before transform:", data.shape)  # (T, 3, H, W)
    data = transform(data)
    print("After transform:", data.shape)
    print("min/max values:", data.min().item(), data.max().item())
    
    data = data.unsqueeze(0)

    if FLAGS.times == 0:
        FLAGS.times = data.shape[1]

    model = get_model(pretrained=True, times=FLAGS.times)
    model.eval()

    subset_seeds = [5, 6, 7, 8, 9]
    n_subsets = len(subset_seeds)

    _model_feats = [[] for _ in range(n_subsets)]
    _model_pre_relu = [[] for _ in range(n_subsets)]
    _model_adaptive_state = [[] for _ in range(n_subsets)]

    def _store_feats(layer, inp, output):
        output = output.cpu().numpy()
        output = np.reshape(output, (len(output), -1))
        for i, seed in enumerate(subset_seeds):
            np.random.seed(seed)
            indices = np.random.choice(output.shape[1], 200, replace=False)
            subset = output[:, indices]
            _model_feats[i].append(subset)

    def _store_pre_relu(layer, inp, output):
        pre_relu = layer._last_pre_relu_state
        if isinstance(pre_relu, torch.Tensor):
            pre_relu = pre_relu.cpu().numpy()
            pre_relu = np.reshape(pre_relu, (len(pre_relu), -1))
            for i, seed in enumerate(subset_seeds):
                np.random.seed(seed)
                indices = np.random.choice(pre_relu.shape[1], 200, replace=False)
                subset = pre_relu[:, indices]
                _model_pre_relu[i].append(subset)

    def _store_adaptive_state(layer, inp, output):
        adaptive = layer._last_adaptive_state
        if isinstance(adaptive, torch.Tensor):
            adaptive = adaptive.cpu().numpy()
            adaptive = np.reshape(adaptive, (len(adaptive), -1))
            for i, seed in enumerate(subset_seeds):
                np.random.seed(seed)
                indices = np.random.choice(adaptive.shape[1], 200, replace=False)
                subset = adaptive[:, indices]
                _model_adaptive_state[i].append(subset)

    try:
        m = model.module
    except:
        m = model
    model_layer = getattr(m, layer)

    model_layer.output.register_forward_hook(_store_feats)
    model_layer.register_forward_hook(_store_pre_relu)
    model_layer.register_forward_hook(_store_adaptive_state)

    with torch.no_grad():
        model(data)
    print(f"[DEBUG] FLAGS.output_path = {FLAGS.output_path}")
    if FLAGS.output_path is not None:
        base = f'CORnet-{FLAGS.model}_{layer}_manual slow 2_sim{sim_index}_pretrained'
        for i in range(n_subsets):
           
            suffix = f'_subset{i+1}'            
            np.save(os.path.join(FLAGS.output_path, f'{base}{suffix}_r_t.npy'),
                    np.concatenate(_model_feats[i], axis=0))
            np.save(os.path.join(FLAGS.output_path, f'{base}{suffix}_h_t.npy'),
                    np.concatenate(_model_pre_relu[i], axis=0))
            np.save(os.path.join(FLAGS.output_path, f'{base}{suffix}_s_t.npy'),
                    np.concatenate(_model_adaptive_state[i], axis=0))

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        _, pred = output.topk(max(topk), dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [correct[:k].sum().item() for k in topk]

if __name__ == '__main__':
    fire.Fire(command=FIRE_FLAGS)
    test(layer=FLAGS.layer)
