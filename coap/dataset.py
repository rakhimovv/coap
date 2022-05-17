import glob
import logging
import os
import os.path as osp
from typing import Optional, List
from typing import Tuple

import numpy as np
import torch
from tqdm import tqdm
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig
from torch.utils import data
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from trimesh import Trimesh

from .coap_body_model import COAPBodyModel
from .tools.libmesh import check_mesh_contains
from .utils import get_world_size

__all__ = ['AmassDataset', 'build_datasets', 'build_loaders']

log = logging.getLogger(__name__)


class AmassDataset(data.Dataset):
    """ AMASS dataset class for occupancy training. """

    def __init__(self, dataset_folder, bm_path, split_file, sampling_config):
        """ Initialization of the the 3D shape dataset.

        Args:
            dataset_folder (str): where dataset is located
            bm_path (str): body model path
            split_file (str): contains list of file names to use
            sampling_config (DictConfig): how to sample points
        """
        # Attributes
        self.dataset_folder = dataset_folder
        self.bm_path = bm_path
        self.split_file = split_file

        # Sampling config
        self.points_uniform_ratio = sampling_config.get('points_uniform_ratio', 0.5)
        self.bbox_padding = sampling_config.get('bbox_padding', 0)
        self.points_padding = sampling_config.get('points_padding', 0.1)
        self.points_sigma = sampling_config.get('points_sigma', 0.01)
        self.n_points_posed = sampling_config.get('n_points_posed', 2048)  # fixme correct ?

        # Get all models
        self.data = self._load_data_files()

    def _load_data_files(self):
        # load SMPL datasets
        self.faces = np.load(osp.join(self.bm_path, 'neutral', 'model.npz'))['f']

        # list files
        data_list = []
        with open(self.split_file, 'r') as f:
            for _sequence in f:
                sequence = _sequence.strip()  # sequence in format dataset/subject/sequence
                sequence = sequence.replace('/', osp.sep)
                points_dir = osp.join(self.dataset_folder, sequence)
                data_files = sorted(glob.glob(osp.join(points_dir, '*.npz')))
                data_list.extend(data_files)

        # filtered_data_list = []
        # for f in tqdm(data_list):
        #     # assert os.path.exists(f), f"Missing: {f}"
        #     if os.path.exists(f):
        #         filtered_data_list.append(f)
        #     else:
        #         print(f"Missing: {f}")
        filtered_data_list = data_list

        return filtered_data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """ Returns an item of the dataset.

        Args:
            idx (int): ID of datasets point
        """
        data_path = self.data[idx]
        np_data = np.load(data_path)

        betas = torch.from_numpy(np_data['betas'].astype(np.float32))
        pose_body = torch.from_numpy(np_data['pose_body'].astype(np.float32))
        pose_hand = torch.from_numpy(np_data['pose_hand'].astype(np.float32))
        gender = np_data['gender'].item()

        bm_path = os.path.join(self.bm_path, gender, 'model.npz')
        coap_body_model = COAPBodyModel(bm_path=bm_path, num_betas=betas.shape[0], init_modules=False)
        coap_body_model.set_parameters(1, betas[None], pose_body[None], pose_hand[None])
        coap_body_model.forward_parametric_model()

        to_ret = {
            'backward_transform': coap_body_model.backward_transforms[0][:22],
            'surface_part_posed_points': coap_body_model.sample_surface_points()[0],
        }

        to_ret.update(self.sample_points(
            Trimesh(coap_body_model.posed_vert[0].numpy(), self.faces),
            self.n_points_posed,
            compute_occupancy=True))

        return to_ret

    def sample_points(self, mesh, n_points, prefix='', compute_occupancy=False):
        # Get extents of model.
        bb_min = np.min(mesh.vertices, axis=0)
        bb_max = np.max(mesh.vertices, axis=0)
        total_size = (bb_max - bb_min).max()

        # Scales all dimensions equally.
        scale = total_size / (1 - self.bbox_padding)
        loc = np.array([(bb_min[0] + bb_max[0]) / 2.,
                        (bb_min[1] + bb_max[1]) / 2.,
                        (bb_min[2] + bb_max[2]) / 2.], dtype=np.float32)

        n_points_uniform = int(n_points * self.points_uniform_ratio)
        n_points_surface = n_points - n_points_uniform

        box_size = 1 + self.points_padding
        points_uniform = np.random.rand(n_points_uniform, 3)
        points_uniform = box_size * (points_uniform - 0.5)
        # Scale points in (padded) unit box back to the original space
        points_uniform *= scale
        points_uniform += loc
        # Sample points around posed-mesh surface
        n_points_surface_cloth = n_points_surface
        points_surface = mesh.sample(n_points_surface_cloth)

        points_surface = points_surface[:n_points_surface_cloth]
        points_surface += np.random.normal(scale=self.points_sigma, size=points_surface.shape)

        # Check occupancy values for sampled points
        query_points = np.vstack([points_uniform, points_surface]).astype(np.float32)

        to_ret = {
            f'{prefix}points': query_points,
            f'{prefix}loc': loc,
            f'{prefix}scale': np.asarray(scale),
        }
        if compute_occupancy:
            to_ret[f'{prefix}occ'] = check_mesh_contains(mesh, query_points).astype(np.float32)

        return to_ret


# todo move things below to system.py

def build_datasets(cfg: DictConfig, stage: str) -> List[Tuple[str, DictConfig, Dataset]]:
    assert stage in ['train', 'val', 'test']
    if stage not in cfg.datasets:
        return []

    datasets_params = cfg.datasets[stage]
    if datasets_params is None:
        return []

    assert isinstance(datasets_params, ListConfig)

    datasets = []

    for dataset_param in datasets_params:
        assert dataset_param.dataset_name not in [d[0] for d in datasets]
        dataset = instantiate(dataset_param.dataset_class)
        log.info(f"Dataset ({dataset_param.dataset_name}) in stage={stage} contains {len(dataset)} elements")
        datasets.append((dataset_param.dataset_name, dataset_param, dataset))

    return datasets


def build_loaders(cfg: DictConfig, datasets: List[Dataset], stage: str) -> Optional[List[DataLoader]]:
    if len(datasets) == 0:
        return None

    assert stage in ['val', 'test'] or (
            stage == 'train' and len(datasets) == 1), f"stage: {stage}, len(datasets)={len(datasets)}"

    batch_size = get_batch_size(cfg.dataloader.total_batch_size)
    num_workers = cfg.dataloader.num_workers
    pin_memory = cfg.dataloader.pin_memory

    shuffle = stage == 'train'
    drop_last = stage == 'train'

    data_loaders = []
    for dataset in datasets:
        dataloader_kwargs = {}
        data_loaders.append(DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            shuffle=shuffle,
            **dataloader_kwargs
        ))

    return data_loaders


def get_batch_size(total_batch_size):
    world_size = get_world_size()
    assert (total_batch_size > 0 and total_batch_size % world_size == 0), \
        f"Total batch size ({total_batch_size}) must be divisible by the number of gpus ({world_size})."
    batch_size = total_batch_size // world_size
    return batch_size
