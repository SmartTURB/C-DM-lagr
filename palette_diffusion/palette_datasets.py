from mpi4py import MPI
import numpy as np
import h5py
from torch.utils.data import DataLoader, Dataset


def load_palette_data(
    *,
    mask_mode,
    dataset_path,
    dataset_name,
    batch_size,
    deterministic=False,
):
    """
    Modification of guided_diffusion.turb_datasets.load_data with two 
    keys in kwargs: "mask" indicates the target region for imputation, 
    and "cond" represents the conditioned input.

    :param mask_mode: mode for generating the mask.
    See guided_diffusion.turb_datasets.load_data for other params.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    rng = np.random.default_rng(seed=rank)

    with h5py.File(dataset_path, 'r') as f:
        len_dataset = f[dataset_name].len()

    chunk_size = len_dataset // size
    start_idx  = rank * chunk_size

    dataset = PaletteDataset(
        mask_mode, rng, dataset_path, dataset_name, start_idx, chunk_size,
    )

    shuffle = False if deterministic else True
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1, drop_last=shuffle
    )  # set shuffle and drop_last to False for sampling

    while True:
        yield from loader


class PaletteDataset(Dataset):
    def __init__(
        self,
        mask_mode,
        rng,
        dataset_path,
        dataset_name,
        start_idx,
        chunk_size,
    ):
        super().__init__()
        self.mask_mode = mask_mode
        self.rng = rng
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.start_idx  = start_idx
        self.chunk_size = chunk_size

    def __len__(self):
        return self.chunk_size

    def __getitem__(self, idx):
        idx += self.start_idx

        with h5py.File(self.dataset_path, 'r') as f:
            data = f[self.dataset_name][idx].astype(np.float32)
            data = np.moveaxis(data, -1, 0)

        shape = data.shape
        mask = get_mask(shape, self.mask_mode).astype(np.float32)
        noise = self.rng.standard_normal(shape).astype(np.float32)

        out_dict = {}
        out_dict["mask"] = mask
        out_dict["cond"] = (1.0 - mask) * data + mask * noise

        return data, out_dict


def get_mask(size, mask_mode):
    mask = np.zeros(size)
    if mask_mode.startswith('center1d'):
        lg = int(mask_mode[8:])
        i0 = (size[1] - lg) // 2
        mask[:, i0:i0+lg] = 1.0
    elif mask_mode.startswith('random1d'):
        lg = int(mask_mode[8:])
        i0 = np.random.randint(size[1] - lg)
        mask[:, i0:i0+lg] = 1.0
    elif mask_mode.startswith('right1d'):
        lg = int(mask_mode[7:])
        mask[:, -lg:] = 1.0
    elif mask_mode.startswith('interp1d'):
        scale_factor = int(mask_mode[8:])
        mask[:, ::scale_factor] = 1.0
        mask = 1.0 - mask
    else:
        raise NotImplementedError()
    return mask
