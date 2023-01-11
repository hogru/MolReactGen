from importlib import reload

import fcd_torch.utils

reload(fcd_torch.utils)

import gc
import os
import warnings

import numpy as np
import torch
from fcd_torch.utils import (  # todevice,
    SmilesDataset,
    calculate_frechet_distance,
    load_imported_model,
)
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from torch.utils.data import DataLoader


class FCD:
    """
    Computes Frechet ChemNet Distance on PyTorch.
    * You can precalculate mean and sigma for further usage,
      e.g. if you use the statistics from the same dataset
      multiple times.
    * Supports GPU and selection of GPU index
    * Multithread SMILES parsing

    Example 1:
        fcd = FCD(device='cuda:0', n_jobs=8)
        smiles_list = ['CCC', 'CCNC']
        fcd(smiles_list, smiles_list)

    Example 2:
        fcd = FCD(device='cuda:0', n_jobs=8)
        smiles_list = ['CCC', 'CCNC']
        pgen = fcd.precalc(smiles_list)
        fcd(smiles_list, pgen=pgen)
    """

    def __init__(
        self,
        device="cpu",
        n_jobs=1,
        batch_size=512,
        model_path=None,
        canonize=True,
        pbar=False,
    ):
        """
        Loads ChemNet on device
        params:
            device: cpu for CPU, cuda:0 for GPU 0, etc.
            n_jobs: number of workers to parse SMILES
            batch_size: batch size for processing SMILES
            model_path: path to ChemNet_v0.13_pretrained.pt
        """
        if model_path is None:
            model_dir = os.path.split(__file__)[0]
            model_path = os.path.join(model_dir, "ChemNet_v0.13_pretrained.pt")

        self.device = device
        self.n_jobs = n_jobs if n_jobs != 1 else 0
        self.batch_size = batch_size
        keras_config = torch.load(model_path)
        self.model = load_imported_model(keras_config)
        self.model.eval()
        self.canonize = canonize
        self.pbar = pbar

    def get_predictions(self, smiles_list):
        if len(smiles_list) == 0:
            return np.zeros((0, 512))
        dataloader = DataLoader(
            SmilesDataset(smiles_list, canonize=self.canonize),
            batch_size=self.batch_size,
            num_workers=self.n_jobs,
        )
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(elapsed_when_finished=True),
            refresh_per_second=2,
            disable=not self.pbar,
        ) as progress:

            task = progress.add_task(
                "Calculate activations...", total=len(dataloader)  # size=0
            )

            """
            with todevice(self.model, self.device), torch.no_grad():
                chemnet_activations = []
                for batch in dataloader:
                    chemnet_activations.append(
                        self.model(
                            batch.transpose(1, 2).float().to(self.device)
                        ).to('cpu').detach().numpy()
                    )
            """

            # This does not help
            self.model.to(self.device)
            self.model.eval()
            chemnet_activations = []
            with torch.inference_mode():
                for batch in dataloader:
                    batch = batch.transpose(1, 2).float().to(self.device)
                    activations = self.model(batch)
                    activations = activations.to("cpu").detach().numpy()
                    chemnet_activations.append(activations)
                    progress.update(task, advance=1)
                    gc.collect()
                    # del batch  # this does not help
                    # del activations

        return np.row_stack(chemnet_activations)

    def precalc(self, smiles_list):
        if len(smiles_list) < 2:
            warnings.warn(
                "Can't compute FCD for less than 2 molecules"
                "({} given)".format(len(smiles_list))
            )
            return {}
        chemnet_activations = self.get_predictions(smiles_list)
        mu = chemnet_activations.mean(0)
        sigma = np.cov(chemnet_activations.T)
        return {"mu": mu, "sigma": sigma}

    def metric(self, pref, pgen):
        if "mu" not in pref or "sigma" not in pgen:
            warnings.warn("Failed to compute FCD (check ref)")
            return np.nan
        if "mu" not in pgen or "sigma" not in pgen:
            warnings.warn("Failed to compute FCD (check gen)")
            return np.nan
        return calculate_frechet_distance(
            pref["mu"], pref["sigma"], pgen["mu"], pgen["sigma"]
        )

    def __call__(self, ref=None, gen=None, pref=None, pgen=None):
        assert (ref is None) != (pref is None), "specify ref xor pref"
        assert (gen is None) != (pgen is None), "specify gen xor pgen"
        if pref is None:
            pref = self.precalc(ref)
        if pgen is None:
            pgen = self.precalc(gen)
        return self.metric(pref, pgen)
