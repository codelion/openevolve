from typing import Optional, Any

import json
from pathlib import Path

import numpy as np
import h5py
import datasets
from huggingface_hub import snapshot_download

from .dataclasses import Equation, Problem

import warnings

REPO_ID = "nnheui/llm-srbench"

def _download(repo_id):
    return snapshot_download(repo_id=repo_id, repo_type="dataset")

class TransformedFeynmanDataModule:
    def __init__(self):
        self._dataset_dir = None
        self._dataset_identifier = 'lsr_transform'
    
    def setup(self):
        self._dataset_dir = Path(_download(repo_id=REPO_ID))
        ds = datasets.load_dataset(REPO_ID)['lsr_transform']
        sample_h5file_path = self._dataset_dir / "lsr_bench_data.hdf5"
        self.problems = []
        with h5py.File(sample_h5file_path, "r") as sample_file:
            for e in ds:
                samples = {k:v[...].astype(np.float64) for k,v in sample_file[f'/lsr_transform/{e["name"]}'].items()}
                self.problems.append(Problem(dataset_identifier=self._dataset_identifier,
                                        equation_idx = e['name'],
                                        gt_equation=Equation(
                                            symbols=e['symbols'],
                                            symbol_descs=e['symbol_descs'],
                                            symbol_properties=e['symbol_properties'],
                                            expression=e['expression'],
                                        ),
                                        samples=samples)
                )
        self.name2id = {p.equation_idx: i for i,p in enumerate(self.problems)}

    @property
    def name(self):
        return "LSR_Transform"

class SynProblem(Problem):
    @property
    def train_samples(self):
        return self.samples['train_data']
    
    @property
    def test_samples(self):
        return self.samples['id_test_data']
    
    @property
    def ood_test_samples(self):
        return self.samples['ood_test_data']

class BaseSynthDataModule:
    def __init__(self, dataset_identifier, short_dataset_identifier, root, default_symbols = None, default_symbol_descs=None):
        self._dataset_dir = Path(root)
        self._dataset_identifier = dataset_identifier
        self._short_dataset_identifier = short_dataset_identifier
        self._default_symbols = default_symbols
        self._default_symbol_descs = default_symbol_descs
    
    def setup(self):
        self._dataset_dir = Path(_download(repo_id=REPO_ID))
        ds = datasets.load_dataset(REPO_ID)[f'lsr_synth_{self._dataset_identifier}']
        sample_h5file_path = self._dataset_dir / "lsr_bench_data.hdf5"
        self.problems = []
        with h5py.File(sample_h5file_path, "r") as sample_file:
            for e in ds:
                samples = {k:v[...].astype(np.float64) for k,v in sample_file[f'/lsr_synth/{self._dataset_identifier}/{e["name"]}'].items()}
                self.problems.append(Problem(dataset_identifier=self._dataset_identifier,
                                        equation_idx = e['name'],
                                        gt_equation=Equation(
                                            symbols=e['symbols'],
                                            symbol_descs=e['symbol_descs'],
                                            symbol_properties=e['symbol_properties'],
                                            expression=e['expression'],
                                        ),
                                        samples=samples)
                )
        self.name2id = {p.equation_idx: i for i,p in enumerate(self.problems)}

    
        self.name2id = {p.equation_idx: i for i,p in enumerate(self.problems)}

    @property
    def name(self):
        return self._dataset_identifier

class MatSciDataModule(BaseSynthDataModule):
    def __init__(self, root):
        super().__init__("matsci", "MatSci", root)

class ChemReactKineticsDataModule(BaseSynthDataModule):
    def __init__(self, root):
        super().__init__("chem_react", "CRK", root,
                         default_symbols=['dA_dt', 't', 'A'],
                         default_symbol_descs=['Rate of change of concentration in chemistry reaction kinetics', 'Time', 'Concentration at time t'])
        
class BioPopGrowthDataModule(BaseSynthDataModule):
    def __init__(self, root):
        super().__init__("bio_pop_growth", "BPG", root,
                         default_symbols=['dP_dt', 't', 'P'],
                         default_symbol_descs=['Population growth rate', 'Time', 'Population at time t'])
        
class PhysOscilDataModule(BaseSynthDataModule):
    def __init__(self, root):
        super().__init__("phys_osc", "PO", root,
                         default_symbols=['dv_dt', 'x', 't', 'v'],
                         default_symbol_descs=['Acceleration in Nonl-linear Harmonic Oscillator', 'Position at time t', 'Time', 'Velocity at time t'])

def get_datamodule(name, root_folder):
    if name == 'bio_pop_growth':
        root = root_folder or "datasets/lsr-synth-bio"
        return BioPopGrowthDataModule(root)
    elif name == 'chem_react':
        root = root_folder or "datasets/lsr-synth-chem"
        return ChemReactKineticsDataModule(root)
    elif name == 'matsci':
        root = root_folder or "datasets/lsr-synth-matsci"
        return MatSciDataModule(root)
    elif name == 'phys_osc':
        root = root_folder or "datasets/lsr-synth-phys"
        return PhysOscilDataModule(root)
    # elif name == 'feynman':
    #     return FeynmanDataModule()
    elif name == 'lsrtransform':
        return TransformedFeynmanDataModule()
    else:
        raise ValueError(f"Unknown datamodule name: {name}")