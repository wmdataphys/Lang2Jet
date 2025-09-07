import uproot
import awkward as ak
import vector
import torch
from omegaconf import OmegaConf
import numpy as np
import os
import glob
from torch.utils.data import Dataset
import h5py


from torch.utils.data import IterableDataset


class MultiClassJetDataset(Dataset):
    def __init__(self, h5FilePath, 
                 n_load = -1
                 ):
        self.h5FilePath = h5FilePath
        self.n_load = n_load
        with h5py.File(self.h5FilePath, "r") as f:
            self.pf_features = torch.from_numpy(f["pf_features"][:n_load].astype('float32'))
            
            if self.pf_features.shape[1] == 17 or self.pf_features.shape[-1] == 128:
                self.pf_features = self.pf_features.permute(0, 2, 1)
                print (f"The input shape is WRONG. Corrected to {self.pf_features.shape}")
            
            self.pf_mask = torch.from_numpy(f["pf_mask"][:n_load])
            
            if self.pf_mask.shape[1] == 1:
                print (f"The mask shape is wrong and is of shape {self.pf_mask.shape}")
                self.pf_mask = self.pf_mask.squeeze(1)
                print (f"Converted the shape of mask to {self.pf_mask.shape}")
            
            self.labels = torch.from_numpy(f["label"][:n_load]).long()
            if "event_id" in f:
                self.event_ids = torch.from_numpy(f["event_id"][:n_load])
            else:
                print ("No event ids found. Creating dummy event ids")
                self.event_ids = torch.arange(len(self.labels))
            # TO DO 
            # MAKE THE SELECTION MORE GENERIC BASED ON LAEBLS. 
            # iF labels [0, 1, 5] is given choose only thsioe for training

    def __len__(self):
        return self.labels.shape[0]
    
    def to(self, device):
        self.pf_features = self.pf_features.to(device)
        self.pf_mask = self.pf_mask.to(device)
        self.labels = self.labels.to(device)
    
    def device(self):
        return self.pf_features.device

    def __getitem__(self, idx):
        return self.pf_features[idx], self.pf_mask[idx], self.labels[idx], self.event_ids[idx]



class JetTorchDatasetShuffled(Dataset):
    def __init__(self, parquet_file, labels, config, n_load=None):
        self.labels = labels
        self.config = config
        self.max_particles = config.max_particles
        self.scaling_factor = config.scaling_factor
        np.random.seed(42)
        data = ak.from_parquet(parquet_file)
        self.max_events = len(data) if n_load is None else n_load

        # --- Balance classes ---
        label_array_2d = ak.to_numpy([[row[label] for label in labels] for row in data])
        label_indices = np.argmax(label_array_2d, axis=1)

        class0_idx = np.where(label_indices == 0)[0]
        class1_idx = np.where(label_indices == 1)[0]
        n_each = self.max_events // 2
        sampled0 = np.random.choice(class0_idx, size=n_each, replace=False)
        sampled1 = np.random.choice(class1_idx, size=n_each, replace=False)
        selected_indices = np.concatenate([sampled0, sampled1])
        np.random.shuffle(selected_indices)

        # --- Preprocess selected jets ---
        selected_data = data[selected_indices]
        self.features, self.masks, self.labels = self._preprocess(selected_data)

        # free memory 
        del data
        del selected_data

    def _preprocess(self, chunk):
        # Jet and particle 4-vectors
        p4 = vector.zip({
            "px": chunk["part"]["px"],
            "py": chunk["part"]["py"],
            "pz": chunk["part"]["pz"],
            "energy": chunk["part"]["energy"]
        })
        p4_jet = ak.sum(p4, axis=1)

        # Raw input features
        raw_data = ak.zip({
            "part_px": chunk["part"]["px"],
            "part_py": chunk["part"]["py"],
            "part_pz": chunk["part"]["pz"],
            "part_eta": p4.eta,
            "part_phi": p4.phi,
            "part_pT": p4.pt,
            "part_etarel": p4.deltaeta(p4_jet),
            "part_phirel": p4.deltaphi(p4_jet),
        })

        # Apply cuts
        cuts = (np.abs(raw_data["part_etarel"]) < 0.8) & (np.abs(raw_data["part_phirel"]) < 0.8)
        raw_data = raw_data[cuts]

        # Filter jets with enough constituents
        particle_count = ak.count(raw_data["part_pT"], axis=1)
        valid_jets = particle_count >= self.config.min_constituent
        raw_data = raw_data[valid_jets]
        label_data = ak.zip({label: chunk[label] for label in self.labels})[valid_jets]

        # Sort and transform
        sorted_indices = ak.argsort(raw_data["part_pT"], ascending=False, axis=1)
        sorted_data = raw_data[sorted_indices]
        sorted_data["part_pT"] = np.log(sorted_data["part_pT"]) - 1.8

        # Pad and stack features
        part_pT = ak.fill_none(ak.pad_none(sorted_data["part_pT"], self.max_particles, clip=True), 0)
        part_etarel = ak.fill_none(ak.pad_none(sorted_data["part_etarel"], self.max_particles, clip=True), 0)
        part_phirel = ak.fill_none(ak.pad_none(sorted_data["part_phirel"], self.max_particles, clip=True), 0)
        features = ak.concatenate([
            part_pT[:, :, None],
            (part_etarel * self.scaling_factor)[:, :, None],
            (part_phirel * self.scaling_factor)[:, :, None]
        ], axis=-1)

        # Labels
        label_array_2d = ak.to_numpy([[row[label] for label in self.labels] for row in label_data])
        labels_array = np.argmax(label_array_2d, axis=1)

        # Convert to tensors
        features_np = ak.to_numpy(features)
        features_tensor = torch.tensor(features_np, dtype=torch.float32)
        labels_tensor = torch.tensor(labels_array, dtype=torch.long)
        masks = (features_tensor[:, :, 0] != 0).float()

        return features_tensor, masks, labels_tensor

    def to(self, device):
        self.features = self.features.to(device)
        self.masks = self.masks.to(device)
        self.labels = self.labels.to(device)
        return self
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "part_features": self.features[idx],
            "part_mask": self.masks[idx],
            "jet_type_labels": self.labels[idx]
        }
        
## Building a jet iterable dataset that is shufflable from ne root files
class JetIterableTorchDatasetShuffled(IterableDataset):
    def __init__(self, parquet_file, labels, config, n_load):
        self.data = ak.from_parquet(parquet_file)
        self.labels = labels
        self.config = config
        self.max_particles = config.max_particles
        self.max_events = len(self.data) if n_load is None else n_load
        self.batch_size = config.batch_size

        # Prepare balanced, shuffled index list
        self._prepare_indices()

    def _prepare_indices(self):
        # One-hot → argmax
        label_array_2d = ak.to_numpy([
            [row[label] for label in self.labels] for row in self.data
        ])
        labels = np.argmax(label_array_2d, axis=1)
        
        class0_idx = np.where(labels == 0)[0]
        class1_idx = np.where(labels == 1)[0]
        print(f"Class 0 count: {len(class0_idx)}, Class 1 count: {len(class1_idx)}")
        n_each = self.max_events // 2
        sampled0 = np.random.choice(class0_idx, size=n_each, replace=False)
        sampled1 = np.random.choice(class1_idx, size=n_each, replace=False)

        self.indices = np.concatenate([sampled0, sampled1])
        np.random.shuffle(self.indices)
        print(f"Total samples prepared: {len(self.indices)}")

    def __iter__(self):
        np.random.shuffle(self.indices)  # Reshuffle each epoch
        for i in range(0, len(self.indices), self.batch_size):
            batch_idx = self.indices[i:i + self.batch_size]
            batch = self.data[batch_idx]
            yield from self.process_chunk(batch)

    def process_chunk(self, chunk):
        p4 = vector.zip({
            "px": chunk["part"]["px"],
            "py": chunk["part"]["py"],
            "pz": chunk["part"]["pz"],
            "energy": chunk["part"]["energy"]
        })
        p4_jet = ak.sum(p4, axis=1)

        raw_data = ak.zip({
            "part_px": chunk["part"]["px"],
            "part_py": chunk["part"]["py"],
            "part_pz": chunk["part"]["pz"],
            "part_eta": p4.eta,
            "part_phi": p4.phi,
            "part_pT": p4.pt,
            "part_etarel": p4.deltaeta(p4_jet),
            "part_phirel": p4.deltaphi(p4_jet),
        })
            
        cuts = (np.abs(raw_data["part_etarel"]) < 0.8) & (np.abs(raw_data["part_phirel"]) < 0.8)
        raw_data = raw_data[cuts]
        # label_data = ak.zip({label: chunk[label] for label in self.labels})[cuts]

        particle_count = ak.count(raw_data["part_pT"], axis=1)
        valid_jets = particle_count >= self.config.min_constituent
        raw_data = raw_data[valid_jets]
        label_data = ak.zip({label: chunk[label] for label in self.labels})
        label_data = label_data[valid_jets] 

        sorted_indices = ak.argsort(raw_data["part_pT"], ascending=False, axis=1)
        sorted_data = raw_data[sorted_indices]
        sorted_data["part_pT"] = np.log(sorted_data["part_pT"]) - 1.8

        part_pT = ak.fill_none(ak.pad_none(sorted_data["part_pT"], self.max_particles, clip=True), 0)
        part_etarel = ak.fill_none(ak.pad_none(sorted_data["part_etarel"], self.max_particles, clip=True), 0)
        part_phirel = ak.fill_none(ak.pad_none(sorted_data["part_phirel"], self.max_particles, clip=True), 0)
        features = ak.concatenate([
            part_pT[:, :, None],
            (part_etarel * self.config.scaling_factor)[:, :, None],
            (part_phirel * self.config.scaling_factor)[:, :, None]
        ], axis=-1)

        features_np = ak.to_numpy(features)
        label_array_2d = ak.to_numpy([
            [row[label] for label in self.labels] for row in label_data
        ])
        labels_array = np.argmax(label_array_2d, axis=1)

        for x, y in zip(features_np, labels_array):
            x_tensor = torch.tensor(x, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.long)
            mask = (x_tensor[:, 0] != 0).float()
            yield {
                "part_features": x_tensor,
                "part_mask": mask,
                "jet_type_labels": y_tensor
            }






# Custom IterableDataset for loading jet data from ROOT files
# for the BERT MLP model because of OOM error
# This dataset will yield batches of jet data on-the-fly, allowing for efficient memory usage.
class JetIterableTorchDataset(IterableDataset):
    def __init__(self, root_path, tree_name, labels, config):
        self.root_files = sorted(glob.glob(os.path.join(root_path, "*.root")))
        assert self.root_files, f"No ROOT files found in: {root_path}"
        self.max_events_per_file = getattr(config, "n_load", None)
        self.tree_name = tree_name
        self.labels = labels
        self.config = config
        self.max_particles = config.max_particles
        self.max_events_per_file = config.data.n_load

    def __iter__(self):
        # import random
        # files = self.root_files.copy()
        # random.shuffle(files)
        for file in self.root_files:
            nevents = 0
            for chunk in uproot.iterate(
                files=file,
                treepath=self.tree_name,
                expressions=[
                    "part_px", "part_py", "part_pz", "part_energy",
                    "jet_eta", "jet_phi", "jet_pt", "jet_energy"
                ] + self.labels,
                library="ak",
                how="zip",
                step_size=1000
            ):
                if self.max_events_per_file is not None:
                    remaining = self.max_events_per_file - nevents
                    if remaining <= 0:
                        break  
                    if len(chunk["jet_pt"]) > remaining:
                        chunk = chunk[:remaining]
                for item in self.process_chunk(chunk):
                    yield item
                    nevents += 1
                    if self.max_events_per_file is not None and nevents >= self.max_events_per_file:
                        break 

    def process_chunk(self, chunk):
        # Label extraction
        label_data = ak.zip({label: chunk[label] for label in self.labels})
        # print("Extracting features")
        # Physics feature processing
        p4 = vector.zip({
            "px": chunk["part"]["px"],
            "py": chunk["part"]["py"],
            "pz": chunk["part"]["pz"],
            "energy": chunk["part"]["energy"]
        })
        p4_jet = ak.sum(p4, axis=1)

        raw_data = ak.zip({
            "part_px": chunk["part"]["px"],
            "part_py": chunk["part"]["py"],
            "part_pz": chunk["part"]["pz"],
            "part_eta": p4.eta,
            "part_phi": p4.phi,
            "part_pT": p4.pt,
            "part_etarel": p4.deltaeta(p4_jet),
            "part_phirel": p4.deltaphi(p4_jet),
        })

        
        cuts = (np.abs(raw_data["part_etarel"]) < 0.8) & (np.abs(raw_data["part_phirel"]) < 0.8)
        raw_data = raw_data[cuts]
        particle_count = ak.count(raw_data["part_pT"], axis=1)
        valid_jets = particle_count >= self.config.min_constituent
        raw_data = raw_data[valid_jets]
        label_data = label_data[valid_jets]

        sorted_indices = ak.argsort(raw_data["part_pT"], ascending=False, axis=1)
        sorted_data = raw_data[sorted_indices]
        sorted_data["part_pT"] = np.log(sorted_data["part_pT"]) - 1.8

        # Pad and convert to tensor
        part_pT = ak.fill_none(ak.pad_none(sorted_data["part_pT"], self.max_particles, clip=True), 0)
        part_etarel = ak.fill_none(ak.pad_none(sorted_data["part_etarel"], self.max_particles, clip=True), 0)
        part_phirel = ak.fill_none(ak.pad_none(sorted_data["part_phirel"], self.max_particles, clip=True), 0)
        features = ak.concatenate([
            part_pT[:, :, None],
            (part_etarel * self.config.scaling_factor)[:, :, None],
            (part_phirel * self.config.scaling_factor)[:, :, None]
        ], axis=-1)

        features_np = ak.to_numpy(features)
        # Convert record array to a list-of-lists of label values
        label_array_2d = ak.to_numpy([
            [record[label] for label in self.labels] for record in label_data
        ])

        # Now do argmax across label dimension
        labels_array = np.argmax(label_array_2d, axis=1)

        # labels_array = ak.to_numpy(ak.argmax(ak.values_astype(label_data, "int32"), axis=1))

        for x, y in zip(features_np, labels_array):
            x_tensor = torch.tensor(x, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.long)
            mask = (x_tensor[:, 0] != 0).float()
            yield {
                "part_features": x_tensor,
                "part_mask": mask,
                "jet_type_labels": y_tensor
            }



from torch.utils.data import Dataset
import torch
import numpy as np

class JetTorchDataset(Dataset):
    def __init__(self, jet_dataset):
        self.raw_data, self.preprocessed_data, self.labels, self.true = jet_dataset.load_all_data()

    def __len__(self):
        return len(self.preprocessed_data)

    def __getitem__(self, idx):
        x = self.preprocessed_data[idx]
        y = self.labels[idx]
        x_tensor = torch.tensor(np.array(x), dtype=torch.float32)  # [128, 3]
        y_tensor = torch.tensor(y, dtype=torch.long)

        # Optional mask, assuming zero-padded particles have zero pt
        mask = (x_tensor[:, 0] != 0).float()  # [128]
        return {
            "part_features": x_tensor,      # [128, 3]
            "part_mask": mask,              # [128]
            "jet_type_labels": y_tensor,    # scalar
        }
