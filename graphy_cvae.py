import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

from pymatgen.core import Structure, DummySpecie, Element
from pymatgen.analysis.local_env import MinimumDistanceNN
from pymatgen.analysis.graphs import StructureGraph

from structure_manipulation import clean_defective_manipulation
from graphy_gnn_copy import get_nodes, get_edges_and_features, get_globals

# ==========
#  MASK
# ==========

def get_mask(pristine_structure):
    masking_dict = {
        "Ga":"In", "Se":"S", "In":"Ga", 
        "B":"C", "N":"C", "P":"N", 
        "W":"Mo", "Mo":"W", "S":"Se"
    }

    the_masks = []

    for site in pristine_structure.sites:
        specie_now = site.specie.symbol

        # Normal site 
        norm = site.specie.Z

        # Vacancy site
        vac = 0

        # Sub site
        allowed_sub = masking_dict[specie_now]
        sub = Element(allowed_sub).Z

        # The row mask
        row_mask = [norm, vac, sub]
        
        the_masks.append(row_mask)

    
    return the_masks

# ====================
# PRISTINE NODES
# ====================
def get_pristine_nodes(pristine_structure):
    
    all_features= []
    for a_site in pristine_structure.sites:
        coords = a_site.coords

        all_features.append(coords)

    return all_features

# ====================
# PRISTINE EDGES
# ====================
def get_pristine_edges(pristine_structure):
    # 2. Generate a StructureGraph using a neighbor strategy (e.g., CrystalNN)
    mdnn = MinimumDistanceNN()
    the_graph = StructureGraph.from_local_env_strategy(pristine_structure, mdnn)

    # 3. Extract edge indices (COO format: [source_indices, target_indices])
    edge_indices = []
    for u, v, _ in the_graph.graph.edges(data=True):
        edge_indices.append([u, v])
        edge_indices.append([v, u])

    edge_indices = np.array(edge_indices).T  # Shape: (2, num_edges)
    return edge_indices


# =======================
# CVAE TARGET
# =======================

def get_target_tensor(full_defective_structure):
    new_z = []
    for site in full_defective_structure.sites:
        the_specie = site.specie

        if the_specie == DummySpecie():
            the_z = 0
        else:
            the_z = the_specie.Z

        new_z.append(the_z)

    return new_z

# =========================
# COMBINE TO GET GRAPHS
# =========================

def get_encoder_data(cloud_structure, clean_defective_structure, ref_structure):
    gnn_nodes = get_nodes(cloud_structure, ref_structure)

    encoder_nodes = []
    for cloud_site, nodes_i in zip(cloud_structure, gnn_nodes):
        
        encoder_nodes.append(np.concatenate([nodes_i, cloud_site.frac_coords]))

    edges, edge_features = get_edges_and_features(gnn_nodes, cloud_structure, ref_structure)
    
    len_reference = len(ref_structure)
    len_clean_defective = len(clean_defective_structure)
    
    global_features = get_globals(gnn_nodes, len_clean_defective, len_reference)

    return (encoder_nodes, edges, edge_features, global_features)

def get_decoder_data(full_defective_structure, ref_structure):
    the_mask = get_mask(ref_structure)
    pristine_nodes = get_pristine_nodes(ref_structure)
    pristine_edges = get_pristine_edges(ref_structure)
    the_target    = get_target_tensor(full_defective_structure)

    return (the_mask, pristine_nodes, pristine_edges, the_target)



def cvae_graphy(target, clean_defective_structure, ref_structure):
    full_defective_structure = clean_defective_manipulation(clean_defective_structure, ref_structure, get_full_defective=True)
    cloud_structure = clean_defective_manipulation(clean_defective_structure, ref_structure, get_cloud=True)

    for_encoder = get_encoder_data(cloud_structure, clean_defective_structure, ref_structure)

    for_decoder = get_decoder_data(full_defective_structure, ref_structure)

    the_data = Data(
        x          =torch.tensor(for_encoder[0], dtype=torch.float),
        edge_index =torch.tensor(for_encoder[1], dtype=torch.long),
        edge_attr  =torch.tensor(for_encoder[2], dtype=torch.float),
        u          =torch.tensor(for_encoder[3], dtype=torch.float).unsqueeze(0), 
        bgv        =torch.tensor([target], dtype=torch.float),
        mask       =torch.tensor(for_decoder[0], dtype=torch.long),
        pristine_x =torch.tensor(for_decoder[1], dtype=torch.float),
        pristine_e =torch.tensor(for_decoder[2], dtype=torch.long),
        y          =torch.tensor(for_decoder[3], dtype=torch.long), 
    )

    return the_data

def fast_cvae_graphy(material_dataset):
    data_list = []
    dataset_materials = list(material_dataset["dataset_material"].unique())    
    
    for material in (dataset_materials):
        ref_structure = Structure.from_file(f"Final_Dataset/ref_cifs/{material}.cif")
        the_mask = get_mask(ref_structure)
        pristine_nodes = get_pristine_nodes(ref_structure)
        pristine_edges = get_pristine_edges(ref_structure)
        
        focus_data = material_dataset[material_dataset["dataset_material"] == material]

        for _, row in focus_data.iterrows():
            the_id = row["_id"]
            the_bgv = row["band_gap_value"]

            clean_defective_structure  = Structure.from_file(f"original_dataset/{material}/cifs/{the_id}.cif")
            cloud_structure = clean_defective_manipulation(clean_defective_structure, ref_structure, get_cloud=True)
            full_defective_structure = clean_defective_manipulation(clean_defective_structure, ref_structure, get_full_defective=True)

            for_encoder = get_encoder_data(cloud_structure, clean_defective_structure, ref_structure)

            the_target    = get_target_tensor(full_defective_structure)
            
            data = Data(
                x          =torch.tensor(for_encoder[0], dtype=torch.float),
                edge_index =torch.tensor(for_encoder[1], dtype=torch.long),
                edge_attr  =torch.tensor(for_encoder[2], dtype=torch.float),
                u          =torch.tensor(for_encoder[3], dtype=torch.float).unsqueeze(0), 
                bgv        =torch.tensor([the_bgv], dtype=torch.float),
                mask       =torch.tensor(the_mask, dtype=torch.long),
                y          =torch.tensor(the_target, dtype=torch.long),
                pristine_x =torch.tensor(pristine_nodes, dtype=torch.float),
                pristine_e =torch.tensor(pristine_edges, dtype=torch.long), 
            )
        
            data_list.append(data)
        
    return data_list

def main():
    # The Data
    comb_df = pd.read_csv("Final_Dataset/combined/combined_data.csv")
    train_set, val_set = train_test_split(comb_df, test_size=0.3, random_state=42, stratify=comb_df["strata"])

    full_train_graphs = fast_cvae_graphy(train_set)
    full_val_graphs = fast_cvae_graphy(val_set)

    torch.save(full_train_graphs, "Final_Dataset/cvae graphs/cvae_train_graphs.pt")
    torch.save(full_val_graphs, "Final_Dataset/cvae graphs/cvae_val_graphs.pt")

if __name__ == "__main__":
    main()