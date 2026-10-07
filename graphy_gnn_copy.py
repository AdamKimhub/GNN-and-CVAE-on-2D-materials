import numpy as np
import pandas as pd
import torch 
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from pymatgen.core import Structure, PeriodicSite, DummySpecie
from pymatgen.analysis.local_env import MinimumDistanceNN, CrystalNN
from structure_manipulation import struct_to_dict, clean_defective_manipulation

"""
# Get formation energy of all elements in the periodiic table
from mp_api.client import MPRester
from pymatgen.core.periodic_table import Element

all_elements = [str(el) for el in Element]

API_KEY = ""

def get_formation(element, API_KEY):
    with MPRester(API_KEY) as mpr:
        results = mpr.materials.summary.search(
            elements=[element],
            num_elements=1,
            fields= ["energy_per_atom"]
        )
        forms_list = [result.energy_per_atom for result in results]
        avg_formation_energy = np.mean(forms_list)

    return avg_formation_energy

formation_energies = {element: get_formation(element, API_KEY) for element in all_elements}
print(formation_energies)
"""
formation_energies = {
    "H":-2.835951846430921, "He":-0.31196323625, "Li":-2.3461654534259258, "Be":-3.6906741816666666,
    "B":-7.002827127491851, "C":-9.316731934117035, "N":-7.233842272742187, "O":-4.676438329750001,
    "F":-2.9296551798214288, "Ne":-1.91282416, "Na":-3.4536177056717374, "Mg":-4.09163935124074,
    "Al":-6.63584847076, "Si":-8.426176753356291, "P":-8.31567715169643, "S":-7.827358545769599,
    "Cl":-6.128105756041667, "Ar":-4.856792372499999, "K":-5.951273052253347, "Ca":-6.8788465471875,
    "Sc":-11.237957669242425, "Ti":-12.802568009213335, "V":-13.922600690000001, "Cr":-14.77669376875,
    "Mn":-14.348056304012763, "Fe":-8.25899893927, "Co":-13.16277911966809, "Ni":-11.659201161666667,
    "Cu":-10.54583238875, "Zn":-8.8732424244375, "Ga":-11.365511257524622, "Ge":-13.659562455970299,
    "As":-14.343474869861112, "Se":-14.136485254729166, "Br":-2.802858366809896, "Kr":-12.577137455166667,
    "Rb":-3.522684412102679, "Sr":-14.872080059242423, "Y":-20.267552732083335, "Zr":-22.357979443055555,
    "Nb":-24.592838769404764, "Mo":-25.65156121359375, "Tc":-25.938798287500003, "Ru":-25.359055447499998,
    "Rh":-23.891668894, "Pd":-22.939867830625, "Ag":-21.344974299333337, "Cd":-20.075181660000002,
    "In":-22.569768360619864, "Sn":-24.623966149094205, "Sb":-25.500892143392857, "Te":-25.35888922212963,
    "I":-5.240995872916667, "Xe":-2.681022067222222, "Cs":-25.08106868751572, "Ba":-1.714427234090909,
    "La":-29.703585855, "Ce":-30.765536582916667, "Pr":-29.360978606000003, "Nd":-29.38402319895833,
    "Pm":-29.532344363055557, "Sm":-29.722466492916666, "Eu":-38.489393041388894, "Gd":-43.33816982966667,
    "Tb":-31.19427302633333, "Dy":-31.949257627833333, "Ho":-32.837879401, "Er":-33.877142851833334,
    "Tm":-35.078865518166666, "Yb":-36.496195276250006, "Lu":-38.04532439777778, "Hf":-45.06808062583333,
    "Ta":-47.144759839058324, "W":-50.771654835674994, "Re":-52.330339372333334, "Os":-51.948078458750004,
    "Ir":-51.251150415, "Pt":-51.429006055, "Au":-50.54936647375, "Hg":-49.346659606686515, "Tl":-53.16799786356322, 
    "Pb":-56.20973389190476, "Bi":-58.259812768529414, "Ac":-68.625217766875, "Th":-73.399498815, 
    "Pa":-76.84324669166666,    "U":-79.60085220729613, "Np":-82.84298793625, "Pu":-86.04870866293301, 
    "Am":0, "Cm":0, "Bk":0, "Cf":0,"Es":0, "Fm":0, "Md":0, "No":0, "Lr":0, "Rf":0, 
    "Db":0, "Sg":0, "Bh":0, "Hs":0, "Mt":0, "Ds":0, "Rg":0, "Cn":0, "Nh":0, "Fl":0,
    "Mc":0, "Lv":0, "Ts":0, "Og":0, "Po": 0, "At":0, "Rn":0, "Fr":0, "Ra":0
}

def fe_site(original, new):
    if new == 0: # For vcancy
        fe_defect = formation_energies[original] * -1

    else: # For substitution
        form_original = formation_energies[original]
        form_new = formation_energies[new]
        fe_defect = (form_original * -1) + form_new

    return fe_defect

def get_ir(e):
    try:
        return float(e.average_ionic_radius)
    except Exception:
        try:
            return float(e.atomic_radius)
        except Exception:
            return 0.0

def get_nodes(cloud_struct, reference_struct):
    mindnn = MinimumDistanceNN()
    # struct to dict
    cloud_dict = struct_to_dict(cloud_struct)
    reference_dict = struct_to_dict(reference_struct)

    # List to add all defect sites
    nodes_list = []

    for cloud_coord, cloud_site in cloud_dict.items():
        # Use the cloud coordinates to get the original sites

        ref_site = reference_dict.get(cloud_coord)

        if ref_site:  # The site is found in both the reference structure and the cloud structure
            ref_specie = ref_site.specie
            cloud_specie = cloud_site.specie

            original_Z = ref_specie.Z
            original_en = ref_specie.X
            original_ir = get_ir(ref_specie)
            original_ar = ref_specie.atomic_radius
            original_row = ref_specie.row
            original_group = ref_specie.group
            original_max_os = max(ref_specie.common_oxidation_states)
            original_ef = ref_specie.electron_affinity

            if cloud_specie == DummySpecie():  # Vacancy
                
                new_z, new_en, new_ir, new_ar = 0.0, 0.0, 0.0, 0.0
                new_row, new_group, new_max_os, new_ef = 0.0, 0.0, 0.0, 0.0

                change_z, change_en, change_ir = -1*(original_Z), -1*(original_en), -1*(original_ir)
                change_ar, change_row, change_group = -1*(original_ar), -1*(original_row), -1*(original_group)

                site_fe = fe_site(ref_site.species_string, 0)
                vacancy_defect = 1.0
                substitution_defect = 0.0
                ref_idx = reference_struct.sites.index(ref_site)
                bonds_broken = mindnn.get_cn(reference_struct, ref_idx)


                the_nodes = [
                    change_z, change_ar, change_en, change_group,
                    change_ir, new_z, new_ar, new_ef, new_en, new_group,
                    new_ir, new_max_os, new_row, original_Z, original_ar, 
                    original_ef, original_en, original_group, original_ir, 
                    original_max_os, original_row, change_row, bonds_broken, 
                    vacancy_defect, substitution_defect, site_fe, ref_idx
                ]

                nodes_list.append(the_nodes)

            else: # Substitution site
                new_z = cloud_specie.Z
                new_en = cloud_specie.X
                new_ir = get_ir(cloud_specie)
                new_ar = cloud_specie.atomic_radius
                new_row = cloud_specie.row
                new_group = cloud_specie.group
                new_max_os = max(cloud_specie.common_oxidation_states) if cloud_specie.common_oxidation_states else 0
                new_ef = cloud_specie.electron_affinity

                change_z = new_z - original_Z
                change_en = new_en - original_en
                change_ir = new_ir - original_ir
                change_ar = new_ar - original_ar
                change_row = new_row - original_row
                change_group = new_group - original_group

                site_fe = fe_site(ref_site.species_string, cloud_site.species_string)
                vacancy_defect = 0.0
                substitution_defect = 1.0
                ref_idx = reference_struct.sites.index(ref_site)
                bonds_broken = 0.0

                the_nodes = [
                    change_z, change_ar, change_en, change_group,
                    change_ir, new_z, new_ar, new_ef, new_en, new_group,
                    new_ir, new_max_os, new_row, original_Z, original_ar, 
                    original_ef, original_en, original_group, original_ir, 
                    original_max_os, original_row, change_row, bonds_broken, 
                    vacancy_defect, substitution_defect, site_fe, ref_idx
                ]
                
                nodes_list.append(the_nodes)

    return nodes_list



def get_edges_and_features(nodes_list, cloud_structure, reference_structure):
    a_lat = float(reference_structure.lattice.a)

    from_edge = []
    to_edge = []
    edges = []
    edge_features = []

    cloud_cart_coords = cloud_structure.cart_coords

    for i, (node_i, site_i) in enumerate(zip(nodes_list, cloud_structure.sites)):
        for j, (node_j, site_j) in enumerate(zip(nodes_list, cloud_structure.sites)):
            if j > i :
                from_edge.append(i)
                to_edge.append(j)

                cart_i = cloud_cart_coords[i]
                cart_j = cloud_cart_coords[j]

                r_vec = cart_j - cart_i

                r_ij = float(np.linalg.norm(r_vec))

                # if r_ij > 12.0 or r_ij<1e-6:
                    # continue

                dist_angstrom = r_ij
                dist_norm = site_i.distance(site_j)
                dist_lattice_units = r_ij/a_lat

                # Formation energy interaction , # index 26
                fe_site_i = node_i[25]
                fe_site_j = node_j[25]

                fe_product = fe_site_i * fe_site_j
                fe_sum = fe_site_i + fe_site_j
                fe_diff = abs(fe_site_i - fe_site_j)

                # Electrostatic interaction # index 11
                q_i = node_i[11]
                q_j = node_j[11]

                # Vacancy # 23
                # Substitution #24

                if node_i[23] == 1:
                    q_i = -q_i

                if node_j[23] == 1:
                    q_j = -q_j

                charge_product = q_i * q_j
                screened_coulomb = (q_i * q_j) / (r_ij) if r_ij > 0 else 0.0

                # Elastic size interaction # index 4
                ir_change_i = node_i[4]
                ir_change_j = node_j[4]

                ir_change_product = ir_change_i * ir_change_j
                elastic_size_interaction = (ir_change_i * ir_change_j) / (r_ij ** 3) if r_ij > 0 else 0.0

                # Angular factor
                if r_ij > 0:
                    cos_theta = r_vec[2]/ r_ij
                    angular_factor = 1.0-3.0 * cos_theta ** 2
                else:
                    angular_factor = 0.0

                # Defect interaction
                if node_i[23] == 1 and node_j[24] == 1:
                    vac_sub = 1
                    vac_vac = 0
                    sub_sub = 0

                elif node_i[24] == 1 and node_j[23] == 1:
                    vac_sub = 1
                    vac_vac = 0
                    sub_sub = 0

                elif node_i[24] == 1 and node_j[24] == 1:
                    vac_sub = 0
                    vac_vac = 0
                    sub_sub = 1

                elif node_i[23] == 1 and node_j[23] == 1:
                    vac_sub = 0
                    vac_vac = 1
                    sub_sub = 0

                else:
                    vac_sub = 0
                    vac_vac = 0
                    sub_sub = 0

                edge_features.append(
                    [
                        dist_angstrom, dist_norm, dist_lattice_units,
                        fe_product, fe_sum, fe_diff,
                        charge_product, screened_coulomb, ir_change_product,
                        elastic_size_interaction, angular_factor, 
                        vac_vac, sub_sub, vac_sub
                    ]
                )

    edges.append(from_edge)
    edges.append(to_edge)

    return edges, edge_features

def get_globals(nodes_list, len_clean_defective, len_reference):
    num_defects = len(nodes_list)
    defect_concentration = num_defects / len_reference

    vacs = 0
    subs = 0
    for node_i in nodes_list:
        if node_i[23] == 1:
            vacs += 1
        else:
            subs+=1

    n_vacancy = vacs
    n_substitution = subs

    global_list = [
        len_reference, len_clean_defective, num_defects,
        defect_concentration, n_vacancy, n_substitution
    ]

    return global_list


# Create graph representation of the structures
def gnn_graphy(target, clean_defective_structure, reference_structure):
    cloud_structure = clean_defective_manipulation(clean_defective_structure, reference_structure, get_cloud=True)

    nodes = get_nodes(cloud_structure, reference_structure)
    edges, edge_features = get_edges_and_features(nodes, cloud_structure, reference_structure)

    len_reference = len(reference_structure)
    len_clean_defective = len(clean_defective_structure)

    global_features = get_globals(nodes, len_clean_defective, len_reference)

    the_data = Data(
        x          = torch.tensor(nodes, dtype=torch.float),
        edge_index = torch.tensor(edges, dtype=torch.long),
        edge_attr  = torch.tensor(edge_features, dtype=torch.float),
        u          = torch.tensor(global_features, dtype=torch.float).unsqueeze(0),
        y          = torch.tensor(target, dtype=torch.float).unsqueeze(0), 
    )
    return the_data

def fast_gnn_graphy(dataset):
    graph_list = []
    unique_dataset_materials = dataset["dataset_material"].unique()


    for i in unique_dataset_materials:
        print(f"Working on {i}...")
        subset = dataset[dataset["dataset_material"] == i].reset_index(drop=True)

        pristine_structure = Structure.from_file(f"Final_Dataset/ref_cifs/{i}.cif")
        
        # Go through every row in the subset
        for _ , row in subset.iterrows():
            the_target = row["band gap value"]
            id = row["_id"]
            clean_defective_structure = Structure.from_file(f"original_dataset/{i}/cifs/{id}.cif")

            data = gnn_graphy(the_target, clean_defective_structure, pristine_structure)
                        
            # Put all graphs together in a list
            graph_list.append(data)
            
    return graph_list

def main():
    comb_df = pd.read_csv("Final_Dataset/combined/combined_data.csv")
    train_set, val_set = train_test_split(comb_df, test_size=0.3, random_state=42, stratify=comb_df["strata"])
    val_set, test_set = train_test_split(val_set, test_size=0.5, random_state=42, stratify=val_set["strata"])

    full_train_graphs = fast_gnn_graphy(train_set)
    full_val_graphs = fast_gnn_graphy(val_set)
    full_test_graphs = fast_gnn_graphy(test_set)

    torch.save(full_train_graphs, "Final_Dataset/gnn graphs/gnn_train_graphs.pt")
    torch.save(full_val_graphs, "Final_Dataset/gnn graphs/gnn_val_graphs.pt")
    torch.save(full_test_graphs, "Final_Dataset/gnn graphs/gnn_test_graphs.pt")


if __name__ == "__main__":
    main()