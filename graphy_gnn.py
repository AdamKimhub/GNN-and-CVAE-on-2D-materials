import numpy as np
import torch 
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from pymatgen.core.periodic_table import Element
from pymatgen.core import Structure, PeriodicSite, DummySpecie
from pymatgen.analysis.local_env import MinimumDistanceNN, CrystalNN



def struct_to_dict(structure):
    rounded_coords = np.round(structure.frac_coords, 3)
    return {tuple(coord): site for coord, site in zip(rounded_coords, structure.sites)}

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

def get_defects_structure(defective_struct, reference_struct):
    mindnn = MinimumDistanceNN()
    # struct to dict
    defective_dict = struct_to_dict(defective_struct)
    reference_dict = struct_to_dict(reference_struct)

    # Get lattice of defective structure
    structure_lattice = defective_struct.lattice

    # List to add all defect sites
    defects_list = []

    # Dictionary to hold properties of each defect site
    defects_properties = {}

    ref_index = 0

    for ref_coord, ref_site in reference_dict.items():
        # Use the reference coordinates to get the defective site
        ref_index = ref_index + 1

        def_site = defective_dict.get(ref_coord)

        if def_site:  # The site is found in both the reference structure and the defective structure
            # But are the species the same?
            ref_specie = ref_site.specie
            def_specie = def_site.specie
            if ref_specie != def_specie:  # Substitution
                # Add site to defects list
                defects_list.append(def_site)

                # Get atomic number change and defect type
                add_property = {
                    "original_Z":ref_specie.Z,
                    "new_Z": def_specie.Z,
                    "Z_change": def_specie.Z - ref_specie.Z,

                    "original_en": ref_specie.X,
                    "new_en": def_specie.X,
                    "en_change": def_specie.X - ref_specie.X,

                    "original_ir": get_ir(ref_specie),
                    "new_ir": get_ir(ref_specie),
                    "ir_change": get_ir(def_specie) - get_ir(ref_specie),

                    "original_ar": ref_specie.atomic_radius,
                    "new_ar": def_specie.atomic_radius,
                    "ar_change": def_specie.atomic_radius - ref_specie.atomic_radius,

                    "original_row": ref_specie.row,
                    "new_row": def_specie.row,
                    "row_change": def_specie.row - ref_specie.row,

                    "original_group": ref_specie.group,
                    "new_group": def_specie.group,
                    "group_change": def_specie.group - ref_specie.group,

                    "original_max_os": max(ref_specie.common_oxidation_states),
                    "new_max_os": max(def_specie.common_oxidation_states) if def_specie.common_oxidation_states else 0,

                    "original_ef": ref_specie.electron_affinity,
                    "new_ef": def_specie.electron_affinity,

                    "vacancy_defect": 0.0,
                    "substitution_defect": 1.0,
                    "bonds_broken": 0.0,
                    "site_fe": fe_site(ref_site.species_string,def_site.species_string),
                    "ref_idx": ref_index-1
                }

                defects_properties[def_site] = add_property

        else: # the site from ref_structure is not found in defective structure
            # This means that the site is a vacancy site
            # Add site to defective structure
            vacant_site = PeriodicSite(
                species= DummySpecie(),
                coords= ref_coord,
                coords_are_cartesian= False,
                lattice= structure_lattice
                )

            # Add site to defects list
            defects_list.append(vacant_site)

            ref_specie = ref_site.specie

            # Get atomic number change and defect type
            add_property={
                "original_Z":ref_specie.Z,
                "new_Z": 0,
                "Z_change": 0 - ref_site.specie.Z,

                "original_en": ref_specie.X,
                "new_en": 0.0,
                "en_change": 0.0 - ref_specie.X,

                "original_ir": get_ir(ref_specie),
                "new_ir": 0.0,
                "ir_change": 0.0 - get_ir(ref_specie),

                "original_ar": ref_specie.atomic_radius,
                "new_ar": 0.0,
                "ar_change": 0.0 - ref_specie.atomic_radius,

                "original_row": ref_specie.row,
                "new_row": 0,
                "row_change": 0 - ref_specie.row,

                "original_group": ref_specie.group,
                "new_group": 0,
                "group_change": 0 - ref_specie.group,

                "original_max_os": max(ref_specie.common_oxidation_states),
                "new_max_os": 0,

                "original_ef": ref_specie.electron_affinity,
                "new_ef": 0.0,

                "vacancy_defect": 1.0,
                "substitution_defect": 0.0,
                "bonds_broken": mindnn.get_cn(reference_struct, reference_struct.sites.index(ref_site)),
                "site_fe": fe_site(ref_site.species_string,0),
                "ref_idx": ref_index-1
            }

            defects_properties[vacant_site] = add_property

    # create a defects structure
    defects_struct = Structure.from_sites(defects_list)

    # Add properties to defects structure
    for a_site in defects_struct.sites:
        if a_site in defects_properties.keys():
            a_site.properties.update(defects_properties[a_site])
        else:
            pass

    return defects_struct

def get_nodes(defects_struct):
    defect_sites = defects_struct.sites

    nodes = []
    for site in defect_sites:
        site_features = [
            site.properties["Z_change"],
            site.properties["ar_change"],
            site.properties["en_change"], # 2

            site.properties["group_change"],
            site.properties["ir_change"],
            site.properties["new_Z"], # 5

            site.properties["new_ar"],
            site.properties["new_ef"],
            site.properties["new_en"], # 8

            site.properties["new_group"],
            site.properties["new_ir"],
            site.properties["new_max_os"], # 11

            site.properties["new_row"],
            site.properties["original_Z"],
            site.properties["original_ar"], # 14

            site.properties["original_ef"],
            site.properties["original_en"],
            site.properties["original_group"], # 17

            site.properties["original_ir"],
            site.properties["original_max_os"],
            site.properties["original_row"], # 20

            site.properties["row_change"],
            site.properties["bonds_broken"],
            site.properties["vacancy_defect"], # 23
            
            site.properties["substitution_defect"], #24
            site.properties["site_fe"], #25
            # site.properties["ve_change"]
        ]
        nodes.append(site_features)

    return nodes


def get_edges_and_features(reference_structure, defects_structure):
    a_lat = float(reference_structure.lattice.a)

    from_edge = []
    to_edge = []
    edges = []
    edge_features = []

    cart_coords_ds = defects_structure.cart_coords

    for i, site_i in enumerate(defects_structure):
        for j, site_j in enumerate(defects_structure):
            if j > i :
                # pass
            # else:
                from_edge.append(i)
                to_edge.append(j)

                cart_i = cart_coords_ds[i]
                cart_j = cart_coords_ds[j]
                r_vec = cart_j - cart_i
                r_ij = float(np.linalg.norm(r_vec))

                # if r_ij > 12.0 or r_ij<1e-6:
                    # continue

                dist_angstrom = r_ij
                dist_norm = site_i.distance(site_j)
                dist_lattice_units = r_ij/a_lat

                # Formation energy interaction
                fe_site_i = site_i.properties["site_fe"]
                fe_site_j = site_j.properties["site_fe"]

                fe_product = fe_site_i * fe_site_j
                fe_sum = fe_site_i + fe_site_j
                fe_diff = abs(fe_site_i - fe_site_j)

                # Electrostatic interaction
                q_i = site_i.properties["new_max_os"]
                q_j = site_j.properties["new_max_os"]

                if site_i.properties["vacancy_defect"] == 1:
                    q_i = -q_i

                if site_j.properties["vacancy_defect"] == 1:
                    q_j = -q_j

                charge_product = q_i * q_j
                screened_coulomb = (q_i * q_j) / (r_ij) if r_ij > 0 else 0.0

                # Elastic size interaction
                ir_change_i = site_i.properties["ir_change"]
                ir_change_j = site_j.properties["ir_change"]

                ir_change_product = ir_change_i * ir_change_j
                elastic_size_interaction = (ir_change_i * ir_change_j) / (r_ij ** 3) if r_ij > 0 else 0.0

                # Angular factor
                if r_ij > 0:
                    cos_theta = r_vec[2]/ r_ij
                    angular_factor = 1.0-3.0 * cos_theta ** 2
                else:
                    angular_factor = 0.0

                # Defect interaction
                if site_i.properties["vacancy_defect"] == 1 and site_j.properties["substitution_defect"] == 1:
                    vac_sub = 1
                    vac_vac = 0
                    sub_sub = 0

                elif site_i.properties["substitution_defect"] == 1 and site_j.properties["vacancy_defect"] == 1:
                    vac_sub = 1
                    vac_vac = 0
                    sub_sub = 0

                elif site_i.properties["substitution_defect"] == 1 and site_j.properties["substitution_defect"] == 1:
                    vac_sub = 0
                    vac_vac = 0
                    sub_sub = 1

                elif site_i.properties["vacancy_defect"] == 1 and site_j.properties["vacancy_defect"] == 1:
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

def get_globals(pristine, defective_structure, defects_structure):
    p_n_species = len(pristine.composition.elements)
    d_n_species = len(defective_structure.composition.elements) 
    # Host composition vector: mean electronegativity, mean atomic radius, etc.

    # ---- Defect configuration summary ----
    n_defects = len(defects_structure)
    defect_concentration = len(defects_structure) / len(pristine)

    vacs = 0
    subs = 0
    for def_site in defects_structure:
        if def_site.properties["vacancy_defect"] == 1:
            vacs += 1
        else:
            subs+=1

    n_vacancy = vacs
    n_substitution = subs

    global_list = [
        p_n_species, d_n_species, n_defects,
        defect_concentration, n_vacancy, n_substitution
    ]

    return global_list

# Add ids and ratios in the data

mean_en = []
idss = []

look_dict = {
    "GaSe": ["In", "S"],
    "InSe": ["Ga", "S"],
    "BN"  : ["C"],
    "P"   : ["N"],
    "WSe2": ["Mo", "S"],
    "MoS2": ["W", "Se"],
    }

for i in look_dict.keys():
    pristine_sample = Structure.from_file(f"Final_Dataset/ref_cifs/high_{i}.cif")
    sample_elements = pristine_sample.composition.elements

    sample_X = [e.X for e in sample_elements]

    sample_en_mean = float(np.round(np.mean(sample_X), 2))
    mean_en.append(sample_en_mean)

    # Get Ids
    possible_elements = look_dict.get(i)
    the_ids = [Element(el).Z for el in possible_elements]
    idss.append([0] + the_ids)

look_up_dict = dict(zip(mean_en, idss))

def get_ids_ratios(graphs):
    for data in graphs:

        # Host average electronegativity
        host_avg_en = np.round(torch.round(data.u[..., 2], decimals=2).item(), 2)
        # idx+=1
        # print(idx)
        # print(host_avg_en)

        ids = look_up_dict.get(host_avg_en)
        # print(ids)

        # Defect types
        defects = data.x[:, 6].cpu().numpy()

        # Count occurrences of each id
        num_defects = [np.sum(defects == item) for item in ids]

        total = sum(num_defects)

        ratios = [count / total for count in num_defects]

        data.ids=torch.tensor(ids, dtype=torch.long)
        data.ratios=torch.tensor(ratios, dtype=torch.float)

    return graphs

# Create graph representation of the structures
def graphy(row, reference_structure):
    material = row["dataset_material"]
    id = row["_id"]
    defective_structure = Structure.from_file(f"original_dataset/{material}/cifs/{id}.cif")

    defects_structure = get_defects_structure(defective_structure, reference_structure)

    nodes = get_nodes(defects_structure)
    edges, edge_features = get_edges_and_features(reference_structure, defects_structure)
    global_features = get_globals(reference_structure, defective_structure, defects_structure)

    target = row["band_gap_value"]
    form_energy = row["Formation_Energy"]

    the_data = Data(
        x=torch.tensor(nodes, dtype=torch.float),
        edge_index=torch.tensor(edges, dtype=torch.long),
        edge_attr=torch.tensor(edge_features, dtype=torch.float),
        u=torch.tensor(global_features, dtype=torch.float).unsqueeze(0),
        y=torch.tensor(target, dtype=torch.float).unsqueeze(0), 
        fe =torch.tensor(form_energy, dtype=torch.float).unsqueeze(0)
    )
    return the_data

def fast_grapy(dataset):
    graph_list = []
    unique_dataset_materials = dataset["dataset_material"].unique()

    all_mat = len(unique_dataset_materials)

    for i in unique_dataset_materials:
        subset = dataset[dataset["dataset_material"] == i].reset_index(drop=True)
        all_rows = len(subset)

        pristine_structure = Structure.from_file(f"Final_Dataset/ref_cifs/{i}.cif")
        
        # Go through every row in the subset
        for index, row in subset.iterrows():
            data = graphy(row, pristine_structure)
                        
            # Put all graphs together in a list
            graph_list.append(data)
            
    return graph_list