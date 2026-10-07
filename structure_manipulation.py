import numpy as np
import matplotlib.pyplot as plt
from pymatgen.io.cif import CifWriter
from pymatgen.io.ase import AseAtomsAdaptor
from ase.visualize.plot import plot_atoms
from pymatgen.core import Structure, PeriodicSite, DummySpecie

def struct_to_dict(structure):
    rounded_coords = np.round(structure.frac_coords, 3)
    return {tuple(coord): site for coord, site in zip(rounded_coords, structure.sites)}

def save_structue(structure, path_folder):
    writer = CifWriter(structure)
    writer.write_file(f"Final_Dataset/new_structures/{path_folder}/{structure.formula}.cif")

def visualize_structure(structure, title=None):
    new_atoms = AseAtomsAdaptor.get_atoms(structure)

    fig, ax = plt.subplots(figsize=(8, 8))

    plot_atoms(new_atoms, ax=ax)

    if title is None:
        title = structure.formula

    ax.set_title(title)
    ax.set_axis_off()

    plt.show()

def align_to_reference_lattice(defective_struct, reference_struct):
    if not np.allclose(
                       defective_struct.lattice.matrix,
                       reference_struct.lattice.matrix,
                       atol=1e-6):
        frac_coords = reference_struct.lattice.get_fractional_coords(defective_struct.cart_coords)
        frac_coords = np.mod(frac_coords, 1.0)
        return Structure(reference_struct.lattice,
                         defective_struct.species,
                         frac_coords,
                         coords_are_cartesian=False)
    return defective_struct



# Partial_Defective --> Cloud and Full defective 
def clean_defective_manipulation(defective_struct, reference_struct, get_cloud=False, get_full_defective=False, save=False):
    # struct to dict
    defective_struct = align_to_reference_lattice(defective_struct, reference_struct)
    defective_dict = struct_to_dict(defective_struct)
    reference_dict = struct_to_dict(reference_struct)

    # Get lattice of defective structure
    structure_lattice = reference_struct.lattice

    cloud_list = []
    full_defective_list = []

    for ref_coord, ref_site in reference_dict.items():
        
        def_site = defective_dict.get(ref_coord)

        if def_site:  # The site is found in both the reference structure and the defective structure
            # But are the species the same?
            ref_specie = ref_site.specie
            def_specie = def_site.specie

            if ref_specie != def_specie:  # Substitution
                # Add site to defects list
                cloud_list.append(def_site)
                full_defective_list.append(def_site)

            else: # Normal
                full_defective_list.append(ref_site)


        else: # the site from ref_structure is not found in defective structure
            # This means that the site is a vacancy site
            # Add site to defective structure
            vacant_site = PeriodicSite(
                species= DummySpecie(),
                coords= ref_site.frac_coords,
                coords_are_cartesian= False,
                lattice= structure_lattice
                )

            # Add site to defects list
            cloud_list.append(vacant_site)
            full_defective_list.append(vacant_site)

    if get_cloud:
        # create a defects structure
        defects_struct = Structure.from_sites(cloud_list)

        if save:
            save_structue(defects_struct, "cloud")

        return defects_struct

    if get_full_defective:
        full_defective_structure = Structure.from_sites(full_defective_list)

        if save:
            save_structue(full_defective_structure, "full_defective")

        return full_defective_structure

def cloud_manipulation(cloud_struct, reference_struct, get_full_defective=False, get_clean_defective=False, save=False):
    full_defective_sites = []
    clean_defective_sites = []

    cloud_dict = struct_to_dict(cloud_struct)
    reference_dict = struct_to_dict(reference_struct)

    for r_coords, r_site in reference_dict.items():
        cloud_site = cloud_dict.get(r_coords)
        if cloud_site: # Defective site
            full_defective_sites.append(cloud_site)

            if cloud_site.specie == DummySpecie(): # Vacant site
                pass
            else: # Substitution site
                clean_defective_sites.append(cloud_site)
        else:# Normal site
            clean_defective_sites.append(r_site)
            full_defective_sites.append(r_site)

    if get_clean_defective:
        clean_defective_struct = Structure.from_sites(clean_defective_sites)

        if save:
            save_structue(clean_defective_struct, "clean_defective")

        return clean_defective_struct
    if get_full_defective:
        full_defective_struct = Structure.from_sites(full_defective_sites)

        if save:
            save_structue(full_defective_struct, "full_defective")

        return full_defective_struct



def full_defective_manipulation(full_defective_struct, reference_struct, get_cloud=False, get_clean_defective=False, save=False):
    clean_sites = []
    cloud_sites = []
        
    for f_site, r_site in zip(full_defective_struct.sites, reference_struct.sites):
        if f_site.specie == DummySpecie(): # Vacant site
            cloud_sites.append(f_site)
        elif f_site.specie != r_site.specie: # Substitution site
            cloud_sites.append(f_site)
            clean_sites.append(f_site)
        else: # Normal site
            clean_sites.append(r_site)

    if get_cloud:
        cloud_struct = Structure.from_sites(cloud_sites)

        if save:
            save_structue(cloud_struct, "cloud")

        return cloud_struct

    if get_clean_defective:
        clean_defective_struct = Structure.from_sites(clean_sites)

        if save:
            save_structue(clean_defective_struct, "clean_defective")

        return clean_defective_struct
