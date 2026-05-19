import numpy as np
import pyvista as pv

def generate_3d_lattice(lattice_type, repeats):
    """
    Generates positions, radii, box size, and species classifications for 3D lattices.
    Ensures zero overlapping artifacts via strict unique-coordinate filtering.
    """
    lattice_type = lattice_type.lower()
    
    # Precise maximum radii limits for a unit cell edge length of a = 1.0
    r_max_mono = {
        'sc': 0.5,               # Touches along cell edge (2R = 1.0)
        'bcc': np.sqrt(3) / 4,   # Touches along body diagonal (4R = sqrt(3))
        'fcc': np.sqrt(2) / 4    # Touches along face diagonal (4R = sqrt(2))
    }

    if isinstance(repeats, int):
        Nx = Ny = Nz = repeats
    else:
        Nx, Ny, Nz = repeats
    box_size = np.array([Nx, Ny, Nz], dtype=float)

    raw_positions = []
    raw_species = []

    # --- Case 1: Bidisperse NaCl Rock Salt ---
    if lattice_type == 'nacl':
        na_basis = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]])
        cl_basis = np.array([[0.5, 0.5, 0.5], [0.0, 0.0, 0.5], [0.0, 0.5, 0.0], [0.5, 0.0, 0.0]])
        
        for x in range(Nx):
            for y in range(Ny):
                for z in range(Nz):
                    shift = np.array([x, y, z])
                    raw_positions.append(na_basis + shift)
                    raw_species.append(np.zeros(len(na_basis), dtype=int))
                    raw_positions.append(cl_basis + shift)
                    raw_species.append(np.ones(len(cl_basis), dtype=int))
                    
        raw_positions = np.vstack(raw_positions)
        raw_species = np.concatenate(raw_species)

    # --- Case 2: Monodisperse Lattices ---
    elif lattice_type in ['sc', 'bcc', 'fcc']:
        if lattice_type == 'sc':
            basis = np.array([[0.0, 0.0, 0.0]])
        elif lattice_type == 'bcc':
            basis = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        else: # fcc
            basis = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]])
            
        for x in range(Nx):
            for y in range(Ny):
                for z in range(Nz):
                    raw_positions.append(basis + np.array([x, y, z]))
                    
        raw_positions = np.vstack(raw_positions)
        raw_species = np.zeros(len(raw_positions), dtype=int)
    else:
        raise ValueError("Lattice must be 'sc', 'bcc', 'fcc', or 'nacl'.")

    # --- Deduplication Filter ---
    # Rounds float coordinates to filter out floating-point boundary double-counting
    _, unique_indices = np.unique(np.round(raw_positions, decimals=5), axis=0, return_index=True)
    positions = raw_positions[unique_indices]
    species = raw_species[unique_indices]

    # Assign Radii dynamically based on final filtered species mapping
    if lattice_type == 'nacl':
        r_cl = 0.5 / (1.0 + 0.695)
        r_na = 0.5 - r_cl
        radii = np.where(species == 0, r_na, r_cl)
    else:
        radii = np.full(len(positions), r_max_mono[lattice_type])

    return positions, radii, box_size, species


def plot_3d_packing_pyvista(positions, radii, box_size, species, is_nacl=False):
    """
    Renders 3D crystal packings with strict 1:1 scale mapping (no overlays).
    """
    plotter = pv.Plotter(window_size=[900, 800])
    plotter.background_color = "#ffffff"

    # CRITICAL FIX: The base geometry template must be set to a base radius of 1.0.
    # When glyphing, scale="radius_scaling" multiplies this template radius directly.
    sphere_template = pv.Sphere(radius=1.0, theta_resolution=40, phi_resolution=40)

    if is_nacl:
        na_mask = species == 0
        cl_mask = species == 1

        # Render Na+ Cations
        na_cloud = pv.PolyData(positions[na_mask])
        na_cloud.point_data["radius_scaling"] = radii[na_mask]
        # CRITICAL FIX: factor=1.0 maps the array value 1:1 onto the sphere radius
        na_glyphs = na_cloud.glyph(scale="radius_scaling", factor=1.0, geom=sphere_template)
        plotter.add_mesh(na_glyphs, color="#e74c3c", smooth_shading=True, 
                         specular=0.4, specular_power=20, label="Na+ Cation")

        # Render Cl- Anions
        cl_cloud = pv.PolyData(positions[cl_mask])
        cl_cloud.point_data["radius_scaling"] = radii[cl_mask]
        cl_glyphs = cl_cloud.glyph(scale="radius_scaling", factor=1.0, geom=sphere_template)
        plotter.add_mesh(cl_glyphs, color="#2ecc71", smooth_shading=True, 
                         specular=0.4, specular_power=20, label="Cl- Anion")
        
        plotter.add_legend(bcolor=None, face="none", size=[0.15, 0.15])
    else:
        # Monodisperse system uniform rendering pass
        point_cloud = pv.PolyData(positions)
        point_cloud.point_data["radius_scaling"] = radii
        all_glyphs = point_cloud.glyph(scale="radius_scaling", factor=1.0, geom=sphere_template)
        plotter.add_mesh(all_glyphs, color="#3498db", smooth_shading=True, 
                         specular=0.4, specular_power=20)

    # Unit boundary box frame outline
    box_outline = pv.Box(bounds=(0, box_size[0], 0, box_size[1], 0, box_size[2]))
    plotter.add_mesh(box_outline, color="#2c3e50", style="wireframe", line_width=2)

    # Shading enhancements to clearly show tangent contact point intersections
    plotter.enable_eye_dome_lighting() 
    plotter.show_axes()
    
    plotter.show_grid(
        color="#bdc3c7", 
        n_xlabels=int(box_size[0]) + 1, 
        n_ylabels=int(box_size[1]) + 1, 
        n_zlabels=int(box_size[2]) + 1,
        xtitle="X (Lattice)",
        ytitle="Y (Lattice)",
        ztitle="Z (Lattice)"
    )
    
    plotter.show()


# =============================================================================
# RUN VERIFICATION
# =============================================================================
if __name__ == "__main__":
    # Test setting 'fcc' or 'nacl' to confirm hard contact boundaries without overlap!
    positions, radii, box_size, species = generate_3d_lattice(lattice_type='bcc', repeats=3)
    plot_3d_packing_pyvista(positions, radii, box_size, species, is_nacl=False)