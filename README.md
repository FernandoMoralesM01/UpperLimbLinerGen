# Liner Mesh Generator

**Liner Mesh Generator** is a Blender add-on designed to generate a structured liner mesh from a 3D scan.

The add-on provides an interactive workflow in Blender where the user can mark the **crest of the scanned geometry** using Vertex Paint, automatically separate the desired side of the scan, and reconstruct a new liner mesh from the resulting geometry.

## Features

* Interactive crest selection using **Vertex Paint**.
* Automatic creation of a working copy of the original scan.
* Red brush configuration for marking the crest.
* Automatic detection of painted vertices.
* Geometric separation of the scan into an upper and lower region.
* Generation of a new mesh from the selected region.
* Configurable mesh resolution.
* Optional base capping.
* Optional spherical-bottom orientation.
* Optional Z-axis alignment.
* Interactive controls through the Blender Sidebar.
* Automatic SciPy availability check and installation helper.

## Workflow

The complete workflow consists of three main steps:

### 1. Prepare the scan

Select the 3D scan and click:

**Prepare Scan (Paint Crest)**

The add-on creates a copy of the selected mesh, assigns a gray material, creates the required vertex-color attribute, and switches Blender to **Vertex Paint mode** with a red brush.

Paint the **crest** of the geometry in red.

The original scan is not modified during this step.

### 2. Cut the scan using the crest

After painting the crest, click:

**Cut by Crest**

The add-on reads the painted vertices and uses them to estimate the crest profile around the main axis of the geometry.

Each vertex is then classified according to whether it is above or below the detected crest. The user can choose which side should be preserved:

* **Below (Body)** — keeps the geometry below the crest.
* **Above** — keeps the geometry above the crest.

A new object containing the selected region is then created.

### 3. Generate the liner mesh

Select the resulting cut object and click:

**Generate Liner Mesh**

The add-on processes the selected geometry and reconstructs the liner using the `linergen` module.

The generated mesh is created as a new Blender object named:

`Liner`

The generation process includes:

1. Centerline computation.
2. Main-axis computation.
3. Alignment to the Z axis.
4. Point ordering.
5. Crest extraction.
6. Optional Z rotation.
7. Mesh reconstruction.

---

## Blender Interface

After installing the add-on, the interface is available in:

**3D Viewport → Sidebar (`N`) → Liner**

The panel is organized into three sections:

```text
Liner Mesh Generator

1) Pintar la cresta
   [Preparar escaneo (pintar cresta)]

2) Cortar por cresta
   Conservar: [Abajo / Arriba]
   Tolerancia rojo: [...]
   Margen de corte: [...]
   [Cortar por cresta]

3) Generar malla
   N slices: [...]
   N bins cresta: [...]
   N puntos seccion: [...]
   Nodos seccion: [...]
   Suavizado cresta: [...]

   [✓] Sellar base
   [ ] Extremo esférico abajo
   [ ] Rotación Z

   [Generar malla del liner]
```

The panel is registered as a Blender View3D Sidebar tab called **Liner**.

---

## Parameters

### Crest cutting parameters

| Parameter           | Description                                                                     |
| ------------------- | ------------------------------------------------------------------------------- |
| **Conservar**       | Selects the region to preserve relative to the painted crest.                   |
| **Tolerancia rojo** | Controls how close a vertex color must be to pure red to be considered painted. |
| **Margen de corte** | Moves the cutting surface above or below the detected crest.                    |

The default values are:

```text
Conservar       = Abajo
Tolerancia rojo = 0.5
Margen de corte = 0.0
```

### Mesh generation parameters

| Parameter                  | Default | Description                                                 |
| -------------------------- | ------: | ----------------------------------------------------------- |
| **N slices**               |      40 | Number of longitudinal slices used to reconstruct the mesh. |
| **N bins cresta**          |      80 | Number of angular bins used to estimate the crest profile.  |
| **N puntos seccion**       |      40 | Number of points per reconstructed section.                 |
| **Nodos seccion**          |      10 | Number of nodes used for each section.                      |
| **Suavizado cresta**       |     1.0 | Controls crest smoothing.                                   |
| **Sellar base**            |  `True` | Closes the base of the generated mesh.                      |
| **Extremo esférico abajo** | `False` | Enables spherical-bottom orientation.                       |
| **Rotación Z**             | `False` | Applies Z-axis rotation based on the minimum position.      |

## These parameters are exposed through the Blender UI and passed to the `linergen.Config` object during mesh generation.

## Installation

### Requirements

* **Blender 5.2.0 or later**
* Python support provided by Blender
* **NumPy**
* **SciPy**

The add-on declares Blender 5.2.0 as its target version.

SciPy is required for the mesh-generation stage.

If SciPy is not available, the add-on displays an **Install SciPy** button that attempts to install it into Blender's Python environment. Blender may need to be restarted after installation.

### Installing the add-on

1. Open Blender.

2. Go to:

   **Edit → Preferences → Add-ons**

3. Select **Install...**

4. Select the add-on package.

5. Enable **Liner Mesh Generator**.

6. Open the 3D Viewport.

7. Press **`N`** to open the Sidebar.

8. Select the **Liner** tab.

---

## Recommended Workflow

For best results, use the following procedure:

```text
3D Scan
   │
   ▼
Prepare Scan
   │
   ▼
Paint the crest in red
   │
   ▼
Cut by Crest
   │
   ▼
Select desired side
   │
   ▼
Generate Liner Mesh
   │
   ▼
Generated Liner
```

### Important

The crest should be painted continuously around the geometry. The algorithm uses the painted vertices to estimate a crest function as a function of the angular position around the object's main axis.

A very small painted region may cause the cutting operation to fail. The add-on requires at least a minimum number of painted vertices before attempting the cut.

---

## Project Structure

The add-on is organized around two main components:

```text
Liner Mesh Generator
│
├── __init__.py
│   └── Blender interface and operators
│
└── linergen.py
    └── Liner geometry and mesh generation
```

The Blender-facing module handles:

* User interface
* Blender operators
* Vertex painting
* Crest detection
* Scan cutting
* Parameter configuration
* Object creation

The `linergen` module handles the underlying liner reconstruction pipeline.

---

## Main Components

### `LinerProps`

Stores the parameters exposed by the Blender interface, including cutting tolerances, mesh resolution, smoothing, base sealing, orientation, and rotation options.

### `LINER_OT_prepare`

Prepares the scan for crest painting by duplicating the original mesh and entering Vertex Paint mode.

### `LINER_OT_cut`

Reads the painted crest, estimates the crest surface, classifies vertices, and generates the cut mesh.

### `LINER_OT_generate`

Runs the liner reconstruction pipeline and creates the final Blender mesh.

### `LINER_PT_panel`

Creates the **Liner** tab in Blender's 3D Viewport Sidebar and exposes the complete workflow to the user.

---

## Output

The final result is generated as a new Blender mesh object:

```text
Liner
```

The mesh is created from the reconstructed vertices and faces generated by the liner-generation pipeline.

The generated object can then be further edited, inspected, or exported using Blender's standard mesh tools.

---

## Troubleshooting

### "Falta SciPy"

SciPy is not available in Blender's Python environment.

Click:

**Install SciPy**

and restart Blender if necessary.

### "Pocos vertices pintados"

Not enough vertices were detected as red.

Try:

* Painting a larger portion of the crest.
* Increasing **Tolerancia rojo**.
* Making sure the active vertex-color layer is being painted.
* Ensuring the brush color is red.

### "El corte dejó un lado casi vacío"

The detected crest produced an invalid or nearly empty region.

Try:

* Checking the painted crest.
* Painting the crest more continuously.
* Adjusting **Margen de corte**.
* Switching between **Abajo** and **Arriba**.

### "Muy pocos vertices"

The input mesh contains too few vertices for the reconstruction pipeline.

Use a sufficiently detailed scan before generating the liner.

---

## Notes

The add-on uses the **world-space coordinates** of the input mesh when performing the crest-based classification and reconstruction setup.

The original scan is preserved during the preparation stage because the add-on creates a separate copy for painting.

Generated meshes are also created as separate Blender objects, allowing the original scan and intermediate cut geometry to remain available.

---

## Version

Current add-on version:

```text
1.1.0
```

Target Blender version:

```text
5.2.0
```

---

## Author

**Fernando Morales Magallón**

---

## License


```text
:)
```

If this project is intended for academic, research, or commercial use, specify the corresponding licensing and usage conditions before distribution.
