"""
Liner Mesh Generator — addon de Blender.

Flujo:
    1) Preparar escaneo    -> duplica el escaneo, lo pinta de gris y entra a
                              Vertex Paint con brocha roja. Pinta la CRESTA.
    2) Cortar por cresta   -> lee lo pintado, ajusta la curva de cresta,
                              clasifica cada vertice arriba/abajo y crea un
                              OBJETO NUEVO con el lado elegido.
    3) Generar malla       -> reconstruye el liner a partir del objeto cortado.
"""

bl_info = {
    "name": "Liner Mesh Generator",
    "author": "Fernando Morales Magallón",
    "version": (1, 1, 0),
    "blender": (5, 2, 0),
    "location": "View3D > Sidebar (N) > Liner",
    "description": "Pinta la cresta, corta el escaneo y reconstruye el liner",
    "category": "Mesh",
}

import numpy as np
import bpy
from bpy.props import (IntProperty, FloatProperty, BoolProperty,
                       EnumProperty, PointerProperty)
from bpy.types import Operator, Panel, PropertyGroup

from . import linergen

PAINT_ATTR = "crest_paint"
GRIS = (0.6, 0.6, 0.6, 1.0)


def puntos_de_objeto(obj):
    mw = np.array(obj.matrix_world)
    n = len(obj.data.vertices)
    co = np.empty(n * 3, dtype=float)
    obj.data.vertices.foreach_get("co", co)
    co = co.reshape(n, 3)
    co_h = np.column_stack([co, np.ones(n)])
    return (co_h @ mw.T)[:, :3]


def scipy_disponible():
    try:
        import scipy  # noqa: F401
        return True
    except Exception:
        return False


def _np_to_matrix(M):
    from mathutils import Matrix
    return Matrix([[float(M[i][j]) for j in range(4)] for i in range(4)])


def _asegurar_material_gris(obj):
    mat = bpy.data.materials.get("Liner_Gris") or bpy.data.materials.new("Liner_Gris")
    mat.use_nodes = False
    mat.diffuse_color = GRIS
    if not obj.data.materials:
        obj.data.materials.append(mat)
    else:
        obj.data.materials[0] = mat


def _asegurar_atributo_pintura(mesh):
    attr = mesh.color_attributes.get(PAINT_ATTR)
    if attr is None:
        attr = mesh.color_attributes.new(name=PAINT_ATTR, type='FLOAT_COLOR', domain='POINT')
        n = len(attr.data)
        buf = np.tile(np.array(GRIS, dtype=np.float32), n)
        attr.data.foreach_set("color", buf)
    mesh.color_attributes.active_color = attr
    return attr


def _leer_pintura(mesh):
    attr = mesh.color_attributes.get(PAINT_ATTR)
    if attr is None or attr.domain != 'POINT':
        return None
    n = len(attr.data)
    cols = np.empty(n * 4, dtype=np.float32)
    attr.data.foreach_get("color", cols)
    return cols.reshape(n, 4)[:, :3]


def _crest_z_function(theta_c, z_c, n_bins=180):
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    idx = np.clip(np.digitize(theta_c, bins) - 1, 0, n_bins - 1)
    zmean = np.full(n_bins, np.nan)
    for b in range(n_bins):
        m = idx == b
        if m.any():
            zmean[b] = z_c[m].mean()
    centers = 0.5 * (bins[:-1] + bins[1:])
    valid = ~np.isnan(zmean)
    tc, zc = centers[valid], zmean[valid]
    text = np.concatenate([tc - 2*np.pi, tc, tc + 2*np.pi])
    zext = np.tile(zc, 3)
    return lambda q: np.interp(q, text, zext)


def clasificar_por_cresta(world_co, painted_mask, lado='ABAJO', tol=0.0):
    ax = linergen._pca_main_axis(world_co)
    R = linergen.rotation_matrix_from_vectors(ax, np.array([0, 0, 1.0]))
    P = world_co @ R.T
    cx, cy = P[:, 0].mean(), P[:, 1].mean()
    Pc = P[painted_mask]
    thc = np.arctan2(Pc[:, 1] - cy, Pc[:, 0] - cx)
    f = _crest_z_function(thc, Pc[:, 2])
    th = np.arctan2(P[:, 1] - cy, P[:, 0] - cx)
    crest_z = f(th)
    above = P[:, 2] > crest_z + tol
    return (~above) if lado == 'ABAJO' else above


class LinerProps(PropertyGroup):
    lado: EnumProperty(
        name="Conservar",
        description="Lado a conservar respecto a la cresta pintada",
        items=[('ABAJO', "Abajo (cuerpo)", "Conserva lo que esta por debajo de la cresta"),
               ('ARRIBA', "Arriba", "Conserva lo que esta por encima de la cresta")],
        default='ABAJO')
    tol_rojo: FloatProperty(name="Tolerancia rojo", default=0.5, min=0.05, max=1.0,
                            description="Que tan cerca del rojo puro cuenta como cresta")
    tol_corte: FloatProperty(name="Margen de corte", default=0.0, min=-5.0, max=5.0,
                             description="Desplaza el corte sobre/bajo la cresta")
    n_slices:   IntProperty(name="N slices", default=40, min=5, max=400)
    n_bins_env: IntProperty(name="N bins cresta", default=80, min=8, max=720)
    n_circ:     IntProperty(name="N puntos seccion", default=40, min=6, max=360)
    n_nodos:    IntProperty(name="Nodos seccion", default=10, min=2, max=60)
    suav_env:   FloatProperty(name="Suavizado cresta", default=1.0, min=0.0, max=50.0)
    sellar_base:     BoolProperty(name="Sellar base", default=True)
    orient_esferico: BoolProperty(name="Extremo esferico abajo", default=False)
    rotacion_z:      BoolProperty(name="Rotacion Z (alinear minimo)", default=False)


class LINER_OT_prepare(Operator):
    bl_idname = "liner.prepare"
    bl_label = "Preparar escaneo (pintar cresta)"
    bl_description = "Duplica el escaneo, lo pinta de gris y entra a Vertex Paint (brocha roja)"

    def execute(self, context):
        src = context.active_object
        if src is None or src.type != 'MESH':
            self.report({'ERROR'}, "Selecciona el objeto del escaneo (malla).")
            return {'CANCELLED'}
        if src.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        new = src.copy()
        new.data = src.data.copy()
        new.name = src.name + "_paint"
        context.collection.objects.link(new)

        _asegurar_material_gris(new)
        _asegurar_atributo_pintura(new.data)

        for o in context.selected_objects:
            o.select_set(False)
        new.select_set(True)
        context.view_layer.objects.active = new

        bpy.ops.object.mode_set(mode='VERTEX_PAINT')
        ts = context.tool_settings
        try:
            ts.vertex_paint.brush.color = (1.0, 0.0, 0.0)
            ts.unified_paint_settings.color = (1.0, 0.0, 0.0)
        except Exception:
            pass
        self.report({'INFO'}, "Pinta la CRESTA de rojo. Luego 'Cortar por cresta'.")
        return {'FINISHED'}


class LINER_OT_cut(Operator):
    bl_idname = "liner.cut"
    bl_label = "Cortar por cresta"
    bl_description = "Corta el escaneo por la cresta pintada y crea un objeto nuevo"

    def execute(self, context):
        import bmesh
        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "Selecciona el objeto pintado.")
            return {'CANCELLED'}
        if obj.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        pr = context.scene.liner_props
        cols = _leer_pintura(obj.data)
        if cols is None:
            self.report({'ERROR'}, "No hay capa de pintura. Usa 'Preparar escaneo' primero.")
            return {'CANCELLED'}

        dist_rojo = np.linalg.norm(cols - np.array([1.0, 0.0, 0.0]), axis=1)
        painted = dist_rojo < pr.tol_rojo
        if painted.sum() < 8:
            self.report({'ERROR'}, "Pocos vertices pintados (%d)." % int(painted.sum()))
            return {'CANCELLED'}

        world = puntos_de_objeto(obj)
        keep = clasificar_por_cresta(world, painted, lado=pr.lado, tol=pr.tol_corte)
        if keep.sum() < 10 or (~keep).sum() < 1:
            self.report({'ERROR'}, "El corte dejo un lado casi vacio. Revisa la cresta / el margen.")
            return {'CANCELLED'}

        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        a_borrar = [v for v in bm.verts if not keep[v.index]]
        bmesh.ops.delete(bm, geom=a_borrar, context='VERTS')

        new_mesh = bpy.data.meshes.new(obj.name + "_cut_mesh")
        bm.to_mesh(new_mesh)
        bm.free()
        new_obj = bpy.data.objects.new(obj.name.replace("_paint", "") + "_cut", new_mesh)
        new_obj.matrix_world = obj.matrix_world
        context.collection.objects.link(new_obj)

        for o in context.selected_objects:
            o.select_set(False)
        new_obj.select_set(True)
        context.view_layer.objects.active = new_obj
        self.report({'INFO'}, "Corte listo (%d vertices). Ya puedes 'Generar malla'." % int(keep.sum()))
        return {'FINISHED'}


class LINER_OT_install_scipy(Operator):
    bl_idname = "liner.install_scipy"
    bl_label = "Instalar SciPy"
    bl_description = "Instala SciPy en el Python de Blender (requiere internet)"

    def execute(self, context):
        import subprocess, sys
        try:
            subprocess.check_call([sys.executable, "-m", "ensurepip", "--user"])
        except Exception:
            pass
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"])
        except Exception as e:
            self.report({'ERROR'}, "No se pudo instalar SciPy: %s" % e)
            return {'CANCELLED'}
        self.report({'INFO'}, "SciPy instalado. Reinicia Blender si el import falla.")
        return {'FINISHED'}


class LINER_OT_generate(Operator):
    bl_idname = "liner.generate"
    bl_label = "Generar malla del liner"
    bl_description = "Reconstruye el liner a partir del objeto seleccionado (el cortado)"

    def execute(self, context):
        if not scipy_disponible():
            self.report({'ERROR'}, "Falta SciPy. Usa el boton 'Instalar SciPy'.")
            return {'CANCELLED'}
        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "Selecciona el objeto cortado (malla).")
            return {'CANCELLED'}
        if obj.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        pr = context.scene.liner_props
        puntos = puntos_de_objeto(obj)
        if len(puntos) < 100:
            self.report({'ERROR'}, "Muy pocos vertices (%d)." % len(puntos))
            return {'CANCELLED'}

        cfg = linergen.Config(
            N_SLICES=pr.n_slices, N_BINS_ENV=pr.n_bins_env, N_CIRC=pr.n_circ,
            N_NODOS_SECCION=pr.n_nodos, SUAVIZADO_ENV=pr.suav_env,
            SELLAR_BASE=pr.sellar_base, ORIENT_SPHERICAL_DOWN=pr.orient_esferico,
            APLICAR_ROTACION_Z=pr.rotacion_z, IFSHOW=False,
        )
        try:
            gen = linergen.LinerGen(cfg, puntos=puntos)
            gen.compute_centerline(); gen.compute_axis()
            gen.align_to_z(); gen.order_by_z()
            gen.crest.extract_all(ifshow=False)
            if cfg.APLICAR_ROTACION_Z:
                gen.align_min_z()
            gen.mesh.build(ifshow=False)
        except Exception as e:
            self.report({'ERROR'}, "Fallo el pipeline: %s" % e)
            return {'CANCELLED'}

        mesh = bpy.data.meshes.new("Liner_mesh")
        verts = [tuple(map(float, v)) for v in gen.mesh.vertices]
        faces = [tuple(int(i) for i in f) for f in gen.mesh.caras]
        mesh.from_pydata(verts, [], faces)
        mesh.validate(clean_customdata=False)
        mesh.update()
        nuevo = bpy.data.objects.new("Liner", mesh)
        context.collection.objects.link(nuevo)

        M = np.eye(4); M[:3, :3] = np.array(gen.R).T
        nuevo.matrix_world = obj.matrix_world @ _np_to_matrix(M)

        for o in context.selected_objects:
            o.select_set(False)
        nuevo.select_set(True)
        context.view_layer.objects.active = nuevo
        self.report({'INFO'}, "Liner generado: %d vertices." % len(gen.mesh.vertices))
        return {'FINISHED'}


class LINER_PT_panel(Panel):
    bl_label = "Liner Mesh Generator"
    bl_idname = "LINER_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Liner"

    def draw(self, context):
        layout = self.layout
        pr = context.scene.liner_props

        box = layout.box()
        box.label(text="1) Pintar la cresta", icon='BRUSH_DATA')
        box.operator("liner.prepare", icon='GREASEPENCIL')

        box = layout.box()
        box.label(text="2) Cortar por cresta", icon='MOD_BEVEL')
        box.prop(pr, "lado")
        row = box.row(align=True)
        row.prop(pr, "tol_rojo"); row.prop(pr, "tol_corte")
        box.operator("liner.cut", icon='MOD_BOOLEAN')

        box = layout.box()
        box.label(text="3) Generar malla", icon='MESH_CYLINDER')
        if not scipy_disponible():
            b = box.box()
            b.label(text="Falta SciPy", icon='ERROR')
            b.operator("liner.install_scipy", icon='CONSOLE')
        col = box.column(align=True)
        col.prop(pr, "n_slices"); col.prop(pr, "n_bins_env")
        col.prop(pr, "n_circ"); col.prop(pr, "n_nodos"); col.prop(pr, "suav_env")
        col = box.column(align=True)
        col.prop(pr, "sellar_base"); col.prop(pr, "orient_esferico"); col.prop(pr, "rotacion_z")
        box.operator("liner.generate", icon='MESH_CYLINDER')


_clases = (LinerProps, LINER_OT_prepare, LINER_OT_cut,
           LINER_OT_install_scipy, LINER_OT_generate, LINER_PT_panel)


def register():
    for c in _clases:
        bpy.utils.register_class(c)
    bpy.types.Scene.liner_props = PointerProperty(type=LinerProps)


def unregister():
    del bpy.types.Scene.liner_props
    for c in reversed(_clases):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()