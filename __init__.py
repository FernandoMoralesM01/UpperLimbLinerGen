"""
Liner Mesh Generator — addon de Blender.

Flujo:
    1) Preparar escaneo    -> duplica el escaneo, lo pinta de gris y entra a
                              Vertex Paint con brocha roja. Pinta la REGION que
                              quieres conservar (segmentacion binaria).
    2) Segmentar (cortar)  -> binariza lo pintado: conserva lo pintado (o lo no
                              pintado) y crea un OBJETO NUEVO con esa segmentacion.
    3) Generar malla       -> reconstruye el liner a partir del objeto segmentado.
"""

bl_info = {
    "name": "Liner Mesh Generator",
    "author": "Fernando Morales Magallón",
    "version": (1, 2, 0),
    "blender": (5, 0, 0),
    "location": "View3D > Sidebar (N) > Liner",
    "description": "Pinta la region, segmenta el escaneo y reconstruye el liner",
    "category": "Mesh",
}

import numpy as np
import bpy
from bpy.props import (IntProperty, FloatProperty, BoolProperty,
                       EnumProperty, PointerProperty)
from bpy.types import Operator, Panel, PropertyGroup

from . import linergen

PAINT_ATTR = "crest_paint"
GRIS = (0.7, 0.7, 0.7, 0.7)
ROJO = (0.0, 0.0, 0.0)


# ----------------------------------------------------------------------
# Utilidades Blender <-> numpy
# ----------------------------------------------------------------------
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
    """Devuelve (N,3) con el color RGB por vertice, o None si no hay capa."""
    attr = mesh.color_attributes.get(PAINT_ATTR)
    if attr is None or attr.domain != 'POINT':
        return None
    n = len(attr.data)
    cols = np.empty(n * 4, dtype=np.float32)
    attr.data.foreach_get("color", cols)
    return cols.reshape(n, 4)[:, :3]


def _mascara_pintada(mesh, tol):
    """Segmentacion binaria: True en los vertices pintados de rojo."""
    cols = _leer_pintura(mesh)
    if cols is None:
        return None
    dist = np.linalg.norm(cols - np.array(ROJO), axis=1)
    return dist < tol


# ----------------------------------------------------------------------
# Propiedades
# ----------------------------------------------------------------------
class LinerProps(PropertyGroup):
    # --- segmentacion ---
    lado: EnumProperty(
        name="Conservar",
        description="Que parte conservar tras la segmentacion binaria",
        items=[('PINTADO', "Lo pintado", "Conserva los vertices pintados"),
               ('NO_PINTADO', "Lo no pintado", "Conserva los vertices SIN pintar")],
        default='PINTADO')
    tol_rojo: FloatProperty(name="Tolerancia rojo", default=0.5, min=0.05, max=1.0,
                            description="Que tan cerca del rojo puro cuenta como pintado")
    solo_mayor_isla: BoolProperty(
        name="Solo la isla mayor", default=False,
        description="Tras segmentar, conserva unicamente el fragmento conectado mas grande")
 
    # --- pipeline ---
    n_slices:   IntProperty(name="# segmentos verticales", default=30, min=5, max=400)
    n_pts_reg:  IntProperty(name="# puntos extrapolación", default=3, min=1, max=10)

    n_bins_env: IntProperty(name="# puntos de la cresta", default=180, min=8, max=720)
    #n_env_fino: IntProperty(name="factor de suavizado cresta", default=500, min=100, max=2000)

    orden_k: IntProperty(name="orden polinomial", default=5, min=1, max=20)
    n_circ:     IntProperty(name="# puntos en cada segmento", default=40, min=6, max=360)
    n_nodos:    IntProperty(name="# nodos splines", default=10, min=2, max=60)
    suav_env:   FloatProperty(name="factor de suavizado cresta", default=1.0, min=0.0, max=50.0)
    #n_z1:       FloatProperty(name="# anillos parte inferior", default=None, min=5.0, max=1000.0)
    #n_z2:       FloatProperty(name="# anillos parte superior", default=None, min=2.0, max=1000.0)

    sellar_base:     BoolProperty(name="Sellar base", default=True)
    orient_esferico: BoolProperty(name="Extremo esferico abajo", default=True)
    rotacion_z:      BoolProperty(name="Rotacion Z (alinear minimo)", default=False)
    frac_casquete: FloatProperty(name="Fracción del largo casquete", default=0.3, min=0.1, max=0.8)

# ----------------------------------------------------------------------
# Operadores
# ----------------------------------------------------------------------
class LINER_OT_prepare(Operator):
    bl_idname = "liner.prepare"
    bl_label = "Preparar escaneo (pintar region)"
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
            ts.vertex_paint.brush.color = ROJO
            ts.unified_paint_settings.color = ROJO
        except Exception:
            pass
        self.report({'INFO'}, "Pinta de rojo la REGION a conservar. Luego 'Segmentar'.")
        return {'FINISHED'}


class LINER_OT_cut(Operator):
    bl_idname = "liner.cut"
    bl_label = "Segmentar por pintura"
    bl_description = "Binariza lo pintado y crea un objeto nuevo con la segmentacion"

    def execute(self, context):
        import bmesh
        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "Selecciona el objeto pintado.")
            return {'CANCELLED'}
        if obj.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        pr = context.scene.liner_props
        painted = _mascara_pintada(obj.data, pr.tol_rojo)
        if painted is None:
            self.report({'ERROR'}, "No hay capa de pintura. Usa 'Preparar escaneo' primero.")
            return {'CANCELLED'}
        if painted.sum() < 3:
            self.report({'ERROR'}, "Casi nada pintado (%d vertices)." % int(painted.sum()))
            return {'CANCELLED'}

        keep = painted if pr.lado == 'PINTADO' else ~painted
        if keep.sum() < 3 or (~keep).sum() < 1:
            self.report({'ERROR'}, "La segmentacion dejo un lado vacio. Revisa la pintura.")
            return {'CANCELLED'}

        # Segmentacion binaria: borrar los vertices NO conservados (arrastra sus caras)
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        a_borrar = [v for v in bm.verts if not keep[v.index]]
        bmesh.ops.delete(bm, geom=a_borrar, context='VERTS')

        if pr.solo_mayor_isla:
            self._quedar_isla_mayor(bm)

        new_mesh = bpy.data.meshes.new(obj.name + "_seg_mesh")
        bm.to_mesh(new_mesh)
        bm.free()
        new_obj = bpy.data.objects.new(obj.name.replace("_paint", "") + "_seg", new_mesh)
        new_obj.matrix_world = obj.matrix_world
        context.collection.objects.link(new_obj)

        for o in context.selected_objects:
            o.select_set(False)
        new_obj.select_set(True)
        context.view_layer.objects.active = new_obj
        self.report({'INFO'}, "Segmentado: %d vertices. Ya puedes 'Generar malla'." % len(new_mesh.vertices))
        return {'FINISHED'}

    @staticmethod
    def _quedar_isla_mayor(bm):
        """Conserva solo el fragmento conectado (isla) con mas vertices."""
        restantes = set(bm.verts)
        islas = []
        visitados = set()
        for v in bm.verts:
            if v in visitados:
                continue
            pila, isla = [v], []
            visitados.add(v)
            while pila:
                w = pila.pop()
                isla.append(w)
                for e in w.link_edges:
                    o = e.other_vert(w)
                    if o not in visitados:
                        visitados.add(o); pila.append(o)
            islas.append(isla)
        if len(islas) <= 1:
            return
        islas.sort(key=len, reverse=True)
        borrar = [v for isla in islas[1:] for v in isla]
        bmesh.ops.delete(bm, geom=borrar, context='VERTS')


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
    bl_description = "Reconstruye el liner a partir del objeto seleccionado (el segmentado)"

    def execute(self, context):
        if not scipy_disponible():
            self.report({'ERROR'}, "Falta SciPy. Usa el boton 'Instalar SciPy'.")
            return {'CANCELLED'}
        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "Selecciona el objeto segmentado (malla).")
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
            N_PTS_REGRESION=pr.n_pts_reg, ORDEN_K=pr.orden_k,  
            N_NODOS_SECCION=pr.n_nodos, SUAVIZADO_ENV=pr.suav_env,
            FRAC_CASQUETE=pr.frac_casquete,
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


# ----------------------------------------------------------------------
# Panel
# ----------------------------------------------------------------------
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
        box.label(text="1) Pintar la region", icon='BRUSH_DATA')
        box.operator("liner.prepare", icon='GREASEPENCIL')

        box = layout.box()
        box.label(text="2) Segmentar por pintura", icon='MOD_MASK')
        box.prop(pr, "lado")
        box.prop(pr, "tol_rojo")
        box.prop(pr, "solo_mayor_isla")
        box.operator("liner.cut", icon='MOD_BOOLEAN')

        box = layout.box()
        box.label(text="3) Generar malla", icon='MESH_CYLINDER')
        if not scipy_disponible():
            b = box.box()
            b.label(text="Falta SciPy", icon='ERROR')
            b.operator("liner.install_scipy", icon='CONSOLE')
        col = box.column(align=True)
        col.prop(pr, "n_slices"); col.prop(pr, "n_bins_env")
        col.prop(pr, "n_pts_reg"); col.prop(pr, "orden_k")
                  
        col.prop(pr, "n_circ"); col.prop(pr, "n_nodos"); col.prop(pr, "suav_env")
        col.prop(pr, "frac_casquete")
        
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
