"""
Retopologia Mesh Generator — addon de Blender.

Flujo:
    1) Preparar escaneo    -> duplica el escaneo, lo pinta de gris y entra a
                              Vertex Paint con brocha roja. Pinta la REGION que
                              quieres conservar (segmentacion binaria).
    2) Segmentar (cortar)  -> binariza lo pintado: conserva lo pintado (o lo no
                              pintado) y crea un OBJETO NUEVO con esa segmentacion.
    3) Generar malla       -> reconstruye el liner a partir del objeto segmentado.
"""

bl_info = {
    "name": "Retopologia Mesh Generator",
    "author": "Fernando Morales Magallón",
    "version": (1, 2, 0),
    "blender": (5, 0, 0),
    "location": "View3D > Sidebar (N) > Retopologia",
    "description": "Pinta la region, segmenta el escaneo y reconstruye la retopologia",
    "category": "Mesh",
}

import numpy as np
import bpy
from bpy.props import (IntProperty, FloatProperty, BoolProperty,
                       EnumProperty, PointerProperty, FloatVectorProperty)
from bpy.types import Operator, Panel, PropertyGroup
from bpy_extras import view3d_utils
from mathutils import Vector

from . import linergen

PAINT_ATTR = "crest_paint"
GRIS = (0.7, 0.7, 0.7, 0.7)
ROJO = (0.0, 0.0, 0.0)
EMPTY_BASE = "Retopologia_PuntoBase"


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


def _habilitar_user_site():
    """Hace visible para Blender el site-packages del usuario de su propio Python."""
    import site
    import sys
    try:
        user_site = site.getusersitepackages()
        if user_site and user_site not in sys.path:
            site.addsitedir(user_site)
    except Exception:
        pass


def scipy_disponible():
    try:
        _habilitar_user_site()
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
# Punto base (extremo inferior del muñon)
# ----------------------------------------------------------------------
def _empty_base(crear=False):
    """Empty que marca el punto base en el viewport."""
    e = bpy.data.objects.get(EMPTY_BASE)
    if e is None and crear:
        e = bpy.data.objects.new(EMPTY_BASE, None)
        e.empty_display_type = 'SPHERE'
        e.show_in_front = True
        bpy.context.collection.objects.link(e)
    return e


def _fijar_punto_base(context, co_mundo, escala=1.0):
    pr = context.scene.retopologia_props
    pr.punto_base = co_mundo
    pr.punto_base_ok = True
    e = _empty_base(crear=True)
    e.location = co_mundo
    e.empty_display_size = max(1e-4, 0.05 * escala)
    return e


def _obtener_punto_base(context):
    """Devuelve el punto base como np.array(3,), o None.

    Si el Empty existe, manda su posicion: asi puedes reajustarlo a mano
    moviendolo en el viewport sin volver a hacer clic.
    """
    pr = context.scene.retopologia_props
    if not pr.punto_base_ok:
        return None
    e = _empty_base()
    if e is not None:
        return np.array(e.matrix_world.translation, dtype=float)
    return np.array(pr.punto_base, dtype=float)


# ----------------------------------------------------------------------
# Propiedades
# ----------------------------------------------------------------------
class RetopologiaProps(PropertyGroup):
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

    # --- punto base ---
    punto_base: FloatVectorProperty(
        name="Punto base", subtype='XYZ', size=3, default=(0.0, 0.0, 0.0),
        description="Punto de hasta abajo del muñon, en coordenadas de mundo")
    punto_base_ok: BoolProperty(name="Punto base definido", default=False)
    usar_base_orient: BoolProperty(
        name="Orientar con el punto base", default=True,
        description="El punto base decide que extremo va abajo, en vez de la "
                    "heuristica de las semiesferas")
    usar_base_apice: BoolProperty(
        name="Sellar la base en ese punto", default=True,
        description="El apice del sellado y el z inferior de la malla salen del punto base")

    # --- zonas en Z (tres splines, de abajo hacia arriba) ---
    n_z1: IntProperty(name="# anillos Z1 (cuerpo)", default=20, min=2, max=400,
                      description="Anillos del cuerpo, el tramo tubular")
    n_z2: IntProperty(name="# anillos Z2 (intermedia)", default=10, min=2, max=400,
                      description="Anillos de la franja entre el cuerpo y la cresta")
    n_z3: IntProperty(name="# anillos Z3 (cresta)", default=10, min=1, max=400,
                      description="Anillos de la transicion que aterriza sobre la cresta")
    frac_z2: FloatProperty(
        name="Fracción Z2", default=0.25, min=0.05, max=0.95,
        description="Franja de arriba del tubo que ocupa Z2. El resto es Z1")
    nodos_z1: IntProperty(name="Nodos spline Z1", default=6, min=1, max=30)
    nodos_z2: IntProperty(name="Nodos spline Z2", default=4, min=1, max=30)
    margen_cresta: FloatProperty(
        name="Margen bajo la cresta", default=0.10, min=0.0, max=0.5,
        description="Altura que se reserva bajo el punto mas bajo de la cresta "
                    "para la banda Z3. Si los anillos se encinan en la cresta, subelo")
    tension_cresta: FloatProperty(
        name="Tensión al llegar a la cresta", default=1.0, min=0.0, max=2.0,
        description="Curvatura del spline de Z3 al aterrizar sobre la cresta")
    zonas_avanzado: BoolProperty(name="Ajustes avanzados de zonas", default=False)

    # --- casquete inferior ---
    n_cap: IntProperty(name="# anillos del casquete", default=6, min=1, max=60,
                       description="Anillos del domo, entre el apice y el cuerpo")
    frac_cap_z: FloatProperty(
        name="Altura del casquete", default=0.15, min=0.0, max=0.6,
        description="Parte del tramo base->cresta que ocupa el casquete")

    sellar_base:     BoolProperty(name="Sellar base (casquete)", default=True)
    orient_esferico: BoolProperty(name="Extremo esferico abajo", default=True)
    rotacion_z:      BoolProperty(name="Rotacion Z (alinear minimo)", default=False)
    frac_casquete: FloatProperty(name="Fracción del largo casquete", default=0.3, min=0.1, max=0.8)

# ----------------------------------------------------------------------
# Operadores
# ----------------------------------------------------------------------
class RETOPOLOGIA_OT_prepare(Operator):
    bl_idname = "retopologia.prepare"
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


class RETOPOLOGIA_OT_cut(Operator):
    bl_idname = "retopologia.cut"
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

        pr = context.scene.retopologia_props
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
        self.report({'INFO'}, "Segmentado: %d vertices. Ya puedes 'Generar retopologia'." % len(new_mesh.vertices))
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


class RETOPOLOGIA_OT_install_scipy(Operator):
    bl_idname = "retopologia.install_scipy"
    bl_label = "Instalar SciPy"
    bl_description = "Instala SciPy en el Python de Blender (requiere internet)"

    def execute(self, context):
        import os
        import subprocess
        import sys
        import site

        # Blender instalado en Program Files normalmente no permite escribir
        # en su propio site-packages sin permisos de administrador. Por eso
        # instalamos SciPy en el site-packages del usuario y lo añadimos
        # explícitamente al sys.path de Blender.
        try:
            user_site = site.getusersitepackages()
            os.makedirs(user_site, exist_ok=True)

            env = os.environ.copy()
            env.pop("PYTHONNOUSERSITE", None)

            subprocess.check_call([
                sys.executable, "-m", "ensurepip", "--user"
            ], env=env)

            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "--user", "--upgrade", "scipy"
            ], env=env)

            if user_site not in sys.path:
                site.addsitedir(user_site)

            # Si SciPy ya estaba cargado de forma incorrecta, quitamos su
            # entrada para que el siguiente import use la instalación nueva.
            for name in list(sys.modules):
                if name == "scipy" or name.startswith("scipy."):
                    del sys.modules[name]

            import scipy
            self.report(
                {'INFO'},
                "SciPy %s instalado correctamente. Ya puedes generar la retopologia."
                % scipy.__version__
            )
            return {'FINISHED'}

        except Exception as e:
            self.report(
                {'ERROR'},
                "No se pudo instalar SciPy: %s" % e
            )
            return {'CANCELLED'}


# ----------------------------------------------------------------------
# Operadores: punto base
# ----------------------------------------------------------------------
class RETOPOLOGIA_OT_pick_base(Operator):
    bl_idname = "retopologia.pick_base"
    bl_label = "Elegir punto base (clic)"
    bl_description = ("Haz clic sobre la malla para marcar el punto de hasta abajo "
                      "del muñon. Esc o clic derecho para cancelar")

    _obj = None

    def invoke(self, context, event):
        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "Selecciona primero el objeto segmentado.")
            return {'CANCELLED'}
        if context.area is None or context.area.type != 'VIEW_3D':
            self.report({'ERROR'}, "Ejecuta esto desde la vista 3D.")
            return {'CANCELLED'}
        if obj.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        self._obj = obj
        context.window.cursor_modal_set('EYEDROPPER')
        context.workspace.status_text_set(
            "Clic izquierdo: fijar el punto base   |   Esc o clic derecho: cancelar")
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def _fin(self, context):
        context.window.cursor_modal_restore()
        context.workspace.status_text_set(None)

    def modal(self, context, event):
        if event.type in {'RIGHTMOUSE', 'ESC'}:
            self._fin(context)
            return {'CANCELLED'}

        # que la navegacion siga funcionando mientras se elige
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:
            return {'PASS_THROUGH'}

        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            co = self._raycast(context, event)
            if co is None:
                self.report({'WARNING'}, "El rayo no toco la malla. Intenta otra vez.")
                return {'RUNNING_MODAL'}
            _fijar_punto_base(context, co, max(self._obj.dimensions))
            self._fin(context)
            self.report({'INFO'}, "Punto base: (%.3f, %.3f, %.3f)" % (co.x, co.y, co.z))
            return {'FINISHED'}

        return {'RUNNING_MODAL'}

    def _raycast(self, context, event):
        region = context.region
        rv3d = context.region_data
        if region is None or rv3d is None:
            return None

        coord = (event.mouse_region_x, event.mouse_region_y)
        origen = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        direccion = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)

        obj = self._obj
        mwi = obj.matrix_world.inverted()
        o = mwi @ origen
        d = (mwi.to_3x3() @ direccion).normalized()

        ok, loc, nor, cara = obj.ray_cast(o, d)
        if not ok:
            return None

        # engancha al vertice mas cercano de la cara tocada
        me = obj.data
        if 0 <= cara < len(me.polygons):
            mejor, dmin = None, 1e30
            for vi in me.polygons[cara].vertices:
                v = me.vertices[vi].co
                dd = (v - loc).length_squared
                if dd < dmin:
                    dmin, mejor = dd, v
            if mejor is not None:
                loc = mejor

        return obj.matrix_world @ Vector(loc)


class RETOPOLOGIA_OT_base_lowest(Operator):
    bl_idname = "retopologia.base_lowest"
    bl_label = "Punto base = vértice más bajo"
    bl_description = "Toma el vertice de menor Z en coordenadas de mundo"

    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "Selecciona el objeto segmentado.")
            return {'CANCELLED'}
        pts = puntos_de_objeto(obj)
        if len(pts) == 0:
            self.report({'ERROR'}, "La malla no tiene vertices.")
            return {'CANCELLED'}
        co = pts[int(np.argmin(pts[:, 2]))]
        _fijar_punto_base(context, Vector((float(co[0]), float(co[1]), float(co[2]))),
                          max(obj.dimensions))
        self.report({'INFO'}, "Punto base = vertice mas bajo en Z.")
        return {'FINISHED'}


class RETOPOLOGIA_OT_base_from_selection(Operator):
    bl_idname = "retopologia.base_from_selection"
    bl_label = "Punto base = selección"
    bl_description = "Usa el vertice seleccionado, o el promedio de la seleccion"

    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "Selecciona el objeto segmentado.")
            return {'CANCELLED'}

        modo = obj.mode
        if modo == 'EDIT':
            bpy.ops.object.mode_set(mode='OBJECT')   # refresca la seleccion
        sel = [v.co.copy() for v in obj.data.vertices if v.select]
        if modo == 'EDIT':
            bpy.ops.object.mode_set(mode='EDIT')

        if not sel:
            self.report({'ERROR'}, "No hay vertices seleccionados.")
            return {'CANCELLED'}

        co = Vector((0.0, 0.0, 0.0))
        for v in sel:
            co += v
        co /= len(sel)
        _fijar_punto_base(context, obj.matrix_world @ co, max(obj.dimensions))
        self.report({'INFO'}, "Punto base desde %d vertices." % len(sel))
        return {'FINISHED'}


class RETOPOLOGIA_OT_base_from_cursor(Operator):
    bl_idname = "retopologia.base_from_cursor"
    bl_label = "Punto base = cursor 3D"
    bl_description = "Usa la posicion actual del cursor 3D"

    def execute(self, context):
        obj = context.active_object
        esc = max(obj.dimensions) if obj is not None and obj.type == 'MESH' else 1.0
        _fijar_punto_base(context, context.scene.cursor.location.copy(), esc)
        return {'FINISHED'}


class RETOPOLOGIA_OT_base_clear(Operator):
    bl_idname = "retopologia.base_clear"
    bl_label = "Borrar punto base"
    bl_description = "Olvida el punto base y vuelve a la deteccion automatica"

    def execute(self, context):
        context.scene.retopologia_props.punto_base_ok = False
        e = _empty_base()
        if e is not None:
            bpy.data.objects.remove(e, do_unlink=True)
        return {'FINISHED'}


class RETOPOLOGIA_OT_generate(Operator):
    bl_idname = "retopologia.generate"
    bl_label = "Generar retopologia"
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

        pr = context.scene.retopologia_props
        puntos = puntos_de_objeto(obj)
        if len(puntos) < 100:
            self.report({'ERROR'}, "Muy pocos vertices (%d)." % len(puntos))
            return {'CANCELLED'}

        cfg = linergen.Config(
            N_SLICES=pr.n_slices, N_BINS_ENV=pr.n_bins_env, N_CIRC=pr.n_circ,
            N_PTS_REGRESION=pr.n_pts_reg, ORDEN_K=pr.orden_k,
            N_NODOS_SECCION=pr.n_nodos, SUAVIZADO_ENV=pr.suav_env,
            FRAC_CASQUETE=pr.frac_casquete,
            N_Z1=pr.n_z1, N_Z2=pr.n_z2, N_Z3=pr.n_z3,
            FRAC_Z2=pr.frac_z2, NODOS_Z1=pr.nodos_z1, NODOS_Z2=pr.nodos_z2,
            MARGEN_CRESTA=pr.margen_cresta, TENSION_CRESTA=pr.tension_cresta,
            N_CAP=pr.n_cap, FRAC_CAP_Z=pr.frac_cap_z,
            PUNTO_BASE=_obtener_punto_base(context),
            USAR_BASE_ORIENTACION=pr.usar_base_orient,
            USAR_BASE_APICE=pr.usar_base_apice,
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

        mesh = bpy.data.meshes.new("Retopologia_mesh")
        verts = [tuple(map(float, v)) for v in gen.mesh.vertices]
        faces = [tuple(int(i) for i in f) for f in gen.mesh.caras]
        mesh.from_pydata(verts, [], faces)
        mesh.validate(clean_customdata=False)
        mesh.update()
        nuevo = bpy.data.objects.new("Retopologia", mesh)
        context.collection.objects.link(nuevo)

        M = np.eye(4); M[:3, :3] = np.array(gen.R).T
        nuevo.matrix_world = obj.matrix_world @ _np_to_matrix(M)

        for o in context.selected_objects:
            o.select_set(False)
        nuevo.select_set(True)
        context.view_layer.objects.active = nuevo
        self.report({'INFO'}, "Retopologia generada: %d vertices." % len(gen.mesh.vertices))
        return {'FINISHED'}


# ----------------------------------------------------------------------
# Panel
# ----------------------------------------------------------------------
class RETOPOLOGIA_PT_panel(Panel):
    bl_label = "Retopologia Mesh Generator"
    bl_idname = "RETOPOLOGIA_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Retopologia"

    def draw(self, context):
        layout = self.layout
        pr = context.scene.retopologia_props

        box = layout.box()
        box.label(text="1) Pintar la region", icon='BRUSH_DATA')
        box.operator("retopologia.prepare", icon='GREASEPENCIL')

        box = layout.box()
        box.label(text="2) Segmentar por pintura", icon='MOD_MASK')
        box.prop(pr, "lado")
        box.prop(pr, "tol_rojo")
        box.prop(pr, "solo_mayor_isla")
        box.operator("retopologia.cut", icon='MOD_BOOLEAN')

        # ---------------- punto base ----------------
        box = layout.box()
        fila = box.row()
        fila.label(text="3) Punto base del muñon", icon='PIVOT_CURSOR')
        if pr.punto_base_ok:
            fila.label(text="", icon='CHECKMARK')

        box.operator("retopologia.pick_base", icon='EYEDROPPER')
        sub = box.row(align=True)
        sub.operator("retopologia.base_lowest", text="Más bajo", icon='SORT_ASC')
        sub.operator("retopologia.base_from_selection", text="Selección", icon='VERTEXSEL')
        sub.operator("retopologia.base_from_cursor", text="Cursor", icon='CURSOR')

        if pr.punto_base_ok:
            col = box.column(align=True)
            col.enabled = False
            col.prop(pr, "punto_base", text="")
            box.label(text="Puedes mover el Empty '%s'." % EMPTY_BASE, icon='INFO')
            box.prop(pr, "usar_base_orient")
            box.prop(pr, "usar_base_apice")
            box.operator("retopologia.base_clear", text="Borrar punto base", icon='X')
        else:
            box.label(text="Sin punto base se usa la heurística de esferas.", icon='ERROR')

        # ---------------- generacion ----------------
        box = layout.box()
        box.label(text="4) Generar malla", icon='MESH_CYLINDER')
        if not scipy_disponible():
            b = box.box()
            b.label(text="Falta SciPy", icon='ERROR')
            b.operator("retopologia.install_scipy", icon='CONSOLE')
        col = box.column(align=True)
        col.prop(pr, "n_slices"); col.prop(pr, "n_bins_env")
        col.prop(pr, "n_pts_reg"); col.prop(pr, "orden_k")

        col.prop(pr, "n_circ"); col.prop(pr, "n_nodos"); col.prop(pr, "suav_env")
        col.prop(pr, "frac_casquete")

        # ---- zonas en Z ----
        zb = box.box()
        zb.label(text="Zonas en Z (3 splines, de abajo a arriba)", icon='IPO_BEZIER')
        col = zb.column(align=True)
        col.prop(pr, "n_z1")
        col.prop(pr, "n_z2")
        col.prop(pr, "n_z3")
        zb.prop(pr, "frac_z2", slider=True)
        zb.label(text="Z2 se lleva el %d%% de arriba del tubo; Z1 el resto"
                      % int(round(pr.frac_z2 * 100)))
        zb.prop(pr, "margen_cresta", slider=True)
        zb.prop(pr, "zonas_avanzado", toggle=True)
        if pr.zonas_avanzado:
            col = zb.column(align=True)
            col.prop(pr, "nodos_z1")
            col.prop(pr, "nodos_z2")
            col.prop(pr, "tension_cresta")

        # ---- casquete inferior ----
        cb = box.box()
        cb.label(text="Sellado inferior", icon='SPHERE')
        cb.prop(pr, "sellar_base")
        col = cb.column(align=True)
        col.enabled = pr.sellar_base
        col.prop(pr, "n_cap")
        col.prop(pr, "frac_cap_z", slider=True)

        n_cap = pr.n_cap if pr.sellar_base else 0
        box.label(text="Anillos totales: %d" % (n_cap + pr.n_z1 + 1 + pr.n_z2 + pr.n_z3),
                  icon='MESH_GRID')

        col = box.column(align=True)
        col.prop(pr, "orient_esferico"); col.prop(pr, "rotacion_z")
        box.operator("retopologia.generate", icon='MESH_CYLINDER')


_clases = (RetopologiaProps,
           RETOPOLOGIA_OT_prepare, RETOPOLOGIA_OT_cut,
           RETOPOLOGIA_OT_pick_base, RETOPOLOGIA_OT_base_lowest,
           RETOPOLOGIA_OT_base_from_selection, RETOPOLOGIA_OT_base_from_cursor,
           RETOPOLOGIA_OT_base_clear,
           RETOPOLOGIA_OT_install_scipy, RETOPOLOGIA_OT_generate,
           RETOPOLOGIA_PT_panel)


def register():
    for c in _clases:
        bpy.utils.register_class(c)
    bpy.types.Scene.retopologia_props = PointerProperty(type=RetopologiaProps)


def unregister():
    del bpy.types.Scene.retopologia_props
    for c in reversed(_clases):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()
