"""
linergen — reconstrucción de un liner (geometría hueca tipo tubo con cresta)
a partir de un escaneo 3D.

Arquitectura (composición, se usa como si fueran subclases):

    gen = LinerGen(Config(RUTA_OBJ="..."))
    gen.load()
    gen.compute_centerline()
    gen.compute_axis()
    gen.align_to_z()
    gen.order_by_z()
    gen.crest.extract_all()      # <- subcomponente 'crest'
    gen.mesh.build()             # <- subcomponente 'mesh'
    gen.mesh.export("out.obj")

O todo de una:  gen.run_all()

Los imports de 'vedo' son perezosos (solo se cargan al leer/mostrar/exportar
mallas), de modo que la parte numérica funciona aunque vedo no esté instalado.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline, splprep, splev
from scipy.signal import argrelextrema
from scipy.optimize import differential_evolution


# ======================================================================
# Configuración
# ======================================================================
@dataclass
class Config:
    # --- Entrada ---
    RUTA_OBJ: str = r"escanos/ESCANEO SIN RELLENOS.obj"

    # --- Centerline ---
    N_SLICES: int        = 40      # nº de rebanadas para el centerline
    SUAVIZADO_CL: float  = 200.0   # factor de suavizado del centerline
    CL_OFFSET_MIN: float = 10.0    # recorte del extremo mínimo al rebanar
    CL_OFFSET_MAX: float = 1.0     # recorte del extremo máximo al rebanar
    N_PTS_REGRESION: int = 5       # puntos finales usados para extrapolar la recta

    # --- Cresta / envolvente ---
    N_BINS_ENV: int      = 80      # sectores angulares para la envolvente superior
    SUAVIZADO_ENV: float = 1.0     # factor s del spline periódico (× nº de puntos)
    N_ENV_FINO: int      = 500     # nº de puntos al evaluar la cresta suavizada
    VENTANA_COBERTURA: int = 1000  # ventana deslizante del análisis de cobertura
    ORDER_EXTREMOS: int  = 5       # vecindad para detectar máx/mín locales

    # --- Mallado ---
    ORDEN_K: int         = 10      # grado de los splines de regresión (se limita a 3 donde aplica)
    N_CIRC: int          = 40      # puntos por sección transversal
    N_NODOS_SECCION: int = 10      # nodos internos del spline por sección
    N_Z1: int | None     = None    # anillos en la Parte 1 (None -> N_SLICES)
    N_Z2: int | None     = None    # filas en la Parte 2 (None -> max(4, N_SLICES//3))
    SELLAR_BASE: bool    = True    # cerrar la base con un ápice en el punto mínimo

    # --- Visualización ---
    IFSHOW: bool = True            # por defecto, ¿mostrar las gráficas? (se puede sobre-escribir por llamada)

    # --- Orientación (opcional) ---
    APLICAR_ROTACION_Z: bool      = False  # girar en Z para alinear el mínimo con el centerline
    ORIENT_SPHERICAL_DOWN: bool   = False  # poner el extremo más esférico abajo
    FRAC_CASQUETE: float          = 0.15   # fracción del largo que cuenta como casquete de extremo

    def n_z1(self):
        return self.N_SLICES if self.N_Z1 is None else self.N_Z1

    def n_z2(self):
        return max(4, self.N_SLICES // 3) if self.N_Z2 is None else self.N_Z2


# ======================================================================
# Utilidades geométricas
# ======================================================================
def _pca_main_axis(X):
    """Primer componente principal (numpy, sin sklearn)."""
    Xc = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[0]


def _fit_line(t, Y):
    """Regresión lineal de Y (n,3) sobre t (n,). Devuelve (slope(3,), intercept(3,))."""
    t = np.asarray(t, float)
    A = np.column_stack([t, np.ones_like(t)])
    sol, *_ = np.linalg.lstsq(A, Y, rcond=None)   # (2,3): fila0=slope, fila1=intercept
    return sol[0], sol[1]


def rotation_matrix_from_vectors(vec_from, vec_to):
    """Matriz de rotación que alinea vec_from con vec_to (ambos se normalizan)."""
    a = np.asarray(vec_from, float); a = a / np.linalg.norm(a)
    b = np.asarray(vec_to, float);   b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s < 1e-10:                       # ya alineados o exactamente opuestos
        if c > 0:
            return np.eye(3)
        ortho = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
        v = np.cross(a, ortho); v /= np.linalg.norm(v)
        return 2 * np.outer(v, v) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))


def rotate_z(points, theta, pivot):
    """Rota points alrededor del eje z que pasa por pivot. z se conserva."""
    c, s = np.cos(theta), np.sin(theta)
    Rz = np.array([[c, -s], [s, c]])
    out = np.asarray(points, float).copy()
    out[:, :2] = (out[:, :2] - pivot[:2]) @ Rz.T + pivot[:2]
    return out


def ajustar_esfera(P):
    """Ajuste lineal de esfera (Coope). Devuelve (centro, radio, rmse_relativo)."""
    P = np.asarray(P, float)
    A = np.hstack([2 * P, np.ones((len(P), 1))])
    b = (P ** 2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    centro = sol[:3]
    radio = np.sqrt(max(sol[3] + centro @ centro, 1e-12))
    d = np.linalg.norm(P - centro, axis=1)
    rmse = np.sqrt(np.mean((d - radio) ** 2))
    return centro, radio, rmse / radio


def cobertura_angular(capa_xy):
    """Fracción del círculo (0 a 1) cubierta por los puntos de la capa."""
    if len(capa_xy) < 3:
        return 0.0
    c = capa_xy.mean(axis=0)
    ang = np.sort(np.arctan2(capa_xy[:, 1] - c[1], capa_xy[:, 0] - c[0]))
    gaps = np.diff(ang)
    gap_cierre = (ang[0] + 2 * np.pi) - ang[-1]
    gap_max = max(gaps.max() if len(gaps) else 0.0, gap_cierre)
    return 1 - gap_max / (2 * np.pi)


def construir_caras_quad(n_filas, n_cols, cerrado_angular=True):
    """Índices de caras (quads) para malla estructurada (filas=Z, cols=ángulo)."""
    caras = []
    for i in range(n_filas - 1):
        for j in range(n_cols):
            if not cerrado_angular and j == n_cols - 1:
                continue
            j2 = (j + 1) % n_cols if cerrado_angular else j + 1
            caras.append([i * n_cols + j, i * n_cols + j2,
                          (i + 1) * n_cols + j2, (i + 1) * n_cols + j])
    return np.array(caras)


# ======================================================================
# Subcomponente: CREST (cresta / envolvente superior)
# ======================================================================
class Crest:
    """Extrae la cresta (envolvente de z máximo), la suaviza, la cose a la nube
    y localiza sus extremos. Opera sobre el estado del LinerGen padre."""

    def __init__(self, parent: "LinerGen"):
        self.p = parent
        self.covs = None
        self.corte_top = None
        self._sph = None
        self.env = None          # cresta cruda (por sector)
        self.env_suave = None    # cresta suavizada (spline periódico)
        self.min_3d = None
        self.max_3d = None
        self.env_2d = None       # perfil desenrollado (arco, z)

    def _want_show(self, ifshow):
        """Resuelve el flag: None -> usa Config.IFSHOW; True/False -> fuerza."""
        return self.p.cfg.IFSHOW if ifshow is None else ifshow

    # --- 1) cobertura angular por capas -> codo (dónde empieza la cresta) ---
    def coverage(self):
        cfg = self.p.cfg
        xy = self.p.xy_ordenado
        w = cfg.VENTANA_COBERTURA
        self.covs = np.array([cobertura_angular(xy[i:i + w])
                              for i in range(0, len(xy) - w)])
        x = np.arange(len(self.covs))
        covs = self.covs

        def objetivo(knots):
            try:
                spl = LSQUnivariateSpline(x, covs, np.sort(knots), k=1)
                return np.mean((spl(x) - covs) ** 2)
            except Exception:
                return 1e10

        result = differential_evolution(objetivo, [(x[1], x[-2])])
        self.corte_top = int(np.sort(result.x)[0])
        print("Codo (corte de la zona superior):", self.corte_top)
        return self.corte_top

    # --- 2) cresta cruda: z máximo por sector angular ---
    def raw(self):
        cfg = self.p.cfg
        P = self.p.puntos_rot_ordenado[:self.corte_top, :3]
        xp, yp, zp = P[:, 0], P[:, 1], P[:, 2]
        cx, cy = xp.mean(), yp.mean()
        theta = np.arctan2(yp - cy, xp - cx)

        bins = np.linspace(-np.pi, np.pi, cfg.N_BINS_ENV + 1)
        idx = np.clip(np.digitize(theta, bins) - 1, 0, cfg.N_BINS_ENV - 1)
        env_idx = []
        for b in range(cfg.N_BINS_ENV):
            m = np.where(idx == b)[0]
            if m.size:
                env_idx.append(m[np.argmax(zp[m])])
        env_idx = np.array(env_idx)
        env = P[env_idx]
        self.env = env[np.argsort(theta[env_idx])]
        print("Envolvente cruda:", self.env.shape)
        return self.env

    # --- 3) cresta suavizada: spline periódico ---
    def smooth(self):
        cfg = self.p.cfg
        xe, ye, ze = self.env[:, 0], self.env[:, 1], self.env[:, 2]
        tck, _ = splprep([xe, ye, ze], s=len(xe) * cfg.SUAVIZADO_ENV, per=1)
        u = np.linspace(0, 1, cfg.N_ENV_FINO)
        self.env_suave = np.column_stack(splev(u, tck))
        print("Cresta suavizada:", self.env_suave.shape)
        return self.env_suave

    # --- 4) coser la cresta a la nube (quita el borde ruidoso de arriba) ---
    def stitch(self, tol=0.0):
        pts = self.p.puntos_rot_ordenado
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        cx, cy = x.mean(), y.mean()

        th_pts = np.arctan2(y - cy, x - cx)
        th_env = np.arctan2(self.env_suave[:, 1] - cy, self.env_suave[:, 0] - cx)
        o = np.argsort(th_env)
        th_s, z_s = th_env[o], self.env_suave[o, 2]
        th_ext = np.concatenate([th_s - 2*np.pi, th_s, th_s + 2*np.pi])
        z_ext = np.concatenate([z_s, z_s, z_s])
        z_env_en_pts = np.interp(th_pts, th_ext, z_ext)

        mask_sup = z >= (z_env_en_pts - tol)
        sin_cresta = pts[~mask_sup]

        ncols = pts.shape[1]
        if self.env_suave.shape[1] >= ncols:
            cresta = self.env_suave[:, :ncols]
        else:
            cresta = np.hstack([self.env_suave,
                                np.zeros((len(self.env_suave), ncols - self.env_suave.shape[1]))])
        self.p.puntos_rot = np.vstack([sin_cresta, cresta])
        print("Nube tras cosido:", self.p.puntos_rot.shape)
        return self.p.puntos_rot

    # --- 5) extremos (máximos/mínimos) de la cresta ---
    def extrema(self):
        cfg = self.p.cfg
        d = np.diff(self.env_suave[:, :2], axis=0)
        s = np.concatenate([[0], np.cumsum(np.sqrt((d**2).sum(axis=1)))])
        self.env_2d = np.column_stack([s, self.env_suave[:, 2]])
        zp = self.env_2d[:, 1]
        Nz = len(zp)
        z3 = np.concatenate([zp, zp, zp])

        imax = argrelextrema(z3, np.greater, order=cfg.ORDER_EXTREMOS)[0]
        imin = argrelextrema(z3, np.less,    order=cfg.ORDER_EXTREMOS)[0]
        imax = np.unique(imax[(imax >= Nz) & (imax < 2*Nz)] - Nz)
        imin = np.unique(imin[(imin >= Nz) & (imin < 2*Nz)] - Nz)
        self.idx_max, self.idx_min = imax, imin
        self.max_3d = self.env_suave[imax]
        self.min_3d = self.env_suave[imin]
        print("Máximos:", self.max_3d.shape, "| Mínimos:", self.min_3d.shape)
        return self.min_3d, self.max_3d

    # --- pipeline completo de la cresta ---
    def extract_all(self, ifshow=None):
        self.coverage()
        self.raw()
        self.smooth()
        self.stitch()
        self.extrema()
        if self._want_show(ifshow):
            self.plot_envelope()
            self.plot_extrema()
        return self

    # --- visualización (matplotlib) ---
    def plot_envelope(self, ifshow=None):
        if not self._want_show(ifshow):
            return
        import matplotlib.pyplot as plt
        P = self.p.puntos_rot_ordenado[:self.corte_top]
        fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=P[:, 2], cmap="jet", s=1, alpha=0.2)
        ax.plot(self.env[:, 0], self.env[:, 1], self.env[:, 2], 'k.', ms=3, alpha=0.5, label="cruda")
        ax.plot(self.env_suave[:, 0], self.env_suave[:, 1], self.env_suave[:, 2],
                'r-', lw=2, label="suavizada")
        ax.legend(); plt.show()

    def plot_extrema(self, ifshow=None):
        if not self._want_show(ifshow):
            return
        import matplotlib.pyplot as plt
        zp = self.env_2d[:, 1]
        plt.figure(figsize=(9, 4))
        plt.plot(self.env_2d[:, 0], zp, 'k-', lw=1)
        plt.plot(self.env_2d[self.idx_max, 0], zp[self.idx_max], 'r^', label="máximos")
        plt.plot(self.env_2d[self.idx_min, 0], zp[self.idx_min], 'bv', label="mínimos")
        plt.xlabel("longitud de arco"); plt.ylabel("z"); plt.legend(); plt.show()


# ======================================================================
# Subcomponente: MESH (mallado en 2 partes + sellado de base)
# ======================================================================
class Mesh:
    """Genera la malla estructurada en 2 partes (tubo + transición a la cresta),
    construye las caras y sella la base. Opera sobre el estado del padre."""

    def __init__(self, parent: "LinerGen"):
        self.p = parent
        self.theta_grid = None
        self.CX = self.CY = None
        self.rows1 = None
        self.rows2 = None
        self.malla = None
        self.vertices = None
        self.caras = None
        self.mesh_vedo = None

    def _want_show(self, ifshow):
        """Resuelve el flag: None -> usa Config.IFSHOW; True/False -> fuerza."""
        return self.p.cfg.IFSHOW if ifshow is None else ifshow

    # --- rejilla angular + resampleo por splines ---
    def _setup(self):
        cfg = self.p.cfg
        pr = self.p.puntos_rot
        self.CX, self.CY = pr[:, 0].mean(), pr[:, 1].mean()
        self.theta_grid = np.linspace(-np.pi, np.pi, cfg.N_CIRC, endpoint=False)
        assert np.all(np.diff(self.theta_grid) > 0)

    def resample_ring_spline(self, p, n_nodos=None):
        """Resamplea un anillo (x,y,z) en theta_grid con regresión de splines
        periódicos por coordenada (nodos internos fijos)."""
        cfg = self.p.cfg
        n_nodos = cfg.N_NODOS_SECCION if n_nodos is None else n_nodos
        tg = self.theta_grid
        th = np.arctan2(p[:, 1] - self.CY, p[:, 0] - self.CX)
        o = np.argsort(th); th, p = th[o], p[o]
        th_e = np.concatenate([th - 2*np.pi, th, th + 2*np.pi])
        thu, iu = np.unique(th_e, return_index=True)
        lo, hi = tg.min() - np.pi/4, tg.max() + np.pi/4
        sel = (thu >= lo) & (thu <= hi)
        thu_s = thu[sel]
        n_nodos = min(n_nodos, max(1, len(thu_s) // 4))
        nodos = np.linspace(thu_s[0], thu_s[-1], n_nodos + 2)[1:-1]
        out = np.empty((len(tg), 3))
        for k in range(3):
            col = np.tile(p[:, k], 3)[iu][sel]
            spl = LSQUnivariateSpline(thu_s, col, nodos, k=min(cfg.ORDEN_K, 3))
            out[:, k] = spl(tg)
        return out

    # --- Parte 1: anillos cerrados de la base al inicio de la cresta ---
    def _part1(self):
        cfg = self.p.cfg
        pr = self.p.puntos_rot
        env_suave = self.p.crest.env_suave
        z_bottom = np.percentile(pr[:, 2], 1)
        z_crest_start = env_suave[:, 2].min()
        self.z_bottom, self.z_crest_start = z_bottom, z_crest_start
        self.z_crest_top = env_suave[:, 2].max()
        print(f"z_bottom={z_bottom:.2f}  z_crest_start={z_crest_start:.2f}  "
              f"z_crest_top={self.z_crest_top:.2f}")

        n_z1 = cfg.n_z1()
        z_levels = np.linspace(z_bottom, z_crest_start, n_z1)
        espesor = (z_crest_start - z_bottom) / n_z1 * 1.5
        rows = []
        for zc in z_levels:
            slab = pr[np.abs(pr[:, 2] - zc) < espesor]
            if len(slab) < cfg.N_NODOS_SECCION + 4:
                continue
            rows.append(self.resample_ring_spline(slab))
        self.rows1 = np.array(rows)
        if len(self.rows1) < 2:
            raise ValueError("Parte 1 sin suficientes anillos. Sube 'espesor' o baja N_Z1.")
        print("Parte 1:", self.rows1.shape)

    # --- Parte 2: transición alineada a la cresta ---
    def _part2(self):
        cfg = self.p.cfg
        env_suave = self.p.crest.env_suave
        crest_cols = self.resample_ring_spline(env_suave[:, :3])
        last_ring = self.rows1[-1]
        f_lin = np.linspace(0, 1, cfg.n_z2() + 1)[1:]
        fs = f_lin * f_lin * (3 - 2 * f_lin)          # smoothstep
        self.rows2 = np.array([(1 - f) * last_ring + f * crest_cols for f in fs])
        print("Parte 2:", self.rows2.shape,
              "| borde superior = cresta:", np.allclose(self.rows2[-1], crest_cols))

    # --- ensamblar malla + caras ---
    def _assemble(self):
        cfg = self.p.cfg
        self.malla = np.vstack([self.rows1, self.rows2])
        self.vertices = self.malla.reshape(-1, 3)
        self.caras = list(construir_caras_quad(len(self.malla), cfg.N_CIRC, cerrado_angular=True))
        if cfg.SELLAR_BASE:
            self.seal_base()
        print("malla:", self.malla.shape, "| vértices:", len(self.vertices),
              "| caras:", len(self.caras))

    # --- sellar la base en el punto mínimo real de la nube ---
    def seal_base(self):
        cfg = self.p.cfg
        pr = self.p.puntos_rot
        apice = pr[np.argmin(pr[:, 2])]
        idx_apice = len(self.vertices)
        self.vertices = np.vstack([self.vertices, apice])
        tapa = [[idx_apice, j, (j + 1) % cfg.N_CIRC] for j in range(cfg.N_CIRC)]
        self.caras = list(self.caras) + tapa
        print("Ápice base en z =", round(float(apice[2]), 3))

    # --- pipeline completo del mallado ---
    def build(self, ifshow=None):
        self._setup()
        self._part1()
        self._part2()
        self._assemble()
        if self._want_show(ifshow):
            self.plot_columns()
        return self

    # --- salida vedo / export ---
    def to_vedo(self, color="lightblue", alpha=0.85):
        from vedo import Mesh as VMesh
        self.mesh_vedo = (VMesh([self.vertices, self.caras])
                          .c(color).alpha(alpha).lw(1).lc("black"))
        return self.mesh_vedo

    def export(self, ruta="malla_liner.obj"):
        if self.mesh_vedo is None:
            self.to_vedo()
        self.mesh_vedo.write(ruta)
        print("Exportada a", ruta)
        return ruta

    def plot_columns(self, ifshow=None):
        if not self._want_show(ifshow):
            return
        import matplotlib.pyplot as plt
        cfg = self.p.cfg
        plt.figure(figsize=(5, 5))
        for j in [0, cfg.N_CIRC // 4, cfg.N_CIRC // 2, 3 * cfg.N_CIRC // 4]:
            plt.plot(self.malla[:, j, 0], self.malla[:, j, 1], '.-', label=f"col {j}")
        plt.axis("equal"); plt.legend()
        plt.title("columnas ordenadas (líneas limpias)"); plt.show()


# ======================================================================
# Clase principal
# ======================================================================
class LinerGen:
    """Orquesta el pipeline completo y expone los subcomponentes .crest y .mesh."""

    def __init__(self, cfg: Config | None = None, puntos: np.ndarray | None = None):
        self.cfg = cfg or Config()
        self.puntos = None if puntos is None else np.asarray(puntos, float)
        self.crest = Crest(self)
        self.mesh = Mesh(self)

    # --- 1) cargar la nube desde el OBJ ---
    def load(self, ruta=None):
        import vedo
        ruta = ruta or self.cfg.RUTA_OBJ
        m = vedo.load(ruta)
        self.puntos = np.asarray(m.points)
        self.centroide = self.puntos.mean(axis=0)
        print("Nº de puntos:", len(self.puntos))
        return self.puntos

    # --- 2) centerline por PCA + rebanado + suavizado ---
    def compute_centerline(self):
        cfg = self.cfg
        main_axis = _pca_main_axis(self.puntos)
        proj = (self.puntos - self.puntos.mean(axis=0)) @ main_axis
        bins = np.linspace(proj.min() + cfg.CL_OFFSET_MIN,
                           proj.max() - cfg.CL_OFFSET_MAX, cfg.N_SLICES)
        cl = []
        for i in range(len(bins) - 1):
            mask = (proj >= bins[i]) & (proj < bins[i + 1])
            if mask.sum() > 0:
                cl.append(self.puntos[mask].mean(axis=0))
        cl = np.array(cl)
        self.centerline_points = self._smooth_centerline(cl, cfg.SUAVIZADO_CL)
        print("Centerline:", self.centerline_points.shape)
        return self.centerline_points

    @staticmethod
    def _smooth_centerline(cl, s):
        t = np.linspace(0, 1, len(cl))
        out = np.zeros_like(cl)
        for i in range(3):
            out[:, i] = UnivariateSpline(t, cl[:, i], s=s)(t)
        return out

    # --- 3) eje recto por regresión (extrapolación) ---
    def compute_axis(self):
        cfg = self.cfg
        cl = self.centerline_points
        t = np.arange(0, -cfg.N_PTS_REGRESION, -1, dtype=float)
        slope, intercept = _fit_line(t, cl[-cfg.N_PTS_REGRESION:])
        self.direccion = slope / np.linalg.norm(slope)
        t_fut = np.arange(0, -cl.shape[0], -1, dtype=float)
        pp = t_fut[:, None] * slope[None, :] + intercept[None, :]
        pp -= pp[-1] - cl[-1]                     # anclar al último punto
        self.puntos_predichos = pp
        self.error = pp - cl
        print("Dirección:", self.direccion)
        return self.direccion

    # --- 4) alinear el eje con Z ---
    def align_to_z(self):
        self.R = rotation_matrix_from_vectors(self.direccion, np.array([0, 0, 1.0]))
        self.puntos_rot = self.puntos @ self.R.T
        self.centerline_rot = self.centerline_points @ self.R.T
        self.puntos_predichos_rot = self.puntos_predichos @ self.R.T
        if self.cfg.ORIENT_SPHERICAL_DOWN:
            self.orient_spherical_down()
        return self.puntos_rot

    # --- 5) ordenar por Z descendente ---
    def order_by_z(self):
        o = np.argsort(self.puntos_rot[:, 2])[::-1]
        self.puntos_rot_ordenado = self.puntos_rot[o]
        self.xy_ordenado = self.puntos_rot_ordenado[:, :2]
        self.z_val = self.puntos_rot_ordenado[:, 2]
        return self.puntos_rot_ordenado

    # ------------------------------------------------------------------
    # Orientación opcional
    # ------------------------------------------------------------------
    def end_sphericity(self):
        """Ajusta una esfera al casquete de cada extremo del centerline (en el
        marco rotado) y devuelve métricas. Menor error_rel = más esférico."""
        cl = self.centerline_rot
        pr = self.puntos_rot
        A, B = cl[0], cl[-1]
        eje = B - A; L = np.linalg.norm(eje); eje = eje / L
        proj = (pr - A) @ eje
        f = self.cfg.FRAC_CASQUETE
        capa_A = pr[proj < f * L]
        capa_B = pr[proj > (1 - f) * L]
        cenA, rA, eA = ajustar_esfera(capa_A)
        cenB, rB, eB = ajustar_esfera(capa_B)
        info = {"A": {"radio": rA, "error_rel": eA, "z": A[2], "n": len(capa_A)},
                "B": {"radio": rB, "error_rel": eB, "z": B[2], "n": len(capa_B)},
                "mas_esferico": "A" if eA < eB else "B"}
        print(f"Extremo A: r={rA:.2f} err={eA:.4f} (n={len(capa_A)})")
        print(f"Extremo B: r={rB:.2f} err={eB:.4f} (n={len(capa_B)})")
        print("Más esférico:", info["mas_esferico"])
        return info

    def orient_spherical_down(self):
        """Voltea 180° (si hace falta) para que el extremo más esférico quede
        en la parte baja (menor z). Rotación propia sobre el eje X."""
        info = self.end_sphericity()
        z_esf = info[info["mas_esferico"]]["z"]
        z_otro = info["B" if info["mas_esferico"] == "A" else "A"]["z"]
        if z_esf > z_otro:                        # el esférico está arriba -> voltear
            flip = np.diag([1.0, -1.0, -1.0])     # 180° sobre X (det +1)
            self.puntos_rot = self.puntos_rot @ flip.T
            self.centerline_rot = self.centerline_rot @ flip.T
            self.puntos_predichos_rot = self.puntos_predichos_rot @ flip.T
            print("Figura volteada: extremo más esférico abajo.")
        else:
            print("El extremo más esférico ya estaba abajo.")
        return self.puntos_rot

    def align_min_z(self):
        """Gira en Z para que la línea mínimo->centerline caiga sobre el eje X.
        Rota de forma consistente nube, centerline, cresta y extremos."""
        min_3d = self.crest.min_3d
        P0 = min_3d[0, :3]
        C = self.puntos_predichos_rot[
            np.argmin(np.linalg.norm(self.puntos_predichos_rot - P0, axis=1))]
        phi = np.arctan2((P0 - C)[1], (P0 - C)[0])
        theta = -phi
        self.puntos_rot = rotate_z(self.puntos_rot, theta, C)
        self.puntos_predichos_rot = rotate_z(self.puntos_predichos_rot, theta, C)
        self.centerline_rot = rotate_z(self.centerline_rot, theta, C)
        self.crest.env_suave = rotate_z(self.crest.env_suave, theta, C)
        self.crest.min_3d = rotate_z(self.crest.min_3d, theta, C)
        self.crest.max_3d = rotate_z(self.crest.max_3d, theta, C)
        print("Rotación en Z aplicada. theta =", round(float(theta), 4))
        return self.puntos_rot

    # ------------------------------------------------------------------
    # Pipeline completo
    # ------------------------------------------------------------------
    def run_all(self, export_path=None, ifshow=None):
        if self.puntos is None:
            self.load()
        self.compute_centerline()
        self.compute_axis()
        self.align_to_z()
        self.order_by_z()
        self.crest.extract_all(ifshow=ifshow)
        if self.cfg.APLICAR_ROTACION_Z:
            self.align_min_z()
        self.mesh.build(ifshow=ifshow)
        if export_path:
            self.mesh.export(export_path)
        return self

    # --- visualización final (vedo) ---
    def show(self):
        from vedo import Points, Line, show as vshow
        actores = [Points(self.puntos_rot, r=2, c="red", alpha=0.25)]
        if self.crest.env_suave is not None:
            actores.append(Line(self.crest.env_suave[:, :3], c="green", lw=2))
        if self.mesh.malla is not None:
            actores.append(self.mesh.to_vedo())
        vshow(actores, axes=1, bg="white")
