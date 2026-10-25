"""Relief de la Tarentaise en fonction analytique z(x, y), Lambert-93.

    t = Terrain.load("tarentaise.npz")
    t(VILLES["x"], VILLES["y"]), t.grad(x, y)
"""

import os
import time
import urllib.parse
import urllib.request

import numpy as np
from scipy import fft as sfft
from scipy import ndimage


# ===========================================================================
#   LES 5 PARAMÈTRES À RÉGLER. Tout le reste du fichier peut rester tel quel.
# ===========================================================================

# Emprise en Lambert-93 (X_min, Y_min, X_max, Y_max), en mètres.
# Actuellement Albertville -> Val d'Isère. Descends Y_min à 6_470_000 pour
# inclure la vallée des Belleville (Les Menuires, Val Thorens).
BBOX = (955_000.0, 6_480_000.0, 1_013_000.0, 6_522_000.0)

# RES = pas d'échantillonnage du MNT en mètres. C'EST LE PLAFOND DE DÉTAIL :
# la fonction ne pourra JAMAIS représenter quoi que ce soit de plus fin que
# 2 x RES de longueur d'onde. Pour plus de détail, c'est ici qu'on baisse.
# Le coût grimpe en 1/RES² :
#     RES = 25  ->   2320 x 1680 =  3.9 M px,   2 tuiles,  cache  16 Mo
#     RES = 10  ->   5800 x 4200 = 24.4 M px,   9 tuiles,  cache  98 Mo   <-- ici
#     RES =  5  ->  11600 x 8400 = 97.4 M px,  30 tuiles,  cache 390 Mo
#     RES =  2  ->  déraisonnable (609 M px, ~2.4 Go)
RES = 5

# LAYER DOIT SUIVRE RES : la BD ALTI est native à 25 m, le RGE ALTI à 1 m.
# Demander du 10 m à la BD ALTI ne ferait qu'interpoler du 25 m.
#     RES >= 25  ->  "ELEVATION.ELEVATIONGRIDCOVERAGE"           (BD ALTI)
#     RES <  25  ->  "ELEVATION.ELEVATIONGRIDCOVERAGE.HIGHRES"   (RGE ALTI)
# Si le script s'arrête sur "tuile vide", c'est que la couche n'est pas servie
# à ce pas-là : essaie l'autre valeur ci-dessus.
LAYER = "ELEVATION.ELEVATIONGRIDCOVERAGE.HIGHRES"

# RMSE_CIBLE = erreur quadratique visée, en mètres. C'est ELLE qui fixe le
# nombre de coefficients M de la fonction (calculé exactement par Parseval,
# pas par essai-erreur). En gros M ∝ RMSE⁻² :
#     RMSE 30 m ->    ~400 termes        RMSE  8 m ->  ~6 000 termes
#     RMSE 15 m ->  ~1 600 termes        RMSE  4 m -> ~27 000 termes
# Plancher utile ~5 m : en montagne le RGE ALTI vient de corrélation radar,
# précision verticale ~7 m sur les fortes pentes. En dessous on ajuste du bruit.
RMSE_CIBLE = 5.0

# PIX = côté en pixels de l'image du plot. Comme la fonction est analytique,
# elle est RECALCULÉE à cette résolution sur la fenêtre visible à chaque zoom :
# on ne voit donc jamais de pixels, quel que soit le niveau d'agrandissement.
# Monte à 1400 si l'écran est grand, baisse à 600 si le zoom est lent.
PIX = 900

# ===========================================================================


WMS = "https://data.geopf.fr/wms-r/wms"
TILE = 2000
NPZ = "tarentaise.npz"
CACHE = f"mnt_tarentaise_{RES:g}m.npy"

_A = 6_378_137.0
_E = np.sqrt(2 / 298.257222101 - (1 / 298.257222101) ** 2)
_P1, _P2, _P0, _L0 = np.radians([44.0, 49.0, 46.5, 3.0])


def _mm(p):
    return np.cos(p) / np.sqrt(1 - _E ** 2 * np.sin(p) ** 2)


def _tt(p):
    return np.tan(np.pi / 4 - p / 2) / ((1 - _E * np.sin(p)) / (1 + _E * np.sin(p))) ** (_E / 2)


_N = np.log(_mm(_P1) / _mm(_P2)) / np.log(_tt(_P1) / _tt(_P2))
_F = _mm(_P1) / (_N * _tt(_P1) ** _N)
_R0 = _A * _F * _tt(_P0) ** _N


def wgs84_to_l93(lon, lat):
    r = _A * _F * _tt(np.radians(lat)) ** _N
    th = _N * (np.radians(lon) - _L0)
    return 700_000.0 + r * np.sin(th), 6_600_000.0 + _R0 - r * np.cos(th)


_VILLES_WGS84 = [
    ("Albertville", 45.6757, 6.3925), ("Gilly-sur-Isère", 45.6614, 6.3494),
    ("Tours-en-Savoie", 45.6664, 6.4614), ("La Bâthie", 45.6338, 6.4514),
    ("Esserts-Blay", 45.6350, 6.4382), ("Cevins", 45.5967, 6.4571),
    ("Rognaix", 45.5913, 6.4496), ("Saint-Paul-sur-Isère", 45.5921, 6.4489),
    ("Feissons-sur-Isère", 45.5511, 6.4740), ("La Léchère", 45.5233, 6.4767),
    ("Aigueblanche", 45.5176, 6.4848), ("Moûtiers", 45.4847, 6.5311),
    ("Salins-les-Thermes", 45.4703, 6.5308), ("Brides-les-Bains", 45.4539, 6.5692),
    ("Méribel", 45.3969, 6.5656), ("Saint-Martin-de-Belleville", 45.3722, 6.5058),
    ("Courchevel", 45.4147, 6.6347), ("Bozel", 45.4436, 6.6489),
    ("Pralognan-la-Vanoise", 45.3822, 6.7239), ("Champagny-en-Vanoise", 45.4631, 6.7061),
    ("Aime", 45.5544, 6.6489), ("Macôt", 45.5406, 6.6689),
    ("La Plagne", 45.5075, 6.6764), ("Bellentre", 45.5556, 6.7011),
    ("Landry", 45.5722, 6.7331), ("Peisey-Nancroix", 45.5497, 6.7639),
    ("Bourg-Saint-Maurice", 45.6183, 6.7694), ("Les Arcs 1800", 45.5811, 6.8000),
    ("Séez", 45.6244, 6.7986), ("La Rosière", 45.6294, 6.8511),
    ("Villaroger", 45.5746, 6.8819), ("Sainte-Foy-Tarentaise", 45.5928, 6.8917),
    ("Tignes", 45.4692, 6.9058), ("Val d'Isère", 45.4489, 6.9781),
]

VILLES = np.array(
    [(n,) + tuple(float(c) for c in wgs84_to_l93(lo, la)) for n, la, lo in _VILLES_WGS84],
    dtype=[("nom", "U26"), ("x", "f8"), ("y", "f8")],
)


def _decode(raw, w, h):
    if len(raw) < 4 * w * h:
        raise RuntimeError(f"réponse tronquée ({len(raw)} o) : {raw[:200]!r}")
    best, score = None, -1.0
    for dt in ("<f4", ">f4"):
        a = np.frombuffer(raw[:4 * w * h], dtype=dt).reshape(h, w)
        s = (np.isfinite(a) & (a > -500) & (a < 5000)).mean()
        if s > score:
            best, score = a, s
    if score < 0.5:
        raise RuntimeError("altitudes invraisemblables (erreur XML du service ?)")
    return best


def _tile(x0, y0, x1, y1, w, h):
    q = {"SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetMap", "LAYERS": LAYER,
         "STYLES": "", "FORMAT": "image/x-bil;bits=32", "CRS": "EPSG:2154",
         "BBOX": f"{x0},{y0},{x1},{y1}", "WIDTH": str(w), "HEIGHT": str(h),
         "EXCEPTIONS": "text/xml"}
    url = WMS + "?" + urllib.parse.urlencode(q, safe=";=")
    for k in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "tarentaise/1.0"})
            with urllib.request.urlopen(req, timeout=300) as r:
                return _decode(r.read(), w, h)
        except Exception as e:
            if k == 2:
                raise RuntimeError(f"échec WMS : {e}\n{url}") from e
            time.sleep(3 * (k + 1))


def download_dem(bbox=BBOX, res=RES, cache=CACHE):
    x0, y0, x1, y1 = bbox
    nx, ny = int(round((x1 - x0) / res)), int(round((y1 - y0) / res))
    if os.path.exists(cache):
        Z = np.load(cache)
    else:
        Z = np.empty((ny, nx), np.float32)
        n = 0
        for i0 in range(0, ny, TILE):
            for j0 in range(0, nx, TILE):
                i1, j1 = min(i0 + TILE, ny), min(j0 + TILE, nx)
                t0 = time.time()
                t = _tile(x0 + j0 * res, y1 - i1 * res, x0 + j1 * res, y1 - i0 * res,
                          j1 - j0, i1 - i0)
                ok = np.isfinite(t) & (t > -400) & (t < 5000)
                if ok.mean() < 0.5 or t[ok].std() < 1e-6:
                    raise RuntimeError(
                        f"tuile vide en ({i0},{j0}) : la couche {LAYER} n'est sans "
                        f"doute pas servie au pas de {res:g} m, change LAYER")
                Z[i0:i1, j0:j1] = t
                n += 1
                print(f"  tuile {n} : {j1-j0}x{i1-i0} px, "
                      f"{t[ok].min():.0f}-{t[ok].max():.0f} m, {time.time()-t0:.1f} s")
        Z = Z[::-1]
        np.save(cache, Z)
    Z = np.where(np.isfinite(Z) & (Z > -400) & (Z < 5000), Z, np.nan)
    return x0 + res * (np.arange(nx) + 0.5), y0 + res * (np.arange(ny) + 0.5), Z


def fill_nodata(Z):
    valid = np.isfinite(Z)
    if valid.all():
        return Z
    _, idx = ndimage.distance_transform_edt(~valid, return_indices=True)
    Zf = Z[tuple(idx)]
    return np.where(valid, Zf, ndimage.gaussian_filter(Zf, 2.0))


class Terrain:
    def __init__(self, kx, ky, coef, Xa, Ya, Lx, Ly, shape):
        self.kx, self.ky, self.coef = kx, ky, coef
        self.Xa, self.Ya, self.Lx, self.Ly = Xa, Ya, Lx, Ly
        self.shape = tuple(shape)

    def _uv(self, x, y):
        return ((np.ravel(np.asarray(x, float)) - self.Xa) / self.Lx,
                (np.ravel(np.asarray(y, float)) - self.Ya) / self.Ly)

    def __call__(self, x, y, chunk=400):
        u, v = self._uv(x, y)
        out = np.zeros(u.size)
        for s in range(0, self.coef.size, chunk):
            b = slice(s, s + chunk)
            out += (np.cos(np.pi * np.outer(u, self.kx[b])) *
                    np.cos(np.pi * np.outer(v, self.ky[b])) * self.coef[b]).sum(1)
        return out.reshape(np.shape(x))

    def grad(self, x, y, chunk=400):
        u, v = self._uv(x, y)
        gx, gy = np.zeros(u.size), np.zeros(u.size)
        for s in range(0, self.coef.size, chunk):
            b = slice(s, s + chunk)
            au, av = np.pi * np.outer(u, self.kx[b]), np.pi * np.outer(v, self.ky[b])
            gx -= (self.coef[b] * np.pi * self.kx[b] / self.Lx * np.sin(au) * np.cos(av)).sum(1)
            gy -= (self.coef[b] * np.pi * self.ky[b] / self.Ly * np.cos(au) * np.sin(av)).sum(1)
        return gx.reshape(np.shape(x)), gy.reshape(np.shape(x))

    def rendre(self, x0, x1, y0, y1, n=PIX):
        u = (np.linspace(x0, x1, n) - self.Xa) / self.Lx
        v = (np.linspace(y0, y1, n) - self.Ya) / self.Ly
        K, L = int(self.kx.max()) + 1, int(self.ky.max()) + 1
        C = np.zeros((L, K))
        C[self.ky, self.kx] = self.coef
        return (np.cos(np.pi * np.outer(v, np.arange(L))) @ C
                @ np.cos(np.pi * np.outer(u, np.arange(K))).T)

    def grid(self):
        ny, nx = self.shape
        ay = np.full(ny, np.sqrt(2 / ny)); ay[0] = np.sqrt(1 / ny)
        ax = np.full(nx, np.sqrt(2 / nx)); ax[0] = np.sqrt(1 / nx)
        C = np.zeros(self.shape)
        C[self.ky, self.kx] = self.coef / (ay[self.ky] * ax[self.kx])
        return sfft.idctn(C, type=2, norm="ortho")

    def save(self, path=NPZ):
        np.savez_compressed(path, kx=self.kx, ky=self.ky, coef=self.coef, Xa=self.Xa,
                            Ya=self.Ya, Lx=self.Lx, Ly=self.Ly, shape=np.array(self.shape))
        return path

    @staticmethod
    def load(path=NPZ):
        d = np.load(path)
        return Terrain(d["kx"], d["ky"], d["coef"], float(d["Xa"]), float(d["Ya"]),
                       float(d["Lx"]), float(d["Ly"]), tuple(d["shape"]))


def fit(x, y, Z, rmse_cible=RMSE_CIBLE):
    ny, nx = Z.shape
    dx, dy = x[1] - x[0], y[1] - y[0]
    C = sfft.dctn(np.asarray(Z, np.float64), type=2, norm="ortho")
    cum = np.cumsum(np.sort(C.ravel() ** 2))
    M = max(C.size - int(np.searchsorted(cum, rmse_cible ** 2 * C.size)), 1)
    idx = np.argpartition(np.abs(C).ravel(), C.size - M)[C.size - M:]
    ky, kx = np.unravel_index(idx, C.shape)
    ay = np.full(ny, np.sqrt(2 / ny)); ay[0] = np.sqrt(1 / ny)
    ax = np.full(nx, np.sqrt(2 / nx)); ax[0] = np.sqrt(1 / nx)
    return Terrain(kx.astype(np.int32), ky.astype(np.int32), C[ky, kx] * ay[ky] * ax[kx],
                   x[0] - dx / 2, y[0] - dy / 2, nx * dx, ny * dy, (ny, nx))


def _image(t, x0, x1, y0, y1, n, vmin, vmax, cmap):
    Z = t.rendre(x0, x1, y0, y1, n)
    gy, gx = np.gradient(Z, (y1 - y0) / n, (x1 - x0) / n)
    slope, aspect = np.pi / 2 - np.arctan(np.hypot(gx, gy)), np.arctan2(-gx, gy)
    hs = (np.sin(np.radians(45.0)) * np.sin(slope) +
          np.cos(np.radians(45.0)) * np.cos(slope) * np.cos(np.radians(135.0) - aspect))
    rgb = cmap(np.clip((Z - vmin) / (vmax - vmin), 0, 1))[..., :3]
    return np.clip(rgb * (0.35 + 0.65 * hs[..., None]), 0, 1)


def plot(t):
    import matplotlib.pyplot as plt
    x0, y0 = t.Xa, t.Ya
    x1, y1 = t.Xa + t.Lx, t.Ya + t.Ly
    cmap = plt.get_cmap("terrain")
    Zg = t.grid()
    vmin, vmax = float(Zg.min()), float(Zg.max())
    del Zg

    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(_image(t, x0, x1, y0, y1, PIX, vmin, vmax, cmap),
                   origin="lower", extent=[x0, x1, y0, y1], interpolation="bilinear")
    ax.plot(VILLES["x"], VILLES["y"], "o", ms=4, color="crimson")
    for v in VILLES:
        ax.annotate(v["nom"], (v["x"], v["y"]), fontsize=7,
                    xytext=(5, 3), textcoords="offset points")
    fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax)),
                 ax=ax, label="altitude (m)")
    ax.set_xlabel("X Lambert-93 (m)")
    ax.set_ylabel("Y Lambert-93 (m)")
    ax.set_aspect("equal")
    ax.set_autoscale_on(False)

    busy = [False]

    def refresh(_ax=None):
        if busy[0]:
            return
        busy[0] = True
        xa, xb = ax.get_xlim()
        ya, yb = ax.get_ylim()
        im.set_data(_image(t, xa, xb, ya, yb, PIX, vmin, vmax, cmap))
        im.set_extent([xa, xb, ya, yb])
        busy[0] = False

    ax.callbacks.connect("xlim_changed", refresh)
    ax.callbacks.connect("ylim_changed", refresh)
    fig.tight_layout()
    plt.show()
    return refresh


if __name__ == "__main__":
    x, y, Z = download_dem()
    print(f"MNT {Z.shape[1]} x {Z.shape[0]} = {Z.size/1e6:.1f} M points @ {RES:g} m")
    Zf = fill_nodata(Z)
    del Z
    t = fit(x, y, Zf)
    e = t.grid() - Zf
    print(f"{t.coef.size} termes, RMSE {np.sqrt((e ** 2).mean()):.2f} m, "
          f"max {np.abs(e).max():.1f} m, lambda_min "
          f"{2*t.Lx/max(int(t.kx.max()), 1):.0f} m -> {t.save()}")
    del e, Zf
    for v, z in zip(VILLES, t(VILLES["x"], VILLES["y"])):
        print(f"  {v['nom']:26s} {z:7.1f} m")
    plot(t)