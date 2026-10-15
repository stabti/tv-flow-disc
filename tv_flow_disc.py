"""Total variation (TV) flow on two discs (R = 30 and R = 15): the discs do not shrink, they lose contrast
linearly and vanish in finite time t* = h R / 2 (the small one first).

TV flow = implicit Euler scheme -> one ROF problem per time step, solved in its dual form.
Usage: python tv_flow_disc.py [--lang en|fr] [--out tv_disc.mp4] [--clean] [--scheme crofton|iso]
--clean  : image only (no title, plot or annotation)
--scheme : discretisation of TV.
           crofton = 16 neighbours with Cauchy-Crofton weights, ~isotropic perimeter, round discs (default)
           iso     = forward differences (faster, but edges drift towards axis-aligned, discs look polygonal)
"""
import argparse
from math import atan2, hypot, pi
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

SHAPE, H = (128, 160), 1.0
DISCS = [(64, 40, 30), (64, 125, 15)]  # (row, column, radius)
DT, N_STEPS, INNER = 0.25, 80, 1500


# --- scheme "iso": forward differences, isotropic norm of the gradient (Chambolle 2004) ---
def grad(u):  # forward differences, u = 0 outside the frame on all 4 sides (Dirichlet), as on R^2
    h, w = u.shape
    P = np.pad(u, 1)
    return P[:h + 1, 1:] - P[:h + 1, :w + 1], P[1:, :w + 1] - P[:h + 1, :w + 1]


def div(px, py):  # minus the adjoint of grad
    h, w = px.shape[0] - 1, px.shape[1] - 1
    Q = np.zeros((h + 2, w + 2))
    Q[:h + 1, 1:] += px; Q[:h + 1, :w + 1] -= px
    Q[1:, :w + 1] += py; Q[:h + 1, :w + 1] -= py
    return -Q[1:h + 1, 1:w + 1]


def rof(f, lam_inv, n_iter=INNER, tau=0.125):
    """argmin_u TV(u) + ||u - f||^2 / (2 lam_inv), Chambolle 2004."""
    px = np.zeros((f.shape[0] + 1, f.shape[1] + 1)); py = np.zeros_like(px)
    for _ in range(n_iter):
        gx, gy = grad(div(px, py) - f / lam_inv)
        nrm = 1 + tau * np.hypot(gx, gy)
        px = (px + tau * gx) / nrm
        py = (py + tau * gy) / nrm
    return f - lam_inv * div(px, py)


# --- scheme "crofton": 16 neighbours, TV(u) = sum_k w_k |u(x+e_k) - u(x)|, w_k = angular gap / (2 |e_k|) ---
OFFS = [(0, 1), (1, 0), (1, 1), (1, -1), (1, 2), (2, 1), (1, -2), (2, -1)]  # (dy, dx)
_ang = sorted((atan2(dy, dx) % pi, (dy, dx)) for dy, dx in OFFS)
WEIGHTS = {o: ((_ang[(i + 1) % 8][0] + (pi if i == 7 else 0)) - (_ang[i - 1][0] - (pi if i == 0 else 0))) / 2
           / (2 * hypot(*o)) for i, (_, o) in enumerate(_ang)}
PAD = 3


def gradw(u):  # list of w_k * (u(x+e_k) - u(x)), u = 0 outside the frame
    C = np.pad(u, PAD)
    return [w * (np.roll(C, (-dy, -dx), (0, 1)) - C) for (dy, dx), w in WEIGHTS.items()]


def gradw_T(rs):  # adjoint of gradw
    out = 0
    for r, ((dy, dx), w) in zip(rs, WEIGHTS.items()):
        out = out + w * (np.roll(r, (dy, dx), (0, 1)) - r)
    return out[PAD:-PAD, PAD:-PAD]


def rof_crofton(f, dt, rs=None, n_iter=400):
    """argmin_u TV16(u) + ||u - f||^2 / (2 dt), dual solved by FISTA (r_k in [-dt, dt]).
    Returns (u, rs); pass rs back in to warm start the next time step."""
    step = 1 / (2 * sum(WEIGHTS.values())) ** 2
    shp = (f.shape[0] + 2 * PAD, f.shape[1] + 2 * PAD)
    r = [np.zeros(shp) for _ in OFFS] if rs is None else rs
    y, tk = [x.copy() for x in r], 1.0
    for _ in range(n_iter):
        g = gradw(f - gradw_T(y))
        r_new = [np.clip(yk + step * gk, -dt, dt) for yk, gk in zip(y, g)]
        t_new = (1 + np.sqrt(1 + 4 * tk * tk)) / 2
        y = [rn + (tk - 1) / t_new * (rn - ro) for rn, ro in zip(r_new, r)]
        r, tk = r_new, t_new
    return f - gradw_T(r), r


def run(scheme="crofton"):
    yy, xx = np.mgrid[:SHAPE[0], :SHAPE[1]]
    u = np.zeros(SHAPE)
    for cy, cx, r in DISCS:
        u[np.hypot(xx - cx, yy - cy) <= r] = H
    frames, t = [u.copy()], [0.0]
    rs = None
    for k in range(N_STEPS):
        if scheme == "crofton":
            u, rs = rof_crofton(u, DT, rs)
        else:
            u = rof(u, DT)
        frames.append(u.copy()); t.append((k + 1) * DT)
    return np.array(frames), np.array(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="en", choices=["en", "fr"]); ap.add_argument("--out", default="tv_disc.mp4")
    ap.add_argument("--clean", action="store_true"); ap.add_argument("--scheme", default="crofton", choices=["crofton", "iso"])
    a = ap.parse_args()
    L = dict(fr=dict(img="Image u(t)", c="Contraste au centre du disque", th="théorie : h - 2t/R", sim="simulation",
                     tit="Flot de variation totale : les disques ne rétrécissent pas, ils s'effacent (le petit d'abord)"),
             en=dict(img="Image u(t)", c="Contrast at disc centre", th="theory: h - 2t/R", sim="simulation",
                     tit="Total variation flow: the discs do not shrink, they fade out (small one first)"))[a.lang]
    F, t = run(a.scheme)
    centers = [F[:, cy, cx] for cy, cx, _ in DISCS]
    for (cy, cx, r), c in zip(DISCS, centers):
        print(f"R={r}: theoretical t* = {H * r / 2:.1f}, final contrast = {c[-1]:.3f}")

    hold = 20  # still frames at the start and at the end
    idx = [0] * hold + list(range(len(t))) + [len(t) - 1] * hold
    if a.clean:
        fig = plt.figure(figsize=(10, 8), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        im = ax.imshow(F[0], cmap="gray", vmin=0, vmax=H, interpolation="bilinear")
        FuncAnimation(fig, lambda i: [im.set_data(F[idx[i]]), im][1:], frames=len(idx)).save(
            a.out, writer=FFMpegWriter(fps=12, bitrate=4000))
        print("wrote", a.out)
        return

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(12.8, 7.2), dpi=100, gridspec_kw=dict(width_ratios=[1.2, 1]))
    fig.suptitle(L["tit"], fontsize=17)
    im = a0.imshow(F[0], cmap="gray", vmin=0, vmax=H); a0.axis("off"); a0.set_title(L["img"])
    tt = np.linspace(0, t[-1], 200)
    lines = []
    for i, ((cy, cx, r), ci) in enumerate(zip(DISCS, ["C0", "C3"])):
        a1.plot(tt, np.clip(H - 2 * tt / r, 0, None), "--", c="gray", lw=1.5, label=L["th"] if i == 0 else None)
        (ln,) = a1.plot([], [], c=ci, lw=3, label=f"{L['sim']} R={r}")
        lines.append(ln)
        a1.axvline(H * r / 2, c=ci, ls=":"); a1.text(H * r / 2 + .3, .9 - .08 * i, f"t* = {H * r / 2:g}", color=ci)
        a0.text(cx, cy - r - 4, f"R={r}", color=ci, ha="center", fontsize=13)
    a1.set(xlim=(0, t[-1]), ylim=(0, 1.05), xlabel="t", ylabel=L["c"]); a1.legend(loc="lower left")

    def upd(i):
        k = idx[i]; im.set_data(F[k])
        for ln, c in zip(lines, centers):
            ln.set_data(t[:k + 1], c[:k + 1])
        return [im, *lines]

    FuncAnimation(fig, upd, frames=len(idx)).save(a.out, writer=FFMpegWriter(fps=12, bitrate=4000))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
