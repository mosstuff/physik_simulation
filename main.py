"""
Fractal Explorer
================
Interactive fractal renderer with adjustable parameters.

Requirements:
    pip install numpy pillow

Run:
    python fractal_explorer.py
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image, ImageTk
import threading


# ---------------------------------------------------------------------------
# Fractal math
# ---------------------------------------------------------------------------

COLORMAPS = {
    "Inferno":   [(0,0,0),(50,0,50),(150,0,100),(255,100,0),(255,220,50),(255,255,255)],
    "Ocean":     [(0,0,30),(0,30,100),(0,100,180),(0,200,220),(150,240,255),(255,255,255)],
    "Fire":      [(0,0,0),(80,0,0),(200,50,0),(255,150,0),(255,240,100),(255,255,255)],
    "Mint":      [(0,10,20),(0,40,60),(0,100,100),(0,200,150),(100,240,200),(255,255,255)],
    "Grayscale": [(0,0,0),(64,64,64),(128,128,128),(192,192,192),(230,230,230),(255,255,255)],
}


def colorize(iters, max_iter, palette_name):
    """Map iteration counts → RGB using smooth coloring."""
    stops = COLORMAPS[palette_name]
    n = len(stops) - 1

    # smooth normalized value in [0, 1]
    t = np.where(iters == max_iter, 1.0, iters / max_iter)
    t = np.clip(t ** 0.45, 0.0, 1.0)

    idx = np.clip((t * n).astype(int), 0, n - 1)
    frac = (t * n) - idx

    rgb = np.zeros((*iters.shape, 3), dtype=np.uint8)
    for c in range(3):
        lo = np.array([stops[i][c] for i in range(n + 1)])
        hi = np.array([stops[min(i + 1, n)][c] for i in range(n + 1)])
        v = lo[idx] + frac * (hi[idx] - lo[idx])
        rgb[..., c] = np.clip(v, 0, 255).astype(np.uint8)
    return rgb


def render_mandelbrot(width, height, cx, cy, zoom, max_iter, julia=False, jc=(-0.7, 0.27)):
    """Render Mandelbrot or Julia set using vectorized numpy."""
    x = np.linspace(cx - zoom, cx + zoom, width)
    y = np.linspace(cy - zoom * height / width, cy + zoom * height / width, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y

    if julia:
        Z = C.copy()
        Jc = complex(*jc)
    else:
        Z = np.zeros_like(C)
        Jc = C

    iters = np.zeros(C.shape, dtype=float)
    mask = np.ones(C.shape, dtype=bool)

    for i in range(max_iter):
        Z[mask] = Z[mask] ** 2 + (Jc[mask] if not julia else Jc)
        escaped = mask & (np.abs(Z) > 2)
        # smooth iteration count
        iters[escaped] = i + 1 - np.log2(np.log2(np.abs(Z[escaped])))
        mask[escaped] = False
        if not mask.any():
            break

    iters = np.clip(iters, 0, max_iter)
    return iters, max_iter


def render_burning_ship(width, height, cx, cy, zoom, max_iter):
    x = np.linspace(cx - zoom, cx + zoom, width)
    y = np.linspace(cy - zoom * height / width, cy + zoom * height / width, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    Z = np.zeros_like(C)
    iters = np.zeros(C.shape, dtype=float)
    mask = np.ones(C.shape, dtype=bool)

    for i in range(max_iter):
        Z[mask] = (np.abs(Z[mask].real) + 1j * np.abs(Z[mask].imag)) ** 2 + C[mask]
        escaped = mask & (np.abs(Z) > 2)
        iters[escaped] = i + 1.0
        mask[escaped] = False

    return iters, max_iter


def render_tricorn(width, height, cx, cy, zoom, max_iter):
    x = np.linspace(cx - zoom, cx + zoom, width)
    y = np.linspace(cy - zoom * height / width, cy + zoom * height / width, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    Z = np.zeros_like(C)
    iters = np.zeros(C.shape, dtype=float)
    mask = np.ones(C.shape, dtype=bool)

    for i in range(max_iter):
        Z[mask] = np.conj(Z[mask]) ** 2 + C[mask]
        escaped = mask & (np.abs(Z) > 2)
        iters[escaped] = i + 1.0
        mask[escaped] = False

    return iters, max_iter


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class FractalExplorer:
    W, H = 700, 520   # canvas size

    def __init__(self, root):
        self.root = root
        root.title("Fractal Explorer")
        root.resizable(False, False)
        root.configure(bg="#1a1a2e")

        self._render_thread = None
        self._render_id = 0  # incremented to cancel stale renders

        # ── state ──────────────────────────────────────────────────────────
        self.cx = tk.DoubleVar(value=-0.5)
        self.cy = tk.DoubleVar(value=0.0)
        self.zoom = tk.DoubleVar(value=1.5)
        self.max_iter = tk.IntVar(value=100)
        self.fractal_type = tk.StringVar(value="Mandelbrot")
        self.colormap = tk.StringVar(value="Inferno")
        self.julia_re = tk.DoubleVar(value=-0.7)
        self.julia_im = tk.DoubleVar(value=0.27)

        # drag state
        self._drag_start = None

        self._build_ui()
        self.schedule_render()

    # ── layout ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = self.root

        # left: canvas
        canvas_frame = tk.Frame(root, bg="#0f0f1a", bd=0)
        canvas_frame.pack(side=tk.LEFT, padx=0, pady=0)

        self.canvas = tk.Label(canvas_frame, bg="#0f0f1a", cursor="crosshair")
        self.canvas.pack()

        self.status = tk.Label(canvas_frame, text="Rendering…",
                               bg="#0f0f1a", fg="#888899",
                               font=("Courier", 10), anchor="w")
        self.status.pack(fill=tk.X, padx=8, pady=4)

        # mouse
        self.canvas.bind("<ButtonPress-1>",   self._on_press)
        self.canvas.bind("<B1-Motion>",       self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<MouseWheel>",      self._on_scroll)
        self.canvas.bind("<Button-4>",        self._on_scroll)
        self.canvas.bind("<Button-5>",        self._on_scroll)

        # right: controls
        ctrl = tk.Frame(root, bg="#1a1a2e", width=220)
        ctrl.pack(side=tk.RIGHT, fill=tk.Y, padx=12, pady=12)
        ctrl.pack_propagate(False)

        self._label(ctrl, "FRACTAL EXPLORER", size=13, bold=True, fg="#c8b8f8")
        self._sep(ctrl)

        self._label(ctrl, "Fractal type")
        for name in ("Mandelbrot", "Julia", "Burning Ship", "Tricorn"):
            tk.Radiobutton(ctrl, text=name, variable=self.fractal_type, value=name,
                           command=self._on_type_change,
                           bg="#1a1a2e", fg="#ddd", selectcolor="#2e2e50",
                           activebackground="#1a1a2e", activeforeground="#fff",
                           font=("Helvetica", 11)).pack(anchor="w", padx=8)

        self._sep(ctrl)
        self._label(ctrl, "Colormap")
        cmap_menu = ttk.Combobox(ctrl, textvariable=self.colormap,
                                 values=list(COLORMAPS.keys()),
                                 state="readonly", width=18)
        cmap_menu.pack(padx=8, pady=4, fill=tk.X)
        cmap_menu.bind("<<ComboboxSelected>>", lambda _: self.schedule_render())

        self._sep(ctrl)
        self._label(ctrl, "Parameters")

        self.slider_cx = self._slider(ctrl, "Center X", self.cx,   -2.5, 2.5)
        self.slider_cy = self._slider(ctrl, "Center Y", self.cy,   -2.0, 2.0)
        self.slider_z  = self._slider(ctrl, "Zoom",     self.zoom, 0.05, 3.0,  res=0.01)
        self.slider_it = self._slider(ctrl, "Max iter", self.max_iter, 20, 500, res=1, integer=True)

        self._sep(ctrl)
        self.julia_frame = tk.Frame(ctrl, bg="#1a1a2e")
        self.julia_frame.pack(fill=tk.X)
        self._label(self.julia_frame, "Julia constant")
        self.slider_jr = self._slider(self.julia_frame, "Re", self.julia_re, -2.0, 2.0)
        self.slider_ji = self._slider(self.julia_frame, "Im", self.julia_im, -2.0, 2.0)
        self.julia_frame.pack_forget()

        self._sep(ctrl)
        tk.Button(ctrl, text="⟳  Reset view", command=self._reset,
                  bg="#2e2e50", fg="#c8b8f8", relief=tk.FLAT,
                  font=("Helvetica", 11), padx=6, pady=4,
                  activebackground="#3e3e70", cursor="hand2").pack(fill=tk.X, padx=8)

    def _label(self, parent, text, size=11, bold=False, fg="#aaa8c8"):
        tk.Label(parent, text=text, bg="#1a1a2e", fg=fg,
                 font=("Helvetica", size, "bold" if bold else "normal"),
                 anchor="w").pack(fill=tk.X, padx=8, pady=(8, 2))

    def _sep(self, parent):
        tk.Frame(parent, bg="#2e2e50", height=1).pack(fill=tk.X, padx=8, pady=6)

    def _slider(self, parent, label, var, lo, hi, res=0.01, integer=False):
        row = tk.Frame(parent, bg="#1a1a2e")
        row.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(row, text=label, bg="#1a1a2e", fg="#aaa8c8",
                 font=("Helvetica", 10), width=7, anchor="w").pack(side=tk.LEFT)
        val_lbl = tk.Label(row, bg="#1a1a2e", fg="#e0d8ff",
                           font=("Courier", 10), width=7, anchor="e")
        val_lbl.pack(side=tk.RIGHT)

        def on_change(v):
            if integer:
                val_lbl.config(text=str(int(float(v))))
            else:
                val_lbl.config(text=f"{float(v):.3f}")
            self.schedule_render()

        s = tk.Scale(parent, variable=var, from_=lo, to=hi, resolution=res,
                     orient=tk.HORIZONTAL, showvalue=False, command=on_change,
                     bg="#1a1a2e", fg="#c8b8f8", troughcolor="#2e2e50",
                     activebackground="#6a5acd", highlightthickness=0, bd=0,
                     sliderlength=16, length=190)
        s.pack(padx=8, fill=tk.X)
        # init label
        if integer:
            val_lbl.config(text=str(int(var.get())))
        else:
            val_lbl.config(text=f"{var.get():.3f}")
        return s

    # ── type change ────────────────────────────────────────────────────────

    def _on_type_change(self):
        if self.fractal_type.get() == "Julia":
            self.julia_frame.pack(fill=tk.X)
        else:
            self.julia_frame.pack_forget()
        self._reset(rerender=False)
        self.schedule_render()

    # ── mouse interaction ──────────────────────────────────────────────────

    def _on_press(self, e):
        self._drag_start = (e.x, e.y, self.cx.get(), self.cy.get())

    def _on_drag(self, e):
        if not self._drag_start:
            return
        x0, y0, cx0, cy0 = self._drag_start
        scale = 2 * self.zoom.get() / self.W
        self.cx.set(cx0 - (e.x - x0) * scale)
        self.cy.set(cy0 - (e.y - y0) * scale * (self.H / self.W))
        self.schedule_render()

    def _on_release(self, _e):
        self._drag_start = None

    def _on_scroll(self, e):
        if e.num == 4 or e.delta > 0:
            self.zoom.set(max(0.05, self.zoom.get() * 0.85))
        else:
            self.zoom.set(min(3.0, self.zoom.get() * 1.15))
        self.schedule_render()

    # ── reset ──────────────────────────────────────────────────────────────

    def _reset(self, rerender=True):
        self.cx.set(-0.5)
        self.cy.set(0.0)
        self.zoom.set(1.5)
        if rerender:
            self.schedule_render()

    # ── rendering ──────────────────────────────────────────────────────────

    def schedule_render(self, *_):
        """Cancel any pending render and start a new one after a short delay."""
        self._render_id += 1
        rid = self._render_id
        self.root.after(80, lambda: self._start_render(rid))

    def _start_render(self, rid):
        if rid != self._render_id:
            return   # superseded
        params = dict(
            width=self.W, height=self.H,
            cx=self.cx.get(), cy=self.cy.get(),
            zoom=self.zoom.get(), max_iter=self.max_iter.get(),
            fractal=self.fractal_type.get(),
            cmap=self.colormap.get(),
            julia_re=self.julia_re.get(),
            julia_im=self.julia_im.get(),
        )
        self.status.config(text="Rendering…")
        t = threading.Thread(target=self._render_worker, args=(params, rid), daemon=True)
        t.start()

    def _render_worker(self, p, rid):
        try:
            ftype = p["fractal"]
            if ftype == "Mandelbrot":
                iters, mi = render_mandelbrot(p["width"], p["height"],
                                              p["cx"], p["cy"], p["zoom"], p["max_iter"])
            elif ftype == "Julia":
                iters, mi = render_mandelbrot(p["width"], p["height"],
                                              p["cx"], p["cy"], p["zoom"], p["max_iter"],
                                              julia=True, jc=(p["julia_re"], p["julia_im"]))
            elif ftype == "Burning Ship":
                iters, mi = render_burning_ship(p["width"], p["height"],
                                                p["cx"], p["cy"], p["zoom"], p["max_iter"])
            else:
                iters, mi = render_tricorn(p["width"], p["height"],
                                           p["cx"], p["cy"], p["zoom"], p["max_iter"])

            rgb = colorize(iters, mi, p["cmap"])
            img = Image.fromarray(rgb, "RGB")
        except Exception as ex:
            self.root.after(0, lambda: self.status.config(text=f"Error: {ex}"))
            return

        self.root.after(0, lambda: self._display(img, p, rid))

    def _display(self, img, p, rid):
        if rid != self._render_id:
            return
        photo = ImageTk.PhotoImage(img)
        self.canvas.config(image=photo)
        self.canvas._photo = photo   # keep reference
        self.status.config(
            text=f"{p['fractal']}  |  center ({p['cx']:.4f}, {p['cy']:.4f})"
                 f"  |  zoom {p['zoom']:.3f}  |  iter {p['max_iter']}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    root = tk.Tk()

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TCombobox",
                    fieldbackground="#2e2e50", background="#2e2e50",
                    foreground="#e0d8ff", selectbackground="#3e3e70",
                    bordercolor="#2e2e50", arrowcolor="#c8b8f8")

    app = FractalExplorer(root)
    root.mainloop()


if __name__ == "__main__":
    main()