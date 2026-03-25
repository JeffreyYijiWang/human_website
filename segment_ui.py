import ctypes, os

# Tell Windows to use the NVIDIA GPU for this process
try:
    ctypes.windll.ntdll.NtSetInformationProcess(
        ctypes.windll.kernel32.GetCurrentProcess(),
        0x27, ctypes.byref(ctypes.c_ulong(1)), ctypes.sizeof(ctypes.c_ulong)
    )
except Exception:
    pass

# Also set NVIDIA env vars as a fallback
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
# segment_editor_full.py
# Outputs:
#   threshold_images/
#     saved_mask/
#     clipped_image/

from pathlib import Path
import cv2
import numpy as np


# ---------------------- color + mask helpers ----------------------

def to_hsv(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

def hue_dist(h, h0):
    d = np.abs(h.astype(np.int16) - int(h0))
    return np.minimum(d, 180 - d).astype(np.uint8)

def mask_for_target(hsv, target_hsv, tol_h, tol_s, tol_v):
    h, s, v = cv2.split(hsv)
    th, ts, tv = target_hsv
    dh = hue_dist(h, th)
    m = (dh <= tol_h) & (np.abs(s.astype(np.int16) - int(ts)) <= tol_s) & (np.abs(v.astype(np.int16) - int(tv)) <= tol_v)
    return (m.astype(np.uint8) * 255)

def postprocess_mask(mask, open_k=0, close_k=0):
    m = mask
    if open_k > 0:
        if open_k % 2 == 0: open_k += 1
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    if close_k > 0:
        if close_k % 2 == 0: close_k += 1
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return m

def overlay_mask(img_bgr, mask):
    overlay = img_bgr.copy()
    overlay[mask > 0] = (0, 255, 0)
    return cv2.addWeighted(img_bgr, 0.75, overlay, 0.25, 0)

def cutout_bgr(img_bgr, mask, bg=(0, 0, 0)):
    out = np.empty_like(img_bgr)
    out[:] = bg
    out[mask > 0] = img_bgr[mask > 0]
    return out


# ---------------------- UI state ----------------------

class State:
    def __init__(self):
        self.targets = []            # list[(H,S,V)]
        self.swatch_enabled = []     # list[bool]
        self.active_idx = -1
        self.last_pick = None        # (x,y)

        self.paint_mode = None       # None | "erase" | "add"
        self.painting = False
        self.brush_r = 18

        self.mask = None
        self.hsv = None
        self.img = None

        self.swatch_ui = []          # hitboxes
        self.paths = []
        self.idx = 0

state = State()


# ---------------------- recompute mask (AUTO) ----------------------

def recompute_mask():
    if state.img is None:
        return
    H, W = state.img.shape[:2]
    if not state.targets:
        state.mask = np.zeros((H, W), np.uint8)
        return

    tol_h = cv2.getTrackbarPos("tol_h", "controls")
    tol_s = cv2.getTrackbarPos("tol_s", "controls")
    tol_v = cv2.getTrackbarPos("tol_v", "controls")
    open_k = cv2.getTrackbarPos("open_k", "controls")
    close_k = cv2.getTrackbarPos("close_k", "controls")

    m = np.zeros((H, W), np.uint8)
    for t, en in zip(state.targets, state.swatch_enabled):
        if not en:
            continue
        m = cv2.bitwise_or(m, mask_for_target(state.hsv, t, tol_h, tol_s, tol_v))
    m = postprocess_mask(m, open_k=open_k, close_k=close_k)
    state.mask = m


# ---------------------- swatch button UI ----------------------

def draw_targets_text(panel, x=10, y=20):
    cv2.putText(panel,
                "Pick: L-click image | Paint add: Shift+L-drag | Erase: R-drag",
                (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    y += 18
    cv2.putText(panel,
                "Swatches: L-click square=active | R-click square=toggle | X=delete active | DEL=pop",
                (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 255), 1, cv2.LINE_AA)

def draw_swatch_buttons(panel, origin=(10, 75), cell=22, pad=6, cols=10):
    state.swatch_ui = []
    x0, y0 = origin
    x, y = x0, y0

    for i, (th, ts, tv) in enumerate(state.targets):
        sw = np.uint8([[[th, ts, tv]]])
        bgr = cv2.cvtColor(sw, cv2.COLOR_HSV2BGR)[0, 0].tolist()
        bgr = tuple(int(c) for c in bgr)

        if not state.swatch_enabled[i]:
            bgr = tuple(int(c * 0.25) for c in bgr)

        x1, y1 = x + cell, y + cell
        cv2.rectangle(panel, (x, y), (x1, y1), bgr, -1)

        if i == state.active_idx:
            cv2.rectangle(panel, (x - 2, y - 2), (x1 + 2, y1 + 2), (255, 255, 255), 2)
        else:
            cv2.rectangle(panel, (x, y), (x1, y1), (40, 40, 40), 1)

        label = str(i + 1)
        cv2.putText(panel, label, (x + 4, y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(panel, label, (x + 4, y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        state.swatch_ui.append(((x, y, x1, y1), i))

        if (i + 1) % cols == 0:
            x = x0
            y += cell + pad
        else:
            x += cell + pad

def swatch_hit_test(x, y):
    for (x0, y0, x1, y1), i in state.swatch_ui:
        if x0 <= x <= x1 and y0 <= y <= y1:
            return i
    return None


# ---------------------- mouse callback ----------------------

def on_mouse(event, x, y, flags, param):
    if state.img is None or state.mask is None:
        return
    h, w = state.mask.shape
    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))

    shift = (flags & cv2.EVENT_FLAG_SHIFTKEY) != 0

    # swatch bar interactions
    if event == cv2.EVENT_LBUTTONDOWN:
        hit = swatch_hit_test(x, y)
        if hit is not None:
            state.active_idx = hit
            print(f"[active] swatch {hit + 1}")
            return

    if event == cv2.EVENT_RBUTTONDOWN:
        hit = swatch_hit_test(x, y)
        if hit is not None:
            state.swatch_enabled[hit] = not state.swatch_enabled[hit]
            print(f"[toggle] swatch {hit + 1} -> {'ON' if state.swatch_enabled[hit] else 'OFF'}")
            recompute_mask()
            return

    # paint add / erase
    if event == cv2.EVENT_LBUTTONDOWN and shift:
        state.paint_mode = "add"
        state.painting = True
        cv2.circle(state.mask, (x, y), state.brush_r, 255, -1)
        return

    if event == cv2.EVENT_RBUTTONDOWN:
        state.paint_mode = "erase"
        state.painting = True
        cv2.circle(state.mask, (x, y), state.brush_r, 0, -1)
        return

    if event == cv2.EVENT_MOUSEMOVE and state.painting:
        if state.paint_mode == "add":
            cv2.circle(state.mask, (x, y), state.brush_r, 255, -1)
        elif state.paint_mode == "erase":
            cv2.circle(state.mask, (x, y), state.brush_r, 0, -1)
        return

    if event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
        state.painting = False
        state.paint_mode = None
        return

    # normal left click (not shift): add swatch from pixel
    if event == cv2.EVENT_LBUTTONDOWN and not shift:
        th, ts, tv = state.hsv[y, x].tolist()
        state.targets.append((th, ts, tv))
        state.swatch_enabled.append(True)
        state.active_idx = len(state.targets) - 1
        state.last_pick = (x, y)
        print(f"[pick +] HSV={state.targets[-1]}  (total={len(state.targets)})")
        recompute_mask()
        return


# ---------------------- delete helpers ----------------------

def delete_swatch(i):
    if not (0 <= i < len(state.targets)):
        return
    removed = state.targets.pop(i)
    state.swatch_enabled.pop(i)
    print(f"[pick -] removed idx {i+1}: {removed}")
    if not state.targets:
        state.active_idx = -1
    else:
        state.active_idx = min(state.active_idx, len(state.targets) - 1)
    recompute_mask()


# ---------------------- output folders (NEW) ----------------------

def ensure_output_dirs(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    (root / "saved_mask").mkdir(parents=True, exist_ok=True)
    (root / "clipped_image").mkdir(parents=True, exist_ok=True)

def save_current(root: Path, filename: str, mask: np.ndarray, clipped: np.ndarray):
    cv2.imwrite(str(root / "saved_mask" / filename), mask)
    cv2.imwrite(str(root / "clipped_image" / filename), clipped)


# ---------------------- batch export (swatches only) ----------------------

def batch_export(root: Path):
    if not state.targets:
        print("[batch] no swatches selected")
        return

    ensure_output_dirs(root)

    tol_h = cv2.getTrackbarPos("tol_h", "controls")
    tol_s = cv2.getTrackbarPos("tol_s", "controls")
    tol_v = cv2.getTrackbarPos("tol_v", "controls")
    open_k = cv2.getTrackbarPos("open_k", "controls")
    close_k = cv2.getTrackbarPos("close_k", "controls")

    print("[batch] exporting all images using swatches (manual paint ignored)")
    for j, p in enumerate(state.paths):
        imgj = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if imgj is None:
            print("[skip] can't read", p)
            continue
        hsvj = to_hsv(imgj)

        m = np.zeros((imgj.shape[0], imgj.shape[1]), np.uint8)
        for t, en in zip(state.targets, state.swatch_enabled):
            if not en:
                continue
            m = cv2.bitwise_or(m, mask_for_target(hsvj, t, tol_h, tol_s, tol_v))
        m = postprocess_mask(m, open_k=open_k, close_k=close_k)

        clipped = cutout_bgr(imgj, m, bg=(0, 0, 0))
        save_current(root, p.name, m, clipped)

        if j % 50 == 0 or j == len(state.paths) - 1:
            print(f"  {j+1}/{len(state.paths)} {p.name}")


# ---------------------- main ----------------------

def main():
    in_dir = Path("abdomen_png")   # change if needed
    state.paths = sorted(in_dir.glob("*.png"))
    if not state.paths:
        raise FileNotFoundError(f"No PNGs in {in_dir}")

    # output root folder (NEW)
    out_root = Path("threshold_images")
    ensure_output_dirs(out_root)

    # load first image
    state.idx = 0
    state.img = cv2.imread(str(state.paths[state.idx]), cv2.IMREAD_COLOR)
    if state.img is None:
        raise RuntimeError(f"Failed to load {state.paths[state.idx]}")
    state.hsv = to_hsv(state.img)
    state.mask = np.zeros((state.img.shape[0], state.img.shape[1]), np.uint8)

    # windows
    cv2.namedWindow("overlay", cv2.WINDOW_NORMAL)
    cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("cutout", cv2.WINDOW_NORMAL)
    cv2.namedWindow("controls", cv2.WINDOW_NORMAL)

    # controls (AUTO recompute on changes)
    def on_controls_change(_=None):
        if state.img is not None:
            recompute_mask()

    def tb(name, v, vmax):
        cv2.createTrackbar(name, "controls", v, vmax, on_controls_change)

    tb("tol_h", 14, 90)
    tb("tol_s", 70, 255)
    tb("tol_v", 70, 255)
    tb("open_k", 0, 31)
    tb("close_k", 0, 31)

    cv2.createTrackbar("brush", "controls", 18, 120, lambda _=None: None)

    cv2.setMouseCallback("overlay", on_mouse)

    print("Controls:")
    print("  L-click image: add swatch (union)")
    print("  L-click swatch square: set active (white outline)")
    print("  R-click swatch square: toggle enable/disable (dims)")
    print("  Shift + L-drag: paint ADD to mask")
    print("  R-drag: paint ERASE from mask (block out)")
    print("  DEL/BKSP: remove last swatch | 1-9: remove swatch by index | X: delete active")
    print("  N/P: next/prev image (recompute from swatches) | C: clear swatches+mask")
    print("  S: save current -> threshold_images/ | B: batch export all -> threshold_images/")
    print("  ESC: quit")

    while True:
        state.brush_r = max(1, cv2.getTrackbarPos("brush", "controls"))

        overlay = overlay_mask(state.img, state.mask)
        draw_targets_text(overlay)
        draw_swatch_buttons(overlay, origin=(10, 75), cell=22, pad=6, cols=10)

        if state.last_pick is not None:
            cv2.circle(overlay, state.last_pick, 6, (255, 255, 255), 2)

        cv2.putText(overlay, state.paths[state.idx].name, (10, state.img.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2, cv2.LINE_AA)

        clipped = cutout_bgr(state.img, state.mask, bg=(0, 0, 0))

        cv2.imshow("overlay", overlay)
        cv2.imshow("mask", state.mask)
        cv2.imshow("cutout", clipped)

        key = cv2.waitKey(16) & 0xFF
        if key == 27:
            break

        if key in (ord('n'), ord('N')):
            state.idx = (state.idx + 1) % len(state.paths)
            state.img = cv2.imread(str(state.paths[state.idx]), cv2.IMREAD_COLOR)
            state.hsv = to_hsv(state.img)
            recompute_mask()
            state.last_pick = None

        if key in (ord('p'), ord('P')):
            state.idx = (state.idx - 1) % len(state.paths)
            state.img = cv2.imread(str(state.paths[state.idx]), cv2.IMREAD_COLOR)
            state.hsv = to_hsv(state.img)
            recompute_mask()
            state.last_pick = None

        if key in (8, 127):  # backspace/delete
            if state.targets:
                delete_swatch(len(state.targets) - 1)

        if ord('1') <= key <= ord('9'):
            delete_swatch(key - ord('1'))

        if key in (ord('x'), ord('X')):
            if 0 <= state.active_idx < len(state.targets):
                delete_swatch(state.active_idx)

        if key in (ord('c'), ord('C')):
            state.targets.clear()
            state.swatch_enabled.clear()
            state.active_idx = -1
            state.mask[:] = 0
            print("[clear] swatches cleared; mask cleared")

        if key in (ord('s'), ord('S')):
            ensure_output_dirs(out_root)
            name = state.paths[state.idx].name
            save_current(out_root, name, state.mask, clipped)
            print("[save] -> threshold_images/", name)

        if key in (ord('b'), ord('B')):
            batch_export(out_root)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()