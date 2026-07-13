#!/usr/bin/env python3
"""
Manual eye annotation tool — click 5+ points on pupil + iris edges.
Uses live-refresh so clicks show immediately.

Usage:
  python manual_annotate.py --image_dir images/0

Controls:
  Left click  = add point to current mode (pupil or iris)
  p           = switch to PUPIL mode (green dots)
  i           = switch to IRIS mode (blue dots)
  c           = clear all points on current image
  n / SPACE   = save & next image
  b           = save & previous image
  q / ESC     = quit & save
"""
import cv2, json, os, sys, glob, argparse, numpy as np

# ── Globals for mouse callback ──────────────────────────────────
g_mode = 'pupil'
g_pupil_pts = []
g_iris_pts = []
g_img = None
g_needs_redraw = False

def on_mouse(event, x, y, flags, param):
    global g_pupil_pts, g_iris_pts, g_needs_redraw
    if event == cv2.EVENT_LBUTTONDOWN:
        if g_mode == 'pupil':
            g_pupil_pts.append((x, y))
        else:
            g_iris_pts.append((x, y))
        g_needs_redraw = True


def fit_ellipse(pts):
    if len(pts) < 5:
        return None
    try:
        e = cv2.fitEllipse(np.array(pts, dtype=np.float32))
        (cx, cy), (MA, ma), angle = e
        return [float(cx), float(cy), float(MA/2), float(ma/2), float(angle)]
    except:
        return None


def draw(img):
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Pupil points + ellipse
    for pt in g_pupil_pts:
        cv2.circle(vis, pt, 3, (0, 255, 0), -1)
    pup = fit_ellipse(g_pupil_pts)
    if pup is not None:
        cv2.ellipse(vis, ((int(pup[0]), int(pup[1])), (int(pup[2]*2), int(pup[3]*2)), pup[4]),
                    (0, 255, 0), 2)

    # Iris points + ellipse
    for pt in g_iris_pts:
        cv2.circle(vis, pt, 3, (255, 0, 0), -1)
    iri = fit_ellipse(g_iris_pts)
    if iri is not None:
        cv2.ellipse(vis, ((int(iri[0]), int(iri[1])), (int(iri[2]*2), int(iri[3]*2)), iri[4]),
                    (255, 0, 0), 2)

    # HUD
    h, w = vis.shape[:2]
    mode_color = (0, 255, 0) if g_mode == 'pupil' else (255, 0, 0)
    cv2.putText(vis, f'Mode: {g_mode.upper()} | Pupil={len(g_pupil_pts)}pts Iris={len(g_iris_pts)}pts',
                (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 1)
    cv2.putText(vis, 'p=pupil i=iris c=clear n=next b=back q=quit',
                (5, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1)
    return vis


def main():
    global g_mode, g_pupil_pts, g_iris_pts, g_img, g_needs_redraw

    ap = argparse.ArgumentParser()
    ap.add_argument('--image_dir', required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.image_dir, '*.png')))
    if not files:
        print(f'No .png files in {args.image_dir}')
        sys.exit(1)

    json_path = os.path.join(args.image_dir, 'annotations.json')
    saved = json.load(open(json_path)) if os.path.exists(json_path) else {}

    cv2.namedWindow('Eye Annotator')
    cv2.setMouseCallback('Eye Annotator', on_mouse)

    idx = 0
    print(f'{len(files)} images | Saves to: {json_path}')
    print('Click pupil edge 5x → press i → click iris edge 5x → press n')

    while True:
        fname = os.path.basename(files[idx])

        # Load image
        img_raw = cv2.imread(files[idx], cv2.IMREAD_GRAYSCALE)
        if img_raw is None:
            idx += 1
            continue
        g_img = img_raw

        # Restore previous annotation for this file
        if fname in saved:
            g_pupil_pts = [(int(p[0]), int(p[1])) for p in saved[fname].get('pupil_pts', [])]
            g_iris_pts  = [(int(p[0]), int(p[1])) for p in saved[fname].get('iris_pts', [])]
        else:
            g_pupil_pts = []
            g_iris_pts  = []

        # Show current
        cv2.imshow('Eye Annotator', draw(img_raw))
        g_needs_redraw = False

        # Show file info
        print(f'[{idx+1}/{len(files)}] {fname}  |  p=pupil i=iris c=clear n=next b=back q=quit', end='\r')

        # Wait for key
        while True:
            key = cv2.waitKey(30) & 0xFF

            # Redraw if mouse clicked
            if g_needs_redraw:
                cv2.imshow('Eye Annotator', draw(g_img))
                g_needs_redraw = False

            if key == 255:  # no key
                continue
            elif key == ord('q') or key == 27:  # q or ESC
                # Save current before quit
                pup = fit_ellipse(g_pupil_pts)
                iri = fit_ellipse(g_iris_pts)
                saved[fname] = {'pupil_ellipse': pup, 'iris_ellipse': iri,
                                'pupil_pts': [[int(x),int(y)] for x,y in g_pupil_pts],
                                'iris_pts':  [[int(x),int(y)] for x,y in g_iris_pts]}
                with open(json_path, 'w') as f:
                    json.dump(saved, f, indent=2)
                print(f'\nSaved {len(saved)} annotations to {json_path}')
                cv2.destroyAllWindows()
                return
            elif key == ord('p'):
                g_mode = 'pupil'
                cv2.imshow('Eye Annotator', draw(g_img))
                break
            elif key == ord('i'):
                g_mode = 'iris'
                cv2.imshow('Eye Annotator', draw(g_img))
                break
            elif key == ord('c'):
                g_pupil_pts = []
                g_iris_pts  = []
                cv2.imshow('Eye Annotator', draw(g_img))
                break
            elif key == ord('n') or key == 32:  # n or SPACE
                pup = fit_ellipse(g_pupil_pts)
                iri = fit_ellipse(g_iris_pts)
                saved[fname] = {'pupil_ellipse': pup, 'iris_ellipse': iri,
                                'pupil_pts': [[int(x),int(y)] for x,y in g_pupil_pts],
                                'iris_pts':  [[int(x),int(y)] for x,y in g_iris_pts]}
                idx = min(idx + 1, len(files) - 1)
                break
            elif key == ord('b'):
                pup = fit_ellipse(g_pupil_pts)
                iri = fit_ellipse(g_iris_pts)
                saved[fname] = {'pupil_ellipse': pup, 'iris_ellipse': iri,
                                'pupil_pts': [[int(x),int(y)] for x,y in g_pupil_pts],
                                'iris_pts':  [[int(x),int(y)] for x,y in g_iris_pts]}
                idx = max(idx - 1, 0)
                break

        # Periodic save
        if idx % 10 == 0:
            with open(json_path, 'w') as f:
                json.dump(saved, f, indent=2)


if __name__ == '__main__':
    main()
