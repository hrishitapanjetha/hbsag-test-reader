import difflib
import math
import re

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

import ctypes.util
import glob
import os

_orig_find_library = ctypes.util.find_library


def _find_library(name):
    found = _orig_find_library(name)
    if found or name != "zbar":
        return found
    for pattern in ("/opt/homebrew/lib/libzbar*.dylib", "/usr/local/lib/libzbar*.dylib"):
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[0]
    return None


ctypes.util.find_library = _find_library

try:
    from pyzbar.pyzbar import decode as zbar_decode
    HAVE_ZBAR = True
except Exception:
    HAVE_ZBAR = False

COLORS_BGR = {"barcode": (255, 140, 0), "control-line": (0, 0, 255), "handwritten-id": (0, 140, 255),
              "test-cassette": (0, 200, 0), "test-line": (255, 0, 255), "test-type": (200, 200, 0)}
RES_BGR = {"positive": (0, 170, 0), "negative": (200, 110, 0), "invalid": (0, 0, 220)}
BC_ANGLES = (0, 90, 180, 270, 10, -10, 20, -20)
CONFUSE = str.maketrans({"O": "0", "o": "0", "D": "0", "Q": "0", "I": "1", "l": "1", "|": "1", "i": "1",
                         "Z": "2", "z": "2", "S": "5", "s": "5", "B": "8", "g": "9", "q": "9"})


def digits_of(s):
    return re.sub(r"\D", "", s or "")


def to_digits(s):
    return re.sub(r"\D", "", (s or "").translate(CONFUSE))


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def inside(pt, b):
    return b[0] <= pt[0] <= b[2] and b[1] <= pt[1] <= b[3]


def inter(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def iou(a, b):
    i = inter(a, b)
    u = area(a) + area(b) - i
    return i / u if u > 0 else 0.0


def contain_ratio(small, big):
    return inter(small, big) / area(small) if area(small) > 0 else 0.0


def union_box(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def expand(box, fx, fy, shape):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    H, W = shape[:2]
    return [int(max(0, x1 - fx * w)), int(max(0, y1 - fy * h)), int(min(W, x2 + fx * w)), int(min(H, y2 + fy * h))]


def group_by_proximity(items, factor=4.0):
    n = len(items)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            a, b = items[i]["box"], items[j]["box"]
            ca, cb = center(a), center(b)
            scale = max(a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1])
            if math.hypot(ca[0] - cb[0], ca[1] - cb[1]) < factor * scale:
                parent[find(i)] = find(j)
    g = {}
    for i in range(n):
        g.setdefault(find(i), []).append(items[i])
    return list(g.values())


def top_boxes(dets, cls, n):
    return sorted([d for d in dets if d["cls"] == cls], key=lambda d: -d["conf"])[:n]


def hbsag_score(text):
    toks = re.findall(r"[A-Za-z]+", text.lower())
    return max([difflib.SequenceMatcher(None, t, "hbsag").ratio() for t in toks], default=0.0)


def photo_quality(img):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    bright = float(gray.mean())
    warnings = []
    if min(h, w) < 480:
        warnings.append("Low resolution: the photograph is small. Please retake at full camera resolution.")
    if sharp < 15:
        warnings.append("The photograph looks very blurred. Hold the camera steady and focus on the device.")
    if bright < 60:
        warnings.append("The photograph is too dark. Add light or avoid shadows.")
    elif bright > 220:
        warnings.append("The photograph is overexposed. Reduce glare and strong light.")
    return dict(width=w, height=h, sharpness=sharp, brightness=bright, warnings=warnings)


class HBsAgPipeline:
    def __init__(self, weights, conf=0.25, imgsz=1280, photo_rule="any_positive", two_same_label_positive=False, load_readers=True):
        self.conf = conf
        self.imgsz = imgsz
        self.photo_rule = photo_rule
        self.two_same_label_positive = two_same_label_positive
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.det = YOLO(weights)
        self.reader = None
        self.trocr = None
        self.trocr_proc = None
        if load_readers:
            self._load_readers()

    def _load_readers(self, easyocr_reader=True, handwriting=True):
        if easyocr_reader and self.reader is None:
            import easyocr
            self.reader = easyocr.Reader(["en"], gpu=(self.device == "cuda"), verbose=False)
        if handwriting and self.trocr is None:
            from transformers import RobertaTokenizer, ViTImageProcessor, TrOCRProcessor, VisionEncoderDecoderModel
            name = "microsoft/trocr-base-handwritten"
            tok = RobertaTokenizer.from_pretrained(name)
            proc = ViTImageProcessor.from_pretrained(name)
            self.trocr_proc = TrOCRProcessor(image_processor=proc, tokenizer=tok)
            self.trocr = VisionEncoderDecoderModel.from_pretrained(name).to(self.device).eval()

    def detect(self, path):
        r = self.det.predict(source=path, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        return [dict(cls=r.names[int(b.cls[0])], conf=float(b.conf[0]), box=[float(v) for v in b.xyxy[0].tolist()]) for b in r.boxes]

    def build_devices(self, dets):
        cass = sorted([d for d in dets if d["cls"] == "test-cassette"], key=lambda d: -d["conf"])
        keep = []
        for c in cass:
            if all(contain_ratio(c["box"], k["box"]) < 0.8 for k in keep):
                keep.append(c)
        lines = sorted([d for d in dets if d["cls"] in ("control-line", "test-line")], key=lambda d: -d["conf"])
        uniq, dropped = [], 0
        for l in lines:
            if any(l["cls"] != u["cls"] and iou(l["box"], u["box"]) > 0.6 for u in uniq):
                dropped += 1
                continue
            uniq.append(l)
        devs = [dict(box=c["box"], conf=c["conf"], virtual=False, controls=[], tests=[]) for c in keep]
        orphans = []
        for l in uniq:
            cands = [i for i, dv in enumerate(devs) if inside(center(l["box"]), dv["box"])]
            if cands:
                i = min(cands, key=lambda i: area(devs[i]["box"]))
                devs[i]["controls" if l["cls"] == "control-line" else "tests"].append(l)
            else:
                orphans.append(l)
        for g in group_by_proximity(orphans):
            b = union_box([x["box"] for x in g])
            w, h = b[2] - b[0], b[3] - b[1]
            devs.append(dict(box=[b[0] - 0.3 * w, b[1] - 0.3 * h, b[2] + 0.3 * w, b[3] + 0.3 * h], conf=0.0, virtual=True,
                             controls=[x for x in g if x["cls"] == "control-line"], tests=[x for x in g if x["cls"] == "test-line"]))
        devs.sort(key=lambda d: (center(d["box"])[1], center(d["box"])[0]))
        return devs, dropped

    def evaluate_device(self, dv):
        nc, nt = len(dv["controls"]), len(dv["tests"])
        double_c = nc >= 2 and nt == 0
        double_t = nt >= 2 and nc == 0
        if self.two_same_label_positive and (double_c or double_t):
            res = "positive"
        elif nc == 0:
            res = "invalid"
        elif nt >= 1:
            res = "positive"
        else:
            res = "negative"
        dv.update(result=res, double_control=double_c, double_test=double_t,
                  control_conf=max([x["conf"] for x in dv["controls"]], default=0.0),
                  test_conf=max([x["conf"] for x in dv["tests"]], default=0.0))
        return dv

    def photo_decision(self, devs):
        valid = [d["result"] for d in devs if d["result"] != "invalid"]
        if not valid:
            return "invalid", False
        discordant = len(set(valid)) > 1
        if self.photo_rule == "any_positive":
            final = "positive" if "positive" in valid else "negative"
        else:
            final = "positive" if all(r == "positive" for r in valid) else "negative"
        return final, discordant

    def decode_barcode(self, img, box):
        if not HAVE_ZBAR:
            return None
        x1, y1, x2, y2 = expand(box, 0.08, 0.20, img.shape)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        up = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        for g in (gray, up, cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(up)):
            pil = Image.fromarray(g)
            for ang in BC_ANGLES:
                out = zbar_decode(pil if ang == 0 else pil.rotate(ang, expand=True, fillcolor=255))
                if out:
                    t = out[0].data.decode("utf-8", "ignore")
                    if digits_of(t):
                        return t
        return None

    def barcode_digits_ocr(self, img, box):
        x1, y1, x2, y2 = box
        h = y2 - y1
        e = expand([x1, y1, x2, y2 + 0.5 * h], 0.05, 0.05, img.shape)
        crop = img[e[1]:e[3], e[0]:e[2]]
        if crop.size == 0:
            return None
        best = None
        for _, text, conf in self.reader.readtext(crop, allowlist="0123456789", detail=1, rotation_info=[90, 180, 270]):
            d = digits_of(text)
            if len(d) >= 8 and (best is None or conf > best[1]):
                best = (d, float(conf))
        return best[0] if best else None

    def read_handwritten(self, img, box):
        x1, y1, x2, y2 = expand(box, 0.05, 0.10, img.shape)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return "", "", 0.0
        h, w = crop.shape[:2]
        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        angles = [0, 180] + ([90, 270] if h > 1.5 * w else [])
        pv = self.trocr_proc(images=[pil.rotate(a, expand=True) for a in angles], return_tensors="pt").pixel_values.to(self.device)
        with torch.no_grad():
            gen = self.trocr.generate(pv, output_scores=True, return_dict_in_generate=True, max_new_tokens=20)
        texts = self.trocr_proc.batch_decode(gen.sequences, skip_special_tokens=True)
        probs = torch.stack(gen.scores, dim=1).softmax(-1).max(-1).values
        best = None
        for i, t in enumerate(texts):
            conf = float(probs[i].mean())
            d = to_digits(t)
            score = conf * min(1.0, len(d) / 9)
            if best is None or score > best[0]:
                best = (score, t.strip(), d, conf)
        return best[1], best[2], best[3]

    def read_test_type(self, img, box):
        e = expand(box, 0.10, 0.10, img.shape)
        crop = img[e[1]:e[3], e[0]:e[2]]
        if crop.size == 0:
            return "", 0.0
        text = " ".join(t for _, t, _ in self.reader.readtext(crop, detail=1, rotation_info=[90, 180, 270])).strip()
        return text, hbsag_score(text)

    def read_ids(self, img, dets, use_handwriting=True):
        out = dict(barcode_id="", barcode_source="", hw_raw="", hw_digits="", hw_conf=0.0, tt_text="", tt_score=0.0)
        bcs = top_boxes(dets, "barcode", 2)
        for d in bcs:
            t = self.decode_barcode(img, d["box"])
            if t:
                out["barcode_id"], out["barcode_source"] = digits_of(t), "barcode scan"
                break
        if not out["barcode_id"] and self.reader is not None:
            for d in bcs:
                t = self.barcode_digits_ocr(img, d["box"])
                if t:
                    out["barcode_id"], out["barcode_source"] = t, "printed digits"
                    break
        if use_handwriting and self.trocr is not None:
            best = None
            for d in top_boxes(dets, "handwritten-id", 3):
                raw, dig, conf = self.read_handwritten(img, d["box"])
                sc = conf * min(1.0, len(dig) / 9)
                if best is None or sc > best[0]:
                    best = (sc, raw, dig, conf)
            if best:
                out.update(hw_raw=best[1], hw_digits=best[2], hw_conf=best[3])
        if self.reader is not None:
            best = None
            for d in top_boxes(dets, "test-type", 2):
                t, sc = self.read_test_type(img, d["box"])
                if best is None or sc > best[1]:
                    best = (t, sc)
            if best:
                out.update(tt_text=best[0], tt_score=best[1])
        return out

    def process(self, path, read_ids=True, use_handwriting=True):
        img = cv2.imread(path)
        quality = photo_quality(img)
        dets = self.detect(path)
        devs, dropped = self.build_devices(dets)
        for dv in devs:
            self.evaluate_device(dv)
        photo, discordant = self.photo_decision(devs)
        ids = self.read_ids(img, dets, use_handwriting) if read_ids else dict(barcode_id="", barcode_source="", hw_raw="", hw_digits="", hw_conf=0.0, tt_text="", tt_score=0.0)
        double = sum(d["double_control"] or d["double_test"] for d in devs)
        invalid = sum(d["result"] == "invalid" for d in devs)
        reasons = []
        if not devs:
            reasons.append("No test device was found in the photograph.")
        if invalid and devs:
            reasons.append(f"{invalid} device(s) show no control line, so the result is invalid for them.")
        if double:
            reasons.append(f"{double} device(s) have two lines of the same type, so the lines could not be told apart.")
        if discordant:
            reasons.append("The devices in this photograph disagree with each other.")
        if read_ids and not ids["barcode_id"]:
            reasons.append("The barcode could not be read.")
        if ids["barcode_id"] and ids["hw_digits"] and ids["barcode_id"] != ids["hw_digits"]:
            reasons.append("The handwritten ID does not match the barcode. Please confirm the specimen ID.")
        reasons += quality["warnings"]
        hard = bool(not devs or invalid or double or discordant)
        row = dict(
            photo_result=photo if devs else "invalid",
            n_devices=len(devs),
            device_results=[d["result"] for d in devs],
            double_lines=int(double),
            discordant=bool(discordant),
            test_conf=max([d["test_conf"] for d in devs if d["controls"]], default=0.0),
            specimen_id=ids["barcode_id"] or "",
            specimen_id_source=ids["barcode_source"],
            hw_raw=ids["hw_raw"], hw_digits=ids["hw_digits"], hw_conf=ids["hw_conf"],
            tt_text=ids["tt_text"], tt_score=ids["tt_score"],
            test_type="HBsAg" if ids["tt_score"] >= 0.7 else "not confirmed",
            hw_enabled=bool(read_ids and use_handwriting and self.trocr is not None),
            ids_match=bool(ids["barcode_id"] and ids["barcode_id"] == ids["hw_digits"]),
            needs_review=bool(reasons),
            hard_review=hard,
            reasons=reasons,
            quality=quality,
        )
        return row, dict(dets=dets, devices=devs, image=img)


def draw_final(img, ex):
    im = img.copy()
    s = max(im.shape[:2]) / 1600
    t = max(2, int(3 * s))
    for d in ex["dets"]:
        if d["cls"] in ("barcode", "handwritten-id", "test-type"):
            x1, y1, x2, y2 = map(int, d["box"])
            cv2.rectangle(im, (x1, y1), (x2, y2), COLORS_BGR[d["cls"]], max(1, t - 1))
    for k, dv in enumerate(ex["devices"], 1):
        col = RES_BGR[dv["result"]]
        x1, y1, x2, y2 = map(int, dv["box"])
        cv2.rectangle(im, (x1, y1), (x2, y2), col, t + 2)
        cv2.putText(im, f"device {k}: {dv['result'].upper()}", (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9 * s + 0.35, col, max(2, int(2.5 * s)))
        for l in dv["controls"] + dv["tests"]:
            lx1, ly1, lx2, ly2 = map(int, l["box"])
            cv2.rectangle(im, (lx1, ly1), (lx2, ly2), COLORS_BGR[l["cls"]], max(1, t - 1))
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
