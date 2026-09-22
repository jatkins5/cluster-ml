"""Read the LoVoCCS peak catalogue without adding a dependency.

The catalogue ships as .xlsx and the venv has no openpyxl. An xlsx is a zip
of XML, so the stdlib can do it: sheet1.xml holds the cells, and string cells
are indices into sharedStrings.xml. Read-only, and nothing is installed.
"""
import argparse
import glob
import os
import re
import zipfile
from xml.etree import ElementTree as ET

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def col_index(ref):
    """'BC12' -> 54 (0-based column)."""
    letters = re.match(r"([A-Z]+)", ref).group(1)
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def read_sheet(path, sheet=1):
    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.findall(f"{NS}si"):
                shared.append("".join(t.text or "" for t in si.iter(f"{NS}t")))
        root = ET.fromstring(z.read(f"xl/worksheets/sheet{sheet}.xml"))

    rows = []
    for row in root.iter(f"{NS}row"):
        cells = {}
        for c in row.findall(f"{NS}c"):
            v = c.find(f"{NS}v")
            if v is None or v.text is None:
                # inline string?
                is_ = c.find(f"{NS}is")
                if is_ is None:
                    continue
                val = "".join(t.text or "" for t in is_.iter(f"{NS}t"))
            elif c.get("t") == "s":
                val = shared[int(v.text)]
            else:
                val = v.text
            cells[col_index(c.get("r", "A1"))] = val
        if cells:
            width = max(cells) + 1
            rows.append([cells.get(i, "") for i in range(width)])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=None)
    ap.add_argument("--max-rows", type=int, default=25)
    args = ap.parse_args()

    path = args.path or glob.glob(
        "/oscar/data/idellant/Clusters/gen3_processing/Peak catalogue/*.xlsx")[0]
    print(f"{os.path.basename(path)}\n")
    rows = read_sheet(path)
    if not rows:
        raise SystemExit("no rows parsed")

    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    header, body = rows[0], rows[1:]
    print(f"{len(body)} rows x {width} columns")
    print(f"columns: {[h for h in header if h]}\n")

    w = [min(max(len(str(r[i])) for r in rows[:args.max_rows + 1]), 18)
         for i in range(width)]
    for r in rows[:args.max_rows + 1]:
        print("  ".join(str(c)[:w[i]].ljust(w[i]) for i, c in enumerate(r)))
    if len(body) > args.max_rows:
        print(f"... {len(body) - args.max_rows} more rows")

    # Which column holds a cluster name, and do our LOFAR targets appear?
    import csv
    names = set()
    with open("lovoccs_wl_masses.csv") as f:
        for row in csv.DictReader(f):
            if row["has_lotss"] == "True":
                names.add(row["key"])
    norm = lambda s: re.sub(r"[\s_]+", "", str(s)).upper()
    for i in range(width):
        vals = {norm(r[i]) for r in body if r[i]}
        hit = vals & names
        if hit:
            print(f"\ncolumn {i} ({header[i]!r}) matches {len(hit)} "
                  f"LOFAR targets, {len(vals)} distinct values")


if __name__ == "__main__":
    main()
