import re

def get_lattice_1d(path):
    m = re.search(r"N(\d+)", str(path))
    return int(m.group(1))

def get_J2(path):
    m = re.search(r"J2_(\d{3})", str(path))
    return float(m.group(1))/100

