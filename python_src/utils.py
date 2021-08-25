import re

def get_lattice_1d(path):
    m = re.search(r"N(\d+)", str(path))
    return int(m.group(1))

def get_J2(path):
    m = re.search(r"J2_(\d{3})", str(path))
    return float(m.group(1))/100

def dataset_total_and_save(N, dim):
    total_dataset = 512*150*1024
    save_dataset = 1024*512
    return total_dataset, save_dataset

