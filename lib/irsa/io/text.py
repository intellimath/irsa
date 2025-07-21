import os
import numpy as np
from datetime import date
from irsa.preprocess import utils
from irsa.spectra import SpectraSeries, Spectra, SpectraCollection
# from recordclass import make_dataclass

_default_keys = [
    'дата', 
    'вид_бактерий', 'штамм_бактерий', 
    'резистентность', 'отсечки_по_молекулярной_массе',
    'номер_цикла', 'номер_эксперимента_в_цикле', 
    'номер_повтора', 'тип_измерения_спектров',
    'начальная_концентрация_клеток_в_пробе', 'номер_подложки',
    'капля', 'вода', 'отмывка_фильтров',
    "комментарий"]

def parse_file_name(fname, attr_names):
    items = items.split('_')

    attrs = {}
    j = 0
    for item in items:
        name = attr_names[j]
        j += 1
        if name == 'комментарий':
            if item.startswith('!'):
                if item.endswith('!'):
                    item = item[1:-1]
                else:
                    item = item[1:]
            name = attr_names[j]
            continue
        elif name in ('дата', 'дата_подложки'):
            item = date.fromisoformat(item)
        attrs[name] = item
        j += 1

    return attrs

# def find_spectra_info_all(root):
#     for root_entry in os.scandir(root):
#         if not root_entry.is_dir():
#             continue
#         for entry in os.scandir(f"{root}/root_entry.name"):
#             if not entry.is_dir():
#                 continue
        
        
# SpectraAttrs = make_dataclass(
#     "SpectraAttrs", 
#     "date cls subcls resistance n_cycle n_try n_retry concentration wave_length laser_intensity series comment")

def read_spectra_attrs(dirname):
    ret = {}
    for line in open(f"{dirname}/attrs.txt", "rt"):
        if line[0] == "\ufeff":
            line = line[1:]
        line = line.strip()
        if not line:
            continue

        if line[0] == '#':
            continue

        if ":" not in line:
            raise ValueError("В строке отсутствует ':'")
        name, value = line.split(":")
        
        name = name.strip()
        if "  " in name:
            name = name.replace("  ", " ")
        if " " in name:
            name = name.replace(" ", "_")
        
        value = value.strip()
        if value == "no_date":
            value = ""
        # if "," in value:
        #     value = [v.strip() for v in value.split(",")]

        ret[name] = value

    return ret

def load_txt_spectra(path, delimiter="\t", skiprows=0):  
    xy = np.loadtxt(path, delimiter=delimiter, skiprows=skiprows)

    x = xy[:,0]
    ys = xy[:,1:]
    
    if ys.shape[1] == 1:
        ys = np.ascontiguousarray(ys[:,0])
    elif ys.shape[1] > 1:
        ys = np.ascontiguousarray(ys.T)
    
    x = np.ascontiguousarray(x)

    return x, ys

def load_txt_dir(path, delimiter="\t", skiprows=0):
    """
    """
    import os

    Xs = []
    Ys = []
    fnames = os.listdir(path)
    fnames = [fname for fname in fnames if fname.endswith(".txt") and fname != "attrs.txt"]
    fnames.sort()
    for fname in fnames:

        x, ys = load_txt_spectra(f"{path}/{fname}", delimiter=delimiter, skiprows=skiprows)

        # xy = np.loadtxt(f"{path}/{fname}", delimiter=delimiter)

        # x = xy[:,0]
        # ys = xy[:,1:]
        
        # if ys.shape[1] == 1:
        #     ys = np.ascontiguousarray(ys[:,0])
        # elif ys.shape[1] > 1:
        #     ys = np.ascontiguousarray(ys.T)
        
        # x = np.ascontiguousarray(x)    
            
        Xs.append(x)
        # Ys.append(np.power(ys, 0.25))
        Ys.append(ys)
        
    return Xs, Ys

def load_spectra(root, options, clear=True, skiprows=0):
    import os

    # if clear:
    #     dd.clear()
    dd = {}
    for entry in os.scandir(root):
        if not entry.is_dir():
            continue
        
        dirname = entry.name
        # print(dirname)
        dirpath = f"{root}/{dirname}"
    
        ret = load_experiment_spectra_all(dirpath, options, skiprows=skiprows)

        for key in ret:
            dd[key] = ret[key]
        # dd.update(ret)
    
    return SpectraCollection(dd)

def load_experiment_spectra_all(root, options=None, skiprows=0):
    dd = {}
    for entry in os.scandir(root):
        if not entry.is_dir():
            continue

        dirname = entry.name
        # print("\t", dirname)
        dirpath = f"{root}/{dirname}"

        spectra = load_experiment_spectra(dirpath, options, skiprows=skiprows)
        if spectra is None:
            continue

        attrs = spectra.attrs
        attr_names = _default_keys
        key = "_".join(
            attrs[k] for k in attr_names if k in attrs)
        dd[key] = spectra
        spectra.attrs["key"] = key
        spectra.attrs["source"] = root

    return SpectraCollection(dd)

def load_experiment_spectra(dirpath, options=None, skiprows=0):
    attrs = read_spectra_attrs(dirpath)

    is_ok = True
    if options:
        for key, vals in options.items():
            if attrs.get(key, None) not in vals:
                is_ok = False
                break
    if not is_ok:
        return None
    
    Xs, Ys = load_txt_dir(dirpath, skiprows=skiprows)
    # print(len(Xs), len(Ys))
    print(dirpath)
    # print(os.path.split(dirpath)[-1], Xs[0].shape, Ys[0].shape, {k:v for k,v in attrs.items() if k in options})

    mesure_type = attrs["тип_измерения_спектров"]

    if mesure_type == "SE":
        # print(type(Ys[0]))
        if len(Ys[0].shape) > 1:
            return SpectraSeries(Xs, Ys, attrs)
        else:
            return Spectra(Xs, Ys, attrs)
    elif mesure_type == "SS":
            return SpectraSeries(Xs, Ys, attrs)

def collect_attr_values(root):
    attrs = {}
    for entry in os.scandir(root):
        if not entry.is_dir():
            continue
        
        dirname = entry.name
        # print(dirname)
        dirpath = f"{root}/{dirname}"
    
        collect_experiment_attrs(dirpath, attrs)

        # for key, vals in ret.items():
        #     vals.union( attrs.setdefault(key, set()) )

    return {key:list(sorted(vals)) for key,vals in attrs.items()}

def collect_experiment_attrs(root, attrs):
    for entry in os.scandir(root):
        if not entry.is_dir():
            continue
        
        dirname = entry.name
        # print("\t", dirname)
        dirpath = f"{root}/{dirname}"

        ret = read_spectra_attrs(dirpath)
        for key,val in ret.items():
            # print(key, val)
            vals = attrs.setdefault(key, set())
            if val not in vals:
                vals.add(val)
