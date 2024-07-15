import os
import numpy as np
from datetime import date
from irsa.preprocess import utils
from irsa.spectra.objects import ExperimentSpectrasSeries, ExperimentSpectras
# from recordclass import make_dataclass

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

def read_spectras_attrs(dirname):
    ret = {}
    for line in open(f"{dirname}/attrs.txt", "rt"):
        if line[0] == "\ufeff":
            line = line[1:]
        line = line.strip()
        if not line:
            continue

        if ":" not in line:
            raise ValueError("В строке отсутствует ':'")
        name, value = line.split(":")
        
        name = name.strip()
        if " " in name:
            name = name.replace(" ", "_")
        
        value = value.strip()
        if "," in value:
            value = [v.strip() for v in value.split(",")]

        ret[name] = value

    return ret

def load_txt_dir(path, delimiter="\t"):
    """
    """
    import os

    Xs = []
    Ys = []
    for fname in os.listdir(path):
        if not fname.endswith(".txt"):
            continue

        if fname == "attrs.txt":
            continue

        xy = np.loadtxt(f"{path}/{fname}", delimiter=delimiter)

        x = xy[:,0]
        ys = xy[:,1:]
        
        if ys.shape[1] == 1:
            ys = np.ascontiguousarray(ys[:,0])
        elif ys.shape[1] > 1:
            ys = np.ascontiguousarray(ys.T)
        
        x = np.ascontiguousarray(x)    
            
        Xs.append(x)
        Ys.append(ys)
        
    return Xs, Ys

def load_spectras(root, options=None):
    import os

    dd = {}
    for entry in os.scandir(root):
        if not entry.is_dir():
            continue
        
        dirname = entry.name
        # print(dirname)
        dirpath = f"{root}/{dirname}"
    
        ret = load_experiment_spectras_all(dirpath, options)

        dd.update(ret)

    dd = {k:dd[k] for k in sorted(dd.keys())}
    
    return dd

def load_experiment_spectras(dirpath, options=None):
    attrs = read_spectras_attrs(dirpath)

    is_ok = True
    if options:
        for key, val in options.items():
            if attrs.get(key, None) != val:
                is_ok = False
                break
    if not is_ok:
        return None
    
    Xs, Ys = load_txt_dir(dirpath)
    print(os.path.split(dirpath)[-1])


    if len(Ys[0].shape) > 1:
        return ExperimentSpectrasSeries(Xs, Ys, attrs)
    else:
        return ExperimentSpectras(Xs, Ys, attrs)

def load_experiment_spectras_all(root, options=None):
    import os

    dd = {}
    for entry in os.scandir(root):
        if not entry.is_dir():
            continue
        
        dirname = entry.name
        # print("\t", dirname)
        dirpath = f"{root}/{dirname}"

        spectras = load_experiment_spectras(dirpath, options)
        if spectras is None:
            continue

        attrs = spectras.attrs
        attr_names = (
            "вид_бактерий", "штамм_бактерий", "резистентность", 
            "отсечки_по_молекулярной_массе", "начальная_концентрация_клеток_в_пробе", "дата", "комментарий"
        )
        key = "_".join(
            attrs[k] for k in attr_names)
        dd[key] = spectras

    return dd