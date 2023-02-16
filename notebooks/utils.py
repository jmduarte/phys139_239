import awkward as ak
import numpy as np
import uproot

# fix for XRootD/uproot4 issue: https://github.com/scikit-hep/uproot4/discussions/355


def get_file_handler(file_name):
    xrootd_src = file_name.startswith("root://")
    if not xrootd_src:
        return {"file_handler": uproot.MultithreadedFileSource}  # otherwise the memory maps overload available Vmem
    elif xrootd_src:
        # uncomment below for MultithreadedXRootDSource
        return {"xrootd_handler": uproot.source.xrootd.MultithreadedXRootDSource}
    return {}


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def to_np_array(ak_array, max_n=100, pad=0):
    return ak.fill_none(ak.pad_none(ak_array, max_n, clip=True, axis=-1), pad).to_numpy()


def get_features_labels(file_name, features, spectators, labels, remove_mass_pt_window=True, entry_stop=None):
    # load file
    root_file = uproot.open(file_name, **get_file_handler(file_name))
    tree = root_file["deepntuplizer/tree"]
    feature_array = tree.arrays(features, entry_stop=entry_stop, library="np")
    spec_array = tree.arrays(spectators, entry_stop=entry_stop, library="np")
    label_array_all = tree.arrays(labels, entry_stop=entry_stop, library="np")

    feature_array = np.stack([feature_array[feat] for feat in features], axis=1)
    spec_array = np.stack([spec_array[spec] for spec in spectators], axis=1)

    njets = feature_array.shape[0]

    label_array = np.zeros((njets, 2))
    label_array[:, 0] = label_array_all["sample_isQCD"] * (
        label_array_all["label_QCD_b"]
        + label_array_all["label_QCD_bb"]
        + label_array_all["label_QCD_c"]
        + label_array_all["label_QCD_cc"]
        + label_array_all["label_QCD_others"]
    )
    label_array[:, 1] = label_array_all["label_H_bb"]

    # remove samples outside mass/pT window
    if remove_mass_pt_window:
        feature_array = feature_array[
            (spec_array[:, 0] > 40) & (spec_array[:, 0] < 200) & (spec_array[:, 1] > 300) & (spec_array[:, 1] < 2000)
        ]
        label_array = label_array[
            (spec_array[:, 0] > 40) & (spec_array[:, 0] < 200) & (spec_array[:, 1] > 300) & (spec_array[:, 1] < 2000)
        ]
        spec_array = spec_array[
            (spec_array[:, 0] > 40) & (spec_array[:, 0] < 200) & (spec_array[:, 1] > 300) & (spec_array[:, 1] < 2000)
        ]

    # remove unlabeled data
    feature_array = feature_array[np.sum(label_array, axis=1) == 1]
    spec_array = spec_array[np.sum(label_array, axis=1) == 1]
    label_array = label_array[np.sum(label_array, axis=1) == 1]

    return feature_array, label_array, spec_array


def make_image(feature_array, n_pixels=224, img_ranges=[[-0.8, 0.8], [-0.8, 0.8]]):
    wgt = feature_array[:, 0]  # ptrel
    x = feature_array[:, 1]  # etarel
    y = feature_array[:, 2]  # phirel
    img = np.zeros(shape=(len(wgt), n_pixels, n_pixels))
    for i in range(len(wgt)):
        hist2d, xedges, yedges = np.histogram2d(x[i], y[i], bins=[n_pixels, n_pixels], range=img_ranges, weights=wgt[i])
        img[i] = hist2d
    return np.expand_dims(img, axis=-1)
