"""Case-bound NP datasets; lazy exports keep CLI setup ahead of NumPy."""
from importlib import import_module

_EXPORTS = {
    "_archive": ("ArchiveContractError", "CaseArchive", "ROLE_SUFFIXES", "discover_members",
                 "inspect_case", "open_params_npz"),
    "_build": ("BuildContractError", "reference_build"),
    "_cbin": ("CbinContractError", "CbinHeader", "N_RECORDED", "ZipCbinStream", "parse_ch_header"),
    "_labels": ("LABEL_FIELDS", "LabelContractError", "build_labels", "frozen_unit_main_channels",
                "labels_metadata", "load_gt_times_units", "static_map_channels",
                "template_reference_allowed", "unit_group_masks"),
    "_meta": ("N_AP", "MetaContractError", "active_geometry", "sample_rate_hz"),
    "_registry": ("CONTRACT_CASES", "CaseSpec", "RegistryError", "default_registry_path", "load_registry"),
    "_template_reference": ("LABEL_EXPOSURE", "PRODUCTION_REFERENCE", "SOURCE_NAME", "SOURCE_SHA256",
                            "ReferenceContractError", "build_template_reference"),
}
__all__ = [name for names in _EXPORTS.values() for name in names]


def __getattr__(name):
    for module, names in _EXPORTS.items():
        if name in names:
            value = getattr(import_module("." + module, __name__), name)
            globals()[name] = value
            return value
    raise AttributeError(name)
