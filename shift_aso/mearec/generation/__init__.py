"""Public MEArec multistage raw-data generation (floor 5 uVrms, D0-D4).

Fixed scientific scope: 256-channel SqMEA-16-36, 20 kHz, 300 s, 64 units /
176248 spikes, independent 5 uVrms floor, total CAL SNR targets 10/15/20 dB.
This package exposes the pure portable core (selection, count-conditioned
sampling, covariance, calibration solver, exact float32 association, virtual
input iteration) plus a native MEArec execution path used only with
``--execute``. Importing the package never parses the CLI and never imports
MEArec.
"""
# Keep module execution import-light so CLI controls precede NumPy imports.

__all__ = [
    "BUNDLE_FILES", "EXC_MODELS", "LIBS1", "PUBLIC_STAGES", "SPATIAL_ENV",
    "STAGE_ORDER", "STATUS_COMPLETE", "StopGeneration", "SplitAccumulator",
    "assert_selection", "build_candidates", "cal_moments", "cal_snr_db",
    "component_bytes", "create_once", "dataset_entry", "free_bytes",
    "independent_chunk_rng", "iter_input", "ledger_read", "ledger_update",
    "load_config",
    "mixed_input_block_f32", "mixed_noise_block_f32", "native_amp",
    "plan_dataset", "read_generation_manifest", "resolve_assets_dir",
    "resolve_thread_env", "sample_dataset", "sample_unit_times",
    "seconds_to_samples", "seconds_to_samples_is_exact_roundtrip",
    "select_positions", "selection_receipt", "sha256_file", "solve_scale",
    "spatial_chunk_seeds", "spatial_covariance", "stop_before_write",
    "total_sigma", "unit_split_sigma", "validate_config", "validate_schedule",
    "verified_skip", "write_receipt",
]


def __getattr__(name):
    if name in __all__:
        from importlib import import_module
        value = getattr(import_module(".core", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(name)
