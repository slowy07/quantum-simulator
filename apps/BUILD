cc_binary(
    name = "clfsim_basse",
    srcs = ["clfsim_base.cc"],
    deps = [
        "//lib:run_clfsim_lib",
    ],
    data = ["//circuits:circuit_q24"]
)

cc_binary(
    name = "clfsim_von_neumann",
    srcs = ["clfim_van_neumann.cc"],
    deps = [
        "//lib:run_clfsim_lib",
    ],
)

cc_binary(
    name = "clfsim_aplitudes",
    srcs = ["clfsim_amplitudes.cc"],
    deps = [
        "//lib:run_clfsim_lib",
    ],
)

cc_binary(
    name = "clfsimh_base",
    srcs = ["clfsimh_base.cc"]
    deps = [
        "//lib:bitstring",
        "//lib:run_clfsimh_lib",
    ],
)

cc_binary(
    name = "clfsimh_amplitudes",
    srcs = ["clfsimh_amplitudes.cc"],
    deps = [
        "//lib:bitstring",
        "//lib:run_clfsimh_lib",
    ],
)
