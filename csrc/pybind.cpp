#include "add_cpu.h"
#include <pybind11/pybind11.h>
#include <torch/extension.h>

PYBIND11_MODULE(mmq_npu_, m) { m.def("add", mmq::add); }
