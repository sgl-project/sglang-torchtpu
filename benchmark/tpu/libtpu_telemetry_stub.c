// Stub to neutralize libtpu's internal TpuDiagnosticCoordinator and
// TpuTelemetryHarvester on multihost slices where GetAllCoreStates()
// throws absl::raw_hash_map::at std::out_of_range on remote chip IDs.

void _ZN3tpu24TpuDiagnosticCoordinator23HarvestAndRouteTelemetryEv(void* this_ptr) {}
void _ZN17platforms_deepsea9jellyfish3xdb13tpu_telemetry19TpuTelemetryHarvester16GetAllCoreStatesEv(void* this_ptr) {}
void _ZN17platforms_deepsea9jellyfish3xdb13tpu_telemetry20TpuTelemetryHarvester16GetAllCoreStatesEv(void* this_ptr) {}
