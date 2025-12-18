#pragma once

// Stub file - CPU NTT trace disabled
// All trace functions are no-ops

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

inline void custom_ntt_cpu_trace_start(
  const char* identifier,
  uint32_t logn,
  const uint32_t* input,
  bool is_inverse,
  uint32_t limbs) {}

inline void custom_ntt_cpu_trace_stop() {}

inline bool custom_ntt_cpu_trace_should_capture(uint32_t logn, bool is_inverse) { return false; }

inline void custom_ntt_cpu_trace_stage_dump(
  uint32_t stage,
  const uint32_t* data,
  uint32_t values_per_stage,
  uint32_t limbs) {}

inline void custom_ntt_cpu_trace_set_params(uint32_t max_size, uint32_t subntt_log_size) {}

inline uint32_t custom_ntt_cpu_trace_last_max_size() { return 0; }

inline uint32_t custom_ntt_cpu_trace_last_subntt_log() { return 0; }

inline void custom_ntt_cpu_trace_capture_index_map(const uint32_t* data, uint32_t length) {}

inline uint32_t custom_ntt_cpu_trace_index_map_size() { return 0; }

inline bool custom_ntt_cpu_trace_copy_index_map(uint32_t* out, uint32_t length) { return false; }

inline void custom_ntt_cpu_trace_set_inverse_only(bool inverse_only) {}

#ifdef __cplusplus
}
#endif
