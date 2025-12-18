#include "labrador.h"           // For Zq, Rq, Tq, and the APIs
#include "icicle/hash/keccak.h" // For Hash
#include "examples_utils.h"
#include "icicle/runtime.h"

#include "types.h"
#include "utils.h"
#include "prover.h"
#include "verifier.h"
#include "shared.h"
#include "test_helpers.h"
#include "benchmarking.h"

#include <iostream>
#include <chrono>
#include <vector>
#include <tuple>
#include <iomanip>
#include <fstream>

using namespace icicle::labrador;

// 全局标志：是否使用自定义NTT
bool g_use_custom_ntt = false;

// 全局标志：是否使用自定义MatMul
bool g_use_custom_matmul = false;

// 全局标志：是否使用自定义VecOps
bool g_use_custom_vec_ops = false;

// 全局标志：是否使用自定义Misc Ops (decompose/recompose/jl_projection/matrix_transpose)
bool g_use_custom_misc_ops = false;

// 声明自定义NTT函数
extern "C" {
  void custom_ntt_init_cuda();
  void custom_ntt_init_cuda_with_roots(const uint32_t* psi_limbs, const uint32_t* omega_limbs);
  void custom_ntt_release_cuda();
}

void prover_verifier_trace()
{
  const int64_t q = get_q<Zq>();

  // randomize the witness Si with low norm
  const size_t n = 1 << 9;
  const size_t r = 1 << 5;
  constexpr size_t d = Rq::d;
  const size_t max_value = 2;
  size_t num_eq_const = 10;
  size_t num_cz_const = 10;

  const std::vector<Rq> S = rand_poly_vec(r * n, max_value);
  auto eq_inst = create_rand_eq_inst(n, r, S, num_eq_const);
  std::cout << "Created Eq constraints\n";
  auto const_zero_inst = create_rand_const_zero_inst(n, r, S, num_cz_const);
  std::cout << "Created Cz constraints\n";

  // Use current time (milliseconds since epoch) as a unique Ajtai seed
  auto now = std::chrono::system_clock::now();
  auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
  std::string ajtai_seed_str = std::to_string(millis);
  std::cout << "Ajtai seed = " << ajtai_seed_str << std::endl;

  double beta = sqrt(max_value * n * r * d);
  uint32_t base0 = calc_base0(r, OP_NORM_BOUND, beta);
  LabradorParam param{
    r,
    n,
    {reinterpret_cast<const std::byte*>(ajtai_seed_str.data()),
     reinterpret_cast<const std::byte*>(ajtai_seed_str.data()) + ajtai_seed_str.size()},
    secure_msis_rank(), // kappa
    secure_msis_rank(), // kappa1
    secure_msis_rank(), // kappa2,
    base0,              // base1
    base0,              // base2
    base0,              // base3
    beta,               // beta
  };
  LabradorInstance lab_inst{param};
  lab_inst.add_equality_constraint(eq_inst);
  lab_inst.add_const_zero_constraint(const_zero_inst);

  std::string oracle_seed = "ORACLE_SEED";

  size_t NUM_REC = 3;
  LabradorProver prover{
    lab_inst, S, reinterpret_cast<const std::byte*>(oracle_seed.data()), oracle_seed.size(), NUM_REC};

  std::cout << "Problem param: n,r = " << n << ", " << r << "\n";
  std::cout << "CONSISTENCY_CHECKS = " << (CONSISTENCY_CHECKS ? "true" : "false") << "\n";
  if (CONSISTENCY_CHECKS) { std::cout << "CONSISTENCY_CHECKS TRUE IMPLIES TIMING ESTIMATES ARE INCORRECT\n"; }
  auto [trs, final_proof] = prover.prove();

  // extract all prover_msg from trs vector into a vector prover_msgs
  std::vector<BaseProverMessages> prover_msgs;
  for (const auto& transcript : trs) {
    prover_msgs.push_back(transcript.prover_msg);
  }
  LabradorVerifier verifier{lab_inst,           prover_msgs,
                            final_proof,        reinterpret_cast<const std::byte*>(oracle_seed.data()),
                            oracle_seed.size(), NUM_REC};

  std::cout << "Verification result: \n";
  if (verifier.verify()) {
    std::cout << "Verification passed. \n";
  } else {
    std::cout << "Verification failed. \n";
  }
}

// === Main driver ===

int main(int argc, char* argv[])
{
  ICICLE_LOG_INFO << "Labrador example";
  
  // 1. 加载默认后端
  try_load_and_set_backend_device(argc, argv);

  // 2. 如果使用CUDA，初始化自定义NTT/MatMul/VecOps/MiscOps
  if (argc > 1 && std::string(argv[1]) == "CUDA") {
    // 检查是否启用自定义实现
    for (int i = 2; i < argc; ++i) {
      if (std::string(argv[i]) == "--custom-ntt") {
        g_use_custom_ntt = true;
      }
      if (std::string(argv[i]) == "--custom-matmul") {
        g_use_custom_matmul = true;
      }
      if (std::string(argv[i]) == "--custom-vec-ops") {
        g_use_custom_vec_ops = true;
      }
      if (std::string(argv[i]) == "--custom-misc-ops") {
        g_use_custom_misc_ops = true;
      }
    }
    
    if (g_use_custom_ntt || g_use_custom_matmul || g_use_custom_vec_ops || g_use_custom_misc_ops) {
      if (g_use_custom_ntt) {
        std::cout << "CUSTOM CUDA NTT ENABLED!" << std::endl;
        std::cout << "Using Negacyclic NTT with psi from ICICLE domain\n" << std::endl;
      }
      if (g_use_custom_matmul) {
        std::cout << "CUSTOM CUDA MATMUL ENABLED!" << std::endl;
        std::cout << "Using custom matrix multiplication kernels\n" << std::endl;
      }
      if (g_use_custom_vec_ops) {
        std::cout << "CUSTOM CUDA VEC OPS ENABLED!" << std::endl;
        std::cout << "Using custom vector operations kernels\n" << std::endl;
      }
      if (g_use_custom_misc_ops) {
        std::cout << "CUSTOM CUDA MISC OPS ENABLED!" << std::endl;
        std::cout << "Using custom decompose/recompose/jl_projection/matrix_transpose kernels\n" << std::endl;
      }
    }
    
    if (g_use_custom_ntt) {
      
      // 首先初始化 ICICLE 的 NTT domain
      // 这会自动计算并缓存 twiddle 因子
      using namespace icicle;
      using namespace icicle::labrador;
      
      // 获取 128次单位根作为 primitive root
      Zq psi = Zq::omega(7);  // omega(7) = 2^7 = 128 次单位根
      
      std::cout << "    Initializing ICICLE NTT domain with psi..." << std::endl;
      ICICLE_CHECK(ntt_init_domain(psi, NTTInitDomainConfig{}));
      
          std::cout << "    Got psi (128th root): " << psi << std::endl;

          // 获取 64次单位根作为 NTT 的基础 omega
          Zq omega;
          eIcicleError rou_err = get_root_of_unity_from_domain(6, &omega);  // log2(64) = 6          
          if (rou_err == eIcicleError::SUCCESS) {
            std::cout << "    Got omega (64th root): " << omega << std::endl;
          } else {
            std::cerr << "    Warning: Could not get 64th root from domain, computing psi^2" << std::endl;
            omega = psi * psi;  // psi^2 是 64次单位根
            std::cout << "    Computed omega (64th root): " << omega << std::endl;
          }
          
          // 传递 psi (用于 coset) 和 omega (用于 NTT twiddles) 给自定义 NTT 初始化
          custom_ntt_init_cuda_with_roots(psi.limbs_storage.limbs, omega.limbs_storage.limbs);
      
    } else {
      std::cout << "\nUsing default ICICLE CUDA backend\n" << std::endl;
    }
  }

  // I. Use the following code for examining program trace:
  // prover_verifier_trace();

  // II. To run benchmark uncomment:

  std::vector<std::tuple<size_t, size_t>> arr_nr{{1 << 6, 1 << 3}};
  std::vector<std::tuple<size_t, size_t>> num_constraint{{10, 10}};
  size_t NUM_REP = 1;
  bool SKIP_VERIF = false;
  benchmark_program(arr_nr, num_constraint, NUM_REP, SKIP_VERIF);

  return 0;
}