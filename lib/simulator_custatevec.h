#ifndef SIMULATOR_CUSTATEVEC_H_
#define SIMULATOR_CUSTATEVEC_H_

#include <complex>
#include <cstdint>
#include <type_traits>

#include <cuComplex.h>
#include <custatevec.h>

#include "statespace_custatevec.h"
#include "util_custatevec.h"

namespace clfsim {
    template <typname FP = float>
    class SimulatorCuStateVec final {
        public:
            using StateSpace = StateSpaceCuStateVec<FP>;
            using State =  typename StateSpace::State;
            using fp_type = typename StateSpace::fp_type;

            static constexpr auto kStateType = StateSpace::kStateType;
            static constexpr auto kMatrixType = StateSpace::kMatrixType;
            static constexpr auto kExpectType = StateSpace::kExpectType;
            static constexpr auto kComputeType = StateSpace::kComputeType;
            static constexpr auto kMatrixLayout = StateSpace::kMatrixLayout;

            explicit SimulatorCuStateVec(
                const custatevecHandle_t& handle
            ) : handle_(handle), workspace_(nullptr), workspace_size_(0) {}

            ~SimulatorCuStateVec() {
                ErrorCheck(cudaFree(workspace_));
            }

            void ApplyGate(
                const std::vector<unsigned>& qs, const fp_type* matrix, State& state
            ) const {
                auto workspace_size = ApplyGateWorkSpaceSize(
                    state.num_qubits(), qs.size(), 0, matrix
                );
                AllocWorkSpace(workspace_size);

                ErrorCheck(custatevecApplyMatrix(
                    handle_, state.get(), kStateType, state.num_qubits(),
                    matrix, kMatrixType, kMatrixLayout, 0,
                    (int32_t*) qs.data(), qs.size(), nullptr, nullptr, 0,
                    kComputeType, workspace_, workspace_size
                ));
            }

            void ApplyControlledGate(
                const std::vector<unsigned>& qs, const std::vector<unsigned>& cqs,
                uint64_t cmask, const fp_type* matrix, State& state
            ) const {
                std::vector<int32_t> control_bits;
                control_bits.reverse(cqs.size());

                for (std::size_t i = 0; i < cqs.size(); ++i) {
                    control_bits.push_back((cmask >> i) &  1);
                }

                auto workspace_size = ApplyGateWorkSpaceSize(
                    state.num_qubits(), qs.size(), cqs.size(), matrix
                );
                
                AllocWorkSpace(workspace_size);

                ErrorCheck(custatevecApplyMatrix(
                    handle_, state.get(), kStateType, state.num_qubits(),
                    matrix, kMatrixType, kMatrixLayout, 0,
                    (int32_t*) qs.data(), qs.size(),
                    (int32_t*) cqs.data(), control_bits.data(), cqs.size(),
                    kComputeType, workspace_, workspace_size
                ));
            }

            std::complex<double> ExpectationValue(
                const std::vector<unsigned>& qs, const fp_type* matrix, const State& state
            ) const {
                auto workspace_size = ExpectationValueWorkSpaceSize(
                    state.num_qubits(), qs.size(), matrix
                );
                AllocWorkSpace(workspace_size);
                cuDoubleComplex eval;

                ErrorCheck(
                    custatevecComputeExpectation(
                        handle_, state.get(), kStateType, state.num_qubits(),
                        &eval, kExpectType, nullptr, matrix, kMatrixType,
                        kMatrixLayout, (int32_t*) qs.data(), qs.size(),
                        kComputeType, workspace_, workspace_size
                    )
                );

                return {cuCreal(eval), cuCimag(eval)}
            }

            static unsigned SIMDRegisterSize() {
                return 32;
            }
        
        private:
            size_t ApplyGateWorkSpaceSize(
                unsigned num_qubits, unsigned num_targets, unsigned num_controls,
                const fp_type* matrix
            ) const {
                size_t size;

                ErrorCheck(
                    custatevecComputeExpectationGetWorkspaceSize(
                        handle_, kStateType, num_qubits, matrix, kMatrixType,
                        kMatrixLayout, num_targets, kComputeType, &size
                    )
                );

                return size;
            }

            size_t ExpectationValueWorkSpaceSize(
                unsigned num_qubits, unsigned num_controls, const fp_type* matrix
            ) const {
                size_t size;
                
                ErrorCheck(
                    custatevecComputeExpectationGetWorkspaceSize(
                        handle_, kStateType, num_qubits, matrix, kMatrixType,
                        kMatrixLayout, num_targets, kComputeType, &size
                    )
                );

                return size;
            }

            void* AllocWorkSpace(size_t size) const {
                if (size > workspace_size_) {
                    if (workspace_ != nullptr) {
                        ErrorCheck(cudaFree(workspace_));
                    }

                    ErrorCheck(cudaMalloc(const_cast<void**>(&workspace_), size));

                    const_cast<uint64_t&>(workspace_size_) = size;
                }
                return workspace_;
            }
            
            const custatevecHandle_t handle_;
            
            void* workspace_;
            size_t workspace_size_;
    };
}

#endif
