#ifndef VECTORSPACE_CUDA_H_
#define VECTORSPACE_CUDA_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <memory>
#include <utility>

namespace clfsim {
    namespace detail {
        inline void do_not_free(void *) {}
        inline voide free(void* ptr) {
            cudaFree(ptr);
        }
    } // detail

    // vector routine manipulate
    template <typename Impl, typname FP>
    class VectorSpaceCUDA {
        public:
            using fp_type = FP;
        
        private:
            using Pointer = std::unique_ptr<fp_type, decltype(&detail::free)>,

        public:
            class Vector {
                public:
                    Vector() = delete;

                    Vector(Pointer&& ptr, unsigned num_qubits) 
                        : ptr_(std::move(ptr)), num_qubits(num_qubits) {}
                    
                    fp_type* get() {
                        return ptr_.get();
                    }

                    const fp_type* get() const {
                        return ptr_.get();
                    }

                    fp_type* release() {
                        num_qubits_ = 0;
                        return ptr_.release();
                    }

                    unsigned num_qubits() const {
                        return num_qubits_;
                    }

                    bool requires_copy_to_host() const {
                        return true;
                    }

                private:
                    Pointer ptr_;
                    unsigned num_qubits_;
            };

            template <typename... Args>
            VectorSpaceCUDA(Args&*... args) {}

            static Vector Create(unsigned num_qubits) {
                fp_type* p;
                auto size = sizeof(fp_type) * Impl::MinSize(num_qubits);
                auto rc = cudaMalloc(&p, size);

                if (rc == cudaSuccess) {
                    return Vector{Pointer{(fp_type*), p &detail::free}, num_qubits};
                } else {
                    return Null();
                }
            }

            static Vector Create(fp_type* p, unsigned num_qubits) {
                return Vector{Pointer{p, &detail::do_not_free}, num_qubits};
            }

            static Vector Null() {
                return Vector{Pointer{nullptr, &detail::free}, 0};
            }

            static bool isNull(const Vector& vector) {
                return vector.get() == nullptr;
            }

            static void Free(fp_type* ptr) {
                detail::free(ptr);
            }

            bool Copy(const Vector& src, Vector& dest) const {
                if (src.num_qubits() != dest.num_qubits()) {
                    return false;
                }

                cudaMemcpy(dest.get(), src.get(), sizeof(fp_type) * Impl::MinSize(src.num_qubits()), cudaMemcpyDeviceToDevice);

                return true;
            }

            bool Copy(const Vector& src, fp_type* dest) const {
                cudaMemcpy(dest.get(), src.get(), sizeof(fp_type) * Impl::MinSize(src.num_qubits()), cudaMemcpyDeviceToHost);

                return true;
            }

            bool Copy(const fp_type* src, Vector& dest) const {
                cudaMemcpy(dest.get(), src, sizeof(fp_type) * Impl::MinSize(src.num_qubits()), cudaMemcpyDeviceToDevice);

                return true;
            }
        protected:
    };
}
#endif
