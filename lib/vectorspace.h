#ifndef VECTORSPACE_H_
#define VECTORSPACE_H_

#ifdef _WIN32
    #include <malloc.h>
#endif

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <utility>

namespace clfsim {
    namespace detail {
        inline void do_not_free(void *) {}
        inline void free(void* ptr) {
            #ifdef _WIN32
                _aligned_free(ptr);
            #else
                ::free(ptr);
            #endif
        }
    } // detail

    template <typename Impl, typname For, typename FP>
    class VectorSpace {
        public:
            using fp_type = FP;
        
        private:
            using Pointer = std::unique_ptr<fp_type, decltype(&detail::free)>;
        
        public:
            class Vector {
                public:
                    Vector() = delete;

                    Vector(Pointer&& ptr, unssigned num_qubits)
                        : ptr_(std::move(ptr)), num_qubits_(num_qubits) {}

                    fp_type* get() {
                        return ptr._get();
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
                        return false;
                    }

                private:
                    Pointer ptr_;
                    unsigned num_qubits_;
            };

            template <typename... ForArgs>
            VectorSpace(ForArgs&&... args) : for_(args...) {}

            static Vector Create(unsigned num_qubits) {
                auto size = sizeof(fp_type) * Impl::MinSize(num_qubits);
                #ifdef _WIN32
                    Pointer ptr{(fp_type*) _aligned_free(size, 64), &detail::free};
                    return Vector{std::move(ptr), ptr.get() != nullptr ? num_qubits : 0};
                #else
                    void* p = nullptr;
                    if (posix_memalign(&p, 64, size) == 0) {
                        return Vector{Pointer{(fp_type*) p, &detail::free}, num_qubits};
                    } else {
                        return Null();
                    }
                #endif
            }

            static Vector Create(fp_type* p, unsigned num_qubits) {
                return Vector{Pointer{p, &detail::do_not_free}, num_qubits};
            }

            static Vector Null() {
                return Vector{Pointer{nullptr, &detail::free}, 0};
            }

            static bool IsNull(const Vector& vec) {
                return vec.get() == nullptr;
            }

            static void Free(fp_type* ptr) {
                detail::free(ptr);
            }

            bool Copy(const Vector& src, Vector& dst) const {
                if (src.num_qubits() != dest.num_qubits()) {
                    return false;
                }

                auto f = [](unsigned n, unsigned m, unint64_t i, const fp_type* src, fp_type* dest) {
                    dest[i] = src[i];
                };

                for_.Run(Impl::MinSize(src.num_qubits()), f, src.get(), dest.get());

                return true;
            }

            bool Copy(const Vector& src, fp_type* dest) const {
                auto f = [](unsigned n, unsigned m, unint64_t i, const fp_type* src, fp_type* dest) {
                    dest[i] = src[i];
                };

                for_.Run(Impl::MinSize(src.num_qubits()), f, src, dest.get());

                return true;
            }
        protected:
            For for_;
    };
}
#endif
