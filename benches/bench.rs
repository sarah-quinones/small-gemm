use criterion::*;
use reborrow::*;

extern crate openblas_src;

fn matmul(criterion: &mut Criterion) {
    for m in 0..=32 {
        for n in 0..=32 {
            for k in 0..=32 {
                {
                    let mut dst = faer::Mat::from_fn(m, n, |_, _| 1.0);
                    let lhs = faer::Mat::from_fn(m, k, |_, _| 1.0);
                    let rhs = faer::Mat::from_fn(k, n, |_, _| 1.0);

                    let mut dst = unsafe {
                        small_gemm::mat::from_raw_parts_mut(
                            dst.as_ptr_mut(),
                            dst.nrows(),
                            dst.ncols(),
                            dst.row_stride(),
                            dst.col_stride(),
                        )
                    };
                    let lhs = unsafe {
                        small_gemm::mat::from_raw_parts(
                            lhs.as_ptr(),
                            lhs.nrows(),
                            lhs.ncols(),
                            lhs.row_stride(),
                            lhs.col_stride(),
                        )
                    };
                    let rhs = unsafe {
                        small_gemm::mat::from_raw_parts(
                            rhs.as_ptr(),
                            rhs.nrows(),
                            rhs.ncols(),
                            rhs.row_stride(),
                            rhs.col_stride(),
                        )
                    };

                    criterion.bench_function(&format!("NEW-{m}x{n}x{k}"), |bencher| {
                        bencher
                            .iter(|| small_gemm::matmul(dst.rb_mut(), lhs, rhs))
                    });
                }

                {
                    let mut dst = faer::Mat::from_fn(m, n, |_, _| 1.0);
                    let lhs = faer::Mat::from_fn(m, k, |_, _| 1.0);
                    let rhs = faer::Mat::from_fn(k, n, |_, _| 1.0);

                    criterion.bench_function(&format!("faer-{m}x{n}x{k}"), |bencher| {
                        bencher.iter(|| {
                            faer_core::mul::matmul(
                                dst.as_mut(),
                                lhs.as_ref(),
                                rhs.as_ref(),
                                None,
                                1.0,
                                faer_core::Parallelism::None,
                            );
                        })
                    });
                }

                {
                    let mut dst = nalgebra::DMatrix::from_fn(m, n, |_, _| 1.0);
                    let lhs = nalgebra::DMatrix::from_fn(m, k, |_, _| 1.0);
                    let rhs = nalgebra::DMatrix::from_fn(k, n, |_, _| 1.0);

                    criterion.bench_function(&format!("nalgebra-{m}x{n}x{k}"), |bencher| {
                        bencher.iter(|| {
                            lhs.mul_to(&rhs, &mut dst);
                        })
                    });
                }

                {
                    let mut dst = ndarray::Array2::from_shape_fn((m, n), |(_, _)| 1.0);
                    let lhs = ndarray::Array2::from_shape_fn((m, k), |(_, _)| 1.0);
                    let rhs = ndarray::Array2::from_shape_fn((k, n), |(_, _)| 1.0);

                    criterion.bench_function(&format!("ndarray-{m}x{n}x{k}"), |bencher| {
                        bencher.iter(|| {
                            dst = lhs.dot(&rhs);
                        })
                    });
                }
            }
        }
    }
}

criterion_group!(benches, matmul);
criterion_main!(benches);
