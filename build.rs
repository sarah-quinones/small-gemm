use std::{env, fmt::Display, path::PathBuf};

#[derive(Copy, Clone)]
enum K {
    Const(usize),
    Dyn,
}

impl Display for K {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            K::Const(k) => write!(f, "{k}"),
            K::Dyn => write!(f, "dyn"),
        }
    }
}

impl K {
    fn to_loop(self, body: &str, loop_count: &str, iter_ident: &str) -> String {
        let mut out = String::new();
        match self {
            K::Const(k) => {
                for iter in 0..k {
                    out += &format!("{{ let {iter_ident}: usize = {iter}; {body} }}");
                }
            }
            K::Dyn => {
                out += &format!("for {iter_ident} in 0..{loop_count} {{ {body} }}");
            }
        }
        out
    }
}

struct MicroKernel {
    m: usize,
    n: usize,
    k: K,
    reg_size: usize,
    features: String,
    ty: String,
    mask_ty: String,
    load: String,
    mask_load: String,
    store: String,
    mask_store: String,
    set1: String,
    fmadd: String,
}

impl Display for MicroKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self {
            m,
            n,
            k,
            features,
            mask_ty,
            ty,
            set1,
            fmadd,
            load,
            mask_load,
            store,
            mask_store,
            reg_size,
        } = self;

        let compute_code = k.to_loop(
            &{
                let mut loop_body = String::new();
                loop_body +=
                    "let lhs_ptr = lhs_ptr.wrapping_offset(__iter as isize * lhs_col_stride);";
                for i in 0..*m {
                    if i < m - 1 {
                        loop_body += &format!(
                            "let lhs_reg_{i} = {load}(lhs_ptr.wrapping_add({reg_size} * {i}));\n"
                        );
                    } else {
                        loop_body += &format!(
                    "let lhs_reg_{i} = {mask_load}(lhs_ptr.wrapping_add({reg_size} * {i}), mask);\n"
                );
                    }
                }

                for j in 0..*n {
                    loop_body += "{\n";
                    loop_body += &format!(
                        "let rhs = {set1}(*rhs_ptr.offset(__iter as isize + {j} * rhs_col_stride));"
                    );

                    for i in 0..*m {
                        loop_body +=
                            &format!("acc[{j}][{i}] = {fmadd}(lhs_reg_{i}, rhs, acc[{j}][{i}]);");
                    }
                    loop_body += "}\n";
                }
                loop_body
            },
            "k",
            "__iter",
        );

        let mut store_code = String::new();
        for j in 0..*n {
            for i in 0..*m {
                let store_statement = if i < m - 1 {
                    format!("{store}(dst_ptr, acc[{j}][{i}]);")
                } else {
                    format!("{mask_store}(dst_ptr, mask, acc[{j}][{i}]);")
                };
                store_code += &format!(
                    "\
    {{
        let dst_ptr = dst_ptr.offset({reg_size} * {i} + dst_col_stride * {j});
        {store_statement}
    }}
"
                );
            }
        }

        write!(
            f,
            "\
#[target_feature(enable = \"{features}\")]
#[allow(unused_variables)]
#[allow(unused_mut)]
pub unsafe fn ukr_{ty}_{m}_{n}_{k} (
    dst_ptr: *mut {ty},
    dst_col_stride: isize,
    lhs_ptr: *const {ty},
    lhs_col_stride: isize,
    rhs_ptr: *const {ty},
    rhs_col_stride: isize,
    k: usize,
    mask: *const (),
) {{
    let mask = (mask as *const {mask_ty}).read_unaligned();
    let mut acc = [[{set1}(0.0); {m}]; {n}];

    {compute_code}
    {store_code}
}}
"
        )
    }
}

fn main() {
    let mut code = String::new();

    let max_k = 8usize;
    let avx_max_m = 2usize;
    let avx_max_n = 4usize;

    code += "#[cfg(any(target_arch = \"x86_64\", target_arch = \"x86\"))]";
    {
        let arch_prefix = "core::arch::x86_64::";
        code += &format!("mod avx");
        code += "{\n";
        code += "use super::Kernel;\n";

        for mut kernel in [
            MicroKernel {
                m: 0,
                n: 0,
                k: K::Dyn,
                reg_size: 8,
                features: "avx,avx2,fma".to_string(),
                ty: "f32".to_string(),
                mask_ty: format!("{arch_prefix}__m256i"),
                load: format!("{arch_prefix}_mm256_loadu_ps"),
                mask_load: format!("{arch_prefix}_mm256_maskload_ps"),
                store: format!("{arch_prefix}_mm256_storeu_ps"),
                mask_store: format!("{arch_prefix}_mm256_maskstore_ps"),
                set1: format!("{arch_prefix}_mm256_set1_ps"),
                fmadd: format!("{arch_prefix}_mm256_fmadd_ps"),
            },
            MicroKernel {
                m: 0,
                n: 0,
                k: K::Dyn,
                reg_size: 4,
                features: "avx,avx2,fma".to_string(),
                ty: "f64".to_string(),
                mask_ty: format!("{arch_prefix}__m256i"),
                load: format!("{arch_prefix}_mm256_loadu_pd"),
                mask_load: format!("{arch_prefix}_mm256_maskload_pd"),
                store: format!("{arch_prefix}_mm256_storeu_pd"),
                mask_store: format!("{arch_prefix}_mm256_maskstore_pd"),
                set1: format!("{arch_prefix}_mm256_set1_pd"),
                fmadd: format!("{arch_prefix}_mm256_fmadd_pd"),
            },
        ] {
            for m in 1..=avx_max_m {
                kernel.m = m;
                for n in 1..=avx_max_n {
                    kernel.n = n;
                    for k in (0..max_k).map(K::Const).chain([K::Dyn]) {
                        kernel.k = k;
                        code += &format!("{kernel}");
                    }
                }
            }
        }

        for ty in ["f64", "f32"] {
            code += &format!(
                "#[allow(non_upper_case_globals)] pub static KERNELS_{ty}: [Kernel<{ty}>; {}] = [\n",
                max_k + 1
            );
            for k in (0..max_k).map(K::Const).chain([K::Dyn]) {
                code += &format!("Kernel::<{ty}> {{\n");
                for m in 1..=avx_max_m {
                    for n in 1..=avx_max_n {
                        code += &format!("x{m}x{n}: ukr_{ty}_{m}_{n}_{k},\n");
                    }
                }
                code += &format!("}},\n");
            }

            code += "];\n";

            code += &format!("#[allow(non_upper_case_globals)] pub static REG_SIZE_{ty}: usize = 32 / core::mem::size_of::<{ty}>();");
            code += &format!("#[allow(non_upper_case_globals)] pub static REG_SIZE_SHIFT_{ty}: u32 = REG_SIZE_{ty}.trailing_zeros();");
            code += &format!("#[allow(non_upper_case_globals)] pub static MASK_PTR_{ty}: super::Ptr<i64> = unsafe {{ super::Ptr(super::AVX2_MASK.as_ptr().add(8)) }};");
            code += &format!("#[allow(non_upper_case_globals)] pub static MASK_OFFSET_{ty}: usize = core::mem::size_of::<{ty}>();");
        }

        code += "}\n";
    }

    code += "#[cfg(any(target_arch = \"x86_64\", target_arch = \"x86\"))]";
    {
        let arch_prefix = "core::arch::x86_64::";
        code += &format!("mod avx512");
        code += "{\n";
        code += "use super::Kernel;\n";

        code += "
#[cfg(target_arch = \"x86_64\")]
use core::arch::x86_64::__m512;
#[cfg(target_arch = \"x86\")]
use core::arch::x86::__m512;

#[cfg(target_arch = \"x86_64\")]
use core::arch::x86_64::__m512d;
#[cfg(target_arch = \"x86\")]
use core::arch::x86::__m512d;

#[inline(always)]
unsafe fn maskload_ps(ptr: *const f32, mask: u16) -> __m512 {
    #[cfg(target_arch = \"x86_64\")]
    use core::arch::x86_64::_mm512_maskz_loadu_ps;

    #[cfg(target_arch = \"x86\")]
    use core::arch::x86::_mm512_maskz_loadu_ps;

    _mm512_maskz_loadu_ps(mask, ptr)
}

#[inline(always)]
unsafe fn maskload_pd(ptr: *const f64, mask: u8) -> __m512d {
    #[cfg(target_arch = \"x86_64\")]
    use core::arch::x86_64::_mm512_maskz_loadu_pd;

    #[cfg(target_arch = \"x86\")]
    use core::arch::x86::_mm512_maskz_loadu_pd;

    _mm512_maskz_loadu_pd(mask, ptr)
}
            ";

        for mut kernel in [
            MicroKernel {
                m: 0,
                n: 0,
                k: K::Dyn,
                reg_size: 16,
                features: "avx512f".to_string(),
                ty: "f32".to_string(),
                mask_ty: format!("u16"),
                load: format!("{arch_prefix}_mm512_loadu_ps"),
                mask_load: format!("maskload_ps"),
                store: format!("{arch_prefix}_mm512_storeu_ps"),
                mask_store: format!("{arch_prefix}_mm512_mask_storeu_ps"),
                set1: format!("{arch_prefix}_mm512_set1_ps"),
                fmadd: format!("{arch_prefix}_mm512_fmadd_ps"),
            },
            MicroKernel {
                m: 0,
                n: 0,
                k: K::Dyn,
                reg_size: 8,
                features: "avx512f".to_string(),
                ty: "f64".to_string(),
                mask_ty: format!("u8"),
                load: format!("{arch_prefix}_mm512_loadu_pd"),
                mask_load: format!("maskload_pd"),
                store: format!("{arch_prefix}_mm512_storeu_pd"),
                mask_store: format!("{arch_prefix}_mm512_mask_storeu_pd"),
                set1: format!("{arch_prefix}_mm512_set1_pd"),
                fmadd: format!("{arch_prefix}_mm512_fmadd_pd"),
            },
        ] {
            for m in 1..=avx_max_m {
                kernel.m = m;
                for n in 1..=avx_max_n {
                    kernel.n = n;
                    for k in (0..max_k).map(K::Const).chain([K::Dyn]) {
                        kernel.k = k;
                        code += &format!("{kernel}");
                    }
                }
            }
        }

        for ty in ["f64", "f32"] {
            code += &format!(
                "#[allow(non_upper_case_globals)] pub static KERNELS_{ty}: [Kernel<{ty}>; {}] = [\n",
                max_k + 1
            );
            for k in (0..max_k).map(K::Const).chain([K::Dyn]) {
                code += &format!("Kernel::<{ty}> {{\n");
                for m in 1..=avx_max_m {
                    for n in 1..=avx_max_n {
                        code += &format!("x{m}x{n}: ukr_{ty}_{m}_{n}_{k},\n");
                    }
                }
                code += &format!("}},\n");
            }

            code += "];\n";

            code += &format!("#[allow(non_upper_case_globals)] pub static REG_SIZE_{ty}: usize = 64 / core::mem::size_of::<{ty}>();");
            code += &format!("#[allow(non_upper_case_globals)] pub static REG_SIZE_SHIFT_{ty}: u32 = REG_SIZE_{ty}.trailing_zeros();");
            code += &format!("#[allow(non_upper_case_globals)] pub static MASK_PTR_{ty}: super::Ptr<i64> = unsafe {{ super::Ptr(super::AVX512_MASK_{ty}.as_ptr().add(REG_SIZE_{ty}) as _) }};");
            code += &format!("#[allow(non_upper_case_globals)] pub static MASK_OFFSET_{ty}: usize = REG_SIZE_{ty} / 8;");
        }

        code += "}\n";
    }

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    std::fs::write(&out_path.join("codegen.rs"), code).unwrap();
}
