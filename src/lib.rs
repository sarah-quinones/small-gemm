#![feature(stdsimd)]
#![feature(avx512_target_feature)]

use core::marker::PhantomData;
use core::ops::{Index, IndexMut};
use reborrow::*;

pub struct MatMut<'a, T> {
    data: *mut T,
    nrows: usize,
    ncols: usize,
    row_stride: isize,
    col_stride: isize,
    __marker: PhantomData<&'a mut T>,
}

#[derive(Copy, Clone)]
pub struct MatRef<'a, T> {
    data: *const T,
    nrows: usize,
    ncols: usize,
    row_stride: isize,
    col_stride: isize,
    __marker: PhantomData<&'a T>,
}

impl<'short, T> ReborrowMut<'short> for MatMut<'_, T> {
    type Target = MatMut<'short, T>;

    fn rb_mut(&'short mut self) -> Self::Target {
        unsafe {
            mat::from_raw_parts_mut(
                self.data,
                self.nrows,
                self.ncols,
                self.row_stride,
                self.col_stride,
            )
        }
    }
}
impl<'short, T> Reborrow<'short> for MatMut<'_, T> {
    type Target = MatRef<'short, T>;

    fn rb(&'short self) -> Self::Target {
        unsafe {
            mat::from_raw_parts(
                self.data,
                self.nrows,
                self.ncols,
                self.row_stride,
                self.col_stride,
            )
        }
    }
}

impl<'short, T> ReborrowMut<'short> for MatRef<'_, T> {
    type Target = MatRef<'short, T>;

    fn rb_mut(&'short mut self) -> Self::Target {
        unsafe {
            mat::from_raw_parts(
                self.data,
                self.nrows,
                self.ncols,
                self.row_stride,
                self.col_stride,
            )
        }
    }
}
impl<'short, T> Reborrow<'short> for MatRef<'_, T> {
    type Target = MatRef<'short, T>;

    fn rb(&'short self) -> Self::Target {
        unsafe {
            mat::from_raw_parts(
                self.data,
                self.nrows,
                self.ncols,
                self.row_stride,
                self.col_stride,
            )
        }
    }
}

pub mod mat {
    use super::MatMut;
    use super::MatRef;

    #[inline]
    pub unsafe fn from_raw_parts<'a, T>(
        data: *const T,
        nrows: usize,
        ncols: usize,
        row_stride: isize,
        col_stride: isize,
    ) -> MatRef<'a, T> {
        MatRef {
            data,
            nrows,
            ncols,
            row_stride,
            col_stride,
            __marker: std::marker::PhantomData,
        }
    }

    #[inline]
    pub unsafe fn from_raw_parts_mut<'a, T>(
        data: *mut T,
        nrows: usize,
        ncols: usize,
        row_stride: isize,
        col_stride: isize,
    ) -> MatMut<'a, T> {
        MatMut {
            data,
            nrows,
            ncols,
            row_stride,
            col_stride,
            __marker: std::marker::PhantomData,
        }
    }
}

impl<'a, T> MatRef<'a, T> {
    #[inline]
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    #[inline]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    #[inline]
    pub fn row_stride(&self) -> isize {
        self.row_stride
    }

    #[inline]
    pub fn col_stride(&self) -> isize {
        self.col_stride
    }
}

impl<'a, T> MatMut<'a, T> {
    #[inline]
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    #[inline]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    #[inline]
    pub fn row_stride(&self) -> isize {
        self.row_stride
    }

    #[inline]
    pub fn col_stride(&self) -> isize {
        self.col_stride
    }
}

impl<T> Index<(usize, usize)> for MatRef<'_, T> {
    type Output = T;

    #[inline]
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        assert!(row < self.nrows());
        assert!(col < self.ncols());
        unsafe {
            &*self
                .data
                .offset(row as isize * self.row_stride + col as isize * self.col_stride)
        }
    }
}

impl<T> Index<(usize, usize)> for MatMut<'_, T> {
    type Output = T;

    #[inline]
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        assert!(row < self.nrows());
        assert!(col < self.ncols());
        unsafe {
            &*self
                .data
                .offset(row as isize * self.row_stride + col as isize * self.col_stride)
        }
    }
}

impl<T> IndexMut<(usize, usize)> for MatMut<'_, T> {
    #[inline]
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        assert!(row < self.nrows());
        assert!(col < self.ncols());
        unsafe {
            &mut *self
                .data
                .offset(row as isize * self.row_stride + col as isize * self.col_stride)
        }
    }
}

include!(concat!(env!("OUT_DIR"), "/codegen.rs"));

type MicroKernel<T> = unsafe fn(
    dst_ptr: *mut T,
    dst_col_stride: isize,
    lhs_ptr: *const T,
    lhs_col_stride: isize,
    rhs_ptr: *const T,
    rhs_col_stride: isize,
    k: usize,
    *const (),
);

static AVX2_MASK: [i64; 16] = [-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0];

#[allow(non_upper_case_globals)]
static AVX512_MASK_f64: [u8; 9] = [
    0b11111111, 0b01111111, 0b00111111, 0b00011111, 0b00001111, 0b00000111, 0b00000011, 0b00000001,
    0b00000000,
];
#[allow(non_upper_case_globals)]
static AVX512_MASK_f32: [u16; 17] = [
    0b1111111111111111,
    0b0111111111111111,
    0b0011111111111111,
    0b0001111111111111,
    0b0000111111111111,
    0b0000011111111111,
    0b0000001111111111,
    0b0000000111111111,
    0b0000000011111111,
    0b0000000001111111,
    0b0000000000111111,
    0b0000000000011111,
    0b0000000000001111,
    0b0000000000000111,
    0b0000000000000011,
    0b0000000000000001,
    0b0000000000000000,
];

#[repr(C)]
struct Kernel<T> {
    x1x1: MicroKernel<T>,
    x2x1: MicroKernel<T>,
    x1x2: MicroKernel<T>,
    x2x2: MicroKernel<T>,
    x1x3: MicroKernel<T>,
    x2x3: MicroKernel<T>,
    x1x4: MicroKernel<T>,
    x2x4: MicroKernel<T>,
}

#[repr(transparent)]
pub struct Ptr<T>(*const T);
unsafe impl<T> Sync for Ptr<T> {}
unsafe impl<T> Send for Ptr<T> {}

unsafe fn matmul_with_kernel_impl<T>(
    dst: MatMut<'_, T>,
    lhs: MatRef<'_, T>,
    rhs: MatRef<'_, T>,
    mask_ptr: *const (),
    mask_offset: usize,
    reg_size: usize,
    reg_size_shift: u32,
    kernel: &Kernel<T>,
) {
    debug_assert_eq!(dst.row_stride, 1);
    debug_assert_eq!(lhs.row_stride, 1);
    debug_assert_eq!(rhs.row_stride, 1);

    let m = dst.nrows();
    let n = dst.ncols();
    let k = lhs.ncols();

    if m == 0 || n == 0 {
        return;
    }

    let mod_pow2 = |a: usize, b: usize| a & (b - 1);
    let div_pow2 = |a: usize, b_shift: u32| a >> b_shift;

    let m_rem = mod_pow2(m - 1, 2 * reg_size) + 1;
    let m_loop = div_pow2(m - m_rem, 1 + reg_size_shift);

    let n_rem = (n - 1) % 4 + 1;
    let n_loop = (n - n_rem) / 4;

    let dst_ptr = dst.data;
    let lhs_ptr = lhs.data;
    let rhs_ptr = rhs.data;

    let dst_col_stride = dst.col_stride as isize;
    let lhs_col_stride = lhs.col_stride as isize;
    let rhs_col_stride = rhs.col_stride as isize;

    for i in 0..m_loop {
        for j in 0..n_loop {
            let row = (i << (1 + reg_size_shift)) as isize;
            let col = (j * 4) as isize;
            (kernel.x2x4)(
                dst_ptr.offset(row + col * dst_col_stride),
                dst_col_stride,
                lhs_ptr.offset(row),
                lhs_col_stride,
                rhs_ptr.offset(col * rhs_col_stride),
                rhs_col_stride,
                k,
                AVX2_MASK.as_ptr() as *const (),
            );
        }
    }

    let kernel_bot_tail = match div_pow2(m_rem - 1, reg_size_shift) {
        0 => kernel.x1x4,
        1 => kernel.x2x4,
        _ => unreachable!(),
    };
    let kernel_right_tail = match n_rem - 1 {
        0 => kernel.x2x1,
        1 => kernel.x2x2,
        2 => kernel.x2x3,
        3 => kernel.x2x4,
        _ => unreachable!(),
    };
    let kernel_bot_right_tail = match (div_pow2(m_rem - 1, reg_size_shift), n_rem - 1) {
        (0, 0) => kernel.x1x1,
        (0, 1) => kernel.x1x2,
        (0, 2) => kernel.x1x3,
        (0, 3) => kernel.x1x4,
        (1, 0) => kernel.x2x1,
        (1, 1) => kernel.x2x2,
        (1, 2) => kernel.x2x3,
        (1, 3) => kernel.x2x4,
        _ => unreachable!(),
    };
    let bot_mask =
        (mask_ptr as *const u8).sub(mask_offset * (mod_pow2(m - 1, reg_size) + 1)) as *const ();

    for i in 0..m_loop {
        let row = (i << (1 + reg_size_shift)) as isize;
        let col = (n_loop * 4) as isize;
        kernel_right_tail(
            dst_ptr.offset(row + col * dst_col_stride),
            dst_col_stride,
            lhs_ptr.offset(row),
            lhs_col_stride,
            rhs_ptr.offset(col * rhs_col_stride),
            rhs_col_stride,
            k,
            AVX2_MASK.as_ptr() as *const (),
        );
    }
    for j in 0..n_loop {
        let row = (m_loop << (1 + reg_size_shift)) as isize;
        let col = (j * 4) as isize;
        kernel_bot_tail(
            dst_ptr.offset(row + col * dst_col_stride),
            dst_col_stride,
            lhs_ptr.offset(row),
            lhs_col_stride,
            rhs_ptr.offset(col * rhs_col_stride),
            rhs_col_stride,
            k,
            bot_mask,
        );
    }
    let row = (m_loop << (1 + reg_size_shift)) as isize;
    let col = (n_loop * 4) as isize;
    kernel_bot_right_tail(
        dst_ptr.offset(row + col * dst_col_stride),
        dst_col_stride,
        lhs_ptr.offset(row),
        lhs_col_stride,
        rhs_ptr.offset(col * rhs_col_stride),
        rhs_col_stride,
        k,
        bot_mask,
    );
}

#[derive(Copy, Clone)]
pub struct Plan<T: 'static> {
    mask_ptr: *const (),
    mask_offset: usize,
    reg_size: usize,
    reg_size_shift: u32,
    kernel: Option<&'static Kernel<T>>,
}

unsafe impl<T> Send for Plan<T> {}
unsafe impl<T> Sync for Plan<T> {}

impl Plan<f64> {
    pub fn new_f64(nrows: usize, ncols: usize, depth: usize) -> Self {
        let _ = (nrows, ncols);

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if pulp::x86::V4::is_available() {
            return Self {
                mask_ptr: avx512::MASK_PTR_f64.0 as _,
                mask_offset: avx512::MASK_OFFSET_f64,
                reg_size: avx512::REG_SIZE_f64,
                reg_size_shift: avx512::REG_SIZE_SHIFT_f64,
                kernel: Some(&avx512::KERNELS_f64[depth.min(avx512::KERNELS_f64.len() - 1)]),
            };
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if pulp::x86::V3::is_available() {
            return Self {
                mask_ptr: avx::MASK_PTR_f64.0 as _,
                mask_offset: avx::MASK_OFFSET_f64,
                reg_size: avx::REG_SIZE_f64,
                reg_size_shift: avx::REG_SIZE_SHIFT_f64,
                kernel: Some(&avx::KERNELS_f64[depth.min(avx::KERNELS_f64.len() - 1)]),
            };
        }

        Self {
            mask_ptr: core::ptr::null(),
            mask_offset: 0,
            reg_size: 0,
            reg_size_shift: 0,
            kernel: None,
        }
    }
}

impl Plan<f32> {
    pub fn new_f32(nrows: usize, ncols: usize, depth: usize) -> Self {
        let _ = (nrows, ncols);

        if pulp::x86::V4::is_available() {
            return Self {
                mask_ptr: avx512::MASK_PTR_f32.0 as _,
                mask_offset: avx512::MASK_OFFSET_f32,
                reg_size: avx512::REG_SIZE_f32,
                reg_size_shift: avx512::REG_SIZE_SHIFT_f32,
                kernel: Some(&avx512::KERNELS_f32[depth.min(avx512::KERNELS_f32.len() - 1)]),
            };
        }

        if pulp::x86::V3::is_available() {
            return Self {
                mask_ptr: avx::MASK_PTR_f32.0 as _,
                mask_offset: avx::MASK_OFFSET_f32,
                reg_size: avx::REG_SIZE_f32,
                reg_size_shift: avx::REG_SIZE_SHIFT_f32,
                kernel: Some(&avx::KERNELS_f32[depth.min(avx::KERNELS_f32.len() - 1)]),
            };
        }

        Self {
            mask_ptr: core::ptr::null(),
            mask_offset: 0,
            reg_size: 0,
            reg_size_shift: 0,
            kernel: None,
        }
    }
}

fn matmul_naive_f64(dst: MatMut<'_, f64>, lhs: MatRef<'_, f64>, rhs: MatRef<'_, f64>) {
    let m = dst.nrows();
    let n = dst.ncols();
    let k = lhs.ncols();
    let mut dst = dst;

    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0;
            for depth in 0..k {
                acc = f64::mul_add(lhs[(i, depth)], rhs[(depth, j)], acc);
            }
            dst[(i, j)] = acc;
        }
    }
}

fn matmul_naive_f32(dst: MatMut<'_, f32>, lhs: MatRef<'_, f32>, rhs: MatRef<'_, f32>) {
    let m = dst.nrows();
    let n = dst.ncols();
    let k = lhs.ncols();
    let mut dst = dst;

    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0;
            for depth in 0..k {
                acc = f32::mul_add(lhs[(i, depth)], rhs[(depth, j)], acc);
            }
            dst[(i, j)] = acc;
        }
    }
}

#[inline]
pub unsafe fn matmul_with_plan<T: 'static>(
    dst: MatMut<'_, T>,
    lhs: MatRef<'_, T>,
    rhs: MatRef<'_, T>,
    plan: &Plan<T>,
) {
    if let Some(kernel) = plan.kernel {
        matmul_with_kernel_impl(
            dst,
            lhs,
            rhs,
            plan.mask_ptr,
            plan.mask_offset,
            plan.reg_size,
            plan.reg_size_shift,
            kernel,
        )
    } else {
        use core::any::TypeId;

        if TypeId::of::<f64>() == TypeId::of::<T>() {
            matmul_naive_f64(
                core::mem::transmute(dst),
                core::mem::transmute(lhs),
                core::mem::transmute(rhs),
            );
        } else if TypeId::of::<f32>() == TypeId::of::<T>() {
            matmul_naive_f32(
                core::mem::transmute(dst),
                core::mem::transmute(lhs),
                core::mem::transmute(rhs),
            );
        } else {
            unreachable!();
        }
    }
}

pub fn matmul<T: 'static>(dst: MatMut<'_, T>, lhs: MatRef<'_, T>, rhs: MatRef<'_, T>) {
    assert_eq!(dst.nrows(), lhs.nrows());
    assert_eq!(dst.ncols(), rhs.ncols());
    assert_eq!(lhs.ncols(), rhs.nrows());

    let m = dst.nrows();
    let n = dst.ncols();
    let k = lhs.ncols();

    use core::any::TypeId;

    if TypeId::of::<f64>() == TypeId::of::<T>() {
        unsafe { matmul_with_plan(dst, lhs, rhs, core::mem::transmute(&Plan::new_f64(m, n, k))) }
    } else if TypeId::of::<f32>() == TypeId::of::<T>() {
        unsafe { matmul_with_plan(dst, lhs, rhs, core::mem::transmute(&Plan::new_f32(m, n, k))) }
    } else {
        unimplemented!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub struct Mat {
        pub data: Vec<f64>,
        pub nrows: usize,
        pub ncols: usize,
    }

    impl Mat {
        #[inline]
        pub fn new(data: Vec<f64>, nrows: usize, ncols: usize) -> Self {
            assert_eq!(data.len(), nrows * ncols);
            Self { data, nrows, ncols }
        }

        pub fn as_ref(&self) -> MatRef<'_, f64> {
            unsafe {
                mat::from_raw_parts(
                    self.data.as_ptr(),
                    self.nrows,
                    self.ncols,
                    1,
                    self.nrows as isize,
                )
            }
        }

        pub fn as_mut(&mut self) -> MatMut<'_, f64> {
            unsafe {
                mat::from_raw_parts_mut(
                    self.data.as_mut_ptr(),
                    self.nrows,
                    self.ncols,
                    1,
                    self.nrows as isize,
                )
            }
        }
    }

    impl Index<(usize, usize)> for Mat {
        type Output = f64;

        #[inline]
        fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
            &self.data[row + self.nrows * col]
        }
    }

    impl IndexMut<(usize, usize)> for Mat {
        #[inline]
        fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
            &mut self.data[row + self.nrows * col]
        }
    }

    #[test]
    fn test_matmul() {
        for m in 0..16 {
            for n in 0..16 {
                for k in 0..16 {
                    let mut dst_target = Mat::new(vec![0.0; m * n], m, n);
                    let mut dst = Mat::new(vec![0.0; m * n], m, n);
                    let mut lhs = Mat::new(vec![0.0; m * k], m, k);
                    let mut rhs = Mat::new(vec![0.0; k * n], k, n);

                    for x in &mut lhs.data {
                        *x = rand::random();
                    }
                    for x in &mut rhs.data {
                        *x = rand::random();
                    }

                    matmul_naive_f64(dst_target.as_mut(), lhs.as_ref(), rhs.as_ref());
                    matmul(dst.as_mut(), lhs.as_ref(), rhs.as_ref());

                    assert_eq!(dst_target.data, dst.data);
                }
            }
        }
    }
}
