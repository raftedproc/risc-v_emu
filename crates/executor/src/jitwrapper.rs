use std::ops::{Index, IndexMut};

use cranelift_codegen::ir::{types, AbiParam};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module};

use crate::{memory::{mem_load32, mem_store32, printout_value}, regs_printout};

#[derive(Debug, Copy, Clone)]
pub struct TBCacheEntry {
    pub tb: *const u8,
    pub pc: u32,
    pub unconstrained: bool,
}

impl Default for TBCacheEntry {
    fn default() -> Self {
        Self {
            tb: std::ptr::null(),
            pc: 0,
            unconstrained: false,
        }
    }
}

/// JITModule external imports
pub struct JITHelpers {
    pub mem_load32: FuncId,
    pub mem_store32: FuncId,
    pub regs_printout: FuncId,
    pub printout_value: FuncId,
}

impl JITHelpers {
    pub fn new(jit: &mut JITModule) -> Self {
        let mut mem_load32_sig = jit.make_signature();
        mem_load32_sig.params.push(AbiParam::new(types::I64));
        mem_load32_sig.params.push(AbiParam::new(types::I32));
        mem_load32_sig.returns.push(AbiParam::new(types::I32));

        let mut mem_store32_sig = jit.make_signature();
        mem_store32_sig.params.push(AbiParam::new(types::I64));
        mem_store32_sig.params.push(AbiParam::new(types::I32));
        mem_store32_sig.params.push(AbiParam::new(types::I32));

        let mut printout_value_sig = jit.make_signature();
        printout_value_sig.params.push(AbiParam::new(types::I32));
        printout_value_sig.params.push(AbiParam::new(types::I32));

        let mut regs_printout_sig = jit.make_signature();
        regs_printout_sig.params.push(AbiParam::new(types::I64));

        Self {
            mem_load32: jit
                .declare_function("mem_load32", Linkage::Import, &mut mem_load32_sig)
                .expect("Failed to declare mem_load32 function"),
            mem_store32: jit
                .declare_function("mem_store32", Linkage::Import, &mut mem_store32_sig)
                .expect("Failed to declare mem_store32 function"),
            regs_printout: jit
                .declare_function("regs_printout", Linkage::Import, &mut regs_printout_sig)
                .expect("Failed to declare regs_printout function"),
            printout_value: jit
                .declare_function("printout_value", Linkage::Import, &mut printout_value_sig)
                .expect("Failed to declare printout_value function"),
        }
    }
}

/// State JITWrapper to store module
pub struct JITWrapper {
    /// Common JITModule
    pub jit: JITModule,

    pub helpers: JITHelpers,
}

pub struct TBCache<const S: usize>([Option<TBCacheEntry>; S]);

impl<const S: usize> Default for TBCache<S> {
    fn default() -> Self {
        Self(core::array::from_fn(|_| None))
    }
}

impl<const I: usize> Index<usize> for TBCache<I> {
    type Output = Option<TBCacheEntry>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<const S: usize> IndexMut<usize> for TBCache<S> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

pub const FAST_CACHE_MASK: u32 = 0x3FF;
pub const SLOW_CACHE_MASK: u32 = 0x10000;

pub type FastTBCache = TBCache<1024>;
pub type SlowTBCache = TBCache<16384>;

impl Default for JITWrapper {
    fn default() -> Self {
        let mut builder = JITBuilder::new(cranelift_module::default_libcall_names())
            .expect("failed to create JITBuilder");
        // Create the JITModule first

        builder.symbol("mem_load32", mem_load32 as *const u8);
        builder.symbol("mem_store32", mem_store32 as *const u8);
        builder.symbol("regs_printout", regs_printout as *const u8);
        builder.symbol("printout_value", printout_value as *const u8);

        let mut jit = JITModule::new(builder);

        let helpers = JITHelpers::new(&mut jit);

        Self { jit, helpers }
    }
}

impl Clone for JITWrapper {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl std::fmt::Debug for JITWrapper {
    fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

pub fn dummy_jit_module() -> JITModule {
    let builder = JITBuilder::new(cranelift_module::default_libcall_names()).unwrap();
    JITModule::new(builder)
}
