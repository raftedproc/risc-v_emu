use std::ops::{Index, IndexMut};

use cranelift_jit::{JITBuilder, JITModule};

use crate::memory::{mem_load32, mem_store32};

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

/// State JITWrapper to store module
pub struct JITWrapper {
  /// Common JITModule
  pub jit: JITModule,
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
      let mut builder = JITBuilder::new(cranelift_module::default_libcall_names()).expect("failed to create JITBuilder");
      // Create the JITModule first

      builder.symbol("mem_load32",  mem_load32 as *const u8);
      builder.symbol("mem_store32", mem_store32 as *const u8);
  
      let jit = JITModule::new(builder);
      Self {
          jit,
      }
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