use std::ops::{Index, IndexMut};

use cranelift_codegen::ir::Value;
use cranelift_codegen::ir::{types, AbiParam, InstBuilder};
use cranelift_frontend::FunctionBuilder;
use cranelift_jit::JITModule;
use cranelift_module::{Linkage, Module};
use memmap2::MmapMut;

use crate::jitwrapper::JITWrapper;

/// Holds anon mmap memory allocation.
#[repr(C)]
#[derive(Debug)]
pub struct Memory {
    /// The anon mmap memory allocation.
    pub memory: MmapMut,
}

// 2 << 30 : 4GB / 4
impl Default for Memory {
    fn default() -> Self {
        Memory {
            memory: MmapMut::map_anon(1 << 30).unwrap(),
        }
    }
}

impl Clone for Memory {
    fn clone(&self) -> Self {
        Self {
            memory: MmapMut::map_anon(1 << 30).unwrap(),
        }
    }
}

impl IndexMut<u32> for Memory {
    fn index_mut(&mut self, index: u32) -> &mut Self::Output {
        let bytes = &mut self.memory[index as usize..index as usize + 4];
        // Safety: We know the slice is 4 bytes long and properly aligned since index is 4-byte aligned
        unsafe { &mut *(bytes.as_mut_ptr() as *mut u32) }
    }
}

impl Index<u32> for Memory {
    type Output = u32;

    fn index(&self, index: u32) -> &Self::Output {
        let bytes = &self.memory[index as usize..index as usize + 4];
        // Safety: We know the slice is 4 bytes long and properly aligned since index is 4-byte aligned
        unsafe { &*(bytes.as_ptr() as *const u32) }
    }
}

/// Memory load helper
pub extern "C" fn mem_load32(memory: &mut Memory, addr: u32) -> u32 {
    // println!("mem_load32 addr {} ", addr);
    memory[addr]
}

/// Memory store helper
pub extern "C" fn mem_store32(memory: &mut Memory, addr: u32, val: u32) {
    // println!("mem_store32 addr {} val {}", addr, val);
    memory[addr] = val;
}

/// Memory dbug printout
pub extern "C" fn printout_value(addr: u32, val: u32) {
    println!("LB addr {} val {}", addr, val);
}

/// helper-ы для доступа к памяти: вызываем обычные Rust-функции
pub fn call_mem_load_(
    jit_wrapper: &mut JITWrapper,
    b: &mut FunctionBuilder,
    memory_ptr: Value,
    addr: Value,
) -> Value {
    let func_id = jit_wrapper.helpers.mem_load32;
    let func_ref = jit_wrapper.jit.declare_func_in_func(func_id, &mut b.func);
    let call = b.ins().call(func_ref, &[memory_ptr, addr]);
    b.inst_results(call)[0]
}

pub fn call_mem_store_(
    jit_wrapper: &mut JITWrapper,
    b: &mut FunctionBuilder,
    memory_ptr: Value,
    addr: Value,
    val: Value,
) {
    let func_id = jit_wrapper.helpers.mem_store32;
    let func_ref = jit_wrapper.jit.declare_func_in_func(func_id, &mut b.func);
    b.ins().call(func_ref, &[memory_ptr, addr, val]);
}

pub fn call_printout_value(
    jit_wrapper: &mut JITWrapper,
    b: &mut FunctionBuilder,
    addr: Value,
    val: Value,
) {
    let func_id = jit_wrapper.helpers.printout_value;
    let func_ref = jit_wrapper.jit.declare_func_in_func(func_id, &mut b.func);
    b.ins().call(func_ref, &[addr, val]);
}