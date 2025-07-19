//! Registers for the SP1 zkVM.

use std::ops::{Index, IndexMut};

use cranelift_module::Module;
use serde::{Deserialize, Serialize};

use crate::events::MemoryRecord;
use crate::jitwrapper::JITWrapper;

/// A register stores a 32-bit value used by operations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Register {
    /// %x0
    X0 = 0,
    /// %x1
    X1 = 1,
    /// %x2
    X2 = 2,
    /// %x3
    X3 = 3,
    /// %x4
    X4 = 4,
    /// %x5
    X5 = 5,
    /// %x6
    X6 = 6,
    /// %x7
    X7 = 7,
    /// %x8
    X8 = 8,
    /// %x9
    X9 = 9,
    /// %x10
    X10 = 10,
    /// %x11
    X11 = 11,
    /// %x12
    X12 = 12,
    /// %x13
    X13 = 13,
    /// %x14
    X14 = 14,
    /// %x15
    X15 = 15,
    /// %x16
    X16 = 16,
    /// %x17
    X17 = 17,
    /// %x18
    X18 = 18,
    /// %x19
    X19 = 19,
    /// %x20
    X20 = 20,
    /// %x21
    X21 = 21,
    /// %x22
    X22 = 22,
    /// %x23
    X23 = 23,
    /// %x24
    X24 = 24,
    /// %x25
    X25 = 25,
    /// %x26
    X26 = 26,
    /// %x27
    X27 = 27,
    /// %x28
    X28 = 28,
    /// %x29
    X29 = 29,
    /// %x30
    X30 = 30,
    /// %x31
    X31 = 31,
}

impl Register {
    /// Create a new register from a u32.
    ///
    /// # Panics
    ///
    /// This function will panic if the register is invalid.
    #[inline]
    #[must_use]
    pub fn from_u32(value: u32) -> Self {
        match value {
            0 => Register::X0,
            1 => Register::X1,
            2 => Register::X2,
            3 => Register::X3,
            4 => Register::X4,
            5 => Register::X5,
            6 => Register::X6,
            7 => Register::X7,
            8 => Register::X8,
            9 => Register::X9,
            10 => Register::X10,
            11 => Register::X11,
            12 => Register::X12,
            13 => Register::X13,
            14 => Register::X14,
            15 => Register::X15,
            16 => Register::X16,
            17 => Register::X17,
            18 => Register::X18,
            19 => Register::X19,
            20 => Register::X20,
            21 => Register::X21,
            22 => Register::X22,
            23 => Register::X23,
            24 => Register::X24,
            25 => Register::X25,
            26 => Register::X26,
            27 => Register::X27,
            28 => Register::X28,
            29 => Register::X29,
            30 => Register::X30,
            31 => Register::X31,
            _ => panic!("invalid register {value}"),
        }
    }

    /// This function supplies register to idx mapping.
    #[must_use]
    pub const fn number_of_registers() -> usize {
        Self::X31 as usize + 1
    }
}

#[repr(C)]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
/// Register file
pub struct RegisterFile {
    /// Array of registers
    pub registers: [u32; Register::number_of_registers()],
}

impl Index<Register> for RegisterFile {
    type Output = u32;

    fn index(&self, index: Register) -> &Self::Output {
        &self.registers[index as usize]
    }
}

impl IndexMut<Register> for RegisterFile {
    fn index_mut(&mut self, index: Register) -> &mut Self::Output {
        &mut self.registers[index as usize]
    }
}

/// Register load helper
// pub extern "C" fn reg_store(registers: &mut RegisterFile, reg: u32, val: u32) {
//     registers.registers[reg as usize] = val;
// }

// /// Register load helper
// pub extern "C" fn reg_load(registers: &mut RegisterFile, reg: u32) -> u32 {
//     registers.registers[reg as usize]
// }

/// Register load helper
pub extern "C" fn regs_printout(registers: &mut RegisterFile) {
    print!("Registers: ");
    for i in 0..Register::number_of_registers() {
        print!("{} ", registers.registers[i]);
    }
    println!("");
}

use cranelift_codegen::ir::{types, Value};
use cranelift_codegen::ir::{InstBuilder, MemFlags};
use cranelift_frontend::FunctionBuilder;
use cranelift_frontend::Variable;

/// Helper function to load two registers if needed and not dirty
pub fn load_two_regs(
    b: &mut FunctionBuilder<'_>,
    register_file_ptr: Value,
    regs: &[Variable; 32],
    regs_read_so_far: &mut [bool; 32],
    regs_dirty: &mut [bool; 32],
    rs1: u32,
    rs2: u32,
) -> (usize, usize) {
    let rs1_as_ind: usize = rs1.try_into().unwrap();
    let rs2_as_ind: usize = rs2.try_into().unwrap();
    let rs1 = load_reg_if_needed_and_not_dirty(
        b,
        register_file_ptr,
        rs1_as_ind,
        regs_read_so_far,
        regs_dirty,
        regs,
    );
    let rs2 = load_reg_if_needed_and_not_dirty(
        b,
        register_file_ptr,
        rs2_as_ind,
        regs_read_so_far,
        regs_dirty,
        regs,
    );
    (rs1, rs2)
}

/// Helper function to define a register and mark it as dirty
pub fn define_rd_and_mark_dirty(
    b: &mut FunctionBuilder<'_>,
    regs: &[Variable; 32],
    dirty_regs: &mut [bool; 32],
    rd: u32,
    r: Value,
) {
    // TODO impl Index for both arrays.
    let rd_as_idx: usize = rd.try_into().unwrap();
    b.def_var(regs[rd_as_idx], r);
    dirty_regs[rd_as_idx] = true;
    // println!("define_rd_and_mark_dirty def rd {}", rd);
}

/// Helper function to load a register if needed and not dirty
pub fn load_reg_if_needed_and_not_dirty(
    b: &mut FunctionBuilder<'_>,
    register_file_ptr: Value,
    reg: usize,
    regs_read_so_far: &mut [bool; 32],
    regs_dirty: &mut [bool; 32],
    regs: &[Variable],
) -> usize {
    // println!("load_reg_if_needed_and_not_dirty try to load reg {}", reg);
    if !regs_read_so_far[reg] && !regs_dirty[reg] {
        // println!("load_reg_if_needed_and_not_dirty loading reg {}", reg);
        load_register_from_cpu(b, register_file_ptr, reg, &regs[reg]);
        regs_read_so_far[reg] = true;
    }
    reg
}

/// Helper function to load a register from CPU
pub fn load_register_from_cpu(
    b: &mut FunctionBuilder<'_>,
    register_file_ptr: Value,
    reg: usize,
    reg_var: &Variable,
) {
    // TODO convention first comes regs array.
    // println!("loading reg {} to {}", reg, reg_var);
    let off = (reg * 4) as i64;
    let addr = b.ins().iadd_imm(register_file_ptr, off);
    let val = b.ins().load(types::I32, MemFlags::new(), addr, 0);
    b.def_var(*reg_var, val);
}

/// Helper function to store registers to CPU
pub fn store_registers_to_cpu(
    b: &mut FunctionBuilder<'_>,
    register_file_ptr: Value,
    regs: &[Variable],
    dirty_regs: &[bool],
) {
    // Only store registers that have been modified
    if dirty_regs[0] {
        let reg_val = b.ins().iconst(types::I32, 0);
        let off = 0;
        b.ins().store(MemFlags::new(), reg_val, register_file_ptr, 0);
        // println!("Stored reg {} value back to CPU at offset {}", 0, off);
    }
    for i in 1..32 {
        // Skip registers that haven't been modified
        if !dirty_regs[i] {
            continue;
        }
        let reg_val = b.use_var(regs[i]);
        // println!("R egister {}: retrieved value = {:?}", i, reg_val);
        let off = (i * 4) as i64;

        // Calculate pointer to CPU's regs[i]
        let addr = b.ins().iadd_imm(register_file_ptr, off);

        // Store the register value back to CPU memory
        b.ins().store(MemFlags::new(), reg_val, addr, 0);
        // println!("Stored reg {} value back to CPU at offset {}", i, off);
    }
}

pub fn call_regs_printout(
    jit_wrapper: &mut JITWrapper,
    b: &mut FunctionBuilder,
    register_file_ptr: Value,
) {
    let func_id = jit_wrapper.helpers.regs_printout;
    let func_ref = jit_wrapper.jit.declare_func_in_func(func_id, &mut b.func);
    b.ins().call(func_ref, &[register_file_ptr]);
}
