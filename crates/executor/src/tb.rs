use std::{ops::Shl, ptr::null};

use crate::{
    jitwrapper::{dummy_jit_module, TBCacheEntry, SLOW_CACHE_MASK},
    memory::{call_mem_load_, call_mem_store_},
    store_registers_to_cpu, Opcode,
};
use cranelift_codegen::ir::{condcodes::IntCC, *};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::JITModule;
use cranelift_module::Module;
use log::error;
// use raki::{BaseIOpcode, Decode, Instruction, Isa, OpcodeKind};

use crate::{
    define_rd_and_mark_dirty,
    jitwrapper::{JITWrapper, SlowTBCache},
    load_reg_if_needed_and_not_dirty, load_two_regs,
    register, ExecutionMode, ExecutionState, Executor, Instruction,
};

// TODO merge with populate_fast_cache
fn populate_slow_cache(
    current_pc: u32,
    current_unconstrained: bool,
    tb: *const u8,
    slow_tb_cache: &mut SlowTBCache,
) {
    let idx = (current_pc >> 8 & SLOW_CACHE_MASK) as usize;

    slow_tb_cache[idx] = Some(TBCacheEntry {
        pc: current_pc,
        unconstrained: current_unconstrained,
        tb,
    });
}

pub fn try_to_compile_tb_and_populate_slow_cache<'a>(
    executor: &mut Executor<'a>, // TODO reduce Executor down to necessary attributes
    jit_wrapper: &mut JITWrapper,
    slow_tb_cache: &mut SlowTBCache,
) -> ExecutionMode {
    let (tb_ptr, insns_compiled) = compile_tb(executor, jit_wrapper, 16);

    // WIP this must be refactored  
    if tb_ptr == std::ptr::null() && insns_compiled == 0 {
        println!("rotate module");
        // TODO consider one time replace
        let old_jit = std::mem::replace(&mut jit_wrapper.jit, dummy_jit_module());
        unsafe { old_jit.free_memory(); }
        // println!("before the cleanup {:?}",  jit_wrapper.jit.declarations());
        *jit_wrapper = JITWrapper::default();
        println!("after the cleanup {:?}",  jit_wrapper.jit.declarations());

        return ExecutionMode::Emulator;
    }

    println!("insns_compiled {}", insns_compiled);
    if insns_compiled > 0 {
        populate_slow_cache(
            executor.state.pc,
            executor.unconstrained,
            tb_ptr,
            slow_tb_cache,
        );
        ExecutionMode::TB(tb_ptr)
    } else {
        ExecutionMode::Emulator
    }
}

pub fn compile_tb<'a>(
    executor: &mut Executor<'a>,
    jit_wrapper: &mut JITWrapper,
    max_insns: usize,
) -> (*const u8, usize) {
    // let jit = &mut jit_wrapper.jit;

    let mut ctx = jit_wrapper.jit.make_context();
    ctx.func.signature.params.push(AbiParam::new(types::I64)); // *mut Memory
    ctx.func.signature.params.push(AbiParam::new(types::I64)); // *mut Registers
    ctx.func.signature.returns.push(AbiParam::new(types::I32)); // next PC

    let mut fctx = FunctionBuilderContext::new();
    let mut b = FunctionBuilder::new(&mut ctx.func, &mut fctx);

    let entry = b.create_block();
    b.append_block_params_for_function_params(entry);
    b.switch_to_block(entry);

    let memory_ptr = b.block_params(entry)[0]; // *mut Memory as i64
    let register_file_ptr = b.block_params(entry)[1]; // *mut Registers as i64
    let regs: [Variable; 32] = core::array::from_fn(|i| Variable::from_u32(i as u32));
    // Track which registers have been modified during this TB
    let mut regs_read_or_changed_so_far = [false; 32];
    let mut dirty_regs = [false; 32];

    // объявим x0..x31 как переменные
    // use liveness analysis
    for i in 0..32 {
        b.declare_var(regs[i], types::I32);
    }

    let mut pc = executor.state.pc;

    let mut cnt = 0;
    let mut term_was_added = false;
    while cnt < max_insns && cnt < executor.program.instructions.len() {
        // println!("cnt: {} pc {}", cnt, pc);
        let inst = executor.fetch_at(pc);
        println!("pc {} {:?}", pc, inst);
        match inst.opcode {
            Opcode::ADD => {
                let (rd, v1, v2) = preload_alu(
                    &mut b,
                    register_file_ptr,
                    &regs,
                    &mut regs_read_or_changed_so_far,
                    &mut dirty_regs,
                    inst,
                );

                let r = b.ins().iadd(v1, v2);
                define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, rd, r);
            }
            Opcode::SUB => {
                let (rd, v1, v2) = preload_alu(
                    &mut b,
                    register_file_ptr,
                    &regs,
                    &mut regs_read_or_changed_so_far,
                    &mut dirty_regs,
                    inst,
                );

                let v = b.ins().isub(v1, v2);

                define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, rd, v);
            }
            Opcode::XOR => {
                let (rd, v1, v2) = preload_alu(
                    &mut b,
                    register_file_ptr,
                    &regs,
                    &mut regs_read_or_changed_so_far,
                    &mut dirty_regs,
                    inst,
                );

                let v = b.ins().bxor(v1, v2);

                define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, rd, v);
            }
            Opcode::OR => {
                let (rd, v1, v2) = preload_alu(
                    &mut b,
                    register_file_ptr,
                    &regs,
                    &mut regs_read_or_changed_so_far,
                    &mut dirty_regs,
                    inst,
                );

                let v = b.ins().bor(v1, v2);

                define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, rd, v);
            }
            Opcode::AND => {
                let (rd, v1, v2) = preload_alu(
                    &mut b,
                    register_file_ptr,
                    &regs,
                    &mut regs_read_or_changed_so_far,
                    &mut dirty_regs,
                    inst,
                );

                let v = b.ins().band(v1, v2);

                define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, rd, v);
            }
            Opcode::SLL => {
                let (rd, v1, v2) = preload_alu(
                    &mut b,
                    register_file_ptr,
                    &regs,
                    &mut regs_read_or_changed_so_far,
                    &mut dirty_regs,
                    inst,
                );

                let v = b.ins().ishl(v1, v2);

                define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, rd, v);
            }
            Opcode::SRL => {
                let (rd, v1, v2) = preload_alu(
                    &mut b,
                    register_file_ptr,
                    &regs,
                    &mut regs_read_or_changed_so_far,
                    &mut dirty_regs,
                    inst,
                );

                let v = b.ins().ushr(v1, v2);

                define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, rd, v);
            }
            Opcode::SRA => {
                let (rd, v1, v2) = preload_alu(
                    &mut b,
                    register_file_ptr,
                    &regs,
                    &mut regs_read_or_changed_so_far,
                    &mut dirty_regs,
                    inst,
                );

                let v = b.ins().sshr(v1, v2); // Arithmetic (signed) shift right

                define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, rd, v);
            }
            Opcode::SLT => {
                let (rd, v1, v2) = preload_alu(
                    &mut b,
                    register_file_ptr,
                    &regs,
                    &mut regs_read_or_changed_so_far,
                    &mut dirty_regs,
                    inst,
                );

                // Use icmp_slt to compare if v1 < v2 (signed comparison)
                let cond = b.ins().icmp(IntCC::SignedLessThan, v1, v2);
                let zero = b.ins().iconst(types::I32, 0);
                let one = b.ins().iconst(types::I32, 1);
                let v = b.ins().select(cond, one, zero);

                define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, rd, v);
            }
            Opcode::SLTU => {
                let (rd, v1, v2) = preload_alu(
                    &mut b,
                    register_file_ptr,
                    &regs,
                    &mut regs_read_or_changed_so_far,
                    &mut dirty_regs,
                    inst,
                );

                // Using SLTU for unsigned comparison
                let cond = b.ins().icmp(IntCC::UnsignedLessThan, v1, v2);
                let zero = b.ins().iconst(types::I32, 0);
                let one = b.ins().iconst(types::I32, 1);
                let v = b.ins().select(cond, one, zero);

                define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, rd, v);
            }
            Opcode::LB | Opcode::LBU => {
                let (rd, rs1, imm) = inst.i_type();
                let reg_idx = rs1 as usize;
                let op_b = load_reg_if_needed_and_not_dirty(
                    &mut b,
                    register_file_ptr,
                    reg_idx,
                    &mut regs_read_or_changed_so_far,
                    &mut dirty_regs,
                    &regs,
                );

                let three = b.ins().iconst(types::I32, 3);

                let base = b.use_var(regs[op_b]);
                let addr = b.ins().iadd_imm(base, imm as i64);
                // TODO see if the next op is replaced by `addr & 3`
                let shift_amount = b.ins().band_imm(addr,3);
                let shift_amount = b.ins().ishl_imm(shift_amount, 3);

                let aligned_addr = b.ins().band_not(addr,three);

                // Call load32 (we don't have separate load8)
                let val = call_mem_load_(jit_wrapper, &mut b, memory_ptr, aligned_addr);

                // Extract the needed byte
                let byte_val = b.ins().ushr(val, shift_amount);
                let byte_val = b.ins().band_imm(byte_val, 0xff);

                define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, rd as u32, byte_val);
            }
            Opcode::LH | Opcode::LHU => {
                // Decode instruction fields
                let (rd, rs1, imm) = inst.i_type();
                let reg_idx = rs1 as usize;

                // Ensure source register is available in a Cranelift variable
                let op_b = load_reg_if_needed_and_not_dirty(
                    &mut b,
                    register_file_ptr,
                    reg_idx,
                    &mut regs_read_or_changed_so_far,
                    &mut dirty_regs,
                    &regs,
                );

                let one = b.ins().iconst(types::I32, 1);
                let four = b.ins().iconst(types::I32, 4);
                let mask_16bits = b.ins().iconst(types::I32, 0xFFFF);

                // Effective address: base + immediate
                let base = b.use_var(regs[op_b]);
                let addr = b.ins().iadd_imm(base, imm as i64);

                // Load the 32-bit word containing the requested half-word
                let val = call_mem_load_(jit_wrapper, &mut b, memory_ptr, addr);
                let addr_shifted = b.ins().ushr(addr, one);
                let higher_or_lower = b.ins().band(addr_shifted, one);
                let shift_amount = b.ins().band(higher_or_lower, four);
                let mask = b.ins().ishl(mask_16bits, shift_amount);

                // // Extract the halfword and sign-extend it to 32 bits
                let hw_val = b.ins().band(val,mask);

                define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, rd as u32, hw_val);
            }
            Opcode::LW => {
                let (rd, rs1, imm) = inst.i_type();
                let rs1 = load_reg_if_needed_and_not_dirty(
                    &mut b,
                    register_file_ptr,
                    rs1 as usize,
                    &mut regs_read_or_changed_so_far,
                    &mut dirty_regs,
                    &regs,
                );

                let base = b.use_var(regs[rs1]);
                let addr = b.ins().iadd_imm(base, imm as i64);

                let val = call_mem_load_(jit_wrapper, &mut b, memory_ptr, addr);

                define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, rd as u32, val);
            }
            Opcode::SB => {
                let (rs1, rs2, imm) = inst.s_type();
                let (rs1, rs2) = load_two_regs(
                    &mut b,
                    register_file_ptr,
                    &regs,
                    &mut regs_read_or_changed_so_far,
                    &mut dirty_regs,
                    rs1 as u32,
                    rs2 as u32,
                );

                let three = b.ins().iconst(types::I32, 3);
                let ff = b.ins().iconst(types::I32, 0xFF);

                let base = b.use_var(regs[rs2]);
                let addr = b.ins().iadd_imm(base, imm as i64);
                let shift_amount = b.ins().band_imm(addr,3);

                let shift_amount = b.ins().ishl_imm(shift_amount, 3);

                let aligned_addr = b.ins().band_not(addr,three);
                
                let val = b.use_var(regs[rs1]);
                let byte_val: Value = b.ins().band_imm(val, 0xFF);
                let shifted_byte_val = b.ins().ishl(byte_val, shift_amount);

                let stored_word_mask = b.ins().ishl(ff, shift_amount);
                let memory_word_value = call_mem_load_(jit_wrapper, &mut b, memory_ptr, aligned_addr);
                let new_memory_word_value = b.ins().band_not(memory_word_value, stored_word_mask);
                let new_memory_word_value = b.ins().iadd(new_memory_word_value, shifted_byte_val);

                call_mem_store_(jit_wrapper, &mut b, memory_ptr, aligned_addr, new_memory_word_value);
            }
            Opcode::SH => {
                // let (rs1, rs2, imm) = inst.s_type();
                // let (rs1, rs2) = load_two_regs(
                //     &mut b,
                //     register_file_ptr,
                //     &regs,
                //     &mut regs_read_or_changed_so_far,
                //     &mut dirty_regs,
                //     rs1 as u32,
                //     rs2 as u32,
                // );

                // let base = b.use_var(regs[rs2]);
                // let addr = b.ins().iadd_imm(base, imm as i64);
                // let val = b.use_var(regs[rs1]);

                // call_mem_store(jit, &mut b, memory_ptr, addr, val);
                break;
            }
            Opcode::SW => {
                let (rs1, rs2, imm) = inst.s_type();
                let (rs1, rs2) = load_two_regs(
                    &mut b,
                    register_file_ptr,
                    &regs,
                    &mut regs_read_or_changed_so_far,
                    &mut dirty_regs,
                    rs1 as u32,
                    rs2 as u32,
                );

                let base = b.use_var(regs[rs2]);
                let imm = imm as i64;
                let addr = b.ins().iadd_imm(base, imm);
                let val = b.use_var(regs[rs1]);

                call_mem_store_(jit_wrapper, &mut b, memory_ptr, addr, val);
            }
            Opcode::JAL => {
                // Jump and link - terminal instruction
                let (rd, imm) = inst.j_type();

                // Store return address (PC+4) in rd
                let return_addr = b.ins().iconst(types::I32, (pc + 4) as i64);
                define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, rd as u32, return_addr);

                // Calculate target address
                let imm = imm as i64;
                let target_pc = b.ins().iconst(types::I32, (pc as i64) + imm);

                // Terminate the current translation block and jump to target
                store_registers_to_cpu(&mut b, register_file_ptr, &regs, &dirty_regs);
                b.ins().return_(&[target_pc]);
                term_was_added = true;
                cnt += 1;
                break;
            }
            Opcode::JALR => {
                let (rd, rs1, imm) = inst.i_type();
                let rs1 = load_reg_if_needed_and_not_dirty(
                    &mut b,
                    register_file_ptr,
                    rs1 as usize,
                    &mut regs_read_or_changed_so_far,
                    &mut dirty_regs,
                    &regs,
                );

                let target = b.use_var(regs[rs1]);
                let imm = imm as i64;
                let next = b.ins().iadd_imm(target, imm);
                let const_pc = b.ins().iconst(types::I32, (pc + 4) as i64);
                define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, rd as u32, const_pc);

                // quit early after J-instruction
                store_registers_to_cpu(&mut b, register_file_ptr, &regs, &dirty_regs);

                b.ins().return_(&[next]);
                term_was_added = true;
                cnt += 1;

                break;
            }
            Opcode::BEQ => {
                // Branch if equal - terminal instruction
                let (rs1, rs2, imm) = inst.b_type();

                let (rs1, rs2) = load_two_regs(
                    &mut b,
                    register_file_ptr,
                    &regs,
                    &mut regs_read_or_changed_so_far,
                    &mut dirty_regs,
                    rs1 as u32,
                    rs2 as u32,
                );

                let v1 = b.use_var(regs[rs1]);
                let v2 = b.use_var(regs[rs2]);

                // Compare rs1 and rs2
                let cond = b.ins().icmp(IntCC::Equal, v1, v2);

                // Calculate target and fallthrough addresses
                // Calculate target address (pc + offset) ensuring proper casting
                let imm = imm as i64;
                let target_pc = b.ins().iconst(types::I32, (pc as i64) + imm);
                let fallthrough_pc = b.ins().iconst(types::I32, (pc + 4) as i64);

                // Select which PC to branch to based on condition
                let next_pc = b.ins().select(cond, target_pc, fallthrough_pc);

                // Terminate the current translation block
                store_registers_to_cpu(&mut b, register_file_ptr, &regs, &dirty_regs);
                b.ins().return_(&[next_pc]);
                term_was_added = true;
                cnt += 1;
                break;
            }
            Opcode::BNE => {
                // Branch if not equal - terminal instruction
                let (rs1, rs2, imm) = inst.b_type();
                let (rs1, rs2) = load_two_regs(
                    &mut b,
                    register_file_ptr,
                    &regs,
                    &mut regs_read_or_changed_so_far,
                    &mut dirty_regs,
                    rs1 as u32,
                    rs2 as u32,
                );

                let v1 = b.use_var(regs[rs1]);
                let v2 = b.use_var(regs[rs2]);

                // Compare rs1 and rs2
                let cond = b.ins().icmp(IntCC::NotEqual, v1, v2);

                let imm = imm as i64;
                // Calculate target and fallthrough addresses
                // Calculate target address (pc + offset) ensuring proper casting
                let target_pc = b.ins().iconst(types::I32, (pc as i64) + imm);
                let fallthrough_pc = b.ins().iconst(types::I32, (pc + 4) as i64);

                // Select which PC to branch to based on condition
                let next_pc = b.ins().select(cond, target_pc, fallthrough_pc);

                // Terminate the current translation block
                store_registers_to_cpu(&mut b, register_file_ptr, &regs, &dirty_regs);
                b.ins().return_(&[next_pc]);
                term_was_added = true;
                cnt += 1;
                break;
            }
            Opcode::BLT => {
                // Branch if less than - terminal instruction
                let (rs1, rs2, imm) = inst.b_type();
                let (rs1, rs2) = load_two_regs(
                    &mut b,
                    register_file_ptr,
                    &regs,
                    &mut regs_read_or_changed_so_far,
                    &mut dirty_regs,
                    rs1 as u32,
                    rs2 as u32,
                );

                let v1 = b.use_var(regs[rs1]);
                let v2 = b.use_var(regs[rs2]);

                // Compare rs1 < rs2 (signed)
                let cond = b.ins().icmp(IntCC::SignedLessThan, v1, v2);

                let imm = imm as i64;
                // Calculate target and fallthrough addresses
                // Calculate target address (pc + offset) ensuring proper casting
                let target_pc = b.ins().iconst(types::I32, (pc as i64) + imm);
                let fallthrough_pc = b.ins().iconst(types::I32, (pc + 4) as i64);

                // Select which PC to branch to based on condition
                let next_pc = b.ins().select(cond, target_pc, fallthrough_pc);

                // Terminate the current translation block
                store_registers_to_cpu(&mut b, register_file_ptr, &regs, &dirty_regs);
                b.ins().return_(&[next_pc]);
                term_was_added = true;
                cnt += 1;
                break;
            }
            Opcode::BGE => {
                // Branch if greater than or equal - terminal instruction
                let (rs1, rs2, imm) = inst.b_type();
                let (rs1, rs2) = load_two_regs(
                    &mut b,
                    register_file_ptr,
                    &regs,
                    &mut regs_read_or_changed_so_far,
                    &mut dirty_regs,
                    rs1 as u32,
                    rs2 as u32,
                );

                let v1 = b.use_var(regs[rs1]);
                let v2 = b.use_var(regs[rs2]);

                // Compare rs1 >= rs2 (signed)
                let cond = b.ins().icmp(IntCC::SignedGreaterThanOrEqual, v1, v2);

                // Calculate target and fallthrough addresses
                // Calculate target address (pc + offset) ensuring proper casting

                let imm = imm as i64;
                let target_pc = b.ins().iconst(types::I32, (pc as i64) + imm);
                let fallthrough_pc = b.ins().iconst(types::I32, (pc + 4) as i64);

                // Select which PC to branch to based on condition
                let next_pc = b.ins().select(cond, target_pc, fallthrough_pc);

                // Terminate the current translation block
                store_registers_to_cpu(&mut b, register_file_ptr, &regs, &dirty_regs);
                b.ins().return_(&[next_pc]);
                term_was_added = true;
                cnt += 1;
                break;
            }
            Opcode::BLTU => {
                // Branch if less than unsigned - terminal instruction
                let (rs1, rs2, imm) = inst.b_type();
                let (rs1, rs2) = load_two_regs(
                    &mut b,
                    register_file_ptr,
                    &regs,
                    &mut regs_read_or_changed_so_far,
                    &mut dirty_regs,
                    rs1 as u32,
                    rs2 as u32,
                );

                let v1 = b.use_var(regs[rs1]);
                let v2 = b.use_var(regs[rs2]);

                // Compare rs1 < rs2 (unsigned)
                let cond = b.ins().icmp(IntCC::UnsignedLessThan, v1, v2);

                let imm = imm as i64;
                // Calculate target and fallthrough addresses
                // Calculate target address (pc + offset) ensuring proper casting
                let target_pc = b.ins().iconst(types::I32, (pc as i64) + imm);
                let fallthrough_pc = b.ins().iconst(types::I32, (pc + 4) as i64);

                // Select which PC to branch to based on condition
                let next_pc = b.ins().select(cond, target_pc, fallthrough_pc);

                // Terminate the current translation block
                store_registers_to_cpu(&mut b, register_file_ptr, &regs, &dirty_regs);
                b.ins().return_(&[next_pc]);
                term_was_added = true;
                cnt += 1;
                break;
            }
            Opcode::BGEU => {
                // Branch if greater than or equal unsigned - terminal instruction
                let (rs1, rs2, imm) = inst.b_type();
                let (rs1, rs2) = load_two_regs(
                    &mut b,
                    register_file_ptr,
                    &regs,
                    &mut regs_read_or_changed_so_far,
                    &mut dirty_regs,
                    rs1 as u32,
                    rs2 as u32,
                );

                let v1 = b.use_var(regs[rs1]);
                let v2 = b.use_var(regs[rs2]);

                // Compare rs1 >= rs2 (unsigned)
                let cond = b.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, v1, v2);

                let imm = imm as i64;
                // Calculate target and fallthrough addresses
                // Calculate target address (pc + offset) ensuring proper casting
                let target_pc = b.ins().iconst(types::I32, (pc as i64) + imm);
                let fallthrough_pc = b.ins().iconst(types::I32, (pc + 4) as i64);

                // Select which PC to branch to based on condition
                let next_pc = b.ins().select(cond, target_pc, fallthrough_pc);

                // Terminate the current translation block
                store_registers_to_cpu(&mut b, register_file_ptr, &regs, &dirty_regs);
                b.ins().return_(&[next_pc]);
                term_was_added = true;
                cnt += 1;
                break;
            }
            Opcode::AUIPC => {
                // Add Upper Immediate to PC
                let (rd, imm) = inst.u_type();
                let rd = rd as u32;
                // let imm: i64 = (imm as i64) << 12;
                let result = b.ins().iconst(types::I32, (pc as i64) + (imm as i64));

                define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, rd, result);
            }
            Opcode::MUL => {
                let (rd, v1, v2) = preload_alu(
                    &mut b,
                    register_file_ptr,
                    &regs,
                    &mut regs_read_or_changed_so_far,
                    &mut dirty_regs,
                    inst,
                );
                let v = b.ins().imul(v1, v2);

                define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, rd, v);
            }
            Opcode::MULH => {
                // Multiply high (signed x signed)
                let (rd, v1, v2) = preload_alu(
                    &mut b,
                    register_file_ptr,
                    &regs,
                    &mut regs_read_or_changed_so_far,
                    &mut dirty_regs,
                    inst,
                );

                // We need to cast to i64, multiply, then get the high 32 bits
                let v1_64 = b.ins().sextend(types::I64, v1);
                let v2_64 = b.ins().sextend(types::I64, v2);
                let mul_result = b.ins().imul(v1_64, v2_64);

                // Shift right by 32 to get the high bits
                let shift_amt = b.ins().iconst(types::I64, 32);
                let high_bits = b.ins().ushr(mul_result, shift_amt);

                // Truncate back to 32 bits
                let result_32 = b.ins().ireduce(types::I32, high_bits);

                define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, rd, result_32);
            }
            Opcode::MULHU => {
                // Multiply high (unsigned x unsigned)
                let (rd, v1, v2) = preload_alu(
                    &mut b,
                    register_file_ptr,
                    &regs,
                    &mut regs_read_or_changed_so_far,
                    &mut dirty_regs,
                    inst,
                );

                // We need to cast to u64, multiply, then get the high 32 bits
                let v1_64 = b.ins().uextend(types::I64, v1);
                let v2_64 = b.ins().uextend(types::I64, v2);
                let mul_result = b.ins().imul(v1_64, v2_64);

                // Shift right by 32 to get the high bits
                let shift_amt = b.ins().iconst(types::I64, 32);
                let high_bits = b.ins().ushr(mul_result, shift_amt);

                // Truncate back to 32 bits
                let result_32 = b.ins().ireduce(types::I32, high_bits);

                define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, rd, result_32);
            }
            Opcode::MULHSU => {
                // Multiply high (signed x unsigned)
                let (rd, v1, v2) = preload_alu(
                    &mut b,
                    register_file_ptr,
                    &regs,
                    &mut regs_read_or_changed_so_far,
                    &mut dirty_regs,
                    inst,
                );

                // Mixed sign extension: rs1 is signed, rs2 is unsigned
                let v1_64 = b.ins().sextend(types::I64, v1);
                let v2_64 = b.ins().uextend(types::I64, v2);
                let mul_result = b.ins().imul(v1_64, v2_64);

                // Shift right by 32 to get the high bits
                let shift_amt = b.ins().iconst(types::I64, 32);
                let high_bits = b.ins().ushr(mul_result, shift_amt);

                // Truncate back to 32 bits
                let result_32 = b.ins().ireduce(types::I32, high_bits);

                define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, rd, result_32);
            }
            Opcode::DIV => {
                // Signed division
                let (rd, v1, v2) = preload_alu(
                    &mut b,
                    register_file_ptr,
                    &regs,
                    &mut regs_read_or_changed_so_far,
                    &mut dirty_regs,
                    inst,
                );
                let zero = b.ins().iconst(types::I32, 0);
                let neg_one = b.ins().iconst(types::I32, -1_i32 as i64);

                let is_zero = b.ins().icmp(IntCC::Equal, v2, zero);

                let overflow_block = b.create_block(); // jumps to cont if overflow
                let div_block  = b.create_block();    // executes the divide
                let cont_block = b.create_block();    // join-point, result in a block param

                // Additional block for overflow case
                // Jump either to div_block or directly to cont_block with −1.
                b.ins().brif(is_zero, cont_block, &[BlockArg::Value(neg_one)], overflow_block, &[]);
                
                // ---------- overflow_block ----------
                b.switch_to_block(overflow_block);
                let min_int = b.ins().iconst(types::I32, -2147483648_i32 as i64); // 0x80000000
                let cmp_with_min = b.ins().icmp(IntCC::Equal, v1, min_int);
                let cmp_with_neg_one = b.ins().icmp(IntCC::Equal, v2, neg_one);
                let is_overflow = b.ins().band(cmp_with_min, cmp_with_neg_one);
                b.ins().brif(is_overflow, cont_block, &[BlockArg::Value(min_int)], div_block, &[]);
                
                // ---------- div_block ----------
                b.switch_to_block(div_block);
                let div_res = b.ins().sdiv(v1, v2);
                b.ins().jump(cont_block, &[BlockArg::Value(div_res)]);
                
                // ---------- cont_block ----------
                b.switch_to_block(cont_block);
                let final_result = b.append_block_param(cont_block, types::I32);
                
                define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, rd, final_result);
            }
            Opcode::DIVU => {
                // Unsigned division
                let (rd, v1, v2) = preload_alu(
                    &mut b,
                    register_file_ptr,
                    &regs,
                    &mut regs_read_or_changed_so_far,
                    &mut dirty_regs,
                    inst,
                );
                let zero = b.ins().iconst(types::I32, 0);
                let neg_one = b.ins().iconst(types::I32, -1_i32 as i64);

                let is_zero = b.ins().icmp(IntCC::Equal, v2, zero);

                let div_block  = b.create_block();    // executes the divide
                let cont_block = b.create_block();    // join-point, result in a block param

                // Additional block for overflow case
                // Jump either to div_block or directly to cont_block with −1.
                b.ins().brif(is_zero, cont_block, &[BlockArg::Value(neg_one)], div_block, &[]);
                
                // ---------- div_block ----------
                b.switch_to_block(div_block);
                let div_res = b.ins().udiv(v1, v2);
                b.ins().jump(cont_block, &[BlockArg::Value(div_res)]);
                
                // ---------- cont_block ----------
                b.switch_to_block(cont_block);
                let final_result = b.append_block_param(cont_block, types::I32);
                
                define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, rd, final_result);
            }
            Opcode::REM => {
                // Signed remainder
                let (rd, v1, v2) = preload_alu(
                    &mut b,
                    register_file_ptr,
                    &regs,
                    &mut regs_read_or_changed_so_far,
                    &mut dirty_regs,
                    inst,
                );
                let zero = b.ins().iconst(types::I32, 0);
                let neg_one = b.ins().iconst(types::I32, -1_i32 as i64);

                let is_zero = b.ins().icmp(IntCC::Equal, v2, zero);

                let overflow_block = b.create_block(); // jumps to cont if overflow
                let div_block  = b.create_block();    // executes the divide
                let cont_block = b.create_block();    // join-point, result in a block param

                // Additional block for overflow case
                // Jump either to div_block or directly to cont_block with −1.
                b.ins().brif(is_zero, cont_block, &[BlockArg::Value(v1)], overflow_block, &[]);
                
                // ---------- overflow_block ----------
                b.switch_to_block(overflow_block);
                let min_int = b.ins().iconst(types::I32, -2147483648_i32 as i64); // 0x80000000
                let cmp_with_min = b.ins().icmp(IntCC::Equal, v1, min_int);
                let cmp_with_neg_one = b.ins().icmp(IntCC::Equal, v2, neg_one);
                let is_overflow = b.ins().band(cmp_with_min, cmp_with_neg_one);
                b.ins().brif(is_overflow, cont_block, &[BlockArg::Value(zero)], div_block, &[]);
                
                // ---------- div_block ----------
                b.switch_to_block(div_block);
                let div_res = b.ins().srem(v1, v2);
                b.ins().jump(cont_block, &[BlockArg::Value(div_res)]);
                
                // ---------- cont_block ----------
                b.switch_to_block(cont_block);
                let final_result = b.append_block_param(cont_block, types::I32);
                
                define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, rd, final_result);
            }
            Opcode::REMU => {
                let (rd, v1, v2) = preload_alu(
                    &mut b,
                    register_file_ptr,
                    &regs,
                    &mut regs_read_or_changed_so_far,
                    &mut dirty_regs,
                    inst,
                );
                let zero = b.ins().iconst(types::I32, 0);

                let is_zero = b.ins().icmp(IntCC::Equal, v2, zero);

                let div_block  = b.create_block();    // executes the divide
                let cont_block = b.create_block();    // join-point, result in a block param

                // Additional block for overflow case
                // Jump either to div_block or directly to cont_block with −1.
                b.ins().brif(is_zero, cont_block, &[BlockArg::Value(v1)], div_block, &[]);
                
                // ---------- overflow_block ----------
                // b.switch_to_block(overflow_block);
                // let min_int = b.ins().iconst(types::I32, -2147483648_i32 as i64); // 0x80000000
                // let cmp_with_min = b.ins().icmp(IntCC::Equal, v1, min_int);
                // let cmp_with_neg_one = b.ins().icmp(IntCC::Equal, v2, neg_one);
                // let is_overflow = b.ins().band(cmp_with_min, cmp_with_neg_one);
                // b.ins().brif(is_overflow, cont_block, &[zero], div_block, &[]);
                
                // ---------- div_block ----------
                b.switch_to_block(div_block);
                let div_res = b.ins().urem(v1, v2);
                b.ins().jump(cont_block, &[BlockArg::Value(div_res)]);
                
                // ---------- cont_block ----------
                b.switch_to_block(cont_block);
                let final_result = b.append_block_param(cont_block, types::I32);
                
                define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, rd, final_result);
            }
            Opcode::ECALL => {
                // Placeholder for ECALL as requested
                // ECALL should be a terminal instruction
                // store_registers_to_cpu(&mut b, register_file_ptr, &regs, &dirty_regs);

                // Return a special value to indicate an ECALL (could be handled by the main loop)
                // let ecall_indicator = b.ins().iconst(types::I32, 0xECA11);
                // let rvals = &[b.ins().iconst(types::I32, pc as i64)];
                // b.ins().return_(rvals);
                // term_was_added = true;
                // cnt += 1;
                break;
            }
            Opcode::EBREAK => {
            //     // Environment break - terminal instruction
            //     store_registers_to_cpu(&mut b, cpu_ptr, &regs, &dirty_regs);

            //     // Return a special value to indicate an EBREAK (could be handled by the main loop)
            //     let ebreak_indicator = b.ins().iconst(types::I32, 0xEB8EA);
            //     b.ins().return_(&[ebreak_indicator]);
            //     term_was_added = true;
            //     cnt += 1;
                break;
            }
            _ => unreachable!(""),
        }
        pc += 4;
        cnt += 1;
    }

    // println!("compile_tb ########### cnt: {}", cnt);
    // return next PC if we reached the limit or if there was no terminal instruction
    if !term_was_added || (!term_was_added && cnt == max_insns) {
        println!("compile_tb add terminal cnt: {}", cnt);
        store_registers_to_cpu(&mut b, register_file_ptr, &regs, &dirty_regs);
        let rvals = &[b.ins().iconst(types::I32, pc as i64)];
        b.ins().return_(rvals);
    }

    b.seal_all_blocks();
    // replace with ctx.func.signature ?
    let sign = b.func.signature.clone();
    b.finalize();

    println!("{}", ctx.func.display());

    let id = jit_wrapper.jit.declare_anonymous_function(&sign).unwrap();
    jit_wrapper.jit.define_function(id, &mut ctx).unwrap();

    jit_wrapper.jit.clear_context(&mut ctx);
    // jit.finalize_definitions().expect("must be ok");
    match jit_wrapper.jit.finalize_definitions() {
        Ok(_) => {},
        Err(e) => {
            println!("Error finalizing definitions: {}", e);
            return (std::ptr::null(), 0);
        }
    };
    (jit_wrapper.jit.get_finalized_function(id), cnt)
}

fn preload_alu(
    b: &mut FunctionBuilder<'_>,
    register_file_ptr: Value,
    regs: &[Variable; 32],
    regs_read_or_changed_so_far: &mut [bool; 32],
    dirty_regs: &mut [bool; 32],
    inst: Instruction,
) -> (u32, Value, Value) {
    let (rd, v1, v2) = if !inst.has_imm_c() {
        let (rd, rs1, rs2) = inst.r_type();
        let (v1, v2) = preload_for_bin_op(
            b,
            register_file_ptr,
            regs_read_or_changed_so_far,
            dirty_regs,
            regs,
            rs1 as u32,
            rs2 as u32,
        );
        // let v = b.ins().iadd(v1, v2);
        // define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, rd as u32, v);
        // println!("ADD dirty {:?}", dirty_regs);
        (rd as u32, v1, v2)
    } else if inst.is_addi_instruction() {
        let (rd, rs1, imm) = inst.i_type();
        let rs1 = load_reg_if_needed_and_not_dirty(
            b,
            register_file_ptr,
            rs1 as usize,
            regs_read_or_changed_so_far,
            dirty_regs,
            regs,
        );

        let v1 = b.use_var(regs[rs1]);
        let v2 = b.ins().iconst(types::I32, imm as i64);

        // define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, rd as u32, r);
        // println!("processing ADDI dirty {:?}", dirty_regs);
        (rd as u32, v1, v2)
    } else {
        assert!(inst.imm_b && inst.imm_c);
        let Instruction {
            op_a: rd,
            op_b: imm_l,
            op_c: imm_r,
            ..
        } = inst;

        // Both operands are immediates. Materialise each as an I32 constant and add them.
        let v1 = b.ins().iconst(types::I32, imm_l as i64);
        let v2 = b.ins().iconst(types::I32, imm_r as i64);
        // let sum = b.ins().iadd(v1, v2);
        (rd, v1, v2)
    };
    (rd, v1, v2)
}

// Compiler helpers
fn preload_for_bin_op(
    b: &mut FunctionBuilder<'_>,
    register_file_ptr: Value,
    regs_read_or_changed_so_far: &mut [bool; 32],
    dirty_regs: &mut [bool; 32],
    regs: &[Variable; 32],
    rs1: u32,
    rs2: u32,
) -> (Value, Value) {
    let (rs1, rs2) = load_two_regs(
        b,
        register_file_ptr,
        &regs,
        regs_read_or_changed_so_far,
        dirty_regs,
        rs1,
        rs2,
    );

    let v1 = b.use_var(regs[rs1]);
    let v2 = b.use_var(regs[rs2]);
    (v1, v2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        memory::Memory,
        Program, Register, RegisterFile,
    };
    use cranelift_jit::{JITBuilder, JITModule};
    use cranelift_module::default_libcall_names;

    // fn create_cpu_with_program(program: &[u8]) -> Cpu {
    //     Cpu::new(program)
    // }

    // Helper function to setup test environment
    // fn setup_test_env(program: &[u8]) -> (Cpu, JITModule, usize, u32) {
    //     // Initialize CPU with the test program
    //     let cpu = Cpu::new(program);

    //     setup_test_env_with_cpu(program, cpu)
    // }

    fn setup_test_env_with_cpu(executor: &mut Executor) -> (usize, u32) {
        println!(
            "setup_test_env_with_cpu {} ",
            executor.program.instructions.len()
        );

        let mut jit_wrapper = JITWrapper::default();

        // Compile the translation block
        let (fn_ptr, insns) = compile_tb(
            executor,
            &mut jit_wrapper,
            executor.program.instructions.len(),
        ); // Max instructions based on program size

        // Execute the compiled code
        let tb_executor: extern "C" fn(*mut Memory, *mut RegisterFile) -> u32 =
            unsafe { std::mem::transmute(fn_ptr) };
        let next_pc = tb_executor(
            &mut executor.state.memory_,
            &mut executor.state.register_file,
        );

        (insns, next_pc)
    }

    #[test]
    fn test_add_and_addi() {
        // main:
        //     addi x29, x0, 5
        //     addi x30, x0, 37
        //     add x31, x30, x29
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 5, false, true),
            Instruction::new(Opcode::ADD, 30, 0, 37, false, true),
            Instruction::new(Opcode::ADD, 31, 30, 29, false, false),
        ];
        let program = Program::new(instructions, 0, 0);
        let mut runtime = Executor::new(program);
        let (insns, next_pc) = setup_test_env_with_cpu(&mut runtime);

        println!("registers {:?}", runtime.state.register_file.registers);

        assert_eq!(runtime.register(Register::X29), 5);
        assert_eq!(runtime.register(Register::X30), 37);
        assert_eq!(runtime.register(Register::X31), 42);
        assert_eq!(insns, 3);
        assert_eq!(next_pc, 12);
    }

    #[test]
    fn test_sub() {
        // main:
        //     addi x29, x0, 7
        //     addi x30, x0, 3
        //     sub x31, x30, x29
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 7, false, true),
            Instruction::new(Opcode::ADD, 30, 0, 3, false, true),
            Instruction::new(Opcode::SUB, 31, 30, 29, false, false),
        ];
        let program = Program::new(instructions, 0, 0);
        let mut runtime = Executor::new(program);
        let (insns, next_pc) = setup_test_env_with_cpu(&mut runtime);

        println!("registers {:?}", runtime.state.register_file.registers);

        assert_eq!(runtime.register(Register::X29), 7);
        assert_eq!(runtime.register(Register::X30), 3);
        assert_eq!(runtime.register(Register::X31) as i32, -4);
        assert_eq!(insns, 3);
        assert_eq!(next_pc, 12);
    }

    #[test]
    fn test_xor() {
        // main:
        //     addi x29, x0, 5
        //     addi x30, x0, 37
        //     xor x31, x30, x29
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 5, false, true),
            Instruction::new(Opcode::ADD, 30, 0, 37, false, true),
            Instruction::new(Opcode::XOR, 31, 30, 29, false, false),
        ];
        let program = Program::new(instructions, 0, 0);
        let mut runtime = Executor::new(program);
        let (insns, next_pc) = setup_test_env_with_cpu(&mut runtime);

        assert_eq!(runtime.register(Register::X29), 5);
        assert_eq!(runtime.register(Register::X30), 37);
        assert_eq!(runtime.register(Register::X31), 32);
        assert_eq!(insns, 3);
        assert_eq!(next_pc, 12);
    }

    #[test]
    fn test_or() {
        // main:
        //     addi x29, x0, 9
        //     addi x30, x0, 6
        //     or x31, x30, x29
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 9, false, true),
            Instruction::new(Opcode::ADD, 30, 0, 6, false, true),
            Instruction::new(Opcode::OR, 31, 30, 29, false, false),
        ];
        let program = Program::new(instructions, 0, 0);
        let mut runtime = Executor::new(program);
        let (insns, next_pc) = setup_test_env_with_cpu(&mut runtime);

        assert_eq!(runtime.register(Register::X29), 9);
        assert_eq!(runtime.register(Register::X30), 6);
        assert_eq!(runtime.register(Register::X31), 15);
        assert_eq!(insns, 3);
        assert_eq!(next_pc, 12);
    }

    #[test]
    fn test_and() {
        // main:
        //     addi x29, x0, 15
        //     addi x30, x0, 6
        //     and x31, x30, x29
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 15, false, true),
            Instruction::new(Opcode::ADD, 30, 0, 6, false, true),
            Instruction::new(Opcode::AND, 31, 30, 29, false, false),
        ];
        let program = Program::new(instructions, 0, 0);
        let mut runtime = Executor::new(program);
        let (insns, next_pc) = setup_test_env_with_cpu(&mut runtime);

        assert_eq!(runtime.register(Register::X29), 15);
        assert_eq!(runtime.register(Register::X30), 6);
        assert_eq!(runtime.register(Register::X31), 6);
        assert_eq!(insns, 3);
        assert_eq!(next_pc, 12);
    }

    #[test]
    fn test_sll() {
        // main:
        //     addi x29, x0, 1
        //     addi x30, x0, 2
        //     sll  x31, x30, x29
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 1, false, true),
            Instruction::new(Opcode::ADD, 30, 0, 2, false, true),
            Instruction::new(Opcode::SLL, 31, 30, 29, false, false),
        ];
        let program = Program::new(instructions, 0, 0);
        let mut runtime = Executor::new(program);
        let (insns, next_pc) = setup_test_env_with_cpu(&mut runtime);

        assert_eq!(runtime.register(Register::X29), 1);
        assert_eq!(runtime.register(Register::X30), 2);
        assert_eq!(runtime.register(Register::X31), 4);
        assert_eq!(insns, 3);
        assert_eq!(next_pc, 12);
    }

    #[test]
    fn test_srl() {
        // main:
        //     addi x29, x0, 1
        //     addi x30, x0, 8
        //     srl  x31, x30, x29
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 1, false, true),
            Instruction::new(Opcode::ADD, 30, 0, 8, false, true),
            Instruction::new(Opcode::SRL, 31, 30, 29, false, false),
        ];
        let program = Program::new(instructions, 0, 0);
        let mut runtime = Executor::new(program);
        let (insns, next_pc) = setup_test_env_with_cpu(&mut runtime);

        assert_eq!(runtime.register(Register::X29), 1);
        assert_eq!(runtime.register(Register::X30), 8);
        assert_eq!(runtime.register(Register::X31), 4);
        assert_eq!(insns, 3);
        assert_eq!(next_pc, 12);
    }

    #[test]
    fn test_slt() {
        // main:
        //     addi x29, x0, 5
        //     addi x30, x0, 3
        //     slt  x31, x30, x29   // 3 < 5 -> 1
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 5, false, true),
            Instruction::new(Opcode::ADD, 30, 0, 3, false, true),
            Instruction::new(Opcode::SLT, 31, 30, 29, false, false),
        ];
        let program = Program::new(instructions, 0, 0);
        let mut runtime = Executor::new(program);
        let (insns, next_pc) = setup_test_env_with_cpu(&mut runtime);

        assert_eq!(runtime.register(Register::X29), 5);
        assert_eq!(runtime.register(Register::X30), 3);
        assert_eq!(runtime.register(Register::X31), 1);
        assert_eq!(insns, 3);
        assert_eq!(next_pc, 12);
    }

    #[test]
    fn test_lw_instruction() {
        // Program:
        //   addi x10, x0, 0     # base address 0
        //   addi x31, x0, 42    # value to store
        //   lw   x29, 0(x10)    # load 42 from mem[0]
        let instructions = vec![
            Instruction::new(Opcode::ADD, 10, 0, 44, false, true), // addi x10, x0, 0
            Instruction::new(Opcode::ADD, 31, 0, 42, false, true), // addi x31, x0, 42
            Instruction::new(Opcode::LW, 29, 10, 0, false, true),  // lw x29, 0(x10)
        ];

        let program = Program::new(instructions, 0, 0);
        let mut runtime = Executor::new(program);

        runtime.state.memory_[44] = 42;
        let (insns, next_pc) = setup_test_env_with_cpu(&mut runtime);

        // Verify the results
        assert_eq!(insns, 3, "Should have translated 3 instructions");
        assert_eq!(next_pc, 12, "Next PC should be 12 after execution");
        assert_eq!(runtime.register(Register::X10), 44);
        assert_eq!(runtime.register(Register::X31), 42);
        assert_eq!(
            runtime.register(Register::X29),
            42,
            "Register x29 should load the stored value"
        );
    }

    #[test]
    fn test_byte_load_store_instructions() {
        let addr = 1023;
        let addr_offset_mask = addr & 3 << 3;
        let val = 0xFF;
        // main:
        //     addi x10, x0, 1024      # base address in x10
        //     addi x20, x0, 0xFF      # value to store (0xFF)
        //     sb   x20, 0(x10)        # store byte
        //     lb   x21, 0(x10)        # load signed byte
        //     lbu  x22, 0(x10)        # load unsigned byte
        let instructions = vec![
            // addi x10, x0, 1024
            Instruction::new(Opcode::ADD, 10, 0, addr, false, true),
            // addi x20, x0, 0xFF
            Instruction::new(Opcode::ADD, 20, 0, val, false, true),
            // sb x20, 0(x10)
            Instruction::new(Opcode::SB, 20, 10, 0, false, true),
            // lb x21, 0(x10)
            Instruction::new(Opcode::LB, 21, 10, 0, false, true),
            // lbu x22, 0(x10)
            Instruction::new(Opcode::LBU, 22, 10, 0, false, true),
        ];

        let program = Program::new(instructions, 0, 0);
        let mut runtime = Executor::new(program);
        runtime.state.memory_[addr & !3] = 0xfefefefe;
        let (insns, next_pc) = setup_test_env_with_cpu(&mut runtime);

        // Verify the results:
        // 0xFF as signed byte is -1 when sign-extended to 32 bits
        // 0xFF as unsigned byte is 255
        assert_eq!(insns, 5, "Should have translated all 5 instructions");
        assert_eq!(next_pc, 20, "Next PC should be 20 after execution");

        // x10 should hold the base address 1024
        assert_eq!(runtime.register(Register::X10), addr);
        // Memory at 1024 should contain 0xFF (lowest byte of the word)
        assert_eq!(runtime.state.memory_[addr & !3], 0xfffefefe);
        // x20's low byte is 0xFF
        assert_eq!(runtime.register(Register::X20) & 0xFF, 0xFF);
        // LB sign-extends 0xFF -> -1
        assert_eq!(runtime.register(Register::X21), 0xFF);
        // LBU zero-extends 0xFF -> 255
        assert_eq!(runtime.register(Register::X22), 0xFF);
    }

    #[test]
    fn test_halfword_load_store_instructions() {
        // main:
        //     addi x10, x0, 1024     # base address
        //     addi x20, x0, 0xFFFF   # value to store (0xFFFF = -1)
        //     sh   x20, 0(x10)       # store half-word
        //     lh   x21, 0(x10)       # load signed half-word
        //     lhu  x22, 0(x10)       # load unsigned half-word
        let instructions = vec![
            Instruction::new(Opcode::ADD, 10, 0, 1024, false, true), // addi x10, x0, 1024
            Instruction::new(Opcode::ADD, 20, 0, 0xFFFF, false, true), // addi x20, x0, 0xFFFF (-1)
            Instruction::new(Opcode::SH, 10, 20, 0, false, true),    // sh x20, 0(x10)
            Instruction::new(Opcode::LH, 21, 10, 0, false, true),    // lh x21, 0(x10)
            Instruction::new(Opcode::LHU, 22, 10, 0, false, true),   // lhu x22, 0(x10)
        ];

        let program = Program::new(instructions, 0, 0);
        let mut runtime = Executor::new(program);
        let (insns, next_pc) = setup_test_env_with_cpu(&mut runtime);

        // Verify results
        assert_eq!(insns, 5);
        assert_eq!(next_pc, 20);
        assert_eq!(runtime.register(Register::X10), 1024);
        assert_eq!(runtime.state.memory_[1024], 0xFFFF);
        assert_eq!(runtime.register(Register::X20) & 0xFFFF, 0xFFFF);
        assert_eq!(runtime.register(Register::X21) as i32, -1);
        assert_eq!(runtime.register(Register::X22), 0xFFFF);
    }

    #[test]
    fn test_sw_instruction() {
        // main:
        //     addi x10, x0, 4      # base address 4
        //     addi x20, x0, 123    # value 123
        //     sw   x20, 4(x10)     # store word at address 8
        let instructions = vec![
            Instruction::new(Opcode::ADD, 10, 0, 4, false, true), // addi x10, x0, 4
            Instruction::new(Opcode::ADD, 20, 0, 123, false, true), // addi x20, x0, 123
            Instruction::new(Opcode::SW, 10, 20, 4, false, true), // sw x20, 4(x10)
        ];

        let program = Program::new(instructions, 0, 0);
        let mut runtime = Executor::new(program);
        let (insns, next_pc) = setup_test_env_with_cpu(&mut runtime);

        assert_eq!(insns, 3);
        assert_eq!(next_pc, 12);
        assert_eq!(runtime.register(Register::X10), 4);
        assert_eq!(runtime.register(Register::X20), 123);
        assert_eq!(runtime.state.memory_[8], 123);
    }
    #[test]
    fn test_jal_instruction() {
        // Program:
        //   jal  x0, 8       # jump ahead by 8 bytes (skip next inst)
        //   addi x30, x0, 123  (should be skipped)
        //   addi x31, x0, 42   (would execute after branch target)
        let instructions = vec![
            Instruction::new(Opcode::JAL, 0, 8, 0, false, true), // jal x0, 8
            Instruction::new(Opcode::ADD, 30, 0, 123, false, true), // addi x30, x0, 123
            Instruction::new(Opcode::ADD, 31, 0, 42, false, true), // addi x31, x0, 42
        ];

        let program = Program::new(instructions, 0, 0);
        let mut runtime = Executor::new(program);
        let (insns, next_pc) = setup_test_env_with_cpu(&mut runtime);

        assert_eq!(insns, 1);
        assert_eq!(runtime.register(Register::X30), 0);
        assert_eq!(runtime.register(Register::X31), 0);
        assert_eq!(next_pc, 8);
    }

    #[test]
    fn test_jal_with_link_instruction() {
        // Program:
        //   jal  x1, 8        # jump ahead by 8 and store return addr in x1
        //   addi x30, x0, 123 # skipped
        //   addi x31, x0, 42  # branch target
        let instructions = vec![
            Instruction::new(Opcode::JAL, 1, 8, 0, false, true), // jal x1, 8
            Instruction::new(Opcode::ADD, 30, 0, 123, false, true), // addi x30, x0, 123
            Instruction::new(Opcode::ADD, 31, 0, 42, false, true), // addi x31, x0, 42
        ];

        let program = Program::new(instructions, 0, 0);
        let mut runtime = Executor::new(program);
        let (insns, next_pc) = setup_test_env_with_cpu(&mut runtime);

        assert_eq!(insns, 1);
        assert_eq!(runtime.register(Register::X1), 4); // return address PC+4
        assert_eq!(runtime.register(Register::X30), 0);
        assert_eq!(next_pc, 8);
    }

    #[test]
    fn test_auipc_instruction() {
        // TODO test with rd = X0 (should be a no-op)
        // Program:
        //   auipc x10, 1   # x10 = PC + 1<<12 = 4096
        //   addi  x20, x0, 0  # dummy instruction
        let instructions = vec![
            Instruction::new(Opcode::AUIPC, 10, 1, 0, true, false), // auipc x10, 1
            Instruction::new(Opcode::ADD, 20, 0, 0, false, true),   // addi x20, x0, 0
        ];

        let program = Program::new(instructions, 0, 0);
        let mut runtime = Executor::new(program);
        let (insns, next_pc) = setup_test_env_with_cpu(&mut runtime);

        assert_eq!(insns, 2, "Should have translated all 2 instructions");
        assert_eq!(runtime.register(Register::X10), 4096);
        assert_eq!(runtime.register(Register::X20), 0);
        assert_eq!(next_pc, 8);
    }

    #[test]
    fn test_jalr_instruction() {
        // Program:
        //   addi x10, x0, 20   # load jump target 20
        //   jalr x1, 0(x10)    # jump to x10, link return addr into x1
        //   addi x30, x0, 1    # should be skipped
        let instructions = vec![
            Instruction::new(Opcode::ADD, 10, 0, 20, false, true), // addi x10, x0, 20
            Instruction::new(Opcode::JALR, 1, 10, 0, false, true), // jalr x1, 0(x10)
            Instruction::new(Opcode::ADD, 30, 0, 1, false, true),  // addi x30, x0, 1
        ];

        let program = Program::new(instructions, 0, 0);
        let mut runtime = Executor::new(program);
        let (insns, next_pc) = setup_test_env_with_cpu(&mut runtime);

        assert_eq!(insns, 2, "Should have translated all 2 instructions");
        assert_eq!(next_pc, 20, "Next PC should be 20 after execution");
        assert_eq!(runtime.register(Register::X10), 20);
        assert_eq!(runtime.register(Register::X1), 8); // return address PC+4
        assert_eq!(runtime.register(Register::X30), 0);
    }

    // ... rest of the code remains the same ...
    //             0x13, 0x05, 0x70, 0x00,     // addi x10, x0, 7      # set x10 to 7
    //             0x13, 0x0a, 0x40, 0x00,     // addi x20, x0, 4      # set x20 to 4
    //             0x33, 0x0e, 0x45, 0x03,     // mul  x28, x10, x20   # x28 = x10 * x20
    //          ];

    //         let (cpu, _, insns, next_pc) = setup_test_env(&test_program);

    //         // Verify the results
    //         assert_eq!(insns, 3, "Should have translated all 3 instructions");
    //         assert_eq!(next_pc, 12, "PC should be 12 after execution");
    //         assert_eq!(cpu.regs[10], 7, "Register x10 should be 7");
    //         assert_eq!(cpu.regs[20], 4, "Register x20 should be 4");
    //         assert_eq!(cpu.regs[28], 28, "Register x28 should be 28 (result of 7 * 4)");
    //     }

    //     #[test]
    //     fn test_beq_instruction() {
    //         // Define a program with BEQ instruction
    //         // 1. Set x10 to 5
    //         // 2. Set x20 to 5
    //         // 3. BEQ x10, x20, 8 (branch taken as they're equal)
    //         // 4. Set x30 to 123 (should be skipped)
    //         // 5. Set x31 to 42 (should execute)
    //         let test_program = [
    //             0x13, 0x05, 0x50, 0x00,     // addi x10, x0, 5
    //             0x13, 0x0a, 0x50, 0x00,     // addi x20, x0, 5
    //             0x63, 0x04, 0x45, 0x01,     // beq x10, x20, 8
    //             0x13, 0x0f, 0xb0, 0x07,     // addi x30, x0, 123 (should be skipped)
    //             0x13, 0x0f, 0xa0, 0x02,     // addi x31, x0, 42
    //         ];

    //         let (cpu, _, insns, next_pc) = setup_test_env(&test_program);

    //         // Verify the results
    //         assert_eq!(insns, 3, "Should have translated all 3 instructions");
    //         assert_eq!(cpu.regs[10], 5, "Register x10 should be 5");
    //         assert_eq!(cpu.regs[20], 5, "Register x20 should be 5");
    //         assert_eq!(cpu.regs[30], 0, "Register x30 should still be 0 (skipped instruction)");
    //         // assert_eq!(cpu.regs[31], 42, "Register x31 should be 42");
    //         assert_eq!(next_pc, 20, "Next PC should be 20 after branch");
    //     }

    //     #[test]
    //     fn test_bne_instruction() {
    //         // Define a program with BNE instruction
    //         // 1. Set x10 to 5
    //         // 2. Set x20 to 10
    //         // 3. BNE x10, x20, 8 (branch taken as they're not equal)
    //         // 4. Set x30 to 123 (should be skipped)
    //         // 5. Set x31 to 42 (should execute)
    //         let test_program = [
    //             0x13, 0x05, 0x50, 0x00,     // addi x10, x0, 5
    //             0x13, 0x0a, 0xa0, 0x00,     // addi x20, x0, 10
    //             0x63, 0x14, 0x45, 0x01,     // bne x10, x20, 8
    //             0x13, 0x0f, 0xb0, 0x07,     // addi x30, x0, 123 (should be skipped)
    //             0x13, 0x0f, 0xa0, 0x02,     // addi x31, x0, 42
    //         ];

    //         let (cpu, _, insns, next_pc) = setup_test_env(&test_program);

    //         // Verify the results
    //         assert_eq!(insns, 3, "Should have translated all 3 instructions");
    //         assert_eq!(cpu.regs[10], 5, "Register x10 should be 5");
    //         assert_eq!(cpu.regs[20], 10, "Register x20 should be 10");
    //         assert_eq!(cpu.regs[30], 0, "Register x30 should still be 0 (skipped instruction)");
    //         // assert_eq!(cpu.regs[31], 42, "Register x31 should be 42");
    //         assert_eq!(next_pc, 20, "Next PC should be 20 after branch");
    //     }

    //     #[test]
    //     fn test_blt_instruction() {
    //         // Define a program with BLT instruction (signed comparison)
    //         // 1. Set x10 to -5 (negative value)
    //         // 2. Set x20 to 5 (positive value)
    //         // 3. BLT x10, x20, 8 (branch taken as -5 < 5)
    //         // 4. Set x30 to 123 (should be skipped)
    //         // 5. Set x31 to 42 (should execute)
    //         let test_program = [
    //             0x13, 0x05, 0xb0, 0xff,     // addi x10, x0, -5 (sign extended)
    //             0x13, 0x0a, 0x50, 0x00,     // addi x20, x0, 5
    //             0x63, 0x44, 0x45, 0x01,     // blt x10, x20, 8
    //             0x13, 0x0f, 0xb0, 0x07,     // addi x30, x0, 123 (should be skipped)
    //             0x13, 0x0f, 0xa0, 0x02,     // addi x31, x0, 42
    //         ];

    //         let (cpu, _, insns, next_pc) = setup_test_env(&test_program);

    //         // Verify the results
    //         assert_eq!(insns, 3, "Should have translated all 3 instructions");
    //         assert_eq!(cpu.regs[10] as i32, -5, "Register x10 should be -5");
    //         assert_eq!(cpu.regs[20], 5, "Register x20 should be 5");
    //         assert_eq!(cpu.regs[30], 0, "Register x30 should still be 0 (skipped instruction)");
    //         // assert_eq!(cpu.regs[31], 42, "Register x31 should be 42");
    //         assert_eq!(next_pc, 20, "Next PC should be 20 after branch");
    //     }

    //     #[test]
    //     fn test_bge_instruction() {
    //         // Define a program with BGE instruction (signed comparison)
    //         // 1. Set x10 to 10
    //         // 2. Set x20 to 5
    //         // 3. BGE x10, x20, 8 (branch taken as 10 ≥ 5)
    //         // 4. Set x30 to 123 (should be skipped)
    //         // 5. Set x31 to 42 (should execute)
    //         let test_program = [
    //             0x13, 0x05, 0xa0, 0x00,     // addi x10, x0, 10
    //             0x13, 0x0a, 0x50, 0x00,     // addi x20, x0, 5
    //             0x63, 0x54, 0x45, 0x01,     // bge x10, x20, 8
    //             0x13, 0x0f, 0xb0, 0x07,     // addi x30, x0, 123 (should be skipped)
    //             0x13, 0x0f, 0xa0, 0x02,     // addi x31, x0, 42
    //         ];

    //         let (cpu, _, insns, next_pc) = setup_test_env(&test_program);

    //         // Verify the results
    //         assert_eq!(insns, 3, "Should have translated all 3 instructions");
    //         assert_eq!(cpu.regs[10], 10, "Register x10 should be 10");
    //         assert_eq!(cpu.regs[20], 5, "Register x20 should be 5");
    //         assert_eq!(cpu.regs[30], 0, "Register x30 should still be 0 (skipped instruction)");
    //         // assert_eq!(cpu.regs[31], 42, "Register x31 should be 42");
    //         assert_eq!(next_pc, 20, "Next PC should be 20 after branch");
    //     }

    //     #[test]
    //     fn test_bltu_instruction() {
    //         // Define a program with BLTU instruction (unsigned comparison)
    //         // 1. Set x10 to 5
    //         // 2. Set x20 to 10
    //         // 3. BLTU x10, x20, 8 (branch taken as 5 < 10 unsigned)
    //         // 4. Set x30 to 123 (should be skipped)
    //         // 5. Set x31 to 42 (should execute)
    //         let test_program = [
    //             0x13, 0x05, 0x50, 0x00,     // addi x10, x0, 5
    //             0x13, 0x0a, 0xa0, 0x00,     // addi x20, x0, 10
    //             0x63, 0x64, 0x45, 0x01,     // bltu x10, x20, 8
    //             0x13, 0x0f, 0xb0, 0x07,     // addi x30, x0, 123 (should be skipped)
    //             0x13, 0x0f, 0xa0, 0x02,     // addi x31, x0, 42
    //         ];

    //         let (cpu, _, insns, next_pc) = setup_test_env(&test_program);

    //         // Verify the results
    //         assert_eq!(insns, 3, "Should have translated all 3 instructions");
    //         assert_eq!(cpu.regs[10], 5, "Register x10 should be 5");
    //         assert_eq!(cpu.regs[20], 10, "Register x20 should be 10");
    //         assert_eq!(cpu.regs[30], 0, "Register x30 should still be 0 (skipped instruction)");
    //         // assert_eq!(cpu.regs[31], 42, "Register x31 should be 42");
    //         assert_eq!(next_pc, 20, "Next PC should be 20 after branch");
    //     }

    //     #[test]
    //     fn test_bgeu_instruction() {
    //         // Define a program with BGEU instruction (unsigned comparison)
    //         // 1. Set x10 to 10
    //         // 2. Set x20 to 5
    //         // 3. BGEU x10, x20, 8 (branch taken as 10 ≥ 5 unsigned)
    //         // 4. Set x30 to 123 (should be skipped)
    //         // 5. Set x31 to 42 (should execute)
    //         let test_program = [
    //             0x13, 0x05, 0xa0, 0x00,     // addi x10, x0, 10
    //             0x13, 0x0a, 0x50, 0x00,     // addi x20, x0, 5
    //             0x63, 0x74, 0x45, 0x01,     // bgeu x10, x20, 8
    //             0x13, 0x0f, 0xb0, 0x07,     // addi x30, x0, 123 (should be skipped)
    //             0x13, 0x0f, 0xa0, 0x02,     // addi x31, x0, 42
    //         ];

    //         let (cpu, _, insns, next_pc) = setup_test_env(&test_program);

    //         // Verify the results
    //         assert_eq!(insns, 3, "Should have translated all 3 instructions");
    //         assert_eq!(cpu.regs[10], 10, "Register x10 should be 10");
    //         assert_eq!(cpu.regs[20], 5, "Register x20 should be 5");
    //         assert_eq!(cpu.regs[30], 0, "Register x30 should still be 0 (skipped instruction)");
    //         // assert_eq!(cpu.regs[31], 42, "Register x31 should be 42");
    //         assert_eq!(next_pc, 20, "Next PC should be 20 after branch");
    //     }

    //

    //     #[test]
    //     fn test_sra_instruction() {
    //         // Define a program with SRA instruction (shift right arithmetic - sign extended)
    //         // 1. Set x10 to -8 (negative value)
    //         // 2. Set x20 to 1 (shift amount)
    //         // 3. SRA x30, x10, x20 (shift right arithmetic)
    //         let test_program: [u8; 12] = [
    //             0x13, 0x05, 0x80, 0xff,     // addi x10, x0, -8 (0xfffffff8)
    //             0x13, 0x0a, 0x10, 0x00,       // addi x20, x0, 1
    //             0x33, 0x5e, 0x45, 0x41,     // sra x28, x10, x20
    //         ];

    //         let (cpu, _, insns, next_pc) = setup_test_env(&test_program);

    //         // Verify the results
    //         assert_eq!(insns, 3, "Should have translated all 3 instructions");
    //         assert_eq!(cpu.regs[10] as i32, -8, "Register x10 should be -8");
    //         assert_eq!(cpu.regs[20], 1, "Register x20 should be 1");
    //         assert_eq!(cpu.regs[28] as i32, -4, "SRA should shift right with sign extension (-8 >> 1 = -4)");
    //         assert_eq!(next_pc, 12, "Next PC should be 12 after branch");
    //     }

    //     #[test]
    //     fn test_sltu_instruction() {
    //         // Define a program with SLTU instruction (set less than unsigned)
    //         // 1. Set x10 to -1 (unsigned max value 0xFFFFFFFF)
    //         // 2. Set x20 to 10 (small positive value)
    //         // 3. SLTU x28, x10, x20 (should be 0 as 0xFFFFFFFF > 10 unsigned)
    //         let test_program = [
    //             0x13, 0x05, 0xf0, 0xff,     // addi x10, x0, -1 (0xFFFFFFFF)
    //             0x13, 0x0a, 0xa0, 0x00,     // addi x20, x0, 10
    //             0x33, 0x3e, 0x45, 0x00,     // sltu x28, x10, x20
    //         ];

    //         let (cpu, _, insns, next_pc) = setup_test_env(&test_program);

    //         // Verify the results
    //         // Set if x10 < x20 in unsigned comparison, which is false (0xFFFFFFFF > 10)
    //         assert_eq!(insns, 3, "Should have translated all 3 instructions");
    //         assert_eq!(cpu.regs[10] as i32, -1, "Register x10 should be -1 (0xFFFFFFFF)");
    //         assert_eq!(cpu.regs[20], 10, "Register x20 should be 10");
    //         assert_eq!(cpu.regs[28], 0, "SLTU should set x28 to 0 (0xFFFFFFFF > 10)");
    //         assert_eq!(next_pc, 12, "Next PC should be 12 after execution");
    //     }

    //     #[test]
    //     fn test_mulh_instruction() {
    //         // Define a program with MULH instruction (multiply high signed*signed)
    //         // 1. Set x10 to 0x7FFFFFFF (max signed int)
    //         // 2. Set x20 to 0x7FFFFFFF (max signed int)
    //         // 3. MULH x28, x10, x20 (get high bits of signed multiplication)
    //         let test_program = [
    //             0x93, 0x05, 0xF0, 0x01,   // addi x11, x0, 31
    //             0x13, 0x05, 0x10, 0x00,   // addi x10, x0, 1
    //             0x33, 0x15, 0xB5, 0x00,   // sll  x10, x10, x11
    //             0x13, 0x05, 0xF5, 0xFF,   // addi x10, x10, -1
    //             0x33, 0x0A, 0x05, 0x00,   // add  x20, x10, x0
    //             0x33, 0x1E, 0x45, 0x03,   // mulh x28, x10, x20
    //         ];

    //         let (cpu, _, insns, next_pc) = setup_test_env(&test_program);

    //         // 0x7FFFFFFF * 0x7FFFFFFF = 0x3FFFFFFF00000001
    //         // High 32 bits = 0x3FFFFFFF
    //         println!("{:?}", cpu.regs);
    //         assert_eq!(insns, 6, "Should have translated all 6 instructions");
    //         assert_eq!(cpu.regs[10] as i32, 0x7FFFFFFF, "Register x10 should be 0x7FFFFFFF");
    //         assert_eq!(cpu.regs[20] as i32, 0x7FFFFFFF, "Register x20 should be 0x7FFFFFFF");
    //         assert_eq!(cpu.regs[28] as i32, 0x3FFFFFFF, "MULH should get high 32 bits of signed multiplication");
    //         assert_eq!(next_pc, 24, "Next PC should be 24 after branch")
    //     }

    //     #[test]
    //     fn test_mulhu_instruction() {
    //         // Define a program with MULHU instruction (multiply high unsigned*unsigned)
    //         // 1. Set x10 to 0x7FFFFFFF (max signed int)
    //         // 2. Set x20 to 0x7FFFFFFF (max signed int)
    //         // 3. MULH x28, x10, x20 (get high bits of signed multiplication)

    //         let test_program = [
    //             0x93, 0x05, 0xF0, 0x01,   // addi x11, x0, 31
    //             0x13, 0x05, 0x10, 0x00,   // addi x10, x0, 1
    //             0x33, 0x15, 0xB5, 0x00,   // sll  x10, x10, x11
    //             0x13, 0x05, 0xF5, 0xFF,   // addi x10, x10, -1
    //             0x33, 0x0A, 0x05, 0x00,   // add  x20, x10, x0
    //             0x33, 0x2e, 0x45, 0x03,   // mulhu x28, x10, x20
    //         ];

    //         let (cpu, _, insns, next_pc) = setup_test_env(&test_program);

    //         // 0x7FFFFFFF * 0x7FFFFFFF = 0x3FFFFFFF00000001
    //         // High 32 bits = 0x3FFFFFFF
    //         assert_eq!(insns, 6, "Should have translated all 6 instructions");
    //         assert_eq!(cpu.regs[10] as i32, 0x7FFFFFFF, "Register x10 should be 0x7FFFFFFF");
    //         assert_eq!(cpu.regs[20] as i32, 0x7FFFFFFF, "Register x20 should be 0x7FFFFFFF");
    //         assert_eq!(cpu.regs[28] as i32, 0x3FFFFFFF, "MULH should get high 32 bits of signed multiplication");
    //         assert_eq!(next_pc, 24, "Next PC should be 24 after branch")
    //     }

    //     #[test]
    //     fn test_mulhsu_instruction() {
    //         // Define a program with MULHSU instruction (multiply high signed*unsigned)
    //         // 1. Set x10 to -1 (signed)
    //         // 2. Set x20 to -1 (unsigned 0xFFFFFFFF)
    //         // 3. MULHSU x28, x10, x20
    //         let test_program = [
    //             0x13, 0x05, 0xf0, 0xff,     // addi x10, x0, -1
    //             0x13, 0x0a, 0xf0, 0xff,     // addi x20, x0, -1 (as unsigned: 0xFFFFFFFF)
    //             0x33, 0x2e, 0x45, 0x03,     // mulhsu x28, x10, x20
    //         ];

    //         let (cpu, _, insns, next_pc) = setup_test_env(&test_program);

    //         println!("{:?}", cpu.regs);
    //         // -1 * 0xFFFFFFFF (signed*unsigned) = 0xFFFFFFFF00000001
    //         // High 32 bits = 0xFFFFFFFF
    //         assert_eq!(insns, 3, "Should have translated all 3 instructions");
    //         assert_eq!(cpu.regs[10] as i32, -1, "Register x10 should be -1");
    //         assert_eq!(cpu.regs[20], 0xFFFFFFFF, "Register x20 should be 0xFFFFFFFF");
    //         assert_eq!(cpu.regs[28], 0xFFFFFFFF, "MULHSU should get high 32 bits of signed*unsigned multiplication");
    //         assert_eq!(next_pc, 12, "Next PC should be 12 after execution");
    //     }

        #[test]
        fn test_div_instruction() {
            // main:
            //     addi x10, x0, -10
            //     addi x20, x0, 3
            //     div  x28, x10, x20     # -10 / 3 = -3
            let instructions = vec![
                // addi x10, x0, -10
                Instruction::new(Opcode::ADD, 10, 0, (-10i32) as u32, false, true),
                // addi x20, x0, 3
                Instruction::new(Opcode::ADD, 20, 0, 3, false, true),
                // div x28, x10, x20
                Instruction::new(Opcode::DIV, 28, 10, 20, false, false),
            ];

            let program = Program::new(instructions, 0, 0);
            let mut runtime = Executor::new(program);
            let (insns, next_pc) = setup_test_env_with_cpu(&mut runtime);

            // -10 / 3 = -3 (truncated toward zero)
            assert_eq!(insns, 3, "Should have translated all 3 instructions");
            assert_eq!(runtime.register(Register::X10) as i32, -10, "Register x10 should be -10");
            assert_eq!(runtime.register(Register::X20), 3, "Register x20 should be 3");
            assert_eq!(runtime.register(Register::X28) as i32, -3, "DIV should perform signed division");
            assert_eq!(next_pc, 12, "Next PC should be 12 after execution");
        }

            // #[test]
            // fn test_divu_instruction() {
            //     // Define a program with DIVU instruction (unsigned division)
            //     // 1. Set x10 to -1 (0xFFFFFFFF unsigned)
            //     // 2. Set x20 to 10
            //     // 3. DIVU x28, x10, x20
            //     let test_program = [
            //         0x13, 0x05, 0xf0, 0xff,     // addi x10, x0, -1 (0xFFFFFFFF unsigned)
            //         0x13, 0x0a, 0xa0, 0x00,     // addi x20, x0, 10
            //         0x33, 0x5e, 0x45, 0x03,     // divu x28, x10, x20
            //     ];

            //     // let (cpu, _, insns, next_pc) = setup_test_env(&test_program);

            //     // 0xFFFFFFFF / 10 = 429496729 (unsigned)
            //     assert_eq!(cpu.regs[28], 429496729, "DIVU should perform unsigned division");
            // }

            // #[test]
            // fn test_rem_instruction() {
            //     // Define a program with REM instruction (signed remainder)
            //     // 1. Set x10 to -10
            //     // 2. Set x20 to 3
            //     // 3. REM x28, x10, x20 (result: -1)
            //     let test_program = [
            //         0x13, 0x05, 0x60, 0xff,     // addi x10, x0, -10
            //         0x13, 0x0a, 0x30, 0x00,     // addi x20, x0, 3
            //         0x33, 0x6e, 0x45, 0x03,     // rem x28, x10, x20
            //     ];

            //     // let (cpu, _, insns, next_pc) = setup_test_env(&test_program);

            //     // -10 % 3 = -1
            //     assert_eq!(cpu.regs[28] as i32, -1, "REM should perform signed remainder");
            //     assert_eq!(insns, 3, "Should have translated all 3 instructions");
            //     assert_eq!(next_pc, 12, "Next PC should be 12 after execution");
            // }

            // #[test]
            // fn test_remu_instruction() {
            //     // Define a program with REMU instruction (unsigned remainder)        // 1. Set x10 to -1 (0xFFFFFFFF unsigned)
            //     // 2. Set x20 to 10
            //     // 3. REMU x28, x10, x20
            //     let test_program = [
            //         0x13, 0x05, 0xf0, 0xff,     // addi x10, x0, -1 (0xFFFFFFFF unsigned)
            //         0x13, 0x0a, 0xa0, 0x00,     // addi x20, x0, 10
            //         0x33, 0x7e, 0x45, 0x03,     // remu x28, x10, x20
            //     ];

            //     // let (cpu, _, insns, next_pc) = setup_test_env(&test_program);

            //     // 0xFFFFFFFF % 10 = 5 (unsigned)
            //     assert_eq!(insns, 3, "Should have translated all 3 instructions");
            //     assert_eq!(cpu.regs[10] as i32, -1, "Register x10 should be 0xFFFFFFFF");
            //     assert_eq!(cpu.regs[20], 10, "Register x20 should be 10");
            //     assert_eq!(cpu.regs[28], 5, "REMU should perform unsigned remainder");
            //     assert_eq!(next_pc, 12, "Next PC should be 12 after execution");
            // }

    //     #[test]
    //     fn test_ecall_instruction() {
    //         // ECALL instruction should terminate a translation block
    //         // and return a special value
    //         let test_program = [
    //             0x73, 0x00, 0x00, 0x00,     // ecall
    //             0x13, 0x0f, 0xa0, 0x02,     // addi x30, x0, 42 (should not execute)
    //         ];

    //         let (cpu, _, insns, next_pc) = setup_test_env(&test_program);

    //         assert_eq!(insns, 1, "ECALL should terminate after 1 instruction");
    //         assert_eq!(cpu.regs[30], 0, "Register x30 should not be modified");
    //         assert_eq!(next_pc, 0xECA11, "ECALL should return special indicator value");
    //     }

    //     #[test]
    //     fn test_ebreak_instruction() {
    //         // EBREAK instruction should terminate a translation block
    //         // and return a special value
    //         let test_program = [
    //             0x73, 0x00, 0x10, 0x00,     // ebreak
    //             0x13, 0x0f, 0xa0, 0x02,     // addi x30, x0, 42 (should not execute)
    //         ];

    //         let (cpu, _, insns, next_pc) = setup_test_env(&test_program);

    //         assert_eq!(insns, 1, "EBREAK should terminate after 1 instruction");
    //         assert_eq!(cpu.regs[30], 0, "Register x30 should not be modified");
    //         assert_eq!(next_pc, 0xEB8EA, "EBREAK should return special indicator value");
    //     }
}
