use std::ptr::null;

use crate::{store_registers_to_cpu, Opcode};
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
    memory::call_mem_load,
    register, ExecutionMode, ExecutionState, Executor, Instruction,
};

pub fn try_to_compile_tb_and_populate_slow_cache<'a>(
    executor: &mut Executor<'a>, // TODO reduce Executor down to necessary attributes
    // state: &mut ExecutionState,
    jit_wrapper: &mut JITWrapper,
    _slow_tb_cache: &mut SlowTBCache,
) -> ExecutionMode {
    let _ = compile_tb(executor, /*state,*/ jit_wrapper, 16);

    ExecutionMode::Emulator
}

pub fn compile_tb<'a>(
    executor: &mut Executor<'a>,
    /*state: &mut ExecutionState,*/ jit_wrapper: &mut JITWrapper,
    max_insns: usize,
) -> (*const u8, usize) {
    let jit = &mut jit_wrapper.jit;

    let mut ctx = jit.make_context();
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
    while cnt < max_insns {
        println!("cnt: {}", cnt);
        let inst = executor.fetch_at(pc);
        println!("{:?}", inst);
        match inst.opcode {
            Opcode::ADD => {
                let Instruction {
                    op_a, op_b, op_c, ..
                } = inst;
                if inst.is_addi_instruction() {
                    compile_addi(
                        &mut b,
                        register_file_ptr,
                        &mut regs_read_or_changed_so_far,
                        &mut dirty_regs,
                        &regs,
                        op_b,
                        op_c,
                        op_a,
                    );
                    println!("processing ADDI dirty {:?}", dirty_regs);
                } else {
                    let (v1, v2) = preload_for_bin_op(
                        &mut b,
                        register_file_ptr,
                        &mut regs_read_or_changed_so_far,
                        &mut dirty_regs,
                        &regs,
                        op_b,
                        op_c,
                    );

                    let v = b.ins().iadd(v1, v2);

                    define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, op_a, v);
                    println!("ADD dirty {:?}", dirty_regs);
                }
            }
            Opcode::SUB => {
                let Instruction {
                    op_a, op_b, op_c, ..
                } = inst;
                let (v1, v2) = preload_for_bin_op(
                    &mut b,
                    register_file_ptr,
                    &mut regs_read_or_changed_so_far,
                    &mut dirty_regs,
                    &regs,
                    op_b,
                    op_c,
                );

                let v = b.ins().isub(v1, v2);

                define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, op_a, v);
            }
            Opcode::XOR => {
                let Instruction {
                    op_a, op_b, op_c, ..
                } = inst;
                let (v1, v2) = preload_for_bin_op(
                    &mut b,
                    register_file_ptr,
                    &mut regs_read_or_changed_so_far,
                    &mut dirty_regs,
                    &regs,
                    op_b,
                    op_c,
                );

                let v = b.ins().bxor(v1, v2);

                define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, op_a, v);
            }
            Opcode::OR => {
                let Instruction {
                    op_a, op_b, op_c, ..
                } = inst;
                let (v1, v2) = preload_for_bin_op(
                    &mut b,
                    register_file_ptr,
                    &mut regs_read_or_changed_so_far,
                    &mut dirty_regs,
                    &regs,
                    op_b,
                    op_c,
                );

                let v = b.ins().bor(v1, v2);

                define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, op_a, v);
            }
            Opcode::AND => {
                let Instruction {
                    op_a, op_b, op_c, ..
                } = inst;
                let (v1, v2) = preload_for_bin_op(
                    &mut b,
                    register_file_ptr,
                    &mut regs_read_or_changed_so_far,
                    &mut dirty_regs,
                    &regs,
                    op_b,
                    op_c,
                );

                let v = b.ins().band(v1, v2);

                define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, op_a, v);
            }
            Opcode::SLL => {
                let Instruction {
                    op_a, op_b, op_c, ..
                } = inst;
                let (v1, v2) = preload_for_bin_op(
                    &mut b,
                    register_file_ptr,
                    &mut regs_read_or_changed_so_far,
                    &mut dirty_regs,
                    &regs,
                    op_b,
                    op_c,
                );

                let v = b.ins().ishl(v1, v2);

                define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, op_a, v);
            }
            Opcode::SRL => {
                let Instruction { op_a, op_b, op_c, .. } = inst;
                let (v1, v2) = preload_for_bin_op(
                    &mut b,
                    register_file_ptr,
                    &mut regs_read_or_changed_so_far,
                    &mut dirty_regs,
                    &regs,
                    op_b,
                    op_c,
                );

                let v = b.ins().ushr(v1, v2);

                define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, op_a, v);
            }
            Opcode::SRA => {
                let Instruction { op_a, op_b, op_c, .. } = inst;
                let (v1, v2) = preload_for_bin_op(
                    &mut b,
                    register_file_ptr,
                    &mut regs_read_or_changed_so_far,
                    &mut dirty_regs,
                    &regs,
                    op_b,
                    op_c,
                );

                let v = b.ins().sshr(v1, v2); // Arithmetic (signed) shift right

                define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, op_a, v);
            }
            Opcode::SLT => {
                let Instruction { op_a, op_b, op_c, .. } = inst;
                let (v1, v2) = preload_for_bin_op(
                        &mut b,
                        register_file_ptr,
                        &mut regs_read_or_changed_so_far,
                        &mut dirty_regs,
                        &regs,
                        op_b,
                        op_c,
                    );

                    // Use icmp_slt to compare if v1 < v2 (signed comparison)
                    let cond = b.ins().icmp(IntCC::SignedLessThan, v1, v2);
                    let zero = b.ins().iconst(types::I32, 0);
                    let one = b.ins().iconst(types::I32, 1);
                    let v = b.ins().select(cond, one, zero);

                    define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, op_a, v);
            }
            Opcode::SLTU => {
                let Instruction { op_a, op_b, op_c, .. } = inst;
                let (v1, v2) = preload_for_bin_op(
                    &mut b,
                    register_file_ptr,
                    &mut regs_read_or_changed_so_far,
                    &mut dirty_regs,
                    &regs,
                    op_b,
                    op_c,
                );

                // Using SLTU for unsigned comparison
                let cond = b.ins().icmp(IntCC::UnsignedLessThan, v1, v2);
                let zero = b.ins().iconst(types::I32, 0);
                let one = b.ins().iconst(types::I32, 1);
                let v = b.ins().select(cond, one, zero);

                define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, op_a, v);
            }
            // Opcode::LB => {
            //     let Instruction { op_a, op_b, op_c, .. } = inst;
            //     let reg_idx = op_b.try_into().unwrap();
            //     let op_b = load_reg_if_needed_and_not_dirty(
            //         &mut b,
            //         register_file_ptr,
            //         reg_idx,
            //         &mut regs_read_or_changed_so_far,
            //         &mut dirty_regs,
            //         &regs,
            //     );

            //     let base = b.use_var(regs[op_b]);
            //     let imm = op_c as i64;
            //     let addr = b.ins().iadd_imm(base, imm);

            //     // Call load32 (we don't have separate load8)
            //     let val = call_mem_load(jit, &mut b, memory_ptr, addr);

            //     // Extract the byte and sign-extend it to 32 bits
            //     let shift_amount = b.ins().iconst(types::I32, 24);
            //     let byte_val = b.ins().ishl(val, shift_amount);
            //     let signed_val = b.ins().sshr(byte_val, shift_amount);

            //     define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, op_a, signed_val);
            // }
            //     OpcodeKind::BaseI(BaseIOpcode::LH) => {
            //         let Instruction { rd, rs1, imm, .. } = inst;
            //         let rs1 = load_reg_if_needed_and_not_dirty(
            //             &mut b,
            //             cpu_ptr,
            //             rs1.unwrap(),
            //             &mut regs_read_or_changed_so_far,
            //             &mut dirty_regs,
            //             &regs,
            //         );

            //         let base = b.use_var(regs[rs1]);
            //         let addr = b.ins().iadd_imm(base, imm.unwrap() as i64);

            //         // Call load32 (we don't have separate load16)
            //         let val = call_mem_load(jit, &mut b, cpu_ptr, addr);

            //         // Extract the halfword and sign-extend it to 32 bits
            //         let shift_amount = b.ins().iconst(types::I32, 16);
            //         let hw_val = b.ins().ishl(val, shift_amount);
            //         let signed_val = b.ins().sshr(hw_val, shift_amount);

            //         define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, rd, signed_val);
            //     }
            //     OpcodeKind::BaseI(BaseIOpcode::LBU) => {
            //         let Instruction { rd, rs1, imm, .. } = inst;
            //         let rs1 = load_reg_if_needed_and_not_dirty(
            //             &mut b,
            //             cpu_ptr,
            //             rs1.unwrap(),
            //             &mut regs_read_or_changed_so_far,
            //             &mut dirty_regs,
            //             &regs,
            //         );

            //         let base = b.use_var(regs[rs1]);
            //         let addr = b.ins().iadd_imm(base, imm.unwrap() as i64);

            //         // Call load32 (we don't have separate load8)
            //         let val = call_mem_load(jit, &mut b, cpu_ptr, addr);

            //         // Extract the byte (unsigned)
            //         let mask = b.ins().iconst(types::I32, 0xFF);
            //         let unsigned_val = b.ins().band(val, mask);

            //         define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, rd, unsigned_val);
            //     }
            //     OpcodeKind::BaseI(BaseIOpcode::LHU) => {
            //         let Instruction { rd, rs1, imm, .. } = inst;
            //         let rs1 = load_reg_if_needed_and_not_dirty(
            //             &mut b,
            //             cpu_ptr,
            //             rs1.unwrap(),
            //             &mut regs_read_or_changed_so_far,
            //             &mut dirty_regs,
            //             &regs,
            //         );

            //         let base = b.use_var(regs[rs1]);
            //         let addr = b.ins().iadd_imm(base, imm.unwrap() as i64);

            //         // Call load32 (we don't have separate load16)
            //         let val = call_mem_load(jit, &mut b, cpu_ptr, addr);

            //         // Extract the halfword (unsigned)
            //         let mask = b.ins().iconst(types::I32, 0xFFFF);
            //         let unsigned_val = b.ins().band(val, mask);

            //         define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, rd, unsigned_val);
            //     }
            Opcode::LW => {
                // let (rd, rs1, imm) = instruction.i_type();
                let Instruction { op_a, op_b, op_c, .. } = inst;

                let reg_idx = op_b.try_into().unwrap();
                let op_b = load_reg_if_needed_and_not_dirty(
                    &mut b,
                    register_file_ptr,
                    reg_idx,
                    &mut regs_read_or_changed_so_far,
                    &mut dirty_regs,
                    &regs,
                );

                let base = b.use_var(regs[op_b]);
                let addr = b.ins().iadd_imm(base, op_c as i64);

                let val = call_mem_load(jit, &mut b, memory_ptr, addr);

                define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, op_a, val);
            }
            //     OpcodeKind::BaseI(BaseIOpcode::SB) => {
            //         let Instruction { rs2, rs1, imm, .. } = inst;
            //         let (rs1, rs2) = load_two_regs(
            //             &mut b,
            //             cpu_ptr,
            //             &regs,
            //             &mut regs_read_or_changed_so_far,
            //             &mut dirty_regs,
            //             rs1,
            //             rs2,
            //         );

            //         let base = b.use_var(regs[rs1]);
            //         let addr = b.ins().iadd_imm(base, imm.unwrap() as i64);
            //         let val = b.use_var(regs[rs2]);

            //         call_mem_store(jit, &mut b, cpu_ptr, addr, val);
            //     }
            //     OpcodeKind::BaseI(BaseIOpcode::SH) => {
            //         let Instruction { rs2, rs1, imm, .. } = inst;
            //         let (rs1, rs2) = load_two_regs(
            //             &mut b,
            //             cpu_ptr,
            //             &regs,
            //             &mut regs_read_or_changed_so_far,
            //             &mut dirty_regs,
            //             rs1,
            //             rs2,
            //         );

            //         let base = b.use_var(regs[rs1]);
            //         let addr = b.ins().iadd_imm(base, imm.unwrap() as i64);
            //         let val = b.use_var(regs[rs2]);

            //         call_mem_store(jit, &mut b, cpu_ptr, addr, val);
            //     }
            //     OpcodeKind::BaseI(BaseIOpcode::SW) => {
            //         let Instruction { rs2, rs1, imm, .. } = inst;
            //         let (rs1, rs2) = load_two_regs(
            //             &mut b,
            //             cpu_ptr,
            //             &regs,
            //             &mut regs_read_or_changed_so_far,
            //             &mut dirty_regs,
            //             rs1,
            //             rs2,
            //         );

            //         let base = b.use_var(regs[rs1]);
            //         let addr = b.ins().iadd_imm(base, imm.unwrap() as i64);
            //         let val = b.use_var(regs[rs2]);

            //         call_mem_store(jit, &mut b, cpu_ptr, addr, val);
            //     }
            //     OpcodeKind::BaseI(BaseIOpcode::JALR) => {
            //         let Instruction { rd, rs1, imm, .. } = inst;
            //         let target = b.use_var(regs[rs1.unwrap()]);
            //         let next = b.ins().iadd_imm(target, imm.unwrap() as i64);
            //         let const_pc = b.ins().iconst(types::I32, (pc + 4) as i64);
            //         define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, rd, const_pc);

            //         // quit early after J-instruction
            //         store_registers_to_cpu(&mut b, cpu_ptr, &regs, &dirty_regs);

            //         b.ins().return_(&[next]);
            //         term_was_added = true;
            //         cnt += 1;

            //         break;
            //     }
            //     OpcodeKind::BaseI(BaseIOpcode::BEQ) => {
            //         // Branch if equal - terminal instruction
            //         let Instruction { rs1, rs2, imm, .. } = inst;
            //         let (rs1, rs2) = load_two_regs(
            //             &mut b,
            //             cpu_ptr,
            //             &regs,
            //             &mut regs_read_or_changed_so_far,
            //             &mut dirty_regs,
            //             rs1,
            //             rs2,
            //         );

            //         let v1 = b.use_var(regs[rs1]);
            //         let v2 = b.use_var(regs[rs2]);

            //         // Compare rs1 and rs2
            //         let cond = b.ins().icmp(IntCC::Equal, v1, v2);

            //         // Calculate target and fallthrough addresses
            //         // Calculate target address (pc + offset) ensuring proper casting
            //         pc += 4;
            //         let target_pc = b
            //             .ins()
            //             .iconst(types::I32, (pc as i64) + (imm.unwrap() as i64));
            //         let fallthrough_pc = b.ins().iconst(types::I32, (pc + 4) as i64);

            //         // Select which PC to branch to based on condition
            //         let next_pc = b.ins().select(cond, target_pc, fallthrough_pc);

            //         // Terminate the current translation block
            //         store_registers_to_cpu(&mut b, cpu_ptr, &regs, &dirty_regs);
            //         b.ins().return_(&[next_pc]);
            //         term_was_added = true;
            //         cnt += 1;
            //         break;
            //     }
            //     OpcodeKind::BaseI(BaseIOpcode::BNE) => {
            //         // Branch if not equal - terminal instruction
            //         let Instruction { rs1, rs2, imm, .. } = inst;
            //         let (rs1, rs2) = load_two_regs(
            //             &mut b,
            //             cpu_ptr,
            //             &regs,
            //             &mut regs_read_or_changed_so_far,
            //             &mut dirty_regs,
            //             rs1,
            //             rs2,
            //         );

            //         let v1 = b.use_var(regs[rs1]);
            //         let v2 = b.use_var(regs[rs2]);

            //         // Compare rs1 and rs2
            //         let cond = b.ins().icmp(IntCC::NotEqual, v1, v2);

            //         pc += 4;

            //         // Calculate target and fallthrough addresses
            //         // Calculate target address (pc + offset) ensuring proper casting
            //         let target_pc = b
            //             .ins()
            //             .iconst(types::I32, (pc as i64) + (imm.unwrap() as i64));
            //         let fallthrough_pc = b.ins().iconst(types::I32, (pc + 4) as i64);

            //         // Select which PC to branch to based on condition
            //         let next_pc = b.ins().select(cond, target_pc, fallthrough_pc);

            //         // Terminate the current translation block
            //         store_registers_to_cpu(&mut b, cpu_ptr, &regs, &dirty_regs);
            //         b.ins().return_(&[next_pc]);
            //         term_was_added = true;
            //         cnt += 1;
            //         break;
            //     }
            //     OpcodeKind::BaseI(BaseIOpcode::BLT) => {
            //         // Branch if less than - terminal instruction
            //         let Instruction { rs1, rs2, imm, .. } = inst;
            //         let (rs1, rs2) = load_two_regs(
            //             &mut b,
            //             cpu_ptr,
            //             &regs,
            //             &mut regs_read_or_changed_so_far,
            //             &mut dirty_regs,
            //             rs1,
            //             rs2,
            //         );

            //         let v1 = b.use_var(regs[rs1]);
            //         let v2 = b.use_var(regs[rs2]);

            //         // Compare rs1 < rs2 (signed)
            //         let cond = b.ins().icmp(IntCC::SignedLessThan, v1, v2);

            //         pc += 4;

            //         // Calculate target and fallthrough addresses
            //         // Calculate target address (pc + offset) ensuring proper casting
            //         let target_pc = b
            //             .ins()
            //             .iconst(types::I32, (pc as i64) + (imm.unwrap() as i64));
            //         let fallthrough_pc = b.ins().iconst(types::I32, (pc + 4) as i64);

            //         // Select which PC to branch to based on condition
            //         let next_pc = b.ins().select(cond, target_pc, fallthrough_pc);

            //         // Terminate the current translation block
            //         store_registers_to_cpu(&mut b, cpu_ptr, &regs, &dirty_regs);
            //         b.ins().return_(&[next_pc]);
            //         term_was_added = true;
            //         cnt += 1;
            //         break;
            //     }
            //     OpcodeKind::BaseI(BaseIOpcode::BGE) => {
            //         // Branch if greater than or equal - terminal instruction
            //         let Instruction { rs1, rs2, imm, .. } = inst;
            //         let (rs1, rs2) = load_two_regs(
            //             &mut b,
            //             cpu_ptr,
            //             &regs,
            //             &mut regs_read_or_changed_so_far,
            //             &mut dirty_regs,
            //             rs1,
            //             rs2,
            //         );

            //         let v1 = b.use_var(regs[rs1]);
            //         let v2 = b.use_var(regs[rs2]);

            //         // Compare rs1 >= rs2 (signed)
            //         let cond = b.ins().icmp(IntCC::SignedGreaterThanOrEqual, v1, v2);

            //         // Calculate target and fallthrough addresses
            //         // Calculate target address (pc + offset) ensuring proper casting
            //         pc += 4;

            //         let target_pc = b
            //             .ins()
            //             .iconst(types::I32, (pc as i64) + (imm.unwrap() as i64));
            //         let fallthrough_pc = b.ins().iconst(types::I32, (pc + 4) as i64);

            //         // Select which PC to branch to based on condition
            //         let next_pc = b.ins().select(cond, target_pc, fallthrough_pc);

            //         // Terminate the current translation block
            //         store_registers_to_cpu(&mut b, cpu_ptr, &regs, &dirty_regs);
            //         b.ins().return_(&[next_pc]);
            //         term_was_added = true;
            //         cnt += 1;
            //         break;
            //     }
            //     OpcodeKind::BaseI(BaseIOpcode::BLTU) => {
            //         // Branch if less than unsigned - terminal instruction
            //         let Instruction { rs1, rs2, imm, .. } = inst;
            //         let (rs1, rs2) = load_two_regs(
            //             &mut b,
            //             cpu_ptr,
            //             &regs,
            //             &mut regs_read_or_changed_so_far,
            //             &mut dirty_regs,
            //             rs1,
            //             rs2,
            //         );

            //         let v1 = b.use_var(regs[rs1]);
            //         let v2 = b.use_var(regs[rs2]);

            //         // Compare rs1 < rs2 (unsigned)
            //         let cond = b.ins().icmp(IntCC::UnsignedLessThan, v1, v2);

            //         pc += 4;

            //         // Calculate target and fallthrough addresses
            //         // Calculate target address (pc + offset) ensuring proper casting
            //         let target_pc = b
            //             .ins()
            //             .iconst(types::I32, (pc as i64) + (imm.unwrap() as i64));
            //         let fallthrough_pc = b.ins().iconst(types::I32, (pc + 4) as i64);

            //         // Select which PC to branch to based on condition
            //         let next_pc = b.ins().select(cond, target_pc, fallthrough_pc);

            //         // Terminate the current translation block
            //         store_registers_to_cpu(&mut b, cpu_ptr, &regs, &dirty_regs);
            //         b.ins().return_(&[next_pc]);
            //         term_was_added = true;
            //         cnt += 1;
            //         break;
            //     }
            //     OpcodeKind::BaseI(BaseIOpcode::BGEU) => {
            //         // Branch if greater than or equal unsigned - terminal instruction
            //         let Instruction { rs1, rs2, imm, .. } = inst;
            //         let (rs1, rs2) = load_two_regs(
            //             &mut b,
            //             cpu_ptr,
            //             &regs,
            //             &mut regs_read_or_changed_so_far,
            //             &mut dirty_regs,
            //             rs1,
            //             rs2,
            //         );

            //         let v1 = b.use_var(regs[rs1]);
            //         let v2 = b.use_var(regs[rs2]);

            //         // Compare rs1 >= rs2 (unsigned)
            //         let cond = b.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, v1, v2);

            //         pc += 4;

            //         // Calculate target and fallthrough addresses
            //         // Calculate target address (pc + offset) ensuring proper casting
            //         let target_pc = b
            //             .ins()
            //             .iconst(types::I32, (pc as i64) + (imm.unwrap() as i64));
            //         let fallthrough_pc = b.ins().iconst(types::I32, (pc + 4) as i64);

            //         // Select which PC to branch to based on condition
            //         let next_pc = b.ins().select(cond, target_pc, fallthrough_pc);

            //         // Terminate the current translation block
            //         store_registers_to_cpu(&mut b, cpu_ptr, &regs, &dirty_regs);
            //         b.ins().return_(&[next_pc]);
            //         term_was_added = true;
            //         cnt += 1;
            //         break;
            //     }
            //     OpcodeKind::BaseI(BaseIOpcode::JAL) => {
            //         // Jump and link - terminal instruction
            //         let Instruction { rd, imm, .. } = inst;

            //         // Store return address (PC+4) in rd
            //         let return_addr = b.ins().iconst(types::I32, (pc + 4) as i64);
            //         define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, rd, return_addr);

            //         // Calculate target address
            //         let target_pc = b
            //             .ins()
            //             .iconst(types::I32, (pc as i64) + (imm.unwrap() as i64));

            //         // Terminate the current translation block and jump to target
            //         store_registers_to_cpu(&mut b, cpu_ptr, &regs, &dirty_regs);
            //         b.ins().return_(&[target_pc]);
            //         term_was_added = true;
            //         cnt += 1;
            //         break;
            //     }
            Opcode::AUIPC => {
                // Add Upper Immediate to PC
                let (rd, imm) = inst.u_type();
                let rd = rd as u32;
                let shifted_imm: i64 = imm.into();
                let result = b
                    .ins()
                    .iconst(types::I32, (pc as i64) + (shifted_imm));

                define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, rd, result);
            }
            //     OpcodeKind::M(raki::MOpcode::MUL) => {
            //         let Instruction { rd, rs1, rs2, .. } = inst;
            //         let (rs1, rs2) = load_two_regs(
            //             &mut b,
            //             cpu_ptr,
            //             &regs,
            //             &mut regs_read_or_changed_so_far,
            //             &mut dirty_regs,
            //             rs1,
            //             rs2,
            //         );

            //         let v1 = b.use_var(regs[rs1]);
            //         let v2 = b.use_var(regs[rs2]);
            //         let v = b.ins().imul(v1, v2);

            //         define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, rd, v);
            //     }
            //     OpcodeKind::M(raki::MOpcode::MULH) => {
            //         // Multiply high (signed x signed)
            //         let Instruction { rd, rs1, rs2, .. } = inst;
            //         let (rs1, rs2) = load_two_regs(
            //             &mut b,
            //             cpu_ptr,
            //             &regs,
            //             &mut regs_read_or_changed_so_far,
            //             &mut dirty_regs,
            //             rs1,
            //             rs2,
            //         );

            //         let v1 = b.use_var(regs[rs1]);
            //         let v2 = b.use_var(regs[rs2]);

            //         // We need to cast to i64, multiply, then get the high 32 bits
            //         let v1_64 = b.ins().sextend(types::I64, v1);
            //         let v2_64 = b.ins().sextend(types::I64, v2);
            //         let mul_result = b.ins().imul(v1_64, v2_64);

            //         // Shift right by 32 to get the high bits
            //         let shift_amt = b.ins().iconst(types::I64, 32);
            //         let high_bits = b.ins().ushr(mul_result, shift_amt);

            //         // Truncate back to 32 bits
            //         let result_32 = b.ins().ireduce(types::I32, high_bits);

            //         define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, rd, result_32);
            //     }
            //     OpcodeKind::M(raki::MOpcode::MULHU) => {
            //         // Multiply high (unsigned x unsigned)
            //         let Instruction { rd, rs1, rs2, .. } = inst;
            //         let (rs1, rs2) = load_two_regs(
            //             &mut b,
            //             cpu_ptr,
            //             &regs,
            //             &mut regs_read_or_changed_so_far,
            //             &mut dirty_regs,
            //             rs1,
            //             rs2,
            //         );

            //         let v1 = b.use_var(regs[rs1]);
            //         let v2 = b.use_var(regs[rs2]);

            //         // We need to cast to u64, multiply, then get the high 32 bits
            //         let v1_64 = b.ins().uextend(types::I64, v1);
            //         let v2_64 = b.ins().uextend(types::I64, v2);
            //         let mul_result = b.ins().imul(v1_64, v2_64);

            //         // Shift right by 32 to get the high bits
            //         let shift_amt = b.ins().iconst(types::I64, 32);
            //         let high_bits = b.ins().ushr(mul_result, shift_amt);

            //         // Truncate back to 32 bits
            //         let result_32 = b.ins().ireduce(types::I32, high_bits);

            //         define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, rd, result_32);
            //     }
            //     OpcodeKind::M(raki::MOpcode::MULHSU) => {
            //         // Multiply high (signed x unsigned)
            //         let Instruction { rd, rs1, rs2, .. } = inst;
            //         let (rs1, rs2) = load_two_regs(
            //             &mut b,
            //             cpu_ptr,
            //             &regs,
            //             &mut regs_read_or_changed_so_far,
            //             &mut dirty_regs,
            //             rs1,
            //             rs2,
            //         );

            //         let v1 = b.use_var(regs[rs1]);
            //         let v2 = b.use_var(regs[rs2]);

            //         // Mixed sign extension: rs1 is signed, rs2 is unsigned
            //         let v1_64 = b.ins().sextend(types::I64, v1);
            //         let v2_64 = b.ins().uextend(types::I64, v2);
            //         let mul_result = b.ins().imul(v1_64, v2_64);

            //         // Shift right by 32 to get the high bits
            //         let shift_amt = b.ins().iconst(types::I64, 32);
            //         let high_bits = b.ins().ushr(mul_result, shift_amt);

            //         // Truncate back to 32 bits
            //         let result_32 = b.ins().ireduce(types::I32, high_bits);

            //         define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, rd, result_32);
            //     }
            //     OpcodeKind::M(raki::MOpcode::DIV) => {
            //         // Signed division
            //         let Instruction { rd, rs1, rs2, .. } = inst;
            //         let (rs1, rs2) = load_two_regs(
            //             &mut b,
            //             cpu_ptr,
            //             &regs,
            //             &mut regs_read_or_changed_so_far,
            //             &mut dirty_regs,
            //             rs1,
            //             rs2,
            //         );

            //         let v1 = b.use_var(regs[rs1]);
            //         let v2 = b.use_var(regs[rs2]);

            //         // Check for division by zero
            //         let zero = b.ins().iconst(types::I32, 0);
            //         let is_zero = b.ins().icmp(IntCC::Equal, v2, zero);
            //         let min_int = b.ins().iconst(types::I32, -2147483648_i32 as i64); // 0x80000000
            //         let neg_one = b.ins().iconst(types::I32, -1_i32 as i64);
            //         let cmp_with_min = b.ins().icmp(IntCC::Equal, v1, min_int);
            //         let cmp_with_neg_one = b.ins().icmp(IntCC::Equal, v2, neg_one);
            //         let is_overflow = b.ins().band(cmp_with_min, cmp_with_neg_one);

            //         // Normal division result
            //         let div_result = b.ins().sdiv(v1, v2);

            //         // Select based on special cases
            //         let overflow_result = min_int; // For overflow, return MIN_INT
            //         let zero_result = neg_one; // For div by zero, return -1

            //         let temp_result = b.ins().select(is_overflow, overflow_result, div_result);
            //         let final_result = b.ins().select(is_zero, zero_result, temp_result);

            //         define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, rd, final_result);
            //     }
            //     OpcodeKind::M(raki::MOpcode::DIVU) => {
            //         // Unsigned division
            //         let Instruction { rd, rs1, rs2, .. } = inst;
            //         let (rs1, rs2) = load_two_regs(
            //             &mut b,
            //             cpu_ptr,
            //             &regs,
            //             &mut regs_read_or_changed_so_far,
            //             &mut dirty_regs,
            //             rs1,
            //             rs2,
            //         );

            //         let v1 = b.use_var(regs[rs1]);
            //         let v2 = b.use_var(regs[rs2]);

            //         // Check for division by zero
            //         let zero = b.ins().iconst(types::I32, 0);
            //         let is_zero = b.ins().icmp(IntCC::Equal, v2, zero);

            //         // Normal division result
            //         let div_result = b.ins().udiv(v1, v2);

            //         // For division by zero, return all 1s (UINT_MAX)
            //         let max_uint = b.ins().iconst(types::I32, -1_i32 as i64); // 0xFFFFFFFF

            //         let final_result = b.ins().select(is_zero, max_uint, div_result);

            //         define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, rd, final_result);
            //     }
            //     OpcodeKind::M(raki::MOpcode::REM) => {
            //         // Signed remainder
            //         let Instruction { rd, rs1, rs2, .. } = inst;
            //         let (rs1, rs2) = load_two_regs(
            //             &mut b,
            //             cpu_ptr,
            //             &regs,
            //             &mut regs_read_or_changed_so_far,
            //             &mut dirty_regs,
            //             rs1,
            //             rs2,
            //         );

            //         let v1 = b.use_var(regs[rs1]);
            //         let v2 = b.use_var(regs[rs2]);

            //         // Check for division by zero or special case
            //         let zero = b.ins().iconst(types::I32, 0);
            //         let is_zero = b.ins().icmp(IntCC::Equal, v2, zero);
            //         let min_int = b.ins().iconst(types::I32, -2147483648_i32 as i64); // 0x80000000
            //         let neg_one = b.ins().iconst(types::I32, -1_i32 as i64);

            //         let cmp_with_min = b.ins().icmp(IntCC::Equal, v1, min_int);
            //         let cmp_with_neg_one = b.ins().icmp(IntCC::Equal, v2, neg_one);
            //         let is_special_case = b.ins().band(cmp_with_min, cmp_with_neg_one);

            //         // Normal remainder result
            //         let rem_result = b.ins().srem(v1, v2);

            //         // For div by zero, return the dividend
            //         // For special case, return 0
            //         let special_result = zero;

            //         let temp_result = b.ins().select(is_special_case, special_result, rem_result);
            //         let final_result = b.ins().select(is_zero, v1, temp_result);

            //         define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, rd, final_result);
            //     }
            //     OpcodeKind::M(raki::MOpcode::REMU) => {
            //         // Unsigned remainder
            //         let Instruction { rd, rs1, rs2, .. } = inst;
            //         let (rs1, rs2) = load_two_regs(
            //             &mut b,
            //             cpu_ptr,
            //             &regs,
            //             &mut regs_read_or_changed_so_far,
            //             &mut dirty_regs,
            //             rs1,
            //             rs2,
            //         );

            //         let v1 = b.use_var(regs[rs1]);
            //         let v2 = b.use_var(regs[rs2]);

            //         // Check for division by zero
            //         let zero = b.ins().iconst(types::I32, 0);
            //         let is_zero = b.ins().icmp(IntCC::Equal, v2, zero);

            //         // Normal remainder result
            //         let rem_result = b.ins().urem(v1, v2);

            //         // For div by zero, return the dividend
            //         let final_result = b.ins().select(is_zero, v1, rem_result);

            //         define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, rd, final_result);
            //     }
            //     OpcodeKind::BaseI(BaseIOpcode::ECALL) => {
            //         // Placeholder for ECALL as requested
            //         // ECALL should be a terminal instruction
            //         store_registers_to_cpu(&mut b, cpu_ptr, &regs, &dirty_regs);

            //         // Return a special value to indicate an ECALL (could be handled by the main loop)
            //         let ecall_indicator = b.ins().iconst(types::I32, 0xECA11);
            //         b.ins().return_(&[ecall_indicator]);
            //         term_was_added = true;
            //         cnt += 1;
            //         break;
            //     }
            //     OpcodeKind::BaseI(BaseIOpcode::EBREAK) => {
            //         // Environment break - terminal instruction
            //         store_registers_to_cpu(&mut b, cpu_ptr, &regs, &dirty_regs);

            //         // Return a special value to indicate an EBREAK (could be handled by the main loop)
            //         let ebreak_indicator = b.ins().iconst(types::I32, 0xEB8EA);
            //         b.ins().return_(&[ebreak_indicator]);
            //         term_was_added = true;
            //         cnt += 1;
            //         break;
            //     }
            _ => unimplemented!("demo supports few instrs"),
        }
        pc += 4;
        cnt += 1;
    }

    // return next PC if we reached the limit or if there was no terminal instruction
    if cnt == max_insns || !term_was_added {
        store_registers_to_cpu(&mut b, register_file_ptr, &regs, &dirty_regs);
        let rvals = &[b.ins().iconst(types::I32, pc as i64)];
        b.ins().return_(rvals);
    }

    b.seal_all_blocks();
    // replace with ctx.func.signature ?
    let sign = b.func.signature.clone();
    b.finalize();

    let id = jit.declare_anonymous_function(&sign).unwrap();
    jit.define_function(id, &mut ctx).unwrap();

    println!("{}", ctx.func.display());

    jit.clear_context(&mut ctx);
    jit.finalize_definitions().expect("must be ok");
    (jit.get_finalized_function(id), cnt)
}

// Compiler helpers
fn preload_for_bin_op(
    b: &mut FunctionBuilder<'_>,
    register_file_ptr: Value,
    regs_read_or_changed_so_far: &mut [bool; 32],
    dirty_regs: &mut [bool; 32],
    regs: &[Variable; 32],
    op_b: u32,
    op_c: u32,
) -> (Value, Value) {
    let (op_b, op_c) = load_two_regs(
        b,
        register_file_ptr,
        &regs,
        regs_read_or_changed_so_far,
        dirty_regs,
        op_b,
        op_c,
    );

    let v1 = b.use_var(regs[op_b]);
    let v2 = b.use_var(regs[op_c]);
    (v1, v2)
}

pub fn compile_addi(
    b: &mut FunctionBuilder,
    register_file_ptr: Value,
    regs_read_or_changed_so_far: &mut [bool; 32],
    dirty_regs: &mut [bool; 32],
    regs: &[Variable; 32],
    rs1: u32,
    imm: u32,
    rd: u32,
) {
    let rs1 = load_reg_if_needed_and_not_dirty(
        b,
        register_file_ptr,
        rs1.try_into().unwrap(),
        regs_read_or_changed_so_far,
        dirty_regs,
        regs,
    );

    let v1 = b.use_var(regs[rs1]);
    let imm: i64 = imm.try_into().unwrap();
    let r = b.ins().iadd_imm(v1, imm);

    define_rd_and_mark_dirty(b, &regs, dirty_regs, rd.try_into().unwrap(), r);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        memory::{mem_load32, mem_store32, Memory},
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

        // #[test]
        // fn test_lw_instruction() {
        //     // Define a simple program with LW instruction
        //     // This program will:
        //     // 1. Set x10 to an address (0)
        //     // 2. Store the value 42 at memory address (0)
        //     // 3. Load from that address into x28
        //     let test_program = [
        //         0x13, 0x05, 0x00, 0x04,     // addi x10, x0, 0      # set x10 to address 0
        //         0x93, 0x0F, 0xA0, 0x02,     // addi x31, x0, 42     # set x31 to value 42
        //         0x83, 0x2E, 0x05, 0x00,     // lw   x29, 0(x10)     # load from address 0 into x29
        //     ];

        //     let mut cpu = Cpu::new(&test_program);
        //     cpu.mem[64] = 42;
        //     let (cpu, _, insns, next_pc) = setup_test_env_with_cpu(&test_program, cpu);

        //     // Verify the results
        //     assert_eq!(insns, 3, "Should have translated all 4 instructions");
        //     assert_eq!(next_pc, 12, "PC should be 16 after execution");
        //     assert_eq!(cpu.regs[10], 64, "Register x10 should be 64");
        //     assert_eq!(cpu.regs[29], 42, "Register x29 should be 42 (loaded from memory)");
        // }
        #[test]
        fn test_lw_instruction() {
            // Program:
            //   addi x10, x0, 0     # base address 0
            //   addi x31, x0, 42    # value to store
            //   lw   x29, 0(x10)    # load 42 from mem[0]
            let instructions = vec![
                Instruction::new(Opcode::ADD, 10, 0, 44, false, true),   // addi x10, x0, 0
                Instruction::new(Opcode::ADD, 31, 0, 42, false, true),  // addi x31, x0, 42
                Instruction::new(Opcode::LW, 29, 10, 0, false, true),   // lw x29, 0(x10)
            ];

            let program = Program::new(instructions, 0, 0);
            let mut runtime = Executor::new(program);

            runtime.state.memory_.store32(44, 42);
            let (insns, next_pc) = setup_test_env_with_cpu(&mut runtime);
    
            // Verify the results
            assert_eq!(insns, 3, "Should have translated 3 instructions");
            assert_eq!(next_pc, 12, "Next PC should be 12 after execution");
            assert_eq!(runtime.register(Register::X10), 44);
            assert_eq!(runtime.register(Register::X31), 42);
            assert_eq!(runtime.register(Register::X29), 42, "Register x29 should load the stored value");
        }

    //     #[test]
    //     fn test_sw_instruction() {
    //         // Define a simple program with SW instruction
    //         // This program will:
    //         // 1. Set x10 to an address (4)
    //         // 2. Set x20 to a value (123)
    //         // 3. Store x20 to the address in x10 + 4
    //         let test_program = [
    //             0x13, 0x05, 0x40, 0x00,     // addi x10, x0, 4      # set x10 to address 4
    //             0x13, 0x0A, 0xB0, 0x07,     // addi x20, x0, 123    # set x20 to value 123
    //             0x23, 0x22, 0x45, 0x01,     // sw   x20, 4(x10)     # store 123 at address 8
    //         ];

    //         let (cpu, _, insns, next_pc) = setup_test_env(&test_program);

    //         // Verify the results
    //         assert_eq!(insns, 3, "Should have translated all 3 instructions");
    //         assert_eq!(next_pc, 12, "PC should be 12 after execution");
    //         assert_eq!(cpu.regs[10], 4, "Register x10 should be 4");
    //         assert_eq!(cpu.regs[20], 123, "Register x20 should be 123");
    //         assert_eq!(cpu.mem[8], 123, "Register x28 should be 123 (loaded from memory)");
    //     }


    

    //     #[test]
    //     fn test_jalr_instruction() {
    //         // Define a simple program with JALR instruction
    //         // This program will:
    //         // 1. Set x10 to an address (20)
    //         // 2. Jump to that address and link the return address to x1
    //         // We expect the TB to end at the JALR, so we'll check the next_pc value
    //         // Additionally, x1 should contain the return address (PC+4)
    //         let test_program = [
    //             0x13, 0x05, 0x40, 0x01,     // addi x10, x0, 20     # set x10 to address 20
    //             0x67, 0x00, 0x05, 0x00,     // jalr x1, 0(x10)      # jump to address in x10
    //             // The following should not be executed:
    //             0x13, 0x0F, 0x10, 0x00,     // addi x30, x0, 1      # set x30 to 1
    //         ];

    //         let (cpu, _, insns, next_pc) = setup_test_env(&test_program);

    //         println!("next_pc {}", next_pc);
    //         // Verify the results
    //         assert_eq!(insns, 2, "Should have translated 2 instructions (until JALR)");
    //         assert_eq!(next_pc, 20, "PC should be 20 after execution (jumped to x10)");
    //         assert_eq!(cpu.regs[10], 20, "Register x10 should be 20");
    //         assert_eq!(cpu.regs[0], 8, "Register x1 should be 8 (return address: PC+4)");
    //         assert_eq!(cpu.regs[30], 0, "Register x30 should be 0 (instruction after JALR not executed)");
    //     }

    //     #[test]
    //     fn test_mul_instruction() {
    //         // Define a simple program with MUL instruction
    //         let test_program = [
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

    //     #[test]
    //     fn test_jal_instruction() {
    //         // Define a program with JAL instruction
    //         // 1. JAL to skip over the next instruction (PC+8)
    //         // 2. Set x30 to 123 (should be skipped)
    //         // 3. Set x31 to 42 (should execute)
    //         let test_program = [
    //             0x6f, 0x00, 0x80, 0x00,     // jal x0, 8 (jump to PC+8)
    //             0x13, 0x0f, 0xb0, 0x07,     // addi x30, x0, 123 (should be skipped)
    //             0x13, 0x0f, 0xa0, 0x02,     // addi x31, x0, 42
    //         ];

    //         let (cpu, _, insns, next_pc) = setup_test_env(&test_program);

    //         // Verify the results
    //         assert_eq!(insns, 1, "Should have translated only 1 instruction");
    //         assert_eq!(cpu.regs[30], 0, "Register x30 should still be 0 (skipped instruction)");
    //         assert_eq!(cpu.regs[31], 0, "Register x31 should be 0"); // JAL forces quit
    //         assert_eq!(next_pc, 8, "Next PC should be 8 after JAL");
    //     }

    //     #[test]
    //     fn test_jal_with_link_instruction() {
    //         // Define a program with JAL instruction that stores return address
    //         // 1. JAL to skip over the next instruction, storing PC+4 in x1 (ra)
    //         // 2. Set x30 to 123 (should be skipped)
    //         // 3. Set x31 to 42 (should execute)
    //         let test_program = [
    //             0xef, 0x00, 0x80, 0x00,     // jal x1, 8 (jump to PC+8, store PC+4 in x1)
    //             0x13, 0x0f, 0xb0, 0x07,     // addi x30, x0, 123 (should be skipped)
    //             0x13, 0x0f, 0xa0, 0x02,     // addi x31, x0, 42
    //         ];

    //         let (cpu, _, insns, next_pc) = setup_test_env(&test_program);

    //         // Verify the results
    //         assert_eq!(insns, 1, "Should have translated all 1 instructions");
    //         assert_eq!(cpu.regs[1], 4, "Register x1 should contain return address (PC+4)");
    //         assert_eq!(cpu.regs[30], 0, "Register x30 should still be 0 (skipped instruction)");
    //         assert_eq!(next_pc, 8, "Next PC should be 8 after JAL");
    //     }

    //     #[test]
    //     fn test_auipc_instruction() {
    //         // Define a program with AUIPC instruction
    //         // 1. AUIPC x10, 1 (set x10 to PC + (1 << 12))
    //         // 2. Set x20 to expected value for verification
    //         let test_program = [
    //             0x17, 0x15, 0x00, 0x00,     // auipc x10, 1 (PC + 4096)
    //             0x13, 0x0a, 0x00, 0x00,     // addi x20, x0, 0 (we'll manually check x10 value)
    //         ];

    //         let (cpu, _, insns, next_pc) = setup_test_env(&test_program);

    //         // PC starts at 0, so x10 should be 0 + (1 << 12) = 4096
    //         assert_eq!(insns, 2, "Should have translated all 2 instructions");
    //         assert_eq!(cpu.regs[10], 4096, "AUIPC should set x10 to PC + (1 << 12)");
    //         assert_eq!(cpu.regs[20], 0, "Register x20 should be 0");
    //         assert_eq!(next_pc, 8, "Next PC should be 8 after execution");
    //     }

    //     #[test]
    //     fn test_byte_load_store_instructions() {
    //         // Define a program to test LB, LBU, SB instructions
    //         // 1. Set x10 to address 1024
    //         // 2. Set x20 to 0xFF (will be sign-extended but we only store the bottom byte)
    //         // 3. Store byte from x20 to address in x10 (SB)
    //         // 4. Load signed byte from address in x10 to x21 (LB)
    //         // 5. Load unsigned byte from address in x10 to x22 (LBU)
    //         let test_program = [
    //             0x13, 0x05, 0x00, 0x40,     // addi x10, x0, 1024
    //             0x13, 0x0a, 0xf0, 0x0f,     // addi x20, x0, 0xff (will be sign-extended)
    //             0x23, 0x00, 0x45, 0x01,     // sb x20, 0(x10)
    //             0x83, 0x0a, 0x05, 0x00,     // lb x21, 0(x10)
    //             0x03, 0x4b, 0x05, 0x00,     // lbu x22, 0(x10)
    //         ];

    //         let (cpu, _, insns, next_pc) = setup_test_env(&test_program);

    //         // Verify the results:
    //         // 0xFF as signed byte is -1 when sign-extended to 32 bits
    //         // 0xFF as unsigned byte is 255
    //         assert_eq!(insns, 5, "Should have translated all 5 instructions");
    //         assert_eq!(cpu.regs[10], 1024, "Register x10 should be 1024");
    //         assert_eq!(cpu.mem[1024], 0xFF, "Memory at x10 should be 0xFF");
    //         assert_eq!(cpu.regs[20] & 0xFF, 0xFF, "Register x20 low byte should be 0xFF");
    //         assert_eq!(cpu.regs[21] as i32, -1, "LB should sign-extend 0xFF to -1");
    //         assert_eq!(cpu.regs[22], 0xFF, "LBU should zero-extend 0xFF to 255");
    //         assert_eq!(next_pc, 20, "Next PC should be 20 after execution");
    //     }

    //     #[test]
    //     fn test_halfword_load_store_instructions() {
    //         // Define a program to test LH, LHU, SH instructions
    //         // 1. Set x10 to address 1024
    //         // 2. Set x20 to 0xFFFF (will be sign-extended but we only store the bottom halfword)
    //         // 3. Store halfword from x20 to address in x10 (SH)
    //         // 4. Load signed halfword from address in x10 to x21 (LH)
    //         // 5. Load unsigned halfword from address in x10 to x22 (LHU)
    //         let test_program = [
    //             0x13, 0x05, 0x00, 0x40,     // addi x10, x0, 1024
    //             0x13, 0x0a, 0xf0, 0xff,     // addi x20, x0, -1 (0xFFFF sign-extended)
    //             0x23, 0x10, 0x45, 0x01,     // sh x20, 0(x10)
    //             0x83, 0x1a, 0x05, 0x00,     // lh x21, 0(x10)
    //             0x03, 0x5b, 0x05, 0x00,     // lhu x22, 0(x10)
    //         ];

    //         let (cpu, _, insns, next_pc) = setup_test_env(&test_program);

    //         // Verify the results:
    //         // 0xFFFF as signed halfword is -1 when sign-extended to 32 bits
    //         // 0xFFFF as unsigned halfword is 65535
    //         assert_eq!(insns, 5, "Should have translated all 5 instructions");
    //         assert_eq!(cpu.regs[10], 1024, "Register x10 should be 100");
    //         assert_eq!(cpu.regs[20] as i32, -1, "Register x20 should be -1");
    //         assert_eq!(cpu.regs[21] as i32, -1, "LH should sign-extend 0xFFFF to -1");
    //         assert_eq!(cpu.regs[22], 0xFFFF, "LHU should zero-extend 0xFFFF to 65535");
    //         assert_eq!(next_pc, 20, "Next PC should be 20 after execution");
    //     }

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

    //     #[test]
    //     fn test_div_instruction() {
    //         // Define a program with DIV instruction (signed division)
    //         // 1. Set x10 to -10
    //         // 2. Set x20 to 3
    //         // 3. DIV x28, x10, x20 (result: -3)
    //         let test_program = [
    //             0x13, 0x05, 0x60, 0xff,     // addi x10, x0, -10
    //             0x13, 0x0a, 0x30, 0x00,     // addi x20, x0, 3
    //             0x33, 0x4e, 0x45, 0x03,     // div x28, x10, x20
    //         ];

    //         let (cpu, _, insns, next_pc) = setup_test_env(&test_program);

    //         // -10 / 3 = -3 (truncated toward zero)
    //         assert_eq!(insns, 3, "Should have translated all 3 instructions");
    //         assert_eq!(cpu.regs[10] as i32, -10, "Register x10 should be -10");
    //         assert_eq!(cpu.regs[20], 3, "Register x20 should be 3");
    //         assert_eq!(cpu.regs[28] as i32, -3, "DIV should perform signed division");
    //         assert_eq!(next_pc, 12, "Next PC should be 12 after execution");
    //     }

    //     #[test]
    //     fn test_divu_instruction() {
    //         // Define a program with DIVU instruction (unsigned division)
    //         // 1. Set x10 to -1 (0xFFFFFFFF unsigned)
    //         // 2. Set x20 to 10
    //         // 3. DIVU x28, x10, x20
    //         let test_program = [
    //             0x13, 0x05, 0xf0, 0xff,     // addi x10, x0, -1 (0xFFFFFFFF unsigned)
    //             0x13, 0x0a, 0xa0, 0x00,     // addi x20, x0, 10
    //             0x33, 0x5e, 0x45, 0x03,     // divu x28, x10, x20
    //         ];

    //         let (cpu, _, insns, next_pc) = setup_test_env(&test_program);

    //         // 0xFFFFFFFF / 10 = 429496729 (unsigned)
    //         assert_eq!(cpu.regs[28], 429496729, "DIVU should perform unsigned division");
    //     }

    //     #[test]
    //     fn test_rem_instruction() {
    //         // Define a program with REM instruction (signed remainder)
    //         // 1. Set x10 to -10
    //         // 2. Set x20 to 3
    //         // 3. REM x28, x10, x20 (result: -1)
    //         let test_program = [
    //             0x13, 0x05, 0x60, 0xff,     // addi x10, x0, -10
    //             0x13, 0x0a, 0x30, 0x00,     // addi x20, x0, 3
    //             0x33, 0x6e, 0x45, 0x03,     // rem x28, x10, x20
    //         ];

    //         let (cpu, _, insns, next_pc) = setup_test_env(&test_program);

    //         // -10 % 3 = -1
    //         assert_eq!(cpu.regs[28] as i32, -1, "REM should perform signed remainder");
    //         assert_eq!(insns, 3, "Should have translated all 3 instructions");
    //         assert_eq!(next_pc, 12, "Next PC should be 12 after execution");
    //     }

    //     #[test]
    //     fn test_remu_instruction() {
    //         // Define a program with REMU instruction (unsigned remainder)        // 1. Set x10 to -1 (0xFFFFFFFF unsigned)
    //         // 2. Set x20 to 10
    //         // 3. REMU x28, x10, x20
    //         let test_program = [
    //             0x13, 0x05, 0xf0, 0xff,     // addi x10, x0, -1 (0xFFFFFFFF unsigned)
    //             0x13, 0x0a, 0xa0, 0x00,     // addi x20, x0, 10
    //             0x33, 0x7e, 0x45, 0x03,     // remu x28, x10, x20
    //         ];

    //         let (cpu, _, insns, next_pc) = setup_test_env(&test_program);

    //         // 0xFFFFFFFF % 10 = 5 (unsigned)
    //         assert_eq!(insns, 3, "Should have translated all 3 instructions");
    //         assert_eq!(cpu.regs[10] as i32, -1, "Register x10 should be 0xFFFFFFFF");
    //         assert_eq!(cpu.regs[20], 10, "Register x20 should be 10");
    //         assert_eq!(cpu.regs[28], 5, "REMU should perform unsigned remainder");
    //         assert_eq!(next_pc, 12, "Next PC should be 12 after execution");
    //     }

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
