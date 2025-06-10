use std::{
    fs::File,
    io::{Seek, Write}, ops::{Index, IndexMut},
};

use memmap2::{MmapMut};

use hashbrown::HashMap;
use serde::{Deserialize, Serialize};

use crate::{events::MemoryRecord, syscalls::SyscallCode, ExecutorMode, Register};


/// Holds anon mmap memory allocation.
#[derive(Debug)]
pub struct Memory {
    /// The anon mmap memory allocation.
    pub memory: MmapMut,
}

// 2 << 30 : 4GB / 4
impl Default for Memory {
    fn default() -> Self {
        Memory { memory: MmapMut::map_anon(1 << 30).unwrap() }
    }
}

impl Clone for Memory {
    fn clone(&self) -> Self {
        Self { memory: MmapMut::map_anon(1 << 30).unwrap() }
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

/// Holds data describing the current state of a program's execution.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[repr(C)]
pub struct ExecutionState {
    /// The program counter.
    pub pc: u32,

    /// The shard clock keeps track of how many shards have been executed.
    pub current_shard: u32,

    /// The memory which instructions operate over. Values contain the memory value and last shard
    /// + timestamp that each memory address was accessed.
    pub memory: HashMap<u32, MemoryRecord>,

    /// Memory repr using anon mmap.
    #[serde(skip)]
    pub memory_: Memory,

    /// (cnt, length)
    #[serde(skip)]
    pub pot_tb: HashMap<u32, (u32, u32)>,

    /// is_jmp
    #[serde(skip)]
    pub is_jmp: bool,

    /// prev_pc
    #[serde(skip)]
    pub prev_pc: u32,

    /// tb_len
    #[serde(skip)]
    pub tb_len: u32,

    /// Registers file which instructions operate over.
    pub register_file: [MemoryRecord; Register::number_of_registers()],

    /// The global clock keeps track of how many instructions have been executed through all shards.
    pub global_clk: u64,

    /// The clock increments by 4 (possibly more in syscalls) for each instruction that has been
    /// executed in this shard.
    pub clk: u32,

    /// Uninitialized memory addresses that have a specific value they should be initialized with.
    /// `SyscallHintRead` uses this to write hint data into uninitialized memory.
    pub uninitialized_memory: HashMap<u32, u32>,

    /// A stream of input values (global to the entire program).
    pub input_stream: Vec<Vec<u8>>,

    /// A ptr to the current position in the input stream incremented by `HINT_READ` opcode.
    pub input_stream_ptr: usize,

    /// A ptr to the current position in the proof stream, incremented after verifying a proof.
    pub proof_stream_ptr: usize,

    /// A stream of public values from the program (global to entire program).
    pub public_values_stream: Vec<u8>,

    /// A ptr to the current position in the public values stream, incremented when reading from
    /// `public_values_stream`.
    pub public_values_stream_ptr: usize,

    /// Keeps track of how many times a certain syscall has been called.
    pub syscall_counts: HashMap<SyscallCode, u64>,
}

impl ExecutionState {
    #[must_use]
    /// Create a new [`ExecutionState`].
    pub fn new(pc_start: u32) -> Self {
        Self {
            global_clk: 0,
            // Start at shard 1 since shard 0 is reserved for memory initialization.
            current_shard: 1,
            clk: 0,
            pc: pc_start,
            memory: HashMap::with_capacity(2398076),
            uninitialized_memory: HashMap::new(),
            input_stream: Vec::new(),
            input_stream_ptr: 0,
            public_values_stream: Vec::new(),
            public_values_stream_ptr: 0,
            proof_stream_ptr: 0,
            syscall_counts: HashMap::new(),
            register_file: [MemoryRecord::default(); Register::number_of_registers()],
            memory_: Memory::default(),
            pot_tb: HashMap::new(),
            is_jmp: false,
            prev_pc: 0,
            tb_len: 0,
        }
    }
}

/// Holds data to track changes made to the runtime since a fork point.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct ForkState {
    /// The `global_clk` value at the fork point.
    pub global_clk: u64,
    /// The original `clk` value at the fork point.
    pub clk: u32,
    /// The original `pc` value at the fork point.
    pub pc: u32,
    /// All memory changes since the fork point.
    pub memory_diff: HashMap<u32, Option<MemoryRecord>>,
    // /// The original memory access record at the fork point.
    // pub op_record: MemoryAccessRecord,
    // /// The original execution record at the fork point.
    // pub record: ExecutionRecord,
    /// Whether `emit_events` was enabled at the fork point.
    pub executor_mode: ExecutorMode,
}

impl ExecutionState {
    /// Save the execution state to a file.
    pub fn save(&self, file: &mut File) -> std::io::Result<()> {
        let mut writer = std::io::BufWriter::new(file);
        bincode::serialize_into(&mut writer, self).unwrap();
        writer.flush()?;
        writer.seek(std::io::SeekFrom::Start(0))?;
        Ok(())
    }
}
