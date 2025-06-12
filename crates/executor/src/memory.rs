use std::ops::{Index, IndexMut};

use memmap2::MmapMut;


/// Holds anon mmap memory allocation.
#[repr(C)]
#[derive(Debug)]
pub struct Memory {
    /// The anon mmap memory allocation.
    pub memory: MmapMut,
}

impl Memory {
    /// Load a 32-bit value from memory.
    pub fn load32(&self, addr: u32) -> u32 {
        let i = addr as usize;
        // println!("load32 addr {} val {:x}", i, u32::from_le_bytes(self.mem[i..i + 4].try_into().unwrap()));
        u32::from_le_bytes(self.memory[i..i + 4].try_into().unwrap())
    }
    /// Store a 32-bit value to memory.
    pub fn store32(&mut self, addr: u32, val: u32) {
        let i = addr as usize;
        // println!("store32 addr {} val {:x}", i, val);
        self.memory[i..i + 4].copy_from_slice(&val.to_le_bytes());
    }
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

/// Memory load helper 
pub extern "C" fn mem_load32(memory: &mut Memory, addr: u32) -> u32 {
    memory.load32(addr)
}

/// Memory store helper 
pub extern "C" fn mem_store32(memory: &mut Memory, addr: u32, val: u32) {
    memory.store32(addr, val)
}