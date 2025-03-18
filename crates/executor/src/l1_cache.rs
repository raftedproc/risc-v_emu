//! Cache and execution state management for the RISC-V emulator.
//!
//! This module implements an 8-way set-associative L1 cache with hits replacement
//! policy, along with the core execution state tracking for the emulator.

use serde::{Deserialize, Serialize};
use sp1_primitives::consts::WORD_SIZE;
use std::{arch::x86_64::*, ops::Neg};

use crate::{events::MemoryRecord, Memory};

/// A cache line in the L1 cache, representing a block of memory.
///
/// Each cache line contains:
/// - A tag for address matching
/// - An array of memory records (the actual cached data)
/// - An hits counter for replacement decisions
/// - A valid bit indicating if the line contains valid data
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheLine {
    /// Tag bits for address matching
    tag: u32,
    /// Base address of the cache line in memory
    base_addr: u32,
    /// Cached memory records in this line
    data: [Option<MemoryRecord>; Self::LINE_SIZE],
    /// hits counter for replacement policy
    hits: u8,
    /// Whether this cache line contains valid data
    valid: bool,
}

impl CacheLine {
    /// Number of memory records/4 bytes words in each cache line
    const LINE_SIZE: usize = 8;

    /// Creates a new empty cache line
    #[allow(unused)]
    pub fn new() -> Self {
        Self {
            valid: false,
            tag: 0,
            base_addr: 0,
            data: [None; Self::LINE_SIZE],
            hits: 0,
        }
    }

    fn record_from_memory(addr: u32, memory: &mut Memory) -> Option<MemoryRecord> {
        memory.get(&addr).cloned()
    }

    /// Creates a new cache line by loading data from memory
    ///
    /// # Arguments
    /// * `tag` - Tag bits for the cache line
    /// * `base_addr` - Base address of the cache line in memory
    /// * `hits` - Initial hits counter value
    /// * `memory` - Memory to load data from
    fn from_memory(tag: u32, base_addr: u32, memory: &mut Memory) -> Self {
        let valid = true;
        let hits = 0;
        let mut data = [None; Self::LINE_SIZE];
        for offset in 0..Self::LINE_SIZE {
            let addr = addr_from_base_addr_offset(base_addr, offset);
            data[offset] = Self::record_from_memory(addr, memory);
        }
        Self {
            valid,
            tag,
            base_addr,
            data,
            hits,
        }
    }
}

/// L1 cache implementation with 8-way set-associative mapping
///
/// The cache is organized as:
/// - 256 sets (indexed by address bits [5:13])
/// - 8 ways per set (managed by hits replacement with AVX2 SIMD)
/// - 32 words per cache line
///
/// This structure provides efficient memory access through caching while
/// maintaining consistency with main memory through write-back policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct L1Cache {
    /// Cache data organized as sets of cache lines
    pub cache: Vec<Vec<CacheLine>>,
}

impl Default for L1Cache {
    /// Creates a default L1 cache instance
    ///
    /// Delegates to `new()` to create a properly initialized cache
    fn default() -> Self {
        Self::new()
    }
}

impl L1Cache {
    /// Number of cache sets
    const SETS: usize = 256;
    const SETS_MASK: usize = Self::SETS - 1;
    /// Number of ways (associativity) per set
    const WAYS: usize = 8;

    /// Creates a new empty L1 cache with pre-allocated sets and ways
    ///
    /// # Returns
    /// A new L1Cache instance with SETS × WAYS empty cache lines
    pub fn new() -> Self {
        let cache = vec![vec![CacheLine::default(); Self::WAYS]; Self::SETS];
        Self {
            cache,
        }
    }

    #[inline(always)]
    /// Looks up a memory address in the cache using AVX2 SIMD
    pub fn lookup_no_ts_update(&mut self, addr: u32) -> Option<&MemoryRecord> {
        let set: usize = set_from_addr(addr);
        let tag = tag_from_addr(addr);
        let set_lines = unsafe { self.cache.get_unchecked_mut(set) };
        
        unsafe {
            // Load tags and valid bits into AVX2 registers
            let tags = _mm256_setr_epi32(
                set_lines[0].tag as i32,
                set_lines[1].tag as i32,
                set_lines[2].tag as i32,
                set_lines[3].tag as i32,
                set_lines[4].tag as i32,
                set_lines[5].tag as i32,
                set_lines[6].tag as i32,
                set_lines[7].tag as i32,
            );
            
            let valid_bits = _mm256_setr_epi32(
                (set_lines[0].valid as i32).neg(),
                (set_lines[1].valid as i32).neg(),
                (set_lines[2].valid as i32).neg(),
                (set_lines[3].valid as i32).neg(),
                (set_lines[4].valid as i32).neg(),
                (set_lines[5].valid as i32).neg(),
                (set_lines[6].valid as i32).neg(),
                (set_lines[7].valid as i32).neg(),
            );
            

            // println!("tags: {:?}", tags);
            // let b = _mm256_set1_epi32(tag as i32);
            // println!("b: {:?}", b);
            // Compare tags
            let tag_match = _mm256_cmpeq_epi32(tags, _mm256_set1_epi32(tag as i32));
            let valid_match = _mm256_and_si256(tag_match, valid_bits);
            let mask = _mm256_movemask_ps(_mm256_castsi256_ps(valid_match)) as u32;
            // println!("tag: {:?}", tag);
            // println!("tag_match: {:?}", tag_match);
            // println!("valid_match: {:?}", valid_match);
            // println!("mask: {:?}", mask);
            if mask != 0 {
                // Find first matching way
                let way = mask.trailing_zeros() as usize;
                set_lines[way].hits += 1;
                let offset = offset_from_addr(addr);
                return set_lines[way].data[offset].as_ref();
            }
        }
        None
    }

    #[inline(always)]
    pub fn lookup_mut_no_ts_update(&mut self, addr: u32) -> Option<&mut MemoryRecord> {
        let set: usize = set_from_addr(addr);
        let tag = tag_from_addr(addr);
        let set_lines = unsafe { self.cache.get_unchecked_mut(set) };
        
        unsafe {
            // Load tags and valid bits into AVX2 registers
            let tags = _mm256_setr_epi32(
                set_lines[0].tag as i32,
                set_lines[1].tag as i32,
                set_lines[2].tag as i32,
                set_lines[3].tag as i32,
                set_lines[4].tag as i32,
                set_lines[5].tag as i32,
                set_lines[6].tag as i32,
                set_lines[7].tag as i32,
            );
            
            let valid_bits = _mm256_setr_epi32(
                (set_lines[0].valid as i32).neg(),
                (set_lines[1].valid as i32).neg(),
                (set_lines[2].valid as i32).neg(),
                (set_lines[3].valid as i32).neg(),
                (set_lines[4].valid as i32).neg(),
                (set_lines[5].valid as i32).neg(),
                (set_lines[6].valid as i32).neg(),
                (set_lines[7].valid as i32).neg(),
            );
            
            // Compare tags
            let tag_match = _mm256_cmpeq_epi32(tags, _mm256_set1_epi32(tag as i32));
            let valid_match = _mm256_and_si256(tag_match, valid_bits);
            let mask = _mm256_movemask_ps(_mm256_castsi256_ps(valid_match)) as u32;
            
            if mask != 0 {
                // Find first matching way
                let way = mask.trailing_zeros() as usize;
                set_lines[way].hits += 1;
                let offset = offset_from_addr(addr);
                return set_lines[way].data[offset].as_mut();
            }
        }
        None
    }

    #[inline(always)]
    /// Inserts a new cache line for the given address using AVX2 SIMD
    ///
    /// Uses hits-based replacement policy to choose which way to evict
    /// when all ways in a set are occupied. The evicted line is written back to memory
    /// if it contains valid data.
    ///
    /// # Arguments
    /// * `addr` - Memory address to cache
    /// * `memory` - Memory to load data from
    pub fn insert(&mut self, addr: u32, memory: &mut Memory) {
        let set = set_from_addr(addr);
        let tag = tag_from_addr(addr);
        let base_addr = addr & !(CacheLine::LINE_SIZE * 4 - 1) as u32;
        let set_lines = unsafe { self.cache.get_unchecked_mut(set) };

        unsafe {
            // Load hits into AVX2 register
            // let hits = _mm256_setr_epi32(
            //     set_lines[0].hits as i32,
            //     set_lines[1].hits as i32,
            //     set_lines[2].hits as i32,
            //     set_lines[3].hits as i32,
            //     set_lines[4].hits as i32,
            //     set_lines[5].hits as i32,
            //     set_lines[6].hits as i32,
            //     set_lines[7].hits as i32,
            // );

            // Load valid bits into AVX2 register
            let valid_bits = _mm256_setr_epi32(
                (set_lines[0].valid as i32).neg(),
                (set_lines[1].valid as i32).neg(),
                (set_lines[2].valid as i32).neg(),
                (set_lines[3].valid as i32).neg(),
                (set_lines[4].valid as i32).neg(),
                (set_lines[5].valid as i32).neg(),
                (set_lines[6].valid as i32).neg(),
                (set_lines[7].valid as i32).neg(),
            );

            // Find invalid or least recently used way
            let invalid_mask = _mm256_xor_si256(valid_bits, _mm256_set1_epi32(-1));
            let invalid_bits = _mm256_movemask_ps(_mm256_castsi256_ps(invalid_mask)) as u32;

            let way = if invalid_bits != 0 {
                // Use first invalid way if available
                invalid_bits.trailing_zeros() as usize
            } else {
                // Find way with lowest hits
                let mut min_hits = u8::MAX;
                let mut min_way = 0;
                for way in 0..Self::WAYS {
                    if set_lines[way].hits < min_hits {
                        min_hits = set_lines[way].hits;
                        min_way = way;
                    }
                }
                min_way
            };

            store_cacheline_if_needed(&set_lines[way], memory);
            let new_cacheline = CacheLine::from_memory(tag, base_addr, memory);
            set_lines[way] = new_cacheline;
        }
    }

    pub fn insert_and_return(&mut self, addr: u32, memory: &mut Memory) -> Option<&MemoryRecord> {
        let set = set_from_addr(addr);
        let tag = tag_from_addr(addr);
        let offset = offset_from_addr(addr);
        let base_addr = addr & !(CacheLine::LINE_SIZE * 4 - 1) as u32;
        let set_lines = unsafe { self.cache.get_unchecked_mut(set) };

        unsafe {
            // Load hits and valid bits into AVX2 registers
            // let hits = _mm256_setr_epi32(
            //     set_lines[0].hits as i32,
            //     set_lines[1].hits as i32,
            //     set_lines[2].hits as i32,
            //     set_lines[3].hits as i32,
            //     set_lines[4].hits as i32,
            //     set_lines[5].hits as i32,
            //     set_lines[6].hits as i32,
            //     set_lines[7].hits as i32,
            // );

            let valid_bits = _mm256_setr_epi32(
                (set_lines[0].valid as i32).neg(),
                (set_lines[1].valid as i32).neg(),
                (set_lines[2].valid as i32).neg(),
                (set_lines[3].valid as i32).neg(),
                (set_lines[4].valid as i32).neg(),
                (set_lines[5].valid as i32).neg(),
                (set_lines[6].valid as i32).neg(),
                (set_lines[7].valid as i32).neg(),
            );

            // Find invalid or least recently used way
            let invalid_mask = _mm256_xor_si256(valid_bits, _mm256_set1_epi32(-1));
            let invalid_bits = _mm256_movemask_ps(_mm256_castsi256_ps(invalid_mask)) as u32;

            let way = if invalid_bits != 0 {
                // Use first invalid way if available
                invalid_bits.trailing_zeros() as usize
            } else {
                // Find way with lowest hits
                let mut min_hits = u8::MAX;
                let mut min_way = 0;
                for way in 0..Self::WAYS {
                    if set_lines[way].hits < min_hits {
                        min_hits = set_lines[way].hits;
                        min_way = way;
                    }
                }
                min_way
            };

            // println!("insert_and_return addr {:?}, set {} replacing way {}", addr, set, way);

            store_cacheline_if_needed(&set_lines[way], memory);
            let new_cacheline = CacheLine::from_memory(tag, base_addr, memory);
            set_lines[way] = new_cacheline;
            set_lines[way].data[offset].as_ref()
        }
    }

    pub fn insert_and_return_mut(&mut self, addr: u32, memory: &mut Memory) -> Option<&mut MemoryRecord> {
        let set = set_from_addr(addr);
        let tag = tag_from_addr(addr);
        let offset = offset_from_addr(addr);
        let base_addr = addr & !(CacheLine::LINE_SIZE * 4 - 1) as u32;
        let set_lines = unsafe { self.cache.get_unchecked_mut(set) };

        unsafe {
            // Load hits and valid bits into AVX2 registers
            // let hits = _mm256_setr_epi32(
            //     set_lines[0].hits as i32,
            //     set_lines[1].hits as i32,
            //     set_lines[2].hits as i32,
            //     set_lines[3].hits as i32,
            //     set_lines[4].hits as i32,
            //     set_lines[5].hits as i32,
            //     set_lines[6].hits as i32,
            //     set_lines[7].hits as i32,
            // );

            let valid_bits = _mm256_setr_epi32(
                (set_lines[0].valid as i32).neg(),
                (set_lines[1].valid as i32).neg(),
                (set_lines[2].valid as i32).neg(),
                (set_lines[3].valid as i32).neg(),
                (set_lines[4].valid as i32).neg(),
                (set_lines[5].valid as i32).neg(),
                (set_lines[6].valid as i32).neg(),
                (set_lines[7].valid as i32).neg(),
            );

            // Find invalid or least recently used way
            let invalid_mask = _mm256_xor_si256(valid_bits, _mm256_set1_epi32(-1));
            let invalid_bits = _mm256_movemask_ps(_mm256_castsi256_ps(invalid_mask)) as u32;

            let way = if invalid_bits != 0 {
                // Use first invalid way if available
                invalid_bits.trailing_zeros() as usize
            } else {
                // Find way with lowest hits
                let mut min_hits = u8::MAX;
                let mut min_way = 0;
                for way in 0..Self::WAYS {
                    if set_lines[way].hits < min_hits {
                        min_hits = set_lines[way].hits;
                        min_way = way;
                    }
                }
                min_way
            };

            store_cacheline_if_needed(&set_lines[way], memory);
            let new_cacheline = CacheLine::from_memory(tag, base_addr, memory);
            set_lines[way] = new_cacheline;
            set_lines[way].data[offset].as_mut()
        }
    }

    #[inline(always)]
    /// updates MemoryRecord for a specific addr
    /// Used by ExitUnconstrainedSyscall to sync memory
    pub fn update_if_in_cache(&mut self, addr: u32, memory_record: MemoryRecord) {
        let set = set_from_addr(addr);
        let tag = tag_from_addr(addr);

        let set_lines = unsafe { self.cache.get_unchecked_mut(set) };

        let cacheline = if set_lines[0].tag == tag {
            &mut set_lines[0]
        } else if set_lines[1].tag == tag {
            &mut set_lines[1]
        } else {
            return;
        };

        let offset = offset_from_addr(addr);
        cacheline.data[offset] = Some(memory_record);
    }

    #[inline(always)]
    /// erases MemoryRecord for a specific addr
    /// Used by ExitUnconstrainedSyscall to sync memory
    pub fn erase_if_in_cache(&mut self, addr: u32) {
        let memory_record = MemoryRecord::default();
        self.update_if_in_cache(addr, memory_record);
    }
}

/// Writes back a cache line to memory if it contains valid data
///
/// This function is called before evicting a cache line to ensure modified data
/// is not lost. It writes each memory record in the cache line back to its
/// corresponding memory address.
///
/// # Arguments
/// * `cacheline` - Cache line to write back
/// * `memory` - Memory to write data to
fn store_cacheline_if_needed(cacheline: &CacheLine, memory: &mut Memory) {
    if cacheline.valid {
        let base_addr = cacheline.base_addr;
        for (offset, record) in cacheline.data.iter().enumerate() {
            let addr = base_addr | (offset * WORD_SIZE) as u32;
            match record {
                Some(record) => memory.insert(addr, *record),
                None => continue,
            };
        }
    }
}

/// Calculates the cache set index from a memory address
///
/// Extracts bits [5:13] from the address to determine which set to use.
/// With 256 sets, we need 8 bits for the set index.
///
/// # Arguments
/// * `addr` - Memory address to calculate set index for
///
/// # Returns
/// Set index in range [0, 255]
#[inline(always)]
pub fn set_from_addr(addr: u32) -> usize {
    (addr as usize >> 5) & L1Cache::SETS_MASK
}

/// Takes the upper bits [13:31] of the address as the tag.
/// These bits are used to check if a cache line contains the desired address.
#[inline(always)]
pub fn tag_from_addr(addr: u32) -> u32 {
    addr >> 13
}

#[inline(always)]
/// Calculates the offset within a cache line from a memory address
///
/// Uses bits [0:4] of the address to determine the position within a cache line.
/// With 32 words per line, we need 5 bits for the offset.
/// Offset in range [0, 31]
pub fn offset_from_addr(addr: u32) -> usize {
    (addr as usize & 0x1F) / WORD_SIZE
}

#[inline(always)]
pub fn addr_from_base_addr_offset(base_addr: u32, offset: usize) -> u32 {
    base_addr | (offset * WORD_SIZE) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper function to create addresses that map to the same cache set
    fn create_same_set_addresses(base: u32, count: u32) -> Vec<u32> {
        (0..count).map(|i| base + (i << 14)).collect()
    }

    #[test]
    fn test_cacheline_new() {
        let line = CacheLine::new();
        assert!(!line.valid);
        assert_eq!(line.tag, 0);
        assert_eq!(line.hits, 0);
        assert_eq!(line.data.len(), CacheLine::LINE_SIZE);
        assert!(line.data.iter().all(|x| x.is_none()));
    }

    #[test]
    fn test_cacheline_from_memory() {
        let mut memory = Memory::new();
        let addr = 0x1000;
        let tag = tag_from_addr(addr);
        let record = MemoryRecord {
            shard: 42,
            timestamp: 42,
            value: 42,
        };
        memory.insert(addr, record);

        let line = CacheLine::from_memory(tag, addr, &mut memory);
        assert!(line.valid);
        assert_eq!(line.tag, tag);
        assert_eq!(line.hits, 0);
        assert_eq!(line.data[0].as_ref(), Some(&record));
        assert!(line.data[1].is_none());
    }

    #[test]
    fn test_l1cache_new() {
        let cache = L1Cache::new();
        assert_eq!(cache.cache.len(), L1Cache::SETS);
        for set in &cache.cache {
            assert_eq!(set.len(), L1Cache::WAYS);
            for way in set {
                assert!(!way.valid);
                assert!(way.data.iter().all(|x| x.is_none()));
            }
        }
    }

    #[test]
    fn test_cache_to_memory_sync() {
        let mut cache = L1Cache::new();
        let mut memory = Memory::new();

        // Create WAYS + 1 addresses in the same set to test eviction
        let base_addr = 0xfaeff000;
        let mut addresses = Vec::new();
        let mut records = Vec::new();

        // Create WAYS addresses with unique values
        for i in 0..L1Cache::WAYS {
            let addr = (base_addr + (i << 14)) as u32;
            let record = MemoryRecord {
                shard: i as u32,
                timestamp: i as u32,
                value: i as u32 * 42,
            };
            addresses.push(addr);
            records.push(record);
            memory.insert(addr, record);
            cache.insert_and_return(addr, &mut memory);

            // Verify it's cached
            assert!(cache.lookup_no_ts_update(addr).is_some(), 
                   "Address {} should be cached", i);
        }

        // Modify all cached values
        for (i, addr) in addresses.iter().enumerate() {
            if let Some(cached) = cache.lookup_mut_no_ts_update(*addr) {
                cached.value = (i as u32 + 1) * 100;
            }
        }

        // Access first address multiple times to increase its hits
        for &addr in addresses[..L1Cache::WAYS-1].iter() {
            for _ in 0..5 {
                cache.lookup_no_ts_update(addr);
            }
        }

        let target_value =  cache.lookup_no_ts_update(*addresses.last().unwrap()).cloned();
        let set1 = set_from_addr(addresses[0]);
        let set_last = set_from_addr(*addresses.last().unwrap());
        assert_eq!(set1, set_last);

        // Insert one more address to force eviction
        let evicting_addr = (base_addr + (L1Cache::WAYS << 14)) as u32;
        let evicting_record = MemoryRecord {
            shard: L1Cache::WAYS as u32,
            timestamp: L1Cache::WAYS as u32,
            value: L1Cache::WAYS as u32 * 42,
        };
        memory.insert(evicting_addr, evicting_record);
        cache.insert_and_return(evicting_addr, &mut memory);

        // First address should still be cached due to high hits
        assert!(cache.lookup_no_ts_update(addresses[0]).is_some(),
               "First address should still be cached due to high hits");

        assert_eq!(target_value, memory.get(addresses.last().unwrap()).cloned());
    }

    #[test]
    fn test_l1cache_lookup_miss() {
        let mut cache = L1Cache::new();
        let addr = 0x1000;
        assert!(cache.lookup_no_ts_update(addr).is_none());
    }

    #[test]
    fn test_l1cache_sequential_access() {
        let mut cache = L1Cache::new();
        let mut memory = Memory::new();

        // Create sequential addresses in the same cache line
        let base_addr = 0x1000;
        let values = [1, 2, 3, 4];

        // Insert values into memory
        for (i, &val) in values.iter().enumerate() {
            let addr = base_addr + (i * WORD_SIZE) as u32;
            let mut record = MemoryRecord::default();
            record.value = val;
            memory.insert(addr, record);
        }

        // Insert first address - should load entire cache line
        cache.insert(base_addr, &mut memory);

        // Verify all addresses in the cache line are cached
        for (i, &val) in values.iter().enumerate() {
            let addr = base_addr + (i * WORD_SIZE) as u32;
            let record = cache
                .lookup_no_ts_update(addr)
                .expect("Address should be cached");
            assert_eq!(record.value, val, "Cached value mismatch");
        }
    }

    #[test]
    fn test_l1cache_insert_and_lookup() {
        let mut cache = L1Cache::new();
        let mut memory = Memory::new();
        let addr = 0x1000;
        let value = MemoryRecord::default();
        memory.insert(addr, value);

        // Insert into cache
        cache.insert(addr, &mut memory);

        // Verify lookup succeeds
        let result = cache.lookup_no_ts_update(addr);
        assert!(result.is_some());
    }

    #[test]
    fn test_cache_boundary_addresses() {
        let mut cache = L1Cache::new();
        let mut memory = Memory::new();

        // Test cache line boundary addresses
        let addr2 = 0x20; // First offset in next cache line
        let addr3 = 0x1FE0; // Last offset in a set
        let addr4 = 0x2000; // First offset in next set

        for addr in [addr2, addr3, addr4] {
            let record = MemoryRecord {
                shard: 42,
                timestamp: addr,
                value: addr,
            };
            memory.insert(addr, record);
            cache.insert(addr, &mut memory);

            assert!(
                cache.lookup_no_ts_update(addr).is_some(),
                "Failed to cache address {:#x}",
                addr
            );
        }

        // Verify address calculations at boundaries
        assert_eq!(offset_from_addr(addr2), 0x0);
        assert_eq!(set_from_addr(addr3), 0xFF);
        assert_eq!(set_from_addr(addr4), 0x0);
    }

    #[test]
    fn test_cache_aliasing() {
        let mut cache = L1Cache::new();
        let mut memory = Memory::new();

        // Create two addresses that map to the same cache set and offset
        let addr1 = 0x1000;
        let addr2 = addr1 + (1 << 14); // Same set, different tag

        // Insert first address
        let record1 = MemoryRecord {
            shard: 42,
            timestamp: 42,
            value: 42,
        };
        memory.insert(addr1, record1);
        cache.insert(addr1, &mut memory);

        // Insert second address
        let record2 = MemoryRecord {
            shard: 84,
            timestamp: 84,
            value: 84,
        };
        memory.insert(addr2, record2);
        cache.insert(addr2, &mut memory);

        // Both should be in cache (8-way set associative)
        let cached1 = cache
            .lookup_no_ts_update(addr1)
            .expect("First address should be cached");
        assert_eq!(*cached1, record1);
        let cached2 = cache
            .lookup_no_ts_update(addr2)
            .expect("Second address should be cached");
        assert_eq!(*cached2, record2);

        // Add 6 more addresses to the same set to verify 8-way capacity
        for i in 2..8 {
            let addr = addr1 + (i << 14); // Same set, different tag
            let record = MemoryRecord {
                shard: i as u32 * 42,
                timestamp: i as u32 * 42,
                value: i as u32 * 42,
            };
            memory.insert(addr, record);
            cache.insert(addr, &mut memory);

            // Verify it's cached
            let cached = cache
                .lookup_no_ts_update(addr)
                .expect("Address should be cached");
            assert_eq!(*cached, record);

            // Previous addresses should still be cached
            assert!(cache.lookup_no_ts_update(addr1).is_some(), "First address should still be cached");
            assert!(cache.lookup_no_ts_update(addr2).is_some(), "Second address should still be cached");
        }
    }

    #[test]
    fn test_cache_capacity() {
        let mut cache = L1Cache::new();
        let mut memory = Memory::new();
        let mut record = MemoryRecord::default();
        record.value = 42;

        // Try to fill entire cache
        let mut hits = 0;
        for set in 0..L1Cache::SETS {
            for way in 0..L1Cache::WAYS {
                let addr = ((set << 5) | (way << 14)) as u32;
                memory.insert(addr, record);
                cache.insert(addr, &mut memory);
                if cache.lookup_no_ts_update(addr).is_some() {
                    hits += 1;
                }
            }
        }

        // Verify we could use full cache capacity
        assert_eq!(
            hits,
            L1Cache::SETS * L1Cache::WAYS,
            "Cache not utilizing full capacity"
        );
    }

    #[test]
    fn test_cache_uninitialized_memory() {
        let mut cache = L1Cache::new();
        let mut memory = Memory::new();

        // Try to cache an address that doesn't exist in memory
        let addr = 0x1000;
        cache.insert(addr, &mut memory);

        // Should still create a cache line with default values
        assert!(
            cache.lookup_no_ts_update(addr).is_none(),
            "There must be None in memory for the addr {}",
            addr
        );
    }

    #[test]
    fn test_cache_concurrent_access_pattern() {
        let mut cache = L1Cache::new();
        let mut memory = Memory::new();

        // Create interleaved access pattern across different sets
        let addrs = [
            0x1000, // Set 0x80
            0x2000, // Set 0x100
            0x1020, // Set 0x81
            0x2020, // Set 0x101
        ];

        // Initialize memory
        for &addr in &addrs {
            memory.insert(addr, MemoryRecord::default());
        }

        // Access in interleaved pattern
        for &addr in &addrs {
            cache.insert(addr, &mut memory);
        }

        // Verify all addresses still cached
        for &addr in &addrs {
            assert!(
                cache.lookup_no_ts_update(addr).is_some(),
                "Address should be cached {:#x}",
                addr
            );
        }
    }

    #[test]
    fn test_cacheline_modification() {
        let mut cache = L1Cache::new();
        let mut memory = Memory::new();

        // Set up first cache line in the set
        let base_addr = 0x1000; // Set 0x80
        let record = MemoryRecord::default();
        memory.insert(base_addr, record);
        cache.insert(base_addr, &mut memory);

        // Verify first cache hit
        assert!(
            cache.lookup_no_ts_update(base_addr).is_some(),
            "First address should be cached"
        );

        // Add second address to same set
        let second_addr = base_addr + (1 << 14); // Same set, different tag
        memory.insert(second_addr, record);
        cache.insert(second_addr, &mut memory);

        // Both addresses should still be cached (8-way set associative)
        assert!(
            cache.lookup_no_ts_update(base_addr).is_some(),
            "First address should still be cached"
        );
        assert!(
            cache.lookup_no_ts_update(second_addr).is_some(),
            "Second address should be cached"
        );

        // Add 6 more addresses to same set to fill up all 8 ways
        for i in 2..8 {
            let addr = base_addr + (i << 14); // Same set, different tag
            memory.insert(addr, record);
            cache.insert(addr, &mut memory);
            assert!(
                cache.lookup_no_ts_update(addr).is_some(),
                "Address {} should be cached", i
            );
        }

        // Access first address multiple times to increase its hits
        for _ in 0..5 {
            cache.lookup_no_ts_update(base_addr);
        }

        // Add ninth address to same set - should evict the least recently used address
        let ninth_addr = base_addr + (8 << 14); // Same set, different tag
        memory.insert(ninth_addr, record);
        cache.insert(ninth_addr, &mut memory);

        // First address should still be cached due to high hits count
        assert!(
            cache.lookup_no_ts_update(base_addr).is_some(),
            "First address should still be cached due to high hits"
        );
        // Ninth address should be cached
        assert!(
            cache.lookup_no_ts_update(ninth_addr).is_some(),
            "Ninth address should be cached"
        );
    }

    #[test]
    fn test_cache_memory_consistency() {
        let mut cache = L1Cache::new();
        let mut memory = Memory::new();

        // Create a sequence of addresses that will map to the same cache set
        let addrs = create_same_set_addresses(0x1000, L1Cache::WAYS as u32 + 1);

        // Insert initial values into memory
        for &addr in &addrs {
            memory.insert(addr, MemoryRecord::default());
        }

        // Insert all addresses into cache
        for &addr in &addrs {
            cache.insert_and_return(addr, &mut memory);
        }

        // All addresses except the first one should be cached (8-way set)
        for &addr in addrs[1..].iter() {
            println!("addr {}", addr);
            let set = set_from_addr(addr);
            println!("memory {:?}", memory);
            println!("cache {:?}", cache.cache[set]);
            println!("lookup result {:?}", cache.lookup_no_ts_update(addr));
            assert!(
                cache.lookup_no_ts_update(addr).is_some(),
                "Address {} should be cached", addr
            );
        }
        // Last address (WAYS + 1) should have evicted one of the previous addresses
        assert!(
            cache.lookup_no_ts_update(addrs[L1Cache::WAYS]).is_some(),
            "Last address should be cached"
        );
        assert!(
            cache.lookup_no_ts_update(addrs[2]).is_some(),
            "2nd address should still has been evicted"
        );
        assert!(
            cache.lookup_no_ts_update(addrs[2]).is_some(),
            "3d address should still be cached"
        );

        // Verify memory consistency after eviction
        assert!(
            memory.contains_key(&addrs[0]),
            "Evicted address should still be in memory"
        );
    }

    #[test]
    fn test_l1cache_hits_replacement() {
        let mut cache = L1Cache::new();
        let mut memory = Memory::new();

        // Create three addresses that map to the same set
        let addrs = create_same_set_addresses(0x1000, 9);

        // Insert values for all addresses
        for &addr in &addrs {
            memory.insert(addr, MemoryRecord::default());
        }

        // Insert first two addresses
        for addr in addrs.iter().skip(1) {
            cache.insert_and_return(*addr, &mut memory);
        }

        // Access first 2..WAYS-1 address to make them MRU
        for addr in addrs.iter().skip(2) {
            assert!(cache.lookup_no_ts_update(*addr).is_some());
        }

        // Insert evicting address - should evict second address (hits)
        cache.insert_and_return(addrs[0], &mut memory);

        // Verify first and third addresses are in cache
        for addr in addrs.iter().skip(2) {
            assert!(
                cache.lookup_no_ts_update(*addr).is_some(),
                "MRU entry was incorrectly evicted"
            );
        }
        assert!(
            cache.lookup_no_ts_update(addrs[1]).is_none(),
            "MRU entry was not evicted"
        );
    }
}
