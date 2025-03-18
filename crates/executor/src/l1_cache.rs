//! Cache and execution state management for the RISC-V emulator.
//!
//! This module implements an 8-way set-associative L1 cache with hits replacement
//! policy, along with the core execution state tracking for the emulator.

use serde::{Deserialize, Serialize};
use sp1_primitives::consts::WORD_SIZE;
use std::{arch::x86_64::*, ops::Neg};

use crate::{events::MemoryRecord, Memory};

#[derive(Debug)]
pub enum CacheLookupResult<T> {
    Hit(T),
    Miss,
}

fn record_from_memory(addr: u32, memory: &mut Memory) -> Option<MemoryRecord> {
    memory.get(&addr).cloned()
}


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
            data[offset] = record_from_memory(addr, memory);
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
        Self { cache }
    }

    #[inline(always)]
    pub fn get_cacheline_mut(&mut self, addr: u32) -> Option<&mut CacheLine> {
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

            // if addr == 2091468 {
            //     println!("tags: {:?}", tags);
            //     let b = _mm256_set1_epi32(tag as i32);
            //     println!("b: {:?}", b);
            // }
            // Compare tags
            let tag_match = _mm256_cmpeq_epi32(tags, _mm256_set1_epi32(tag as i32));
            let valid_match = _mm256_and_si256(tag_match, valid_bits);
            let mask = _mm256_movemask_ps(_mm256_castsi256_ps(valid_match)) as u32;
            // if addr == 2091468 {
            //     println!("tag: {:?}", tag);
            //     println!("tag_match: {:?}", tag_match);
            //     println!("valid_match: {:?}", valid_match);
            //     println!("mask: {:?}", mask);
            // }

            if mask != 0 {
                // Find first matching way
                let way = mask.trailing_zeros() as usize;
                set_lines[way].hits += 1;
                // let offset = offset_from_addr(addr);
                return Some(&mut set_lines[way]);
            }
        }
        None
    }

    #[inline(always)]
    pub fn get(&mut self, addr: u32) -> CacheLookupResult<Option<&MemoryRecord>> {
        let cacheline = self.get_cacheline_mut(addr);
        match cacheline {
            Some(cacheline) => {
                let offset = offset_from_addr(addr);
                CacheLookupResult::Hit(cacheline.data[offset].as_ref())
            }
            None => CacheLookupResult::Miss,
        }
    }

    #[inline(always)]
    pub fn get_mut(&mut self, addr: u32) -> CacheLookupResult<Option<&mut MemoryRecord>> {
        let cacheline = self.get_cacheline_mut(addr);
        match cacheline {
            Some(cacheline) => {
                let offset = offset_from_addr(addr);
                CacheLookupResult::Hit(cacheline.data[offset].as_mut())
            }
            None => CacheLookupResult::Miss,
        }
    }

    #[inline(always)]
    pub fn insert_and_return(&mut self, addr: u32, memory: &mut Memory) -> Option<&MemoryRecord> {
        let set = set_from_addr(addr);
        let tag = tag_from_addr(addr);
        let offset = offset_from_addr(addr);
        let base_addr = addr & !(CacheLine::LINE_SIZE * 4 - 1) as u32;
        let set_lines = unsafe { self.cache.get_unchecked_mut(set) };

        unsafe {
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

            let tag_match = _mm256_cmpeq_epi32(tags, _mm256_set1_epi32(tag as i32));
            let valid_match = _mm256_and_si256(tag_match, valid_bits);
            let mask = _mm256_movemask_ps(_mm256_castsi256_ps(valid_match)) as u32;
            if mask != 0 {
                // println!("mask: {:08b}", mask);
                // Find first matching way
                let way = mask.trailing_zeros() as usize;
                // let new_cacheline = CacheLine::from_memory(tag, base_addr, memory);
                let memory_record = record_from_memory(addr, memory);
                set_lines[way].data[offset] = memory_record;
                // self.update_if_in_cache(addr, memory_record);
                // store_cacheline_if_needed(&set_lines[way], memory); // WIP
                // set_lines[way] = new_cacheline;
                return set_lines[way].data[offset].as_ref();
            }

            // Find invalid or least recently used way
            let invalid_mask = _mm256_xor_si256(valid_bits, _mm256_set1_epi32(-1));
            // println!("invalid_mask: {:?}", invalid_mask);
            let invalid_bits = _mm256_movemask_ps(_mm256_castsi256_ps(invalid_mask)) as u32;

            // println!("invalid_bits: {:08b}", invalid_bits);
            let way = if invalid_bits != 0 {
                // Use first invalid way if available
                invalid_bits.trailing_zeros() as usize
            } else {
                // Find way with lowest hits or same tag || base_addr.
                let mut min_hits = u8::MAX;
                let mut min_way = 0;
                for way in 0..Self::WAYS {
                    if set_lines[way].hits < min_hits {
                        min_hits = set_lines[way].hits;
                        min_way = way;
                    }
                    // println!(
                    //     "way {} tag {} base_addr {}",
                    //     way, set_lines[way].tag, set_lines[way].base_addr
                    // );
                    if set_lines[way].tag == tag || set_lines[way].base_addr == base_addr {
                        min_way = way;
                        break;
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

    #[inline(always)]
    pub fn insert_and_return_mut(
        &mut self,
        addr: u32,
        memory: &mut Memory,
    ) -> Option<&mut MemoryRecord> {
        let set = set_from_addr(addr);
        let tag = tag_from_addr(addr);
        let offset = offset_from_addr(addr);
        let base_addr = addr & !(CacheLine::LINE_SIZE * 4 - 1) as u32;
        let set_lines = unsafe { self.cache.get_unchecked_mut(set) };

        unsafe {
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

            let tag_match = _mm256_cmpeq_epi32(tags, _mm256_set1_epi32(tag as i32));
            let valid_match = _mm256_and_si256(tag_match, valid_bits);
            let mask = _mm256_movemask_ps(_mm256_castsi256_ps(valid_match)) as u32;
            if mask != 0 {
                let way = mask.trailing_zeros() as usize;
                // let new_cacheline = CacheLine::from_memory(tag, base_addr, memory);
                // store_cacheline_if_needed(&set_lines[way], memory); // WIP
                // set_lines[way] = new_cacheline;
                let memory_record = record_from_memory(addr, memory);
                set_lines[way].data[offset] = memory_record;
                return set_lines[way].data[offset].as_mut();
            }

            // Find invalid or least recently used way
            let invalid_mask = _mm256_xor_si256(valid_bits, _mm256_set1_epi32(-1));
            // println!("invalid_mask: {:?}", invalid_mask);
            let invalid_bits = _mm256_movemask_ps(_mm256_castsi256_ps(invalid_mask)) as u32;

            // println!("invalid_bits: {:08b}", invalid_bits);
            let way = if invalid_bits != 0 {
                // Use first invalid way if available
                invalid_bits.trailing_zeros() as usize
            } else {
                // Find way with lowest hits or same tag || base_addr.
                let mut min_hits = u8::MAX;
                let mut min_way = 0;
                for way in 0..Self::WAYS {
                    if set_lines[way].hits < min_hits {
                        min_hits = set_lines[way].hits;
                        min_way = way;
                    }
                    // println!(
                    //     "way {} tag {} base_addr {}",
                    //     way, set_lines[way].tag, set_lines[way].base_addr
                    // );
                    if set_lines[way].tag == tag || set_lines[way].base_addr == base_addr {
                        min_way = way;
                        break;
                    }
                }
                min_way
            };

            // println!("insert_and_return addr {:?}, set {} replacing way {}", addr, set, way);

            store_cacheline_if_needed(&set_lines[way], memory);
            let new_cacheline = CacheLine::from_memory(tag, base_addr, memory);
            set_lines[way] = new_cacheline;
            set_lines[way].data[offset].as_mut()
        }
    }

    #[inline(always)]
    /// updates MemoryRecord for a specific addr
    /// Used by ExitUnconstrainedSyscall to sync memory
    pub fn update_if_in_cache(&mut self, addr: u32, memory_record: Option<MemoryRecord>) {
        if let Some(cacheline) = self.get_cacheline_mut(addr) {
            let offset = offset_from_addr(addr);
            cacheline.data[offset] = memory_record;
        }
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

    #[test]
    fn test_same_base_addr_mapping() {
        let mut cache = L1Cache::new();
        let mut memory = Memory::new();

        // Create two addresses in the same cache line (same base address)
        let base_addr = 2091468;
        let addr1 = base_addr;

        // Create memory records
        let record1 = MemoryRecord {
            value: 1234,
            shard: 0,
            timestamp: 0,
        };
        memory.insert(addr1, record1.clone());

        // Insert first record and verify it's in cache
        cache.insert_and_return(addr1, &mut memory);
        cache.insert_and_return(addr1, &mut memory);

        let set = set_from_addr(addr1);
        println!("cache set: {:?}", cache.cache[set]);
    }

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
            match cache.get(addr) {
                CacheLookupResult::Hit(Some(_)) => {},
                _ => panic!("Address {} should be cached", i)
            }
        }

        // Modify all cached values
        for (i, addr) in addresses.iter().enumerate() {
            if let CacheLookupResult::Hit(Some(cached)) = cache.get_mut(*addr) {
                cached.value = (i as u32 + 1) * 100;
            }
        }

        // Access first address multiple times to increase its hits
        for &addr in addresses[..L1Cache::WAYS - 1].iter() {
            for _ in 0..5 {
                cache.get(addr);
            }
        }

        let target_value = match cache.get(*addresses.last().unwrap()) {
            CacheLookupResult::Hit(value) => value.cloned(),
            CacheLookupResult::Miss => None
        };
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
        assert!(
            matches!(cache.get(addresses[0]), CacheLookupResult::Hit(Some(_))),
            "First address should still be cached due to high hits"
        );

        assert_eq!(target_value, memory.get(addresses.last().unwrap()).cloned());
    }

    #[test]
    fn test_l1cache_lookup_miss() {
        let mut cache = L1Cache::new();
        let addr = 0x1000;
        assert!(matches!(cache.get(addr), CacheLookupResult::Miss));
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
        cache.insert_and_return(base_addr, &mut memory);

        // Verify all addresses in the cache line are cached
        for (i, &val) in values.iter().enumerate() {
            let addr = base_addr + (i * WORD_SIZE) as u32;
            let record = match cache.get(addr) {
                CacheLookupResult::Hit(Some(record)) => record,
                _ => panic!("Address should be cached")
            };
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
        cache.insert_and_return(addr, &mut memory);

        // Verify lookup succeeds
        let result = cache.get(addr);
        assert!(matches!(result, CacheLookupResult::Hit(Some(_))));
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
            cache.insert_and_return(addr, &mut memory);

            assert!(
                matches!(cache.get(addr), CacheLookupResult::Hit(Some(_))),
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
        cache.insert_and_return(addr1, &mut memory);

        // Insert second address
        let record2 = MemoryRecord {
            shard: 84,
            timestamp: 84,
            value: 84,
        };
        memory.insert(addr2, record2);
        cache.insert_and_return(addr2, &mut memory);

        // Both should be in cache (8-way set associative)
        let cached1 = match cache.get(addr1) {
            CacheLookupResult::Hit(Some(record)) => record,
            _ => panic!("First address should be cached")
        };
        assert_eq!(*cached1, record1);
        let cached2 = match cache.get(addr2) {
            CacheLookupResult::Hit(Some(record)) => record,
            _ => panic!("Second address should be cached")
        };
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
            cache.insert_and_return(addr, &mut memory);

            // Verify it's cached
            let cached = match cache.get(addr) {
                CacheLookupResult::Hit(Some(record)) => record,
                _ => panic!("Address should be cached")
            };
            assert_eq!(*cached, record);

            // Previous addresses should still be cached
            assert!(
                matches!(cache.get(addr1), CacheLookupResult::Hit(Some(_))),
                "First address should still be cached"
            );
            assert!(
                matches!(cache.get(addr2), CacheLookupResult::Hit(Some(_))),
                "Second address should still be cached"
            );
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
                cache.insert_and_return(addr, &mut memory);
                if matches!(cache.get(addr), CacheLookupResult::Hit(Some(_))) {
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
        cache.insert_and_return(addr, &mut memory);

        // Should still create a cache line with default values
        assert!(
            matches!(cache.get(addr), CacheLookupResult::Miss),
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
            cache.insert_and_return(addr, &mut memory);
        }

        // Verify all addresses still cached
        for &addr in &addrs {
            assert!(
                matches!(cache.get(addr), CacheLookupResult::Hit(Some(_))),
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
        cache.insert_and_return(base_addr, &mut memory);

        // Verify first cache hit
        assert!(
            matches!(cache.get(base_addr), CacheLookupResult::Hit(Some(_))),
            "First address should be cached"
        );

        // Add second address to same set
        let second_addr = base_addr + (1 << 14); // Same set, different tag
        memory.insert(second_addr, record);
        cache.insert_and_return(second_addr, &mut memory);

        // Both addresses should still be cached (8-way set associative)
        assert!(
            matches!(cache.get(base_addr), CacheLookupResult::Hit(Some(_))),
            "First address should still be cached"
        );
        assert!(
            matches!(cache.get(second_addr), CacheLookupResult::Hit(Some(_))),
            "Second address should be cached"
        );

        // Add 6 more addresses to same set to fill up all 8 ways
        for i in 2..8 {
            let addr = base_addr + (i << 14); // Same set, different tag
            memory.insert(addr, record);
            cache.insert_and_return(addr, &mut memory);
            assert!(matches!(cache.get(addr), CacheLookupResult::Hit(Some(_))), "Address {} should be cached", i);
        }

        // Access first address multiple times to increase its hits
        for _ in 0..5 {
            cache.get(base_addr);
        }

        // Add ninth address to same set - should evict the least recently used address
        let ninth_addr = base_addr + (8 << 14); // Same set, different tag
        memory.insert(ninth_addr, record);
        cache.insert_and_return(ninth_addr, &mut memory);

        // First address should still be cached due to high hits count
        assert!(
            matches!(cache.get(base_addr), CacheLookupResult::Hit(Some(_))),
            "First address should still be cached due to high hits"
        );
        // Ninth address should be cached
        assert!(
            matches!(cache.get(ninth_addr), CacheLookupResult::Hit(Some(_))),
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
        for (i, &addr) in addrs.iter().enumerate() {
            memory.insert(
                addr,
                MemoryRecord {
                    shard: 0,
                    timestamp: 0,
                    value: (i + 1) as u32,
                },
            );
        }

        // Insert all addresses into cache
        for &addr in &addrs {
            cache.insert_and_return(addr, &mut memory);
        }

        // All addresses except the first one should be cached (8-way set)
        for &addr in addrs[1..].iter() {
            assert!(
                matches!(cache.get(addr), CacheLookupResult::Hit(Some(_))),
                "Address {} should be cached",
                addr
            );
        }
        // Last address (WAYS + 1) should have evicted one of the previous addresses
        assert!(
            matches!(cache.get(addrs[L1Cache::WAYS]), CacheLookupResult::Hit(Some(_))),
            "Last address should be cached"
        );
        assert!(
            matches!(cache.get(addrs[2]), CacheLookupResult::Hit(Some(_))),
            "2nd address should still has been evicted"
        );
        assert!(
            matches!(cache.get(addrs[2]), CacheLookupResult::Hit(Some(_))),
            "3d address should still be cached"
        );

        assert!(
            matches!(cache.get(addrs[0]), CacheLookupResult::Miss),
            "First address must be not in cache"
        );

        // Verify memory consistency after eviction
        assert!(
            memory.contains_key(&addrs[0]),
            "Evicted address should still be in memory"
        );
        println!("memory {:?}", memory);
        // println!("cache {:?}", cache.cache[set]);
        // println!("lookup result {:?}", cache.get(addr));
        println!("addr {}", addrs[0]);
        println!("memory lookup result {:?}", memory.get(&addrs[0]));
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
            assert!(matches!(cache.get(*addr), CacheLookupResult::Hit(Some(_))));
        }

        // Insert evicting address - should evict second address (hits)
        cache.insert_and_return(addrs[0], &mut memory);

        // Verify first and third addresses are in cache
        for addr in addrs.iter().skip(2) {
            assert!(
                matches!(cache.get(*addr), CacheLookupResult::Hit(Some(_))),
                "MRU entry was incorrectly evicted"
            );
        }
        assert!(matches!(cache.get(addrs[1]), CacheLookupResult::Miss), "MRU entry was not evicted");
    }

    #[test]
    fn test_empty_read() {
        let mut cache = L1Cache::new();
        let mut memory = Memory::new();

        // Create three addresses that map to the same set

        let addr = 0x2000;

        cache.insert_and_return(addr, &mut memory);
        memory.insert(addr, MemoryRecord::default());
        let set = set_from_addr(addr);
        println!("cache {:?}", cache.cache[set]);
        println!("memory after {:?}", memory);

        assert!(matches!(cache.get(addr), CacheLookupResult::Hit(Some(_))));
    }
}
