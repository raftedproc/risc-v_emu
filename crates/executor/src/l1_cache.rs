//! Cache and execution state management for the RISC-V emulator.
//!
//! This module implements a 2-way set-associative L1 cache with hits replacement
//! policy, along with the core execution state tracking for the emulator.

use serde::{Deserialize, Serialize};
use sp1_primitives::consts::WORD_SIZE;

use crate::{events::MemoryRecord, Memory, UnInitMemory};

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
    data: [MemoryRecord; Self::LINE_SIZE],
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
            data: [MemoryRecord::default(); Self::LINE_SIZE],
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
    fn from_memory_and_uninit_memory(
        tag: u32,
        base_addr: u32,
        memory: &mut Memory,
        uninit_memory: &UnInitMemory,
    ) -> Self {
        let valid = true;
        let hits = 0;
        let mut data = [MemoryRecord::default(); Self::LINE_SIZE];
        for offset in 0..Self::LINE_SIZE {
            let addr = addr_from_base_addr_offset(base_addr, offset);
            if addr == 9082224 {
                println!(
                    "from_memory_and_uninit_memory: 9082224 memory {:?} uninit memory {:?}",
                    memory.get(&addr),
                    uninit_memory.get(&addr)
                );
            }
            let record = memory.get(&addr).cloned().unwrap_or_else(|| {
                let value = *uninit_memory.get(&addr).unwrap_or(&0);
                MemoryRecord {
                    value,
                    timestamp: 0,
                    shard: 0,
                }
            });
            data[offset] = record;
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

/// L1 cache implementation with 2-way set-associative mapping
///
/// The cache is organized as:
/// - 256 sets (indexed by address bits [5:13])
/// - 2 ways per set (managed by hits replacement)
/// - 32 words per cache line
///
/// This structure provides efficient memory access through caching while
/// maintaining consistency with main memory through write-back policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct L1Cache {
    /// Cache data organized as sets of cache lines
    cache: Vec<Vec<CacheLine>>,
    // /// Tag to set mapping for quick lookups
    // tag_set: HashMap<u32, u32>,
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
    const WAYS: usize = 2;

    /// Creates a new empty L1 cache with pre-allocated sets and ways
    ///
    /// # Returns
    /// A new L1Cache instance with SETS × WAYS empty cache lines
    pub fn new() -> Self {
        let cache = vec![vec![CacheLine::default(); Self::WAYS]; Self::SETS];
        Self {
            cache,
            // tag_set: HashMap::new(),
        }
    }

    #[inline(always)]
    /// Looks up a memory address in the cache
    ///
    /// This function checks both ways in the set-associative cache for a matching tag.
    /// If found, returns a mutable reference to the corresponding memory record.
    ///
    /// # Arguments
    /// * `addr` - Memory address to look up
    ///
    /// # Returns
    /// * `Some(&mut MemoryRecord)` if the address is in cache
    /// * `None` if the address is not in cache (cache miss)
    pub fn lookup(&mut self, addr: u32, timestamp: u32) -> Option<&MemoryRecord> {
        let set: usize = set_from_addr(addr);
        let tag = tag_from_addr(addr);

        let set_lines = unsafe { self.cache.get_unchecked_mut(set) };

        // TODO multiway
        if set_lines[0].valid && set_lines[0].tag == tag {
            set_lines[0].hits += 1;
            let offset = offset_from_addr(addr);
            set_lines[0].data[offset].timestamp = timestamp;
            return Some(&set_lines[0].data[offset]);
        }

        if set_lines[1].valid && set_lines[1].tag == tag {
            set_lines[1].hits += 1;
            let offset = offset_from_addr(addr);
            set_lines[1].data[offset].timestamp = timestamp;
            return Some(&set_lines[1].data[offset]);
        }
        None
    }

    pub fn lookup_no_ts_update(&mut self, addr: u32) -> Option<&MemoryRecord> {
        let set: usize = set_from_addr(addr);
        let tag = tag_from_addr(addr);

        let set_lines = unsafe { self.cache.get_unchecked_mut(set) };

        // TODO multiway
        if set_lines[0].valid && set_lines[0].tag == tag {
            set_lines[0].hits += 1;
            let offset = offset_from_addr(addr);
            return Some(&set_lines[0].data[offset]);
        }

        if set_lines[1].valid && set_lines[1].tag == tag {
            set_lines[1].hits += 1;
            let offset = offset_from_addr(addr);
            return Some(&set_lines[1].data[offset]);
        }
        None
    }

    pub fn lookup_mut(&mut self, addr: u32, timestamp: u32) -> Option<&mut MemoryRecord> {
        // println!("cache lookup_mut: {}", addr);

        let set: usize = set_from_addr(addr);
        let tag = tag_from_addr(addr);

        let set_lines = unsafe { self.cache.get_unchecked_mut(set) };

        // TODO multiway
        if set_lines[0].valid && set_lines[0].tag == tag {
            set_lines[0].hits += 1;
            let offset = offset_from_addr(addr);
            set_lines[0].data[offset].timestamp = timestamp;
            return Some(&mut set_lines[0].data[offset]);
        }

        if set_lines[1].valid && set_lines[1].tag == tag {
            set_lines[1].hits += 1;
            let offset = offset_from_addr(addr);
            set_lines[1].data[offset].timestamp = timestamp;
            return Some(&mut set_lines[1].data[offset]);
        }
        None
    }

    pub fn lookup_mut_no_ts_update(&mut self, addr: u32) -> Option<&mut MemoryRecord> {
        // println!("cache lookup_mut: {}", addr);

        let set: usize = set_from_addr(addr);
        let tag = tag_from_addr(addr);

        let set_lines = unsafe { self.cache.get_unchecked_mut(set) };

        // TODO multiway
        if set_lines[0].valid && set_lines[0].tag == tag {
            set_lines[0].hits += 1;
            let offset = offset_from_addr(addr);
            return Some(&mut set_lines[0].data[offset]);
        }

        if set_lines[1].valid && set_lines[1].tag == tag {
            set_lines[1].hits += 1;
            let offset = offset_from_addr(addr);
            return Some(&mut set_lines[1].data[offset]);
        }
        None
    }

    #[inline(always)]
    /// Inserts a new cache line for the given address
    ///
    /// Uses hits (Least Recently Used) replacement policy to choose which way to evict
    /// when both ways in a set are occupied. The evicted line is written back to memory
    /// if it contains valid data.
    ///
    /// # Arguments
    /// * `addr` - Memory address to cache
    /// * `memory` - Memory to load data from
    pub fn insert(&mut self, addr: u32, memory: &mut Memory, uninit_memory: &UnInitMemory) {
        let set = set_from_addr(addr);
        let tag = tag_from_addr(addr);

        let base_addr = addr & !(CacheLine::LINE_SIZE * 4 - 1) as u32; // WIP

        let set_lines = unsafe { self.cache.get_unchecked_mut(set) };

        // TODO watch out re-insertion
        let cacheline = if set_lines[0].hits <= set_lines[1].hits && !set_lines[0].valid {
            &mut set_lines[0]
        } else {
            &mut set_lines[1]
        };

        store_cacheline_if_needed(set, cacheline, memory);
        let new_cacheline =
            CacheLine::from_memory_and_uninit_memory(tag, base_addr, memory, uninit_memory);
        *cacheline = new_cacheline;
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
        cacheline.data[offset] = memory_record;
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
/// * `set` - Cache set index
/// * `cacheline` - Cache line to write back
/// * `memory` - Memory to write data to
fn store_cacheline_if_needed(set: usize, cacheline: &CacheLine, memory: &mut Memory) {
    // todo clean this up
    // println!(
    //     "store_cacheline_if_needed cacheline addr {} {:b}",
    //     cacheline.base_addr, cacheline.base_addr
    // );
    if cacheline.valid {
        let tag = cacheline.tag;
        let base_addr = cacheline.base_addr;
        for (offset, record) in cacheline.data.iter().enumerate() {
            let addr = base_addr | (offset * WORD_SIZE) as u32;
            // println!(
            //     "store_cacheline_if_needed addr {} cached_word: {:?}",
            //     addr, record
            // );
            memory.insert(addr, *record);
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

/// Reconstructs the base memory address of a cache line from its tag and set
///
/// Combines the tag (upper bits) and set index (middle bits) to form the
/// base address of a cache line. The offset bits are set to 0.
#[inline(always)]
pub fn cacheline_from_tag_set(tag: u32, set: usize) -> u32 {
    tag << 13 | (set << 5) as u32
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
    }

    #[test]
    fn test_cacheline_from_memory_and_uninit_memory() {
        let mut memory = Memory::new();
        let uninit_memory = UnInitMemory::new();
        let addr = 0x1000;
        let tag = tag_from_addr(addr);
        let record = MemoryRecord {
            shard: 42,
            timestamp: 42,
            value: 42,
        };
        memory.insert(addr, record);

        let line = CacheLine::from_memory_and_uninit_memory(tag, addr, &mut memory, &uninit_memory);
        assert!(line.valid);
        assert_eq!(line.tag, tag);
        assert_eq!(line.hits, 0);
        assert_eq!(line.data[0], record);
        assert_eq!(line.data[1], MemoryRecord::default());
    }

    #[test]
    fn test_l1cache_new() {
        let cache = L1Cache::new();
        assert_eq!(cache.cache.len(), L1Cache::SETS);
        for set in &cache.cache {
            assert_eq!(set.len(), L1Cache::WAYS);
        }
        // assert!(cache.tag_set.is_empty());
    }

    #[test]
    fn test_cache_to_memory_sync() {
        let mut cache = L1Cache::new();
        let mut memory = Memory::new();

        // Insert initial value
        let evicting_addr = 0xfaeff000;

        memory.insert(evicting_addr, MemoryRecord::default());
        println!("before first insert {} {:b}", evicting_addr, evicting_addr);
        cache.insert(evicting_addr, &mut memory, &UnInitMemory::new());

        let addr = evicting_addr + (1 << 14); // Different tag, same set
        println!("evicting_addr: {}", addr);
        let record = MemoryRecord {
            shard: 42,
            timestamp: 42,
            value: 42,
        };

        memory.insert(addr, record);
        println!("memory: {:?}", memory);
        println!("before second insert {}", addr);
        cache.insert(addr, &mut memory, &UnInitMemory::new());

        // Modify cached value
        if let Some(cached) = cache.lookup_mut(addr, 0) {
            cached.value = 100;
        }

        // Force writeback by inserting to same set
        let another_evicting_addr = addr + (2 << 14); // Different tag, same set
        println!("before 3d insert {}", another_evicting_addr);
        cache.insert(another_evicting_addr, &mut memory, &UnInitMemory::new());

        println!("memory: {:?}", memory);
        // Verify written back value
        assert_eq!(
            memory.get(&addr).unwrap().value,
            100,
            "Modified value not written back"
        );
    }

    #[test]
    fn test_cache_address_calculations() {
        let addr = 0x12345678;
        let set = set_from_addr(addr);
        let tag = tag_from_addr(addr);
        let offset = offset_from_addr(addr);

        // Verify set is within bounds
        assert!(set < L1Cache::SETS);

        // Verify offset is within bounds
        assert!(offset < CacheLine::LINE_SIZE);

        // Verify we can reconstruct the address (ignoring offset)
        let base_addr = cacheline_from_tag_set(tag, set);
        assert_eq!(addr & !0x1F, base_addr);
    }

    #[test]
    fn test_l1cache_lookup_miss() {
        let mut cache = L1Cache::new();
        let addr = 0x1000;
        assert!(cache.lookup(addr, 0).is_none());
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
        cache.insert(base_addr, &mut memory, &UnInitMemory::new());

        // Verify all addresses in the cache line are cached
        for (i, &val) in values.iter().enumerate() {
            let addr = base_addr + (i * WORD_SIZE) as u32;
            let record = cache.lookup(addr, 0).expect("Address should be cached");
            assert_eq!(record.value, val, "Cached value mismatch");
        }
    }

    #[test]
    fn test_l1cache_insert_and_lookup() {
        let mut cache = L1Cache::new();
        let mut memory = Memory::new();
        let addr = 0x1000;
        // WIP
        let value = MemoryRecord::default();
        memory.insert(addr, value);

        // Insert into cache
        cache.insert(addr, &mut memory, &UnInitMemory::new());

        // Verify lookup succeeds
        let result = cache.lookup(addr, 0);
        assert!(result.is_some());
    }

    #[test]
    fn test_cache_boundary_addresses() {
        let mut cache = L1Cache::new();
        let mut memory = Memory::new();

        // Test cache line boundary addresses
        let addr1 = 0x1F; // Last offset in a cache line
        let addr2 = 0x20; // First offset in next cache line
        let addr3 = 0x1FE0; // Last offset in a set
        let addr4 = 0x2000; // First offset in next set

        for addr in [addr1, addr2, addr3, addr4] {
            memory.insert(addr, MemoryRecord::default());
            cache.insert(addr, &mut memory, &UnInitMemory::new());
            assert!(
                cache.lookup(addr, 0).is_some(),
                "Failed to cache address {:#x}",
                addr
            );
        }

        // Verify address calculations at boundaries
        assert_eq!(offset_from_addr(addr1), 0x7);
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
        cache.insert(addr1, &mut memory, &UnInitMemory::new());

        // Insert second address
        let record2 = MemoryRecord {
            shard: 84,
            timestamp: 84,
            value: 84,
        };
        memory.insert(addr2, record2);
        cache.insert(addr2, &mut memory, &UnInitMemory::new());

        // Both should be in cache (2-way set associative)
        let cached1 = cache
            .lookup(addr1, 0)
            .expect("First address should be cached");
        assert_eq!(*cached1, record1);
        let cached2 = cache
            .lookup(addr2, 0)
            .expect("Second address should be cached");
        assert_eq!(*cached2, record2);
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
                cache.insert(addr, &mut memory, &UnInitMemory::new());
                if cache.lookup(addr, 0).is_some() {
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
        cache.insert(addr, &mut memory, &UnInitMemory::new());

        // Should still create a cache line with default values
        assert!(
            cache.lookup(addr, 0).is_some(),
            "Address should be cached with default values"
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
            cache.insert(addr, &mut memory, &UnInitMemory::new());
        }

        // Verify all addresses still cached
        for &addr in &addrs {
            assert!(
                cache.lookup(addr, 0).is_some(),
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
        cache.insert(base_addr, &mut memory, &UnInitMemory::new());

        // Verify first cache hit
        assert!(
            cache.lookup(base_addr, 0).is_some(),
            "First address should be cached"
        );

        // Add second address to same set
        let second_addr = base_addr + (1 << 14); // Same set, different tag
        memory.insert(second_addr, record);
        cache.insert(second_addr, &mut memory, &UnInitMemory::new());

        // Both addresses should still be cached (2-way set associative)
        assert!(
            cache.lookup(base_addr, 0).is_some(),
            "First address should still be cached"
        );
        assert!(
            cache.lookup(second_addr, 0).is_some(),
            "Second address should be cached"
        );

        // Access first address to update its hits counter
        cache.lookup(base_addr, 0);

        // Add third address to same set - should evict second address (hits)
        let third_addr = second_addr + (1 << 14); // Same set, different tag
        memory.insert(third_addr, record);
        cache.insert(third_addr, &mut memory, &UnInitMemory::new());

        // First and third addresses should be cached, second should be evicted
        assert!(
            cache.lookup(base_addr, 0).is_some(),
            "First address should still be cached"
        );
        assert!(
            cache.lookup(second_addr, 0).is_none(),
            "Second address should have been evicted"
        );
        assert!(
            cache.lookup(third_addr, 0).is_some(),
            "Third address should be cached"
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
            cache.insert(addr, &mut memory, &UnInitMemory::new());
        }

        // Last address should have evicted the first one due to hits policy
        assert!(
            cache.lookup(addrs[0], 0).is_some(),
            "First address should still be cached"
        );
        assert!(
            cache.lookup(addrs[2], 0).is_some(),
            "2nd address should still has been evicted"
        );
        assert!(
            cache.lookup(addrs[2], 0).is_some(),
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
        let addrs = create_same_set_addresses(0x1000, 3);

        // Insert values for all addresses
        for &addr in &addrs {
            memory.insert(addr, MemoryRecord::default());
        }

        // Insert first two addresses
        cache.insert(addrs[0], &mut memory, &UnInitMemory::new());
        cache.insert(addrs[1], &mut memory, &UnInitMemory::new());

        // Access first address to make it MRU
        assert!(cache.lookup(addrs[0], 0).is_some());

        // Insert third address - should evict second address (hits)
        cache.insert(addrs[2], &mut memory, &UnInitMemory::new());

        // Verify first and third addresses are in cache
        assert!(
            cache.lookup(addrs[1], 0).is_none(),
            "hits entry was not evicted"
        );
        assert!(
            cache.lookup(addrs[0], 0).is_some(),
            "MRU entry was incorrectly evicted"
        );
        assert!(
            cache.lookup(addrs[2], 0).is_some(),
            "Newly inserted entry not found"
        );
    }
}
