/*
 * XSigma: High-Performance Quantitative Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * This file is part of XSigma and is licensed under a dual-license model:
 *
 *   - Open-source License (GPLv3):
 *       Free for personal, academic, and research use under the terms of
 *       the GNU General Public License v3.0 or later.
 *
 *   - Commercial License:
 *       A commercial license is required for proprietary, closed-source,
 *       or SaaS usage. Contact us to obtain a commercial agreement.
 *
 * Contact: licensing@xsigma.co.uk
 * Website: https://www.xsigma.co.uk
 */

#include <algorithm>
#include <string>
#include <vector>

#include "logging/logger.h"
#include "util/flat_hash.h"
#include "xsigmaTest.h"

using namespace xsigma;

// ============================================================================
// Basic flat_hash_map Tests
// ============================================================================

/**
 * @brief Test basic flat_hash_map operations
 *
 * Covers: insert, find, erase, size, empty
 */
XSIGMATEST(Core, flat_hash_map_basic)
{
    flat_hash_map<int, std::string> map;

    // Test empty map
    EXPECT_TRUE(map.empty());
    EXPECT_EQ(map.size(), 0);

    // Test insert
    map[1] = "one";
    map[2] = "two";
    map[3] = "three";

    EXPECT_FALSE(map.empty());
    EXPECT_EQ(map.size(), 3);

    // Test find
    auto it1 = map.find(1);
    EXPECT_TRUE(it1 != map.end());
    EXPECT_EQ(it1->second, "one");

    auto it_not_found = map.find(99);
    EXPECT_TRUE(it_not_found == map.end());

    // Test erase
    map.erase(2);
    EXPECT_EQ(map.size(), 2);
    EXPECT_TRUE(map.find(2) == map.end());

    // Test clear
    map.clear();
    EXPECT_TRUE(map.empty());
    EXPECT_EQ(map.size(), 0);

    END_TEST();
}

/**
 * @brief Test flat_hash_map with various key types
 *
 * Covers: int, string, custom types as keys
 */
XSIGMATEST(Core, flat_hash_map_key_types)
{
    // Test with int keys
    flat_hash_map<int, int> int_map;
    int_map[1] = 100;
    int_map[2] = 200;
    EXPECT_EQ(int_map[1], 100);
    EXPECT_EQ(int_map[2], 200);

    // Test with string keys
    flat_hash_map<std::string, int> str_map;
    str_map["hello"] = 1;
    str_map["world"] = 2;
    EXPECT_EQ(str_map["hello"], 1);
    EXPECT_EQ(str_map["world"], 2);

    END_TEST();
}

// ============================================================================
// Basic flat_hash_set Tests
// ============================================================================

/**
 * @brief Test basic flat_hash_set operations
 *
 * Covers: insert, find, erase, size, empty, contains
 */
XSIGMATEST(Core, flat_hash_set_basic)
{
    flat_hash_set<int> set;

    // Test empty set
    EXPECT_TRUE(set.empty());
    EXPECT_EQ(set.size(), 0);

    // Test insert
    set.insert(1);
    set.insert(2);
    set.insert(3);

    EXPECT_FALSE(set.empty());
    EXPECT_EQ(set.size(), 3);

    // Test find
    EXPECT_TRUE(set.find(1) != set.end());
    EXPECT_TRUE(set.find(2) != set.end());
    EXPECT_TRUE(set.find(99) == set.end());

    // Test contains
    EXPECT_TRUE(set.contains(1));
    EXPECT_TRUE(set.contains(2));
    EXPECT_FALSE(set.contains(99));

    // Test duplicate insert
    auto result = set.insert(1);
    EXPECT_FALSE(result.second);  // Should not insert duplicate
    EXPECT_EQ(set.size(), 3);

    // Test erase
    set.erase(2);
    EXPECT_EQ(set.size(), 2);
    EXPECT_FALSE(set.contains(2));

    // Test clear
    set.clear();
    EXPECT_TRUE(set.empty());

    END_TEST();
}

/**
 * @brief Test flat_hash_set with various value types
 *
 * Covers: int, string, custom types
 */
XSIGMATEST(Core, flat_hash_set_value_types)
{
    // Test with int
    flat_hash_set<int> int_set;
    int_set.insert(1);
    int_set.insert(2);
    int_set.insert(3);
    EXPECT_EQ(int_set.size(), 3);

    // Test with string
    flat_hash_set<std::string> str_set;
    str_set.insert("hello");
    str_set.insert("world");
    EXPECT_TRUE(str_set.contains("hello"));
    EXPECT_TRUE(str_set.contains("world"));
    EXPECT_FALSE(str_set.contains("foo"));

    // Test with pair
    END_TEST();
}

// ============================================================================
// Iteration Tests
// ============================================================================

/**
 * @brief Test iteration over flat_hash containers
 *
 * Covers: begin, end, range-based for loops
 */
XSIGMATEST(Core, flat_hash_iteration)
{
    // Test map iteration
    flat_hash_map<int, std::string> map;
    map[1] = "one";
    map[2] = "two";
    map[3] = "three";

    int count = 0;
    for (const auto& pair : map)
    {
        EXPECT_TRUE(pair.first >= 1 && pair.first <= 3);
        count++;
    }
    EXPECT_EQ(count, 3);

    // Test set iteration
    flat_hash_set<int> set;
    set.insert(1);
    set.insert(2);
    set.insert(3);

    count = 0;
    for (int value : set)
    {
        EXPECT_TRUE(value >= 1 && value <= 3);
        count++;
    }
    EXPECT_EQ(count, 3);

    END_TEST();
}

// ============================================================================
// Capacity and Rehashing Tests
// ============================================================================

/**
 * @brief Test capacity management and rehashing
 *
 * Covers: reserve, rehash, load_factor, max_load_factor
 */
XSIGMATEST(Core, flat_hash_capacity)
{
    flat_hash_map<int, int> map;

    // Test reserve
    map.reserve(100);
    EXPECT_GE(map.bucket_count(), 100);

    // Insert elements
    for (int i = 0; i < 50; ++i)
    {
        map[i] = i * 10;
    }
    EXPECT_EQ(map.size(), 50);

    // Test load factor
    float load = map.load_factor();
    EXPECT_GT(load, 0.0f);
    EXPECT_LE(load, map.max_load_factor());

    // Test max_load_factor
    float old_max = map.max_load_factor();
    map.max_load_factor(0.75f);
    EXPECT_EQ(map.max_load_factor(), 0.75f);
    map.max_load_factor(old_max);  // Restore

    END_TEST();
}

// ============================================================================
// Large Dataset Tests
// ============================================================================

/**
 * @brief Test with large datasets
 *
 * Covers: performance with many elements, memory efficiency
 */
XSIGMATEST(Core, flat_hash_large_dataset)
{
    const int count = 10000;

    // Test map with large dataset
    flat_hash_map<int, int> map;
    for (int i = 0; i < count; ++i)
    {
        map[i] = i * 2;
    }
    EXPECT_EQ(map.size(), static_cast<size_t>(count));

    // Verify all elements
    for (int i = 0; i < count; ++i)
    {
        EXPECT_EQ(map[i], i * 2);
    }

    // Test set with large dataset
    flat_hash_set<int> set;
    for (int i = 0; i < count; ++i)
    {
        set.insert(i);
    }
    EXPECT_EQ(set.size(), static_cast<size_t>(count));

    // Verify all elements
    for (int i = 0; i < count; ++i)
    {
        EXPECT_TRUE(set.contains(i));
    }

    XSIGMA_LOG_INFO("Large dataset test: {} elements processed", count);

    END_TEST();
}

// ============================================================================
// Edge Cases Tests
// ============================================================================

/**
 * @brief Test edge cases and boundary conditions
 *
 * Covers: empty containers, single element, duplicate handling
 */
XSIGMATEST(Core, flat_hash_edge_cases)
{
    // Test empty map operations
    flat_hash_map<int, int> empty_map;
    EXPECT_TRUE(empty_map.find(1) == empty_map.end());
    EXPECT_EQ(empty_map.erase(1), 0);

    // Test single element
    flat_hash_map<int, int> single_map;
    single_map[1] = 100;
    EXPECT_EQ(single_map.size(), 1);
    EXPECT_EQ(single_map[1], 100);

    // Test overwriting value
    single_map[1] = 200;
    EXPECT_EQ(single_map.size(), 1);
    EXPECT_EQ(single_map[1], 200);

    // Test empty set operations
    flat_hash_set<int> empty_set;
    EXPECT_FALSE(empty_set.contains(1));
    EXPECT_EQ(empty_set.erase(1), 0);

    // Test duplicate handling in set
    flat_hash_set<int> set;
    set.insert(1);
    set.insert(1);
    set.insert(1);
    EXPECT_EQ(set.size(), 1);

    END_TEST();
}

// ============================================================================
// Copy and Move Semantics Tests
// ============================================================================

/**
 * @brief Test copy and move semantics
 *
 * Covers: copy constructor, move constructor, assignment operators
 */
XSIGMATEST(Core, flat_hash_copy_move)
{
    // Test map copy constructor
    flat_hash_map<int, std::string> map1;
    map1[1] = "one";
    map1[2] = "two";

    flat_hash_map<int, std::string> map2(map1);
    EXPECT_EQ(map2.size(), 2);
    EXPECT_EQ(map2[1], "one");
    EXPECT_EQ(map2[2], "two");

    // Test map move constructor
    flat_hash_map<int, std::string> map3(std::move(map1));
    EXPECT_EQ(map3.size(), 2);
    EXPECT_EQ(map3[1], "one");

    // Test set copy constructor
    flat_hash_set<int> set1;
    set1.insert(1);
    set1.insert(2);

    flat_hash_set<int> set2(set1);
    EXPECT_EQ(set2.size(), 2);
    EXPECT_TRUE(set2.contains(1));
    EXPECT_TRUE(set2.contains(2));

    // Test set move constructor
    flat_hash_set<int> set3(std::move(set1));
    EXPECT_EQ(set3.size(), 2);

    END_TEST();
}

// ============================================================================
// Equality Tests
// ============================================================================

/**
 * @brief Test equality operators
 *
 * Covers: operator==, operator!=
 */
XSIGMATEST(Core, flat_hash_equality)
{
    // Test set equality
    flat_hash_set<int> set1;
    set1.insert(1);
    set1.insert(2);
    set1.insert(3);

    flat_hash_set<int> set2;
    set2.insert(1);
    set2.insert(2);
    set2.insert(3);

    EXPECT_TRUE(set1 == set2);
    EXPECT_FALSE(set1 != set2);

    // Test inequality
    flat_hash_set<int> set3;
    set3.insert(1);
    set3.insert(2);

    EXPECT_FALSE(set1 == set3);
    EXPECT_TRUE(set1 != set3);

    // Test with different order (should still be equal)
    flat_hash_set<int> set4;
    set4.insert(3);
    set4.insert(1);
    set4.insert(2);

    EXPECT_TRUE(set1 == set4);

    END_TEST();
}

// ============================================================================
// Emplace Tests
// ============================================================================

/**
 * @brief Test emplace operations
 *
 * Covers: emplace, try_emplace, emplace_hint
 */
XSIGMATEST(Core, flat_hash_emplace)
{
    // Test map emplace
    flat_hash_map<int, std::string> map;
    auto                            result1 = map.emplace(1, "one");
    EXPECT_TRUE(result1.second);  // Insertion successful
    EXPECT_EQ(result1.first->second, "one");

    // Test duplicate emplace
    auto result2 = map.emplace(1, "ONE");
    EXPECT_FALSE(result2.second);             // Insertion failed (duplicate)
    EXPECT_EQ(result2.first->second, "one");  // Original value unchanged

    // Test set emplace
    flat_hash_set<int> set;
    auto               result3 = set.emplace(42);
    EXPECT_TRUE(result3.second);

    auto result4 = set.emplace(42);
    EXPECT_FALSE(result4.second);  // Duplicate

    END_TEST();
}

// ============================================================================
// Swap Tests
// ============================================================================

/**
 * @brief Test swap functionality
 *
 * Covers: swap member function, std::swap
 */
XSIGMATEST(Core, flat_hash_swap)
{
    // Test map swap
    flat_hash_map<int, std::string> map1;
    map1[1] = "one";
    map1[2] = "two";

    flat_hash_map<int, std::string> map2;
    map2[3] = "three";
    map2[4] = "four";

    map1.swap(map2);

    EXPECT_EQ(map1.size(), 2);
    EXPECT_EQ(map1[3], "three");
    EXPECT_EQ(map2.size(), 2);
    EXPECT_EQ(map2[1], "one");

    // Test set swap
    flat_hash_set<int> set1;
    set1.insert(1);
    set1.insert(2);

    flat_hash_set<int> set2;
    set2.insert(3);
    set2.insert(4);

    set1.swap(set2);

    EXPECT_TRUE(set1.contains(3));
    EXPECT_TRUE(set2.contains(1));

    END_TEST();
}

// ============================================================================
// Count Tests
// ============================================================================

/**
 * @brief Test count functionality
 *
 * Covers: count method for existence checking
 */
XSIGMATEST(Core, flat_hash_count)
{
    // Test map count
    flat_hash_map<int, std::string> map;
    map[1] = "one";
    map[2] = "two";

    EXPECT_EQ(map.count(1), 1);
    EXPECT_EQ(map.count(2), 1);
    EXPECT_EQ(map.count(99), 0);

    // Test set count
    flat_hash_set<int> set;
    set.insert(1);
    set.insert(2);

    EXPECT_EQ(set.count(1), 1);
    EXPECT_EQ(set.count(2), 1);
    EXPECT_EQ(set.count(99), 0);

    END_TEST();
}

// ============================================================================
// Performance Comparison Tests
// ============================================================================

/**
 * @brief Test performance characteristics
 *
 * Covers: insertion, lookup, deletion performance
 */
XSIGMATEST(Core, flat_hash_performance)
{
    const int iterations = 10000;

    // Test insertion performance
    flat_hash_map<int, int> map;
    for (int i = 0; i < iterations; ++i)
    {
        map[i] = i * 2;
    }
    EXPECT_EQ(map.size(), static_cast<size_t>(iterations));

    // Test lookup performance
    int sum = 0;
    for (int i = 0; i < iterations; ++i)
    {
        sum += map[i];
    }
    EXPECT_GT(sum, 0);

    // Test deletion performance
    for (int i = 0; i < iterations / 2; ++i)
    {
        map.erase(i);
    }
    EXPECT_EQ(map.size(), static_cast<size_t>(iterations / 2));

    XSIGMA_LOG_INFO("Performance test: {} operations completed", iterations);

    END_TEST();
}

// ============================================================================
// String Key Tests
// ============================================================================

/**
 * @brief Test with string keys (common use case)
 *
 * Covers: string operations, memory management
 */
XSIGMATEST(Core, flat_hash_string_keys)
{
    flat_hash_map<std::string, int> map;

    // Test with various string keys
    map["short"]                                                = 1;
    map["medium_length_key"]                                    = 2;
    map["very_long_key_that_exceeds_small_string_optimization"] = 3;
    map[""]                                                     = 4;  // Empty string

    EXPECT_EQ(map["short"], 1);
    EXPECT_EQ(map["medium_length_key"], 2);
    EXPECT_EQ(map["very_long_key_that_exceeds_small_string_optimization"], 3);
    EXPECT_EQ(map[""], 4);

    // Test find with string
    EXPECT_TRUE(map.find("short") != map.end());
    EXPECT_TRUE(map.find("nonexistent") == map.end());

    END_TEST();
}

// ============================================================================
// xsigma_map and xsigma_set Alias Tests
// ============================================================================

/**
 * @brief Test xsigma_map and xsigma_set type aliases
 *
 * Covers: type aliases defined in flat_hash.h
 */
XSIGMATEST(Core, flat_hash_xsigma_aliases)
{
    // Test xsigma_map
    xsigma_map<int, std::string> map;
    map[1] = "one";
    map[2] = "two";
    EXPECT_EQ(map.size(), 2);
    EXPECT_EQ(map[1], "one");

    // Test xsigma_set
    xsigma_set<int> set;
    set.insert(1);
    set.insert(2);
    EXPECT_EQ(set.size(), 2);
    EXPECT_TRUE(set.contains(1));

    END_TEST();
}

// ============================================================================
// Platform Independence Tests
// ============================================================================

/**
 * @brief Test platform-independent behavior
 *
 * Covers: consistent behavior across platforms
 */
XSIGMATEST(Core, flat_hash_platform_independence)
{
    // Test with fixed-width types
    flat_hash_map<int32_t, int64_t> map;
    map[INT32_MAX] = INT64_MAX;
    map[INT32_MIN] = INT64_MIN;
    map[0]         = 0;

    EXPECT_EQ(map[INT32_MAX], INT64_MAX);
    EXPECT_EQ(map[INT32_MIN], INT64_MIN);
    EXPECT_EQ(map[0], 0);

    // Test set with fixed-width types
    flat_hash_set<uint64_t> set;
    set.insert(UINT64_MAX);
    set.insert(0);
    set.insert(12345);

    EXPECT_TRUE(set.contains(UINT64_MAX));
    EXPECT_TRUE(set.contains(0));
    EXPECT_TRUE(set.contains(12345));

    END_TEST();
}
