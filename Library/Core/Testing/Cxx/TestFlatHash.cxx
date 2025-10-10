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

#include "util/flat_hash.h"
#include "xsigmaTest.h"

using namespace xsigma;

// ============================================================================
// Basic flat_hash_map Tests
// ============================================================================

XSIGMATEST(FlatHash, map_basic_operations)
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

// ============================================================================
// Basic flat_hash_set Tests
// ============================================================================

XSIGMATEST(FlatHash, set_basic_operations)
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

XSIGMATEST(FlatHash, set_value_types)
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

    END_TEST();
}

// ============================================================================
// Iteration Tests
// ============================================================================

XSIGMATEST(FlatHash, iteration)
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
// Capacity Tests
// ============================================================================

XSIGMATEST(FlatHash, capacity)
{
    flat_hash_map<int, int> map;

    // Test reserve
    map.reserve(100);
    EXPECT_GE(map.bucket_count(), 100);

    // Insert elements
    for (int i = 0; i < 10; ++i)
    {
        map[i] = i * 10;
    }
    EXPECT_EQ(map.size(), 10);

    // Test load factor
    float load = map.load_factor();
    EXPECT_GT(load, 0.0f);
    EXPECT_LE(load, map.max_load_factor());

    END_TEST();
}

// ============================================================================
// Edge Cases Tests
// ============================================================================

// ============================================================================
// Edge Cases Tests
// ============================================================================

XSIGMATEST(FlatHash, edge_cases)
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

XSIGMATEST(FlatHash, copy_move_semantics)
{
    // Test map copy constructor
    flat_hash_map<int, std::string> map1;
    map1[1] = "one";
    map1[2] = "two";

    flat_hash_map<int, std::string> map2(map1);
    EXPECT_EQ(map2.size(), 2);
    EXPECT_EQ(map2[1], "one");
    EXPECT_EQ(map2[2], "two");

    // Test set copy constructor
    flat_hash_set<int> set1;
    set1.insert(1);
    set1.insert(2);

    flat_hash_set<int> set2(set1);
    EXPECT_EQ(set2.size(), 2);
    EXPECT_TRUE(set2.contains(1));
    EXPECT_TRUE(set2.contains(2));

    END_TEST();
}

// ============================================================================
// Emplace Tests
// ============================================================================

// ============================================================================
// Emplace Tests
// ============================================================================

XSIGMATEST(FlatHash, emplace_operations)
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
// xsigma_map and xsigma_set Alias Tests
// ============================================================================

XSIGMATEST(FlatHash, xsigma_aliases)
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
