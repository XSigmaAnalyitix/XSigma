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

#ifndef XSIGMA_TEST_H
#define XSIGMA_TEST_H

#include <iostream>
#include <sstream>
#include <string>
#include <stdexcept>

// Include Google Test if available
#ifdef XSIGMA_GOOGLE_TEST
#include <gtest/gtest.h>
#endif

// Utility macros
#define XSIGMA_UNUSED [[maybe_unused]]

// Logging macros for tests
#define START_LOG_TO_FILE_NAME(name) \
    do { \
        std::cout << "Starting test: " << #name << std::endl; \
    } while(0)

#define END_LOG_TO_FILE_NAME(name) \
    do { \
        std::cout << "Finished test: " << #name << std::endl; \
    } while(0)

#define END_TEST() \
    do { \
        return 0; \
    } while(0)

// Test assertion macros
#ifdef XSIGMA_GOOGLE_TEST
    // Use Google Test macros when available
    #define EXPECT_TRUE(condition) GTEST_EXPECT_TRUE(condition)
    #define EXPECT_FALSE(condition) GTEST_EXPECT_FALSE(condition)
    #define EXPECT_EQ(expected, actual) GTEST_EXPECT_EQ(expected, actual)
    #define EXPECT_NE(expected, actual) GTEST_EXPECT_NE(expected, actual)
    #define EXPECT_LT(val1, val2) GTEST_EXPECT_LT(val1, val2)
    #define EXPECT_LE(val1, val2) GTEST_EXPECT_LE(val1, val2)
    #define EXPECT_GT(val1, val2) GTEST_EXPECT_GT(val1, val2)
    #define EXPECT_GE(val1, val2) GTEST_EXPECT_GE(val1, val2)
    #define ASSERT_TRUE(condition) GTEST_ASSERT_TRUE(condition)
    #define ASSERT_FALSE(condition) GTEST_ASSERT_FALSE(condition)
    #define ASSERT_EQ(expected, actual) GTEST_ASSERT_EQ(expected, actual)
    #define ASSERT_NE(expected, actual) GTEST_ASSERT_NE(expected, actual)
    #define ASSERT_ANY_THROW(statement) GTEST_ASSERT_ANY_THROW(statement)
    
    // Google Test main function
    #define XSIGMATEST(module, name) \
        TEST(module, name)
    
    #define XSIGMATEST_VOID(module, name) \
        TEST(module, name)
    
    #define XSIGMATEST_CALL(module, name) \
        module##_##name()
        
#else
    // Simple assertion macros for non-Google Test builds
    #define EXPECT_TRUE(condition) \
        do { \
            if (!(condition)) { \
                std::cerr << "EXPECT_TRUE failed: " << #condition << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            } \
        } while(0)
    
    #define EXPECT_FALSE(condition) \
        do { \
            if (condition) { \
                std::cerr << "EXPECT_FALSE failed: " << #condition << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            } \
        } while(0)
    
    #define EXPECT_EQ(expected, actual) \
        do { \
            if ((expected) != (actual)) { \
                std::cerr << "EXPECT_EQ failed: " << #expected << " != " << #actual << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            } \
        } while(0)
    
    #define EXPECT_NE(expected, actual) \
        do { \
            if ((expected) == (actual)) { \
                std::cerr << "EXPECT_NE failed: " << #expected << " == " << #actual << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            } \
        } while(0)
    
    #define EXPECT_LT(val1, val2) \
        do { \
            if (!((val1) < (val2))) { \
                std::cerr << "EXPECT_LT failed: " << #val1 << " >= " << #val2 << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            } \
        } while(0)
    
    #define EXPECT_LE(val1, val2) \
        do { \
            if (!((val1) <= (val2))) { \
                std::cerr << "EXPECT_LE failed: " << #val1 << " > " << #val2 << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            } \
        } while(0)
    
    #define EXPECT_GT(val1, val2) \
        do { \
            if (!((val1) > (val2))) { \
                std::cerr << "EXPECT_GT failed: " << #val1 << " <= " << #val2 << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            } \
        } while(0)
    
    #define EXPECT_GE(val1, val2) \
        do { \
            if (!((val1) >= (val2))) { \
                std::cerr << "EXPECT_GE failed: " << #val1 << " < " << #val2 << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            } \
        } while(0)
    
    #define ASSERT_TRUE(condition) \
        do { \
            if (!(condition)) { \
                std::cerr << "ASSERT_TRUE failed: " << #condition << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
                throw std::runtime_error("Assertion failed: " #condition); \
            } \
        } while(0)
    
    #define ASSERT_FALSE(condition) \
        do { \
            if (condition) { \
                std::cerr << "ASSERT_FALSE failed: " << #condition << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
                throw std::runtime_error("Assertion failed: " #condition); \
            } \
        } while(0)
    
    #define ASSERT_EQ(expected, actual) \
        do { \
            if ((expected) != (actual)) { \
                std::cerr << "ASSERT_EQ failed: " << #expected << " != " << #actual << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
                throw std::runtime_error("Assertion failed: " #expected " != " #actual); \
            } \
        } while(0)
    
    #define ASSERT_NE(expected, actual) \
        do { \
            if ((expected) == (actual)) { \
                std::cerr << "ASSERT_NE failed: " << #expected << " == " << #actual << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
                throw std::runtime_error("Assertion failed: " #expected " == " #actual); \
            } \
        } while(0)
    
    #define ASSERT_ANY_THROW(statement) \
        do { \
            bool threw = false; \
            try { \
                statement; \
            } catch (...) { \
                threw = true; \
            } \
            if (!threw) { \
                std::cerr << "ASSERT_ANY_THROW failed: " << #statement << " did not throw at " << __FILE__ << ":" << __LINE__ << std::endl; \
                throw std::runtime_error("Assertion failed: " #statement " did not throw"); \
            } \
        } while(0)
    
    // Standard test main function
    #define XSIGMATEST(module, name) \
        int main(int argc, char* argv[])
    
    #define XSIGMATEST_VOID(module, name) \
        void module##_##name()
    
    #define XSIGMATEST_CALL(module, name) \
        module##_##name()
        
#endif

// Additional utility macros
#define XSIGMA_LOG_INFO(msg) \
    do { \
        std::cout << "[INFO] " << msg << std::endl; \
    } while(0)

#define XSIGMA_LOG_ERROR(msg) \
    do { \
        std::cerr << "[ERROR] " << msg << std::endl; \
    } while(0)

#define XSIGMA_LOG_WARNING(msg) \
    do { \
        std::cout << "[WARNING] " << msg << std::endl; \
    } while(0)

#endif // XSIGMA_TEST_H
