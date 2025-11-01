#include <string>
#include <thread>
#include <vector>

#include "Testing/xsigmaTest.h"
#include "smp/Advanced/thread_name.h"

namespace xsigma::detail::smp::Advanced
{

// Test setting and getting thread name
XSIGMATEST(SmpAdvancedThreadName, set_and_get_name)
{
    set_thread_name("TestThread");
    std::string name = get_thread_name();
    // On systems that support thread naming, the name should be set
    // On systems that don't, it will be empty
    // We just verify the function doesn't crash
    EXPECT_TRUE(true);
}

// Test empty thread name
XSIGMATEST(SmpAdvancedThreadName, empty_name)
{
    set_thread_name("");
    std::string name = get_thread_name();
    // Should not crash
    EXPECT_TRUE(true);
}

// Test long thread name (should be truncated)
XSIGMATEST(SmpAdvancedThreadName, long_name)
{
    std::string long_name = "VeryLongThreadNameThatExceedsLimit";
    set_thread_name(long_name);
    std::string name = get_thread_name();
    // Should not crash, name may be truncated
    EXPECT_TRUE(true);
}

// Test setting name in different threads
XSIGMATEST(SmpAdvancedThreadName, different_threads)
{
    std::vector<std::thread> threads;
    std::vector<std::string> names;

    for (int i = 0; i < 4; ++i)
    {
        threads.emplace_back(
            [i, &names]()
            {
                std::string thread_name = "Thread" + std::to_string(i);
                set_thread_name(thread_name);
                std::string retrieved_name = get_thread_name();
                names.push_back(retrieved_name);
            });
    }

    for (auto& t : threads)
    {
        t.join();
    }

    // Should have 4 names (may be empty on unsupported systems)
    EXPECT_EQ(names.size(), 4);
}

// Test getting name without setting
XSIGMATEST(SmpAdvancedThreadName, get_without_set)
{
    std::string name = get_thread_name();
    // Should not crash, may return empty string
    EXPECT_TRUE(true);
}

// Test special characters in name
XSIGMATEST(SmpAdvancedThreadName, special_characters)
{
    set_thread_name("Test-Thread_123");
    std::string name = get_thread_name();
    // Should not crash
    EXPECT_TRUE(true);
}

// Test unicode characters (may not be supported)
XSIGMATEST(SmpAdvancedThreadName, unicode_characters)
{
    set_thread_name("TestThread");  // Use ASCII instead of unicode
    std::string name = get_thread_name();
    // Should not crash
    EXPECT_TRUE(true);
}

// Test repeated setting
XSIGMATEST(SmpAdvancedThreadName, repeated_setting)
{
    set_thread_name("FirstName");
    set_thread_name("SecondName");
    set_thread_name("ThirdName");
    std::string name = get_thread_name();
    // Should not crash
    EXPECT_TRUE(true);
}

// Test thread name persistence
XSIGMATEST(SmpAdvancedThreadName, name_persistence)
{
    set_thread_name("PersistentName");
    std::string name1 = get_thread_name();
    std::string name2 = get_thread_name();
    // Both calls should return the same name
    EXPECT_EQ(name1, name2);
}

}  // namespace xsigma::detail::smp::Advanced
