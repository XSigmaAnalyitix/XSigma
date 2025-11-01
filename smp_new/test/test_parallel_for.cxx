#include <gtest/gtest.h>

#include <atomic>
#include <vector>

#include "smp_new/parallel/parallel_api.h"

using namespace xsigma::smp_new::parallel;

class ParallelForTest : public ::testing::Test
{
protected:
    void SetUp() override { set_num_interop_threads(4); }
    void TearDown() override {}
};

TEST_F(ParallelForTest, BasicParallelFor)
{
    std::vector<int> data(100);

    parallel_for(
        0,
        100,
        10,
        [&data](int64_t b, int64_t e)
        {
            for (int64_t i = b; i < e; ++i)
            {
                data[i] = i * 2;
            }
        });

    for (int i = 0; i < 100; ++i)
    {
        EXPECT_EQ(data[i], i * 2);
    }
}

TEST_F(ParallelForTest, EmptyRange)
{
    std::atomic<int> counter{0};

    parallel_for(0, 0, 10, [&counter](int64_t b, int64_t e) { counter++; });

    EXPECT_EQ(counter, 0);
}

TEST_F(ParallelForTest, SingleElement)
{
    std::vector<int> data(1);

    parallel_for(
        0,
        1,
        10,
        [&data](int64_t b, int64_t e)
        {
            for (int64_t i = b; i < e; ++i)
            {
                data[i] = 42;
            }
        });

    EXPECT_EQ(data[0], 42);
}

TEST_F(ParallelForTest, LargeRange)
{
    const int        N = 1000000;
    std::vector<int> data(N);

    parallel_for(
        0,
        N,
        10000,
        [&data](int64_t b, int64_t e)
        {
            for (int64_t i = b; i < e; ++i)
            {
                data[i] = static_cast<int>(i);
            }
        });

    // Spot check
    EXPECT_EQ(data[0], 0);
    EXPECT_EQ(data[N / 2], N / 2);
    EXPECT_EQ(data[N - 1], N - 1);
}

TEST_F(ParallelForTest, AutomaticGrainSize)
{
    std::vector<int> data(100);

    // grain_size = 0 should auto-determine
    parallel_for(
        0,
        100,
        0,
        [&data](int64_t b, int64_t e)
        {
            for (int64_t i = b; i < e; ++i)
            {
                data[i] = i;
            }
        });

    for (int i = 0; i < 100; ++i)
    {
        EXPECT_EQ(data[i], i);
    }
}

TEST_F(ParallelForTest, SmallGrainSize)
{
    std::vector<int> data(100);

    // Small grain size should create many tasks
    parallel_for(
        0,
        100,
        1,
        [&data](int64_t b, int64_t e)
        {
            for (int64_t i = b; i < e; ++i)
            {
                data[i] = i * 2;
            }
        });

    for (int i = 0; i < 100; ++i)
    {
        EXPECT_EQ(data[i], i * 2);
    }
}

TEST_F(ParallelForTest, LargeGrainSize)
{
    std::vector<int> data(100);

    // Large grain size should execute serially
    parallel_for(
        0,
        100,
        1000,
        [&data](int64_t b, int64_t e)
        {
            for (int64_t i = b; i < e; ++i)
            {
                data[i] = i * 3;
            }
        });

    for (int i = 0; i < 100; ++i)
    {
        EXPECT_EQ(data[i], i * 3);
    }
}

TEST_F(ParallelForTest, ThreadCount)
{
    auto num_threads = get_num_interop_threads();
    EXPECT_GT(num_threads, 0);
}

TEST_F(ParallelForTest, SetThreadCount)
{
    set_num_interop_threads(2);
    auto num_threads = get_num_interop_threads();
    EXPECT_EQ(num_threads, 2);
}

TEST_F(ParallelForTest, NestedParallelFor)
{
    std::vector<std::vector<int>> data(10, std::vector<int>(10));

    parallel_for(
        0,
        10,
        5,
        [&data](int64_t b, int64_t e)
        {
            for (int64_t i = b; i < e; ++i)
            {
                for (int j = 0; j < 10; ++j)
                {
                    data[i][j] = i * 10 + j;
                }
            }
        });

    for (int i = 0; i < 10; ++i)
    {
        for (int j = 0; j < 10; ++j)
        {
            EXPECT_EQ(data[i][j], i * 10 + j);
        }
    }
}

TEST_F(ParallelForTest, FloatingPointData)
{
    std::vector<float> data(100);

    parallel_for(
        0,
        100,
        10,
        [&data](int64_t b, int64_t e)
        {
            for (int64_t i = b; i < e; ++i)
            {
                data[i] = static_cast<float>(i) * 1.5f;
            }
        });

    for (int i = 0; i < 100; ++i)
    {
        EXPECT_FLOAT_EQ(data[i], static_cast<float>(i) * 1.5f);
    }
}
