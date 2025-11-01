#include <gtest/gtest.h>

#include <vector>

#include "smp_new/parallel/parallel_api.h"

using namespace xsigma::smp_new::parallel;

class ParallelReduceTest : public ::testing::Test
{
protected:
    void SetUp() override { set_num_interop_threads(4); }
    void TearDown() override {}
};

TEST_F(ParallelReduceTest, SumReduction)
{
    std::vector<float> data(100);
    for (int i = 0; i < 100; ++i)
    {
        data[i] = 1.0f;
    }

    float sum = parallel_reduce(
        0,
        100,
        10,
        0.0f,
        [&data](int64_t b, int64_t e, float ident)
        {
            float s = ident;
            for (int64_t i = b; i < e; ++i)
            {
                s += data[i];
            }
            return s;
        },
        [](float a, float b) { return a + b; });

    EXPECT_FLOAT_EQ(sum, 100.0f);
}

TEST_F(ParallelReduceTest, MaxReduction)
{
    std::vector<int> data(100);
    for (int i = 0; i < 100; ++i)
    {
        data[i] = i;
    }

    int max_val = parallel_reduce(
        0,
        100,
        10,
        0,
        [&data](int64_t b, int64_t e, int ident)
        {
            int m = ident;
            for (int64_t i = b; i < e; ++i)
            {
                m = std::max(m, data[i]);
            }
            return m;
        },
        [](int a, int b) { return std::max(a, b); });

    EXPECT_EQ(max_val, 99);
}

TEST_F(ParallelReduceTest, MinReduction)
{
    std::vector<int> data(100);
    for (int i = 0; i < 100; ++i)
    {
        data[i] = i + 1;  // 1 to 100
    }

    int min_val = parallel_reduce(
        0,
        100,
        10,
        INT_MAX,
        [&data](int64_t b, int64_t e, int ident)
        {
            int m = ident;
            for (int64_t i = b; i < e; ++i)
            {
                m = std::min(m, data[i]);
            }
            return m;
        },
        [](int a, int b) { return std::min(a, b); });

    EXPECT_EQ(min_val, 1);
}

TEST_F(ParallelReduceTest, ProductReduction)
{
    std::vector<float> data(10);
    for (int i = 0; i < 10; ++i)
    {
        data[i] = 2.0f;
    }

    float product = parallel_reduce(
        0,
        10,
        2,
        1.0f,
        [&data](int64_t b, int64_t e, float ident)
        {
            float p = ident;
            for (int64_t i = b; i < e; ++i)
            {
                p *= data[i];
            }
            return p;
        },
        [](float a, float b) { return a * b; });

    EXPECT_FLOAT_EQ(product, 1024.0f);  // 2^10
}

TEST_F(ParallelReduceTest, EmptyRange)
{
    float result = parallel_reduce(
        0,
        0,
        10,
        42.0f,
        [](int64_t b, int64_t e, float ident) { return ident; },
        [](float a, float b) { return a + b; });

    EXPECT_FLOAT_EQ(result, 42.0f);
}

TEST_F(ParallelReduceTest, SingleElement)
{
    std::vector<int> data(1);
    data[0] = 100;

    int result = parallel_reduce(
        0,
        1,
        10,
        0,
        [&data](int64_t b, int64_t e, int ident)
        {
            int s = ident;
            for (int64_t i = b; i < e; ++i)
            {
                s += data[i];
            }
            return s;
        },
        [](int a, int b) { return a + b; });

    EXPECT_EQ(result, 100);
}

TEST_F(ParallelReduceTest, LargeRange)
{
    const int N = 1000000;

    float sum = parallel_reduce(
        0,
        N,
        10000,
        0.0f,
        [](int64_t b, int64_t e, float ident)
        {
            float s = ident;
            for (int64_t i = b; i < e; ++i)
            {
                s += 1.0f;
            }
            return s;
        },
        [](float a, float b) { return a + b; });

    EXPECT_FLOAT_EQ(sum, static_cast<float>(N));
}

TEST_F(ParallelReduceTest, AutomaticGrainSize)
{
    std::vector<int> data(100);
    for (int i = 0; i < 100; ++i)
    {
        data[i] = i;
    }

    int sum = parallel_reduce(
        0,
        100,
        0,  // grain_size = 0 should auto-determine
        0,
        [&data](int64_t b, int64_t e, int ident)
        {
            int s = ident;
            for (int64_t i = b; i < e; ++i)
            {
                s += data[i];
            }
            return s;
        },
        [](int a, int b) { return a + b; });

    int expected = 0;
    for (int i = 0; i < 100; ++i)
    {
        expected += i;
    }
    EXPECT_EQ(sum, expected);
}

TEST_F(ParallelReduceTest, StringConcatenation)
{
    std::vector<std::string> data(10);
    for (int i = 0; i < 10; ++i)
    {
        data[i] = "a";
    }

    std::string result = parallel_reduce(
        0,
        10,
        2,
        std::string(""),
        [&data](int64_t b, int64_t e, std::string ident)
        {
            for (int64_t i = b; i < e; ++i)
            {
                ident += data[i];
            }
            return ident;
        },
        [](const std::string& a, const std::string& b) { return a + b; });

    EXPECT_EQ(result.length(), 10);
}
