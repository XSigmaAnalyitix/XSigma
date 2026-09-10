/*
 * XSigma: High-Performance Computational Library
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

#include "ModelsTest.h"
#include "nn/mlp.h"
// Standalone Hartman–Watson log-density: Pirjol (small t) + Talbot (large t).
// Copy this header only. C++17, no project dependencies.
//
// log_distribution(x, t) returns
//   log( θ(r,t) exp(-r - t/8) √(2π) / (r √r) )
// with r = ρ/t,  ρ = x/sinh(x) for x ≤ 0,  ρ = x/sin(x) for x ∈ (0, π).
//
// Regime: t ≤ 2 → Pirjol + C2 + C3;  t > 2 → Talbot Laplace inversion of
// Yor's identity L[θ(r,·)](s) = I_{√(2s)}(r) (Abate–Valkó fixed Talbot,
// M = 24, or M = 64 for r < 1e-3), and a rescaled 32-point Gauss–Legendre
// quadrature of the Yor integral for r > 4 (Talbot loses precision there
// through exponential cancellation).
// Domain: x ∈ (-∞, 0] ∪ (0, π),  t > 0.

#pragma once

#include <cmath>
#include <complex>
#include <limits>
#include <utility>

namespace hartman_watson
{
inline constexpr double k_sqrt_2pi       = 2.506628238584603;   // √(2π)
inline constexpr double k_inverse_sqrt_3 = 0.5773502691896257;  // 1/√3

// D = |h''(saddle)| = ρ cosh(x) - 1  (x < 0)  or  1 - ρ cos(x)  (x > 0).
inline std::pair<double, double> c2_c3(double x, double l, double D, bool negative_branch)
{
    using std::complex;

    const double sqD = std::sqrt(D);
    const double D2  = D * D;
    const double D3  = D2 * D;
    const double D4  = D2 * D2;
    const double D15 = D * sqD;
    const double D25 = D2 * sqD;
    const double D35 = D3 * sqD;

    complex<double> H3, H4, H5, H6, H7, H8;
    complex<double> A1, A2, A3, A4, A5, A6;

    if (negative_branch)
    {
        H3 = {0.0, x / D15};
        H4 = {-l / D2, 0.0};
        H5 = {0.0, -x / D25};
        H6 = {l / D3, 0.0};
        H7 = {0.0, x / D35};
        H8 = {-l / D4, 0.0};

        A1 = {0.0, l / (x * sqD)};
        A2 = {-1.0 / D, 0.0};
        A3 = {0.0, -l / (x * D15)};
        A4 = {1.0 / D2, 0.0};
        A5 = {0.0, l / (x * D25)};
        A6 = {-1.0 / D3, 0.0};
    }
    else
    {
        H3 = {0.0, x / D15};
        H4 = {-l / D2, 0.0};
        H5 = {0.0, x / D25};
        H6 = {-l / D3, 0.0};
        H7 = {0.0, x / D35};
        H8 = {-l / D4, 0.0};

        A1 = {0.0, l / (x * sqD)};
        A2 = {1.0 / D, 0.0};
        A3 = {0.0, l / (x * D15)};
        A4 = {1.0 / D2, 0.0};
        A5 = {0.0, l / (x * D25)};
        A6 = {1.0 / D3, 0.0};
    }

    auto cu = [](complex<double> z) { return z * z * z; };
    auto p4 = [](complex<double> z)
    {
        const auto z2 = z * z;
        return z2 * z2;
    };
    auto p5 = [](complex<double> z)
    {
        const auto z2 = z * z;
        return z2 * z2 * z;
    };
    auto p6 = [](complex<double> z)
    {
        const auto z2 = z * z;
        return z2 * z2 * z2;
    };

    const complex<double> C2 = 385.0 / 1152.0 * p4(H3) - 35.0 / 64.0 * H3 * H3 * H4 +
                               7.0 / 48.0 * H3 * H5 + 35.0 / 384.0 * H4 * H4 - 1.0 / 48.0 * H6 -
                               35.0 / 48.0 * A1 * cu(H3) + 35.0 / 48.0 * A1 * H3 * H4 -
                               1.0 / 8.0 * A1 * H5 + 35.0 / 48.0 * A2 * H3 * H3 -
                               5.0 / 16.0 * A2 * H4 - 5.0 / 12.0 * A3 * H3 + 1.0 / 8.0 * A4;

    const complex<double> C3 =
        85085.0 / 82944.0 * p6(H3) - 25025.0 / 9216.0 * p4(H3) * H4 +
        1001.0 / 1152.0 * cu(H3) * H5 + 5005.0 / 3072.0 * H3 * H3 * H4 * H4 -
        77.0 / 384.0 * H3 * H3 * H6 - 77.0 / 128.0 * H3 * H4 * H5 + 1.0 / 32.0 * H3 * H7 -
        385.0 / 3072.0 * cu(H4) + 7.0 / 128.0 * H4 * H6 + 21.0 / 640.0 * H5 * H5 - 1.0 / 384.0 * H8 -
        5005.0 / 2304.0 * A1 * p5(H3) + 5005.0 / 1152.0 * A1 * cu(H3) * H4 -
        77.0 / 64.0 * A1 * H3 * H3 * H5 - 385.0 / 256.0 * A1 * H3 * H4 * H4 +
        7.0 / 32.0 * A1 * H3 * H6 + 21.0 / 64.0 * A1 * H4 * H5 - 1.0 / 48.0 * A1 * H7 +
        5005.0 / 2304.0 * A2 * p4(H3) - 385.0 / 128.0 * A2 * H3 * H3 * H4 +
        21.0 / 32.0 * A2 * H3 * H5 + 105.0 / 256.0 * A2 * H4 * H4 - 7.0 / 96.0 * A2 * H6 -
        385.0 / 288.0 * A3 * cu(H3) + 35.0 / 32.0 * A3 * H3 * H4 - 7.0 / 48.0 * A3 * H5 +
        35.0 / 64.0 * A4 * H3 * H3 - 35.0 / 192.0 * A4 * H4 - 7.0 / 48.0 * A5 * H3 + 1.0 / 48.0 * A6;

    return {C2.real(), C3.real()};
}

inline double c2(double x, double l, double D, bool negative_branch)
{
    const double D2 = D * D;
    const double D3 = D2 * D;
    const double D4 = D2 * D2;
    const double D5 = D3 * D2;
    const double D6 = D3 * D3;
    const double x2 = x * x;
    const double x4 = x2 * x2;
    const double l2 = l * l;

    return 385.0 / 1152.0 * x4 / D6 - 245.0 / 192.0 * l * x2 / D5 +
           (negative_branch ? 7.0 / 8.0 : -7.0 / 8.0) * x2 / D4 + 105.0 / 128.0 * l2 / D4 +
           (negative_branch ? -7.0 / 8.0 : 7.0 / 8.0) * l / D3 + 1.0 / (8.0 * D2);
}

inline double first_term(double x, double half_angle)
{
    return 0.5 * x * x - x * half_angle;
}

inline double second_term(double x, double D)
{
    return std::sqrt(D) / x;
}

// Pirjol ḡ₂. Pass D = |l - 1|.
inline double third_term(double rho, double l, double D)
{
    return (-1.0 + l * (0.75 - 1.0 / 6.0 * l) + 5.0 / 12.0 * rho * rho) / (D * D * D);
}

// Regular-saddle C₂, C₃ as |x|→0 (generic formulas cancel). With s = sign(x), u = x²:
//   C₂ = 7/11000 + (12/625625) s u − (43/21896875) u² − (1994/5583703125) s u³ + O(u⁴)
//   C₃ = −44081/1051050000 − (7278/1861234375) s u + (71613/990176687500) u²
//        + (114921/2475441718750) s u³ + O(u⁴)
inline double c2_small_x(double x)
{
    const double s  = std::copysign(1.0, x);
    const double u  = x * x;
    const double u2 = u * u;
    return 7.0 / 11000.0 + s * (12.0 / 625625.0) * u - (43.0 / 21896875.0) * u2 -
           s * (1994.0 / 5583703125.0) * u2 * u;
}

inline double c3_small_x(double x)
{
    const double s  = std::copysign(1.0, x);
    const double u  = x * x;
    const double u2 = u * u;
    return -44081.0 / 1051050000.0 - s * (7278.0 / 1861234375.0) * u +
           (71613.0 / 990176687500.0) * u2 + s * (114921.0 / 2475441718750.0) * u2 * u;
}

inline double assemble_correction(double t, double g, double C2, double C3, int order)
{
    if (order < 2)
        return 1.0;

    double correction = 1.0 + 0.5 * t * g + C2 * t * t;
    if (order >= 3)
        correction += C3 * t * t * t;
    return correction;
}

// order = 1: G only; 2: + G₁ t + C₂ t²; 3: + C₃ t³.
// For |x| below these bounds the C₂/C₃ polynomials cancel in double, so use
// their Taylor series. F, G, g keep the tighter |x| < 0.01 series branch.
inline constexpr double k_c2_series_bound = 0.25;
inline constexpr double k_c3_series_bound = 0.40;
inline constexpr double k_tau_switch      = 2.0;

inline double rho_of_x(double x)
{
    const double ax = std::fabs(x);
    if (ax < 0.01)
    {
        const double x2 = x * x;
        return 1.0 + std::copysign(x2 / 6.0, x) + 7.0 / 360.0 * x2 * x2;
    }
    return x < 0.0 ? x / std::sinh(x) : x / std::sin(x);
}

// Complex log-gamma, Lanczos g = 7, n = 9 (about 1e-15 relative on Gamma).
inline std::complex<double> log_gamma(std::complex<double> z)
{
    static constexpr double c[9] = {0.99999999999980993,   676.5203681218851,
                                    -1259.1392167224028,   771.32342877765313,
                                    -176.61502916214059,   12.507343278686905,
                                    -0.13857109526572012,  9.9843695780195716e-6,
                                    1.5056327351493116e-7};
    constexpr double pi = 3.14159265358979323846;
    if (z.real() < 0.5)
        return std::log(pi) - std::log(std::sin(pi * z)) - log_gamma(1.0 - z);
    z -= 1.0;
    std::complex<double> x = c[0];
    for (int i = 1; i < 9; ++i)
        x += c[i] / (z + static_cast<double>(i));
    const std::complex<double> t = z + 7.5;
    return 0.5 * std::log(2.0 * pi) + (z + 0.5) * std::log(t) - t + std::log(x);
}

// I_nu(r) for complex nu, r > 0: (r/2)^nu / Γ(ν+1) · Σ (r²/4)^k / (k! (ν+1)_k).
inline std::complex<double> bessel_i_series(std::complex<double> nu, double r)
{
    const double                 y  = 0.25 * r * r;
    const std::complex<double>   lg = std::log(0.5 * r);
    std::complex<double>         p  = std::exp(nu * lg - log_gamma(nu + 1.0));
    std::complex<double>         s  = p;
    for (int k = 1; k < 400; ++k)
    {
        p *= y / (static_cast<double>(k) * (nu + static_cast<double>(k)));
        s += p;
        if (k > 2 && std::abs(p) < 1e-17 * std::abs(s))
            break;
    }
    return s;
}

// θ(r,t) by Talbot inversion of I_{√(2s)}(r); Abate–Valkó (2004) fixed contour:
//   θ ≈ (2/(5t)) [ ½ e^ρ F(ρ/t) + Σ_{k=1}^{M-1} Re( e^{δ_k} F(δ_k/t) (1 + iσ_k) ) ]
// with ρ = 2M/5, θ_k = kπ/M, δ_k = ρ θ_k (cot θ_k + i), σ_k = θ_k + (θ_k cot θ_k − 1) cot θ_k.
// Double precision: M = 24 gives full accuracy while θ ≳ 1e-13; the absolute
// trapezoid floor is ~1e-17, so for tiny r (θ ≪ 1e-13, far left tail in x)
// M = 64 pushes the floor down to θ ~ 1e-30.
inline double theta_talbot(double r, double t, int M)
{
    constexpr double pi = 3.14159265358979323846;
    const double     rr = 0.4 * static_cast<double>(M);
    std::complex<double> total =
        0.5 * std::exp(rr) * bessel_i_series(std::sqrt(2.0 * rr / t), r);
    for (int k = 1; k < M; ++k)
    {
        const double                 th    = pi * static_cast<double>(k) / static_cast<double>(M);
        const double                 cot   = 1.0 / std::tan(th);
        const std::complex<double>   delta(rr * th * cot, rr * th);
        const double                 sigma = th + (th * cot - 1.0) * cot;
        const std::complex<double>   nu    = std::sqrt(2.0 * delta / t);
        total += std::exp(delta) * bessel_i_series(nu, r) *
                 std::complex<double>(1.0, sigma);
    }
    return (2.0 / (5.0 * t)) * total.real();
}

// θ(r,t) for r > 4: 32-point Gauss–Legendre on the Yor integral, rescaled to
// [0, L] with L = 12/√a, a = r + 1/t, where the integrand is a smooth,
// non-oscillatory hump (the Watson/1-r series is only asymptotic and diverges;
// direct quadrature of the unexpanded integrand converges geometrically).
inline double theta_large_r(double r, double t)
{
    static constexpr double nodes[32] = {
        -9.9726386184948157e-01, -9.8561151154526827e-01, -9.6476225558750639e-01,
        -9.3490607593773967e-01, -8.9632115576605209e-01, -8.4936761373256997e-01,
        -7.9448379596794239e-01, -7.3218211874028971e-01, -6.6304426693021523e-01,
        -5.8771575724076230e-01, -5.0689990893222936e-01, -4.2135127613063533e-01,
        -3.3186860228212767e-01, -2.3928736225213706e-01, -1.4447196158279652e-01,
        -4.8307665687738324e-02,  4.8307665687738324e-02,  1.4447196158279652e-01,
         2.3928736225213706e-01,  3.3186860228212767e-01,  4.2135127613063533e-01,
         5.0689990893222936e-01,  5.8771575724076230e-01,  6.6304426693021523e-01,
         7.3218211874028971e-01,  7.9448379596794239e-01,  8.4936761373256997e-01,
         8.9632115576605209e-01,  9.3490607593773967e-01,  9.6476225558750639e-01,
         9.8561151154526827e-01,  9.9726386184948157e-01};
    static constexpr double weights[32] = {
        7.0186100094691405e-03, 1.6274394730906284e-02, 2.5392065309261930e-02,
        3.4273862913021563e-02, 4.2835898022226898e-02, 5.0998059262376265e-02,
        5.8684093478535621e-02, 6.5822222776361697e-02, 7.2345794108848435e-02,
        7.8193895787070269e-02, 8.3311924226946860e-02, 8.7652093004403894e-02,
        9.1173878695763919e-02, 9.3844399080804483e-02, 9.5638720079274847e-02,
        9.6540088514727854e-02, 9.6540088514727854e-02, 9.5638720079274847e-02,
        9.3844399080804483e-02, 9.1173878695763919e-02, 8.7652093004403894e-02,
        8.3311924226946860e-02, 7.8193895787070269e-02, 7.2345794108848435e-02,
        6.5822222776361697e-02, 5.8684093478535621e-02, 5.0998059262376265e-02,
        4.2835898022226898e-02, 3.4273862913021563e-02, 2.5392065309261930e-02,
        1.6274394730906284e-02, 7.0186100094691405e-03};

    constexpr double pi  = 3.14159265358979323846;
    const double     a   = r + 1.0 / t;
    const double     mid = 6.0 / std::sqrt(a);  // half-length L/2, L = 12/√a

    double s = 0.0;
    for (int i = 0; i < 32; ++i)
    {
        const double xi = mid * (nodes[i] + 1.0);
        s += mid * weights[i] *
             std::exp(-xi * xi / (2.0 * t) - r * std::cosh(xi)) * std::sinh(xi) *
             std::sin(pi * xi / t);
    }
    return r * std::exp(0.5 * pi * pi / t) /
           std::sqrt(2.0 * pi * pi * pi * t) * s;
}

inline double log_distribution_large_t(double x, double t)
{
    const double rho = rho_of_x(x);
    const double r   = rho / t;
    double       theta;
    if (r > 4.0)
        theta = theta_large_r(r, t);
    else
        theta = theta_talbot(r, t, r < 1e-3 ? 64 : 24);
    if (!(theta > 0.0))  // deep left tail: θ below the Talbot floor (~1e-40)
        return -std::numeric_limits<double>::infinity();
    return std::log(theta) + 0.5 * std::log(2.0 * 3.14159265358979323846) -
           1.5 * std::log(r) - r - 0.125 * t;
}

inline double log_distribution_asymptotique(double x, double t, int order = 3)
{
    double F         = 0.0;
    double G_inverse = 0.0;
    double g         = 0.0;
    double rho       = 0.0;
    double C2        = 0.0;
    double C3        = 0.0;
    const double ax  = std::fabs(x);
    const bool   neg = x < 0.0;

    if (ax < 0.01)
    {
        const double x2 = x * x;
        rho             = 1.0 + std::copysign(x2 / 6.0, x) + 7.0 / 360.0 * x2 * x2;

        const double l = neg ? x * std::tanh(0.5 * x) : -x * std::tan(0.5 * x);

        F         = -0.5 * std::copysign(x2, x) - l;
        G_inverse = k_inverse_sqrt_3 *
                    (1.0 + x2 * (std::copysign(1.0 / 30.0, x) + 11.0 / 4200.0 * x2));
        g = -1.0 / 35.0 + 144.0 / 67375.0 * (1.0 / rho - 1.0);
    }
    else if (neg)
    {
        rho               = x / std::sinh(x);
        const double l    = rho * std::cosh(x);
        const double D    = l - 1.0;
        F                 = first_term(x, std::tanh(0.5 * x));
        G_inverse         = -second_term(x, D);
        g                 = third_term(rho, l, D);
        if (ax >= k_c2_series_bound)
            C2 = c2(x, l, D, true);
        if (order >= 3 && ax >= k_c3_series_bound)
            C3 = c2_c3(x, l, D, true).second;
    }
    else
    {
        rho               = x / std::sin(x);
        const double l    = rho * std::cos(x);
        const double D    = 1.0 - l;
        F                 = -first_term(x, std::tan(0.5 * x));
        G_inverse         = second_term(x, D);
        g                 = -third_term(rho, l, D);
        if (ax >= k_c2_series_bound)
            C2 = c2(x, l, D, false);
        if (order >= 3 && ax >= k_c3_series_bound)
            C3 = c2_c3(x, l, D, false).second;
    }

    if (ax < k_c2_series_bound)
        C2 = c2_small_x(x);
    if (order >= 3 && ax < k_c3_series_bound)
        C3 = c3_small_x(x);

    const double correction = assemble_correction(t, g, C2, C3, order);

    return -(F / t + 0.125 * t) +
           std::log(G_inverse * correction / (k_sqrt_2pi * std::sqrt(rho * t)));
}

// Regime switch: Pirjol order 3 for t ≤ 2, Talbot/GL Laplace inversion for t > 2.
inline double log_distribution(double x, double t)
{
    if (t > k_tau_switch)
        return log_distribution_large_t(x, t);
    return log_distribution_asymptotique(x, t, 3);
}

}  // namespace hartman_watson

using namespace models;

TEST(Mlp, rejects_bad_configure)
{
    mlp net;
    EXPECT_TRUE(net.empty());
    const int    sizes[]   = {2, 2};
    const double weights[] = {1.0, 0.0, 0.0, 1.0};
    const double biases[]  = {0.0, 0.0};
    EXPECT_FALSE(net.configure(nullptr, 2, weights, 4, biases, 2));
    EXPECT_FALSE(net.configure(sizes, 1, weights, 4, biases, 2));
    EXPECT_FALSE(net.configure(sizes, 2, weights, 3, biases, 2));
    EXPECT_TRUE(net.empty());
}

TEST(Mlp, identity_linear_layer)
{
    mlp          net;
    const int    sizes[]   = {2, 2};
    const double weights[] = {1.0, 0.0, 0.0, 1.0};
    const double biases[]  = {0.1, -0.2};
    ASSERT_TRUE(net.configure(sizes, 2, weights, 4, biases, 2));
    const double in[] = {1.5, -3.0};
    double       out[2];
    ASSERT_TRUE(net.forward(in, 2, out, 2));
    EXPECT_NEAR(out[0], 1.6, 1.0e-12);
    EXPECT_NEAR(out[1], -3.2, 1.0e-12);
}

TEST(Mlp, relu_hides_negative_preactivation)
{
    mlp          net;
    const int    sizes[]   = {1, 1, 1};
    const double weights[] = {-1.0, 1.0};
    const double biases[]  = {0.0, 0.0};
    ASSERT_TRUE(net.configure(sizes, 3, weights, 2, biases, 2));
    const double in[] = {2.0};
    double       out[1];
    ASSERT_TRUE(net.forward(in, 1, out, 1));
    EXPECT_NEAR(out[0], 0.0, 1.0e-12);
}

TEST(Mlp, rejects_size_mismatch)
{
    mlp          net;
    const int    sizes[]   = {2, 1};
    const double weights[] = {1.0, 1.0};
    const double biases[]  = {0.0};
    ASSERT_TRUE(net.configure(sizes, 2, weights, 2, biases, 1));
    const double in[] = {1.0, 2.0};
    double       out[1];
    EXPECT_FALSE(net.forward(in, 1, out, 1));
    EXPECT_FALSE(net.forward(in, 2, out, 2));
}
