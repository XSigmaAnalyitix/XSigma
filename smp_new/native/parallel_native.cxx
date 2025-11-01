#include "smp_new/native/parallel_native.h"

#include <atomic>
#include <sstream>

#include "smp_new/openmp/parallel_openmp.h"

namespace xsigma::smp_new::native
{

namespace
{

std::atomic<bool>        g_backend_initialized{false};
std::atomic<bool>        g_native_backend_initialized{false};
std::atomic<BackendType> g_current_backend{BackendType::NATIVE};

}  // namespace

void InitializeBackend(BackendType backend)
{
    if (backend == BackendType::AUTO)
    {
        // Auto-select: prefer OpenMP if available, otherwise use native
        backend = openmp::IsOpenMPAvailable() ? BackendType::OPENMP : BackendType::NATIVE;
    }

    g_current_backend.store(backend);

    switch (backend)
    {
    case BackendType::NATIVE:
        InitializeNativeBackend();
        break;
    case BackendType::OPENMP:
        openmp::InitializeOpenMPBackend();
        break;
    case BackendType::AUTO:
        // Should not reach here
        break;
    }

    g_backend_initialized.store(true);
}

void InitializeNativeBackend()
{
    bool expected = false;
    g_native_backend_initialized.compare_exchange_strong(expected, true);
}

void InitializeOpenMPBackend()
{
    openmp::InitializeOpenMPBackend();
}

void ShutdownBackend()
{
    BackendType backend = g_current_backend.load();

    switch (backend)
    {
    case BackendType::NATIVE:
        ShutdownNativeBackend();
        break;
    case BackendType::OPENMP:
        openmp::ShutdownOpenMPBackend();
        break;
    case BackendType::AUTO:
        break;
    }

    g_backend_initialized.store(false);
}

void ShutdownNativeBackend()
{
    g_native_backend_initialized = false;
}

void ShutdownOpenMPBackend()
{
    openmp::ShutdownOpenMPBackend();
}

bool IsBackendInitialized()
{
    return g_backend_initialized.load();
}

bool IsNativeBackendInitialized()
{
    return g_native_backend_initialized.load();
}

bool IsOpenMPBackendInitialized()
{
    return openmp::IsOpenMPBackendInitialized();
}

bool IsOpenMPAvailable()
{
    return openmp::IsOpenMPAvailable();
}

BackendType GetCurrentBackend()
{
    return g_current_backend.load();
}

std::string GetBackendInfo()
{
    std::ostringstream oss;
    oss << "XSigma SMP_NEW Backend Information:\n";
    oss << "  Current Backend: ";

    BackendType backend = g_current_backend.load();
    switch (backend)
    {
    case BackendType::NATIVE:
        oss << "Native (std::thread)\n";
        break;
    case BackendType::OPENMP:
        oss << "OpenMP\n";
        break;
    case BackendType::AUTO:
        oss << "Auto\n";
        break;
    }

    oss << "  Initialized: " << (IsBackendInitialized() ? "Yes" : "No") << "\n";
    oss << "  OpenMP Available: " << (IsOpenMPAvailable() ? "Yes" : "No") << "\n";
    oss << "\n";

    if (backend == BackendType::NATIVE)
    {
        oss << GetNativeBackendInfo();
    }
    else if (backend == BackendType::OPENMP)
    {
        oss << GetOpenMPBackendInfo();
    }

    return oss.str();
}

std::string GetNativeBackendInfo()
{
    std::ostringstream oss;
    oss << "Native Backend (std::thread):\n";
    oss << "  Version: 1.0\n";
    oss << "  Type: std::thread-based\n";
    oss << "  Features:\n";
    oss << "    - Master-worker pattern\n";
    oss << "    - Lazy initialization\n";
    oss << "    - Separate intra-op and inter-op pools\n";
    oss << "    - Exception handling\n";
    oss << "    - NUMA support\n";
    oss << "  Status: " << (IsNativeBackendInitialized() ? "Initialized" : "Not initialized")
        << "\n";
    return oss.str();
}

std::string GetOpenMPBackendInfo()
{
    return openmp::GetOpenMPBackendInfo();
}

}  // namespace xsigma::smp_new::native
