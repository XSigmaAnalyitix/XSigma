

// This file incorporates code from the Visualization Toolkit (VTK) and remains subject to the BSD-3-Clause VTK license.

#include "smp/Common/tools_api.h"

#include <algorithm>  // For std::toupper
#include <cstdlib>    // For std::getenv
#include <iostream>   // For std::cerr
#include <string>     // For std::string

namespace xsigma::detail::smp
{

//------------------------------------------------------------------------------
tools_api::tools_api()
{
#if define(XSIGMA_ENABLE_TBB)
    this->TBBBackend = std::make_unique<tools_impl<BackendType::TBB>>();
#else    
    this->STDThreadBackend = std::make_unique<tools_impl<BackendType::STDThread>>();
#endif

    // Set backend from env if set
    const char* xsigmaSMPBackendInUse = std::getenv("XSIGMA_SMP_BACKEND_IN_USE");
    if (xsigmaSMPBackendInUse != nullptr)
    {
        this->SetBackend(xsigmaSMPBackendInUse);
    }

    // Set max thread number from env
    this->RefreshNumberOfThread();
}

//------------------------------------------------------------------------------
// Must NOT be initialized. Default initialization to zero is necessary.
tools_api* toolsAPIInstanceAsPointer;

//------------------------------------------------------------------------------
tools_api& tools_api::GetInstance()
{
    return *toolsAPIInstanceAsPointer;
}

//------------------------------------------------------------------------------
void tools_api::ClassInitialize()
{
    if (toolsAPIInstanceAsPointer == nullptr)
    {
        toolsAPIInstanceAsPointer = new tools_api;
    }
}

//------------------------------------------------------------------------------
void tools_api::ClassFinalize()
{
    delete toolsAPIInstanceAsPointer;
    toolsAPIInstanceAsPointer = nullptr;
}

//------------------------------------------------------------------------------
BackendType tools_api::GetBackendType()
{
    return this->ActivatedBackend;
}

//------------------------------------------------------------------------------
const char* tools_api::GetBackend()
{
    switch (this->ActivatedBackend)
    {
    case BackendType::STDThread:
        return "STDThread";
    case BackendType::TBB:
        return "TBB";
    }
    return nullptr;
}

//------------------------------------------------------------------------------
bool tools_api::SetBackend(const char* type)
{
    std::string backend(type);
    std::transform(backend.cbegin(), backend.cend(), backend.begin(), ::toupper);
   if (backend == "STDTHREAD" && this->STDThreadBackend)
    {
        this->ActivatedBackend = BackendType::STDThread;
    }
    else if (backend == "TBB" && this->TBBBackend)
    {
        this->ActivatedBackend = BackendType::TBB;
    }
    else
    {
        std::cerr << "WARNING: tried to use a non implemented SMPTools backend \"" << type
                  << "\"!\n";
        std::cerr << "The available backends are:"
                  << (this->STDThreadBackend ? " \"STDThread\"" : "")
                  << (this->TBBBackend ? " \"TBB\"" : "") << "\n";
        std::cerr << "Using " << this->GetBackend() << " instead." << std::endl;
        return false;
    }
    this->RefreshNumberOfThread();
    return true;
}

//------------------------------------------------------------------------------
void tools_api::Initialize(int numThreads)
{
    this->DesiredNumberOfThread = numThreads;
    this->RefreshNumberOfThread();
}

//------------------------------------------------------------------------------
void tools_api::RefreshNumberOfThread()
{
    const int numThreads = this->DesiredNumberOfThread;
    switch (this->ActivatedBackend)
    {
    case BackendType::STDThread:
        this->STDThreadBackend->Initialize(numThreads);
        break;
    case BackendType::TBB:
        this->TBBBackend->Initialize(numThreads);
        break;
    }
}

//------------------------------------------------------------------------------
int tools_api::GetEstimatedDefaultNumberOfThreads()
{
    switch (this->ActivatedBackend)
    {
    case BackendType::STDThread:
        return this->STDThreadBackend->GetEstimatedDefaultNumberOfThreads();
    case BackendType::TBB:
        return this->TBBBackend->GetEstimatedDefaultNumberOfThreads();
    }
    return 0;
}

//------------------------------------------------------------------------------
int tools_api::GetEstimatedNumberOfThreads()
{
    switch (this->ActivatedBackend)
    {
    case BackendType::STDThread:
        return this->STDThreadBackend->GetEstimatedNumberOfThreads();
    case BackendType::TBB:
        return this->TBBBackend->GetEstimatedNumberOfThreads();
    }
    return 0;
}

//------------------------------------------------------------------------------
void tools_api::SetNestedParallelism(bool isNested)
{
    switch (this->ActivatedBackend)
    {
    case BackendType::STDThread:
        this->STDThreadBackend->SetNestedParallelism(isNested);
        break;
    case BackendType::TBB:
        this->TBBBackend->SetNestedParallelism(isNested);
        break;
    }
}

//------------------------------------------------------------------------------
bool tools_api::GetNestedParallelism()
{
    switch (this->ActivatedBackend)
    {
    case BackendType::STDThread:
        return this->STDThreadBackend->GetNestedParallelism();
    case BackendType::TBB:
        return this->TBBBackend->GetNestedParallelism();
    }
    return false;
}

//------------------------------------------------------------------------------
bool tools_api::IsParallelScope()
{
    switch (this->ActivatedBackend)
    {
    case BackendType::STDThread:
        return this->STDThreadBackend->IsParallelScope();
    case BackendType::TBB:
        return this->TBBBackend->IsParallelScope();
    }
    return false;
}

//------------------------------------------------------------------------------
bool tools_api::GetSingleThread()
{
    // Currently, this will work as expected for one parallel area and or nested
    // parallel areas. If there are two or more parallel areas that are not nested,
    // this function will not work properly.
    switch (this->ActivatedBackend)
    {
    case BackendType::STDThread:
        return this->STDThreadBackend->GetSingleThread();
    case BackendType::TBB:
        return this->TBBBackend->GetSingleThread();
    default:
        return false;
    }
}

//------------------------------------------------------------------------------
// Must NOT be initialized. Default initialization to zero is necessary.
unsigned int toolsAPIInitializeCount;

//------------------------------------------------------------------------------
toolsAPIInitialize::toolsAPIInitialize()
{
    if (++toolsAPIInitializeCount == 1)
    {
        tools_api::ClassInitialize();
    }
}

//------------------------------------------------------------------------------
toolsAPIInitialize::~toolsAPIInitialize()
{
    if (--toolsAPIInitializeCount == 0)
    {
        tools_api::ClassFinalize();
    }
}

}  // namespace xsigma::detail::smp
