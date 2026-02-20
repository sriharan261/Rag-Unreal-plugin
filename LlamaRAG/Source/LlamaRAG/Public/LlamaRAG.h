#pragma once

#include "Modules/ModuleManager.h"

class FLlamaRAGModule : public IModuleInterface
{
public:
    /** IModuleInterface implementation */
    virtual void StartupModule() override;
    virtual void ShutdownModule() override;

private:
    // Keep track of loaded DLLs so we can free them when the engine closes
    TArray<void*> DllHandles;
};