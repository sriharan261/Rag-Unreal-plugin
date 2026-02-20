#include "LlamaRAG.h"
#include "Interfaces/IPluginManager.h"
#include "Misc/Paths.h"
#include "HAL/PlatformProcess.h"

#define LOCTEXT_NAMESPACE "FLlamaRAGModule"

void FLlamaRAGModule::StartupModule()
{
    // Get the base directory of this plugin
    FString BaseDir = IPluginManager::Get().FindPlugin("LlamaRAG")->GetBaseDir();
    
    // The path to your compiled DLLs
    FString LibraryPath = FPaths::Combine(*BaseDir, TEXT("ThirdParty/LlamaCpp/bin"));

    // IMPORTANT: DLLs must be loaded in dependency order!
    // ggml libraries must load BEFORE llama.dll
    TArray<FString> DllsToLoad = {
        TEXT("ggml-base.dll"),
        TEXT("ggml-cpu.dll"),
        TEXT("ggml.dll"),
        TEXT("llama.dll")
    };

    for (const FString& DllName : DllsToLoad)
    {
        FString DllPath = FPaths::Combine(*LibraryPath, DllName);
        void* Handle = FPlatformProcess::GetDllHandle(*DllPath);
        
        if (Handle != nullptr)
        {
            DllHandles.Add(Handle);
        }
        else
        {
            // If a DLL fails to load, log it. (Note: It's okay if ggml-base or ggml-cpu 
            // fail if your specific llama.cpp build didn't generate them).
            UE_LOG(LogTemp, Warning, TEXT("LlamaRAG: Did not load DLL (might not exist or failed) -> %s"), *DllPath);
        }
    }
}

void FLlamaRAGModule::ShutdownModule()
{
    // Free DLLs in reverse order
    for (int32 i = DllHandles.Num() - 1; i >= 0; --i)
    {
        if (DllHandles[i])
        {
            FPlatformProcess::FreeDllHandle(DllHandles[i]);
        }
    }
    DllHandles.Empty();
}

#undef LOCTEXT_NAMESPACE

// This tells Unreal Engine that FLlamaRAGModule is the main module class
IMPLEMENT_MODULE(FLlamaRAGModule, LlamaRAG)