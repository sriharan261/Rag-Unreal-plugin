using UnrealBuildTool;
using System.IO;

public class LlamaRAG : ModuleRules
{
    public LlamaRAG(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

        // Add "Projects" to this array so IPluginManager can be linked properly
        PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine", "InputCore", "Projects" });

        string ThirdPartyPath = Path.Combine(ModuleDirectory, "../../ThirdParty/LlamaCpp");
        PublicIncludePaths.Add(Path.Combine(ThirdPartyPath, "include"));

        if (Target.Platform == UnrealTargetPlatform.Win64)
        {
            // Link the static library
            PublicAdditionalLibraries.Add(Path.Combine(ThirdPartyPath, "lib", "llama.lib"));
            
            // Dynamically copy and load ALL DLLs (llama.dll, ggml.dll, etc.)
            string BinPath = Path.Combine(ThirdPartyPath, "bin");
            if (Directory.Exists(BinPath))
            {
                string[] DllFiles = Directory.GetFiles(BinPath, "*.dll");
                foreach (string DllFile in DllFiles)
                {
                    string DllName = Path.GetFileName(DllFile);
                    RuntimeDependencies.Add(Path.Combine("$(BinaryOutputDir)", DllName), DllFile);
                    PublicDelayLoadDLLs.Add(DllName);
                }
            }
        }
    }
}
