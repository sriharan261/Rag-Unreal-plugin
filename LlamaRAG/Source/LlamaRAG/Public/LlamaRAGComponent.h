#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include <vector>
#include <string>
#include "LlamaRAGComponent.generated.h"

struct llama_model;
struct llama_context;

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnRagAnswerGenerated, const FString&, Answer);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnRagStoryIngested, bool, bSuccess);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnModelsLoaded, bool, bSuccess);

USTRUCT(BlueprintType)
struct FStoryChunk
{
    GENERATED_BODY()
    FString TextContent;
    TArray<float> EmbeddingVector;
};

UCLASS( ClassGroup=(Custom), meta=(BlueprintSpawnableComponent) )
class LLAMARAG_API ULlamaRAGComponent : public UActorComponent
{
    GENERATED_BODY()

public:	
    ULlamaRAGComponent();

protected:
    virtual void BeginPlay() override;
    virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

public:
    UPROPERTY(BlueprintAssignable, Category = "Local RAG")
    FOnModelsLoaded OnModelsLoaded;

    UPROPERTY(BlueprintAssignable, Category = "Local RAG")
    FOnRagStoryIngested OnStoryIngested;

    UPROPERTY(BlueprintAssignable, Category = "Local RAG")
    FOnRagAnswerGenerated OnAnswerGenerated;

    // --- Core Functions ---
    UFUNCTION(BlueprintCallable, Category = "Local RAG")
    void LoadModelsAsync(const FString& LLMModelPath, const FString& EmbedModelPath);

    UFUNCTION(BlueprintCallable, Category = "Local RAG")
    void IngestStoryAsync(const FString& StoryText, int32 ChunkSize = 500);

    // NEW: Reads a .txt file from your computer and ingests it
    UFUNCTION(BlueprintCallable, Category = "Local RAG")
    void IngestStoryFromFileAsync(const FString& FilePath, int32 ChunkSize = 500);

    UFUNCTION(BlueprintCallable, Category = "Local RAG")
    void AskQuestionAsync(const FString& Question, int32 TopK = 3);

    // NEW: Helper to easily get the path to your Content folder in Blueprints
    UFUNCTION(BlueprintPure, Category = "Local RAG|Helpers")
    FString GetContentDirectoryPath() const;

private:
    llama_model* LlmModel;
    llama_context* LlmContext;
    llama_model* EmbedModel;
    llama_context* EmbedContext;

    TArray<FStoryChunk> VectorDB;

    TArray<float> GenerateEmbedding(const FString& Text);
    FString GenerateTextInternal(const FString& Prompt);
    float CalculateCosineSimilarity(const TArray<float>& A, const TArray<float>& B);
    
    bool bIsBusy;
};