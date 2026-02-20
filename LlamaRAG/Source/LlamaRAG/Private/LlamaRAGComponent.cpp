#include "LlamaRAGComponent.h"
#include "Async/Async.h"
#include "Misc/Paths.h"
#include "Misc/FileHelper.h"
#include "Math/UnrealMathUtility.h"
#include "llama.h"

DEFINE_LOG_CATEGORY_STATIC(LogLlamaRAG, Log, All);

ULlamaRAGComponent::ULlamaRAGComponent()
{
    PrimaryComponentTick.bCanEverTick = false;
    LlmModel = nullptr; LlmContext = nullptr;
    EmbedModel = nullptr; EmbedContext = nullptr;
    bIsBusy = false;
}

void ULlamaRAGComponent::BeginPlay() { Super::BeginPlay(); llama_backend_init(); }

void ULlamaRAGComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
    if (LlmContext) { llama_free(LlmContext); LlmContext = nullptr; }
    if (LlmModel) { llama_model_free(LlmModel); LlmModel = nullptr; }
    if (EmbedContext) { llama_free(EmbedContext); EmbedContext = nullptr; }
    if (EmbedModel) { llama_model_free(EmbedModel); EmbedModel = nullptr; }
    llama_backend_free();
    Super::EndPlay(EndPlayReason);
}

FString ULlamaRAGComponent::GetContentDirectoryPath() const
{
    return FPaths::ConvertRelativePathToFull(FPaths::ProjectContentDir());
}

void ULlamaRAGComponent::LoadModelsAsync(const FString& LLMModelPath, const FString& EmbedModelPath)
{
    if (bIsBusy) return;
    bIsBusy = true;

    Async(EAsyncExecution::Thread, [this, LLMModelPath, EmbedModelPath]()
    {
        llama_model_params model_params = llama_model_default_params();
        LlmModel = llama_model_load_from_file(TCHAR_TO_UTF8(*LLMModelPath), model_params);
        
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = 2048;
        if (LlmModel) LlmContext = llama_init_from_model(LlmModel, ctx_params);

        llama_model_params embed_m_params = llama_model_default_params();
        EmbedModel = llama_model_load_from_file(TCHAR_TO_UTF8(*EmbedModelPath), embed_m_params);
        
        llama_context_params embed_c_params = llama_context_default_params();
        embed_c_params.embeddings = true;
        if (EmbedModel) EmbedContext = llama_init_from_model(EmbedModel, embed_c_params);

        bool bSuccess = (LlmContext != nullptr && EmbedContext != nullptr);

        AsyncTask(ENamedThreads::GameThread, [this, bSuccess]()
        {
            bIsBusy = false;
            OnModelsLoaded.Broadcast(bSuccess);
        });
    });
}

// NEW: File Reader Implementation
void ULlamaRAGComponent::IngestStoryFromFileAsync(const FString& FilePath, int32 ChunkSize)
{
    FString FileContent;
    if (FFileHelper::LoadFileToString(FileContent, *FilePath))
    {
        IngestStoryAsync(FileContent, ChunkSize);
    }
    else
    {
        UE_LOG(LogLlamaRAG, Error, TEXT("Failed to load TXT file from: %s"), *FilePath);
        OnStoryIngested.Broadcast(false);
    }
}

void ULlamaRAGComponent::IngestStoryAsync(const FString& StoryText, int32 ChunkSize)
{
    if (bIsBusy || !EmbedContext) return;
    bIsBusy = true;
    VectorDB.Empty();

    Async(EAsyncExecution::Thread, [this, StoryText, ChunkSize]()
    {
        TArray<FString> Chunks;
        for (int32 i = 0; i < StoryText.Len(); i += ChunkSize)
        {
            Chunks.Add(StoryText.Mid(i, ChunkSize));
        }

        for (const FString& ChunkStr : Chunks)
        {
            TArray<float> EmbedVector = GenerateEmbedding(ChunkStr);
            if (EmbedVector.Num() > 0)
            {
                FStoryChunk NewChunk;
                NewChunk.TextContent = ChunkStr;
                NewChunk.EmbeddingVector = EmbedVector;
                VectorDB.Add(NewChunk);
            }
        }

        AsyncTask(ENamedThreads::GameThread, [this]()
        {
            bIsBusy = false;
            OnStoryIngested.Broadcast(VectorDB.Num() > 0);
        });
    });
}

void ULlamaRAGComponent::AskQuestionAsync(const FString& Question, int32 TopK)
{
    if (bIsBusy || !LlmContext || VectorDB.Num() == 0) return;
    bIsBusy = true;

    Async(EAsyncExecution::Thread, [this, Question, TopK]()
    {
        TArray<float> QuestionEmbedding = GenerateEmbedding(Question);

        struct FScorePair { float Score; int32 Index; };
        TArray<FScorePair> Scores;
        
        for (int32 i = 0; i < VectorDB.Num(); ++i)
        {
            float SimScore = CalculateCosineSimilarity(QuestionEmbedding, VectorDB[i].EmbeddingVector);
            Scores.Add({SimScore, i});
        }

        Scores.Sort([](const FScorePair& A, const FScorePair& B) { return A.Score > B.Score; });

        FString Context = TEXT("");
        for (int32 i = 0; i < FMath::Min(TopK, Scores.Num()); ++i)
        {
            Context += VectorDB[Scores[i].Index].TextContent + TEXT("\n");
        }

        FString Prompt = FString::Printf(TEXT("Context:\n%s\n\nQuestion: %s\nAnswer:"), *Context, *Question);
        FString FinalAnswer = GenerateTextInternal(Prompt);

        AsyncTask(ENamedThreads::GameThread, [this, FinalAnswer]()
        {
            bIsBusy = false;
            OnAnswerGenerated.Broadcast(FinalAnswer.TrimStartAndEnd());
        });
    });
}

TArray<float> ULlamaRAGComponent::GenerateEmbedding(const FString& Text)
{
    TArray<float> Result;
    if (!EmbedContext) return Result;

    std::string StdText = TCHAR_TO_UTF8(*Text);
    const llama_vocab* vocab = llama_model_get_vocab(EmbedModel);
    
    std::vector<llama_token> tokens(StdText.length() + 1);
    int n_tokens = llama_tokenize(vocab, StdText.c_str(), StdText.length(), tokens.data(), tokens.size(), true, false);
    if (n_tokens < 0) return Result;
    tokens.resize(n_tokens);

    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    if (llama_decode(EmbedContext, batch) != 0) return Result;

    const int n_embd = llama_model_n_embd(EmbedModel);
    const float* embeddings = llama_get_embeddings_seq(EmbedContext, 0);
    
    if (embeddings) { for (int i = 0; i < n_embd; ++i) Result.Add(embeddings[i]); }
    return Result;
}

FString ULlamaRAGComponent::GenerateTextInternal(const FString& Prompt)
{
    std::string StdPrompt = TCHAR_TO_UTF8(*Prompt);
    const llama_vocab* vocab = llama_model_get_vocab(LlmModel);
    
    std::vector<llama_token> tokens(StdPrompt.length() + 1);
    int n_tokens = llama_tokenize(vocab, StdPrompt.c_str(), StdPrompt.length(), tokens.data(), tokens.size(), true, false);
    tokens.resize(n_tokens);

    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    llama_decode(LlmContext, batch);

    FString Output = TEXT("");
    int n_cur = batch.n_tokens;
    int n_max = 512; 

    while (n_cur <= n_max)
    {
        float* logits = llama_get_logits_ith(LlmContext, batch.n_tokens - 1);
        int32_t n_vocab = llama_vocab_n_tokens(vocab);
        
        llama_token new_token_id = 0;
        float max_logit = -1e9f;
        for (int32_t i = 0; i < n_vocab; ++i)
        {
            if (logits[i] > max_logit) { max_logit = logits[i]; new_token_id = i; }
        }

        if (new_token_id == llama_vocab_eos(vocab)) break;

        char buf[128];
        int n_chars = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, false);
        if (n_chars > 0)
        {
            std::string token_str(buf, n_chars);
            Output += FString(UTF8_TO_TCHAR(token_str.c_str()));
        }

        batch = llama_batch_get_one(&new_token_id, 1);
        llama_decode(LlmContext, batch);
        n_cur++;
    }

    return Output;
}

float ULlamaRAGComponent::CalculateCosineSimilarity(const TArray<float>& A, const TArray<float>& B)
{
    if (A.Num() != B.Num() || A.Num() == 0) return 0.0f;
    float Dot = 0.0f, NormA = 0.0f, NormB = 0.0f;
    for (int32 i = 0; i < A.Num(); ++i) { Dot += A[i] * B[i]; NormA += A[i] * A[i]; NormB += B[i] * B[i]; }
    float Denominator = FMath::Sqrt(NormA) * FMath::Sqrt(NormB);
    return (Denominator > 0.0f) ? (Dot / Denominator) : 0.0f;
}