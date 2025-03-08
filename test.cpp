#include "inference.h"
#include "types.h"

#include <iostream>
#include <chrono>
#include <string>
#include <memory>
#include <thread>
#include <fstream>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

typedef IInferenceEngine* (*CreateInferenceEngineFunc)();
typedef void (*DestroyInferenceEngineFunc)(IInferenceEngine*);

// Helper function to generate a string of repeated text to create a long prompt
std::string generateLongPrompt(int targetLength) {
    std::string basePrompt = "This is a test of the context shifting mechanism. ";
    std::string repeated;

    while (repeated.length() < targetLength) {
        repeated += basePrompt;
    }

    return repeated;
}

// Helper function to write result to file for detailed analysis
void writeResultToFile(const std::string& filename, const std::string& content) {
    std::ofstream outFile(filename);
    if (outFile.is_open()) {
        outFile << content;
        outFile.close();
        std::cout << "Result written to " << filename << std::endl;
    }
    else {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
    }
}

int main(int argc, char* argv[])
{
    // Current user and timestamp as per your request
    std::cout << "Test run by: rifkybujana" << std::endl;
    std::cout << "Date/Time: 2025-03-07 17:03:55 UTC" << std::endl;

    std::string modelPath = "C:\\Users\\rifky\\OneDrive\\Documents\\GitHub\\kolosal\\out\\build\\x64-Release-Debug\\models\\qwen2.5-0.5b\\fp16";

#ifdef USE_VULKAN
    std::string libraryName = "InferenceEngineLibVulkan.dll";
#else
    std::string libraryName = "InferenceEngineLib.dll";
#endif

    if (!hModule) {
#ifdef _WIN32
    HMODULE hModule = LoadLibraryA(libraryName.c_str());
    if (hModule == NULL)
    {
        std::cerr << "Failed to load library: " << libraryName << std::endl;
        return 1;
    }

    CreateInferenceEngineFunc* createInferenceEngine = (CreateInferenceEngineFunc*)GetProcAddress(hModule, "createInferenceEngine");
    if (!createInferenceEngine)
    {
        std::cerr << "Failed to get the address of createInferenceEngine" << std::endl;
        FreeLibrary(hModule);
        return 1;
    }
#endif

    InferenceEngine* engine = createInferenceEngine();

    // Set very small context size to force context shifting
    LoadingParameters lParams;
    lParams.n_ctx = 128;  // Small context window
    lParams.n_keep = 1;   // Only keep the first token when shifting

    if (!engine->loadModel(modelPath.c_str(), lParams)) {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }

    try
    {
        std::cout << "\n=== TEST 1: LONG PROMPT CONTEXT SHIFTING ===\n" << std::endl;

        // Create a long prompt that will exceed the context window
        // Assuming each token is roughly 4 characters on average
        std::string longPrompt = generateLongPrompt(300); // ~75 tokens
        longPrompt += "\n\nSummarize the above text in one sentence:";

        std::cout << "Prompt length: " << longPrompt.length() << " characters" << std::endl;
        std::cout << "Expected tokens (approx): " << longPrompt.length() / 4 << std::endl;
        std::cout << "Context window size: " << lParams.n_ctx << " tokens" << std::endl;

        // Test with completion parameters
        CompletionParameters params;
        params.prompt = longPrompt.c_str();
        params.maxNewTokens = 64;
        params.temperature = 0.7;
        params.topP = 0.9;
        params.kvCacheFilePath = "kv_cache_shift_test1.bin";

        auto start = std::chrono::high_resolution_clock::now();
        int jobId = engine->submitCompletionsJob(params);

        std::string fullOutput;
        size_t lastTextSize = 0;
        size_t tokensGenerated = 0;

        // Track generation
        while (!engine->isJobFinished(jobId)) {
            CompletionResult result = engine->getJobResult(jobId);

            if (result.text.size() > lastTextSize) {
                std::string newText = result.text.substr(lastTextSize);
                std::cout << newText << std::flush;
                fullOutput += newText;
                lastTextSize = result.text.size();
            }

            tokensGenerated = result.tokens.size();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        // Check final result
        CompletionResult finalResult = engine->getJobResult(jobId);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;

        std::cout << "\n\nTest 1 Results:" << std::endl;
        std::cout << "  - Tokens generated: " << tokensGenerated << std::endl;
        std::cout << "  - Time taken: " << duration.count() << " seconds" << std::endl;
        std::cout << "  - TPS: " << tokensGenerated / duration.count() << std::endl;

        if (engine->hasJobError(jobId)) {
            std::cerr << "Error in job: " << engine->getJobError(jobId) << std::endl;
        }

        // Write the full output to file for detailed analysis
        writeResultToFile("test1_output.txt", fullOutput);

        std::cout << "\n=== TEST 2: CONTINUOUS GENERATION WITH CONTEXT SHIFTING ===\n" << std::endl;

        // For the second test, we'll use a shorter prompt but generate many tokens
        // to force context shifting during generation
        std::string prompt = "Write a continuous story about a robot learning to feel emotions. Start with:";

        CompletionParameters params2;
        params2.prompt = prompt.c_str();
        params2.maxNewTokens = 512; // Much larger than our context window
        params2.temperature = 0.8;
        params2.topP = 0.9;
        params2.kvCacheFilePath = "kv_cache_shift_test2.bin";

        start = std::chrono::high_resolution_clock::now();
        int jobId2 = engine->submitCompletionsJob(params2);

        fullOutput = "";
        lastTextSize = 0;
        tokensGenerated = 0;
        int tokensDisplayed = 0;

        // Track generation and count shifts
        while (!engine->isJobFinished(jobId2)) {
            CompletionResult result = engine->getJobResult(jobId2);

            if (result.text.size() > lastTextSize) {
                std::string newText = result.text.substr(lastTextSize);
                std::cout << newText << std::flush;
                fullOutput += newText;
                lastTextSize = result.text.size();

                // Count context shifts (approximate detection based on token count)
                if (tokensDisplayed > 0 && tokensDisplayed % 100 == 0) {
                    std::cout << "\n[Context likely shifted around this point]\n" << std::flush;
                }
                tokensDisplayed++;
            }

            tokensGenerated = result.tokens.size();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        // Check final result
        CompletionResult finalResult2 = engine->getJobResult(jobId2);

        end = std::chrono::high_resolution_clock::now();
        duration = end - start;

        std::cout << "\n\nTest 2 Results:" << std::endl;
        std::cout << "  - Tokens generated: " << tokensGenerated << std::endl;
        std::cout << "  - Time taken: " << duration.count() << " seconds" << std::endl;
        std::cout << "  - TPS: " << tokensGenerated / duration.count() << std::endl;
        std::cout << "  - Estimated context shifts: " << (tokensGenerated / lParams.n_ctx) << std::endl;

        if (engine->hasJobError(jobId2)) {
            std::cerr << "Error in job: " << engine->getJobError(jobId2) << std::endl;
        }

        // Write the full output to file for detailed analysis
        writeResultToFile("test2_output.txt", fullOutput);

        std::cout << "\n=== TEST 3: KV CACHE REUSE WITH CONTEXT SHIFTING ===\n" << std::endl;

        // For the third test, we'll use the same KV cache file as test 2 but add more text
        // This tests our KV cache loading with past context shifts
        std::string followupPrompt = fullOutput.substr(0, 100) + "\n\nContinue the story:";

        CompletionParameters params3;
        params3.prompt = followupPrompt.c_str();
        params3.maxNewTokens = 128;
        params3.temperature = 0.8;
        params3.topP = 0.9;
        params3.kvCacheFilePath = "kv_cache_shift_test2.bin"; // Reuse the same KV cache

        start = std::chrono::high_resolution_clock::now();
        int jobId3 = engine->submitCompletionsJob(params3);

        fullOutput = "";
        lastTextSize = 0;
        tokensGenerated = 0;

        // Track generation
        while (!engine->isJobFinished(jobId3)) {
            CompletionResult result = engine->getJobResult(jobId3);

            if (result.text.size() > lastTextSize) {
                std::string newText = result.text.substr(lastTextSize);
                std::cout << newText << std::flush;
                fullOutput += newText;
                lastTextSize = result.text.size();
            }

            tokensGenerated = result.tokens.size();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        end = std::chrono::high_resolution_clock::now();
        duration = end - start;

        std::cout << "\n\nTest 3 Results:" << std::endl;
        std::cout << "  - Tokens generated: " << tokensGenerated << std::endl;
        std::cout << "  - Time taken: " << duration.count() << " seconds" << std::endl;
        std::cout << "  - TPS: " << tokensGenerated / duration.count() << std::endl;

        if (engine->hasJobError(jobId3)) {
            std::cerr << "Error in job: " << engine->getJobError(jobId3) << std::endl;
        }

        // Write the full output to file for detailed analysis
        writeResultToFile("test3_output.txt", fullOutput);

        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Test Error: " << e.what() << std::endl;
        return 1;
    }
}