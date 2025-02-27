#include "inference.h"
#include "types.h"

#include <iostream>
#include <chrono>
#include <string>
#include <memory>
#include <thread>
#include <vector>
#include <iomanip> // For output formatting

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

typedef InferenceEngine* (CreateInferenceEngineFunc)();

int main(int argc, char* argv[])
{
    std::string modelPath = "C:\\Users\\rifky\\OneDrive\\Documents\\GitHub\\kolosal\\out\\build\\x64-Release-Debug\\models\\llama-3.2-1B\\int4";

#ifdef USE_VULKAN
    std::string libraryName = "InferenceEngineLibVulkan.dll";
#else
    std::string libraryName = "InferenceEngineLib.dll";
#endif
    int            mainGpuId = 0;

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
        std::cerr << "Failed to get the address of getInferenceEngine" << std::endl;
        FreeLibrary(hModule);
        return 1;
    }
#endif

    InferenceEngine* engine = createInferenceEngine();
    engine->loadModel(modelPath.c_str(), mainGpuId);

    try
    {
        // Prepare chat completion parameters
        ChatCompletionParameters chatParams1;
        chatParams1.messages = {
            {"user", "Apa itu wololo"} };
        chatParams1.maxNewTokens = 32;

        ChatCompletionParameters chatParams2;
        chatParams2.messages = {
            {"user", "Apa itu game Halo"} };
        chatParams2.maxNewTokens = 32;

        // Perform chat completion
        auto startTime = std::chrono::high_resolution_clock::now();
        int jobId1 = engine->submitChatCompletionsJob(chatParams1);
        int jobId2 = engine->submitChatCompletionsJob(chatParams2);

        std::cout << "Chat completion jobs submitted with IDs: " << jobId1 << " and " << jobId2 << std::endl;
        std::cout << "Processing both jobs concurrently...\n" << std::endl;

        // To store the complete results
        std::string completeOutput1;
        std::string completeOutput2;
        size_t tokensSize1 = 0;
        size_t tokensSize2 = 0;

        bool job1Finished = false;
        bool job2Finished = false;

        // Track individual job completion times
        std::chrono::time_point<std::chrono::high_resolution_clock> job1FinishTime;
        std::chrono::time_point<std::chrono::high_resolution_clock> job2FinishTime;

        // Progress indicator
        const std::vector<char> spinner = { '|', '/', '-', '\\' };
        int spinnerIdx = 0;

        while (!job1Finished || !job2Finished)
        {
            std::cout << "\r" << spinner[spinnerIdx] << " Processing... ";
            spinnerIdx = (spinnerIdx + 1) % spinner.size();

            // Process Job 1
            if (!job1Finished) {
                CompletionResult result1 = engine->getJobResult(jobId1);
                completeOutput1 = result1.text;
                tokensSize1 = result1.tokens.size();

                if (engine->isJobFinished(jobId1)) {
                    job1Finished = true;
                    job1FinishTime = std::chrono::high_resolution_clock::now();
                    std::cout << "Job 1 complete! ";
                }

                if (engine->hasJobError(jobId1)) {
                    std::cerr << "Job 1 Error: " << engine->getJobError(jobId1) << " ";
                    job1Finished = true;
                    job1FinishTime = std::chrono::high_resolution_clock::now();
                }
            }

            // Process Job 2
            if (!job2Finished) {
                CompletionResult result2 = engine->getJobResult(jobId2);
                completeOutput2 = result2.text;
                tokensSize2 = result2.tokens.size();

                if (engine->isJobFinished(jobId2)) {
                    job2Finished = true;
                    job2FinishTime = std::chrono::high_resolution_clock::now();
                    std::cout << "Job 2 complete! ";
                }

                if (engine->hasJobError(jobId2)) {
                    std::cerr << "Job 2 Error: " << engine->getJobError(jobId2) << " ";
                    job2Finished = true;
                    job2FinishTime = std::chrono::high_resolution_clock::now();
                }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        auto endTime = std::chrono::high_resolution_clock::now();

        // Calculate durations
        std::chrono::duration<double> job1Duration = job1FinishTime - startTime;
        std::chrono::duration<double> job2Duration = job2FinishTime - startTime;
        std::chrono::duration<double> totalDuration = endTime - startTime;

        // Clear the progress line
        std::cout << "\r" << std::string(50, ' ') << "\r";

        // Display the results sequentially
        std::cout << "\n=== JOB 1 RESULT ===\n";
        std::cout << "Query: Apa itu wololo\n";
        std::cout << "Response: " << completeOutput1 << std::endl;
        std::cout << "Tokens generated: " << tokensSize1 << std::endl;
        std::cout << "Time taken: " << std::fixed << std::setprecision(3) << job1Duration.count() << " seconds" << std::endl;
        std::cout << "Tokens per second: " << std::fixed << std::setprecision(2)
            << (tokensSize1 / job1Duration.count()) << std::endl;

        std::cout << "\n=== JOB 2 RESULT ===\n";
        std::cout << "Query: Apa itu game Halo\n";
        std::cout << "Response: " << completeOutput2 << std::endl;
        std::cout << "Tokens generated: " << tokensSize2 << std::endl;
        std::cout << "Time taken: " << std::fixed << std::setprecision(3) << job2Duration.count() << " seconds" << std::endl;
        std::cout << "Tokens per second: " << std::fixed << std::setprecision(2)
            << (tokensSize2 / job2Duration.count()) << std::endl;

        std::cout << "\n=== PERFORMANCE METRICS ===\n";
        std::cout << "Total time for both completions: " << std::fixed << std::setprecision(3)
            << totalDuration.count() << " seconds" << std::endl;
        std::cout << "Total tokens generated: " << (tokensSize1 + tokensSize2) << std::endl;
        std::cout << "Combined tokens per second: " << std::fixed << std::setprecision(2)
            << ((tokensSize1 + tokensSize2) / totalDuration.count()) << std::endl;

        // Calculate efficiency gained by parallel processing
        double sequentialTime = job1Duration.count() + job2Duration.count();
        double parallelEfficiency = (sequentialTime / totalDuration.count()) * 100.0;
        std::cout << "Parallel efficiency: " << std::fixed << std::setprecision(1)
            << parallelEfficiency << "% (compared to sequential processing)" << std::endl;

        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "CPU Test Error: " << e.what() << std::endl;
    }
    return 0;
}
