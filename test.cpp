#include "inference.h"
#include "types.h"

#include <iostream>
#include <chrono>
#include <string>
#include <memory>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

typedef IInferenceEngine* (*CreateInferenceEngineFunc)();
typedef void (*DestroyInferenceEngineFunc)(IInferenceEngine*);

int main(int argc, char* argv[])
{
    std::string modelPath =
#ifdef _WIN32
        "C:\\Users\\rifky\\OneDrive\\Documents\\GitHub\\kolosal\\out\\build\\x64-Release-Debug\\models\\llama-3.2-1B\\int4";
#else
        "/path/to/linux/models/llama-3.2-1B/int4"; // Update this Linux path
#endif

#ifdef USE_VULKAN
#ifdef _WIN32
    std::string libraryName = "InferenceEngineLibVulkan.dll";
#else
    std::string libraryName = "libInferenceEngineLibVulkan.so";
#endif
#else
#ifdef _WIN32
    std::string libraryName = "InferenceEngineLib.dll";
#else
    std::string libraryName = "libInferenceEngineLib.so";
#endif
#endif

    int mainGpuId = 0;

    // Platform-specific library loading
#ifdef _WIN32
    HMODULE hModule = LoadLibraryA(libraryName.c_str());
#else
    void* hModule = dlopen(libraryName.c_str(), RTLD_LAZY);
#endif

    if (!hModule) {
#ifdef _WIN32
        std::cerr << "Failed to load library: " << libraryName << " Error: " << GetLastError() << std::endl;
#else
        std::cerr << "Failed to load library: " << libraryName << " Error: " << dlerror() << std::endl;
#endif
        return 1;
    }

    // Get function pointers
#ifdef _WIN32
    auto createInferenceEngine = (CreateInferenceEngineFunc)GetProcAddress(hModule, "createInferenceEngine");
    auto destroyInferenceEngine = (DestroyInferenceEngineFunc)GetProcAddress(hModule, "destroyInferenceEngine");
#else
    auto createInferenceEngine = (CreateInferenceEngineFunc)dlsym(hModule, "createInferenceEngine");
    auto destroyInferenceEngine = (DestroyInferenceEngineFunc)dlsym(hModule, "destroyInferenceEngine");
#endif

    if (!createInferenceEngine || !destroyInferenceEngine) {
#ifdef _WIN32
        std::cerr << "Failed to get function pointers" << std::endl;
        FreeLibrary(hModule);
#else
        std::cerr << "Failed to get function pointers: " << dlerror() << std::endl;
        dlclose(hModule);
#endif
        return 1;
    }

    IInferenceEngine* engine = createInferenceEngine();
    if (!engine->loadModel(modelPath.c_str(), mainGpuId)) {
        std::cerr << "Failed to load model" << std::endl;
        destroyInferenceEngine(engine);
#ifdef _WIN32
        FreeLibrary(hModule);
#else
        dlclose(hModule);
#endif
        return 1;
    }

    try {
        // Prepare chat completion parameters
        ChatCompletionParameters chatParams;
        chatParams.messages = {
            {"user", "Jelaskan dengan detail kenapa 1 + 1 = 2? dan kenapa x^0 = 1?"}
        };
        chatParams.maxNewTokens = 128;
        chatParams.kvCacheFilePath = "kv_cache.bin";

        // Perform chat completion
        auto start = std::chrono::high_resolution_clock::now();
        int jobId = engine->submitChatCompletionsJob(chatParams);

        std::cout << "Chat completion job submitted with ID: " << jobId << "\n" << std::endl;

        size_t lastTextSize = 0;
        size_t tokensSize = 0;
        while (true) {
            CompletionResult result = engine->getJobResult(jobId);

            if (result.text.size() > lastTextSize) {
                std::string newText = result.text.substr(lastTextSize);
                std::cout << newText << std::flush;
                lastTextSize = result.text.size();
            }

            tokensSize = result.tokens.size();

            if (engine->isJobFinished(jobId)) {
                break;
            }

            if (engine->hasJobError(jobId)) {
                std::cerr << "Job error: " << engine->getJobError(jobId) << std::endl;
                break;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> chatDuration = end - start;

        std::cout << std::endl << "Time taken for chat completion: " << chatDuration.count() << " seconds" << std::endl;
        std::cout << "Tokens: " << tokensSize << std::endl;
        std::cout << "TPS: " << tokensSize / chatDuration.count() << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    // Cleanup
    engine->unloadModel();
    destroyInferenceEngine(engine);
#ifdef _WIN32
    FreeLibrary(hModule);
#else
    dlclose(hModule);
#endif

    return 0;
}