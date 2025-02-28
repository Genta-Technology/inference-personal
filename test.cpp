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

typedef InferenceEngine* (CreateInferenceEngineFunc)();

int main(int argc, char* argv[])
{
	std::string modelPath = "C:\\Users\\rifky\\OneDrive\\Documents\\GitHub\\kolosal\\out\\build\\x64-Release-Debug\\models\\llama-3.2-1B\\int4";
	
#ifdef USE_VULKAN
	std::string libraryName = "InferenceEngineLibVulkan.dll";
#else
	std::string libraryName = "InferenceEngineLib.dll";
#endif
	int			mainGpuId = 0;

#ifdef _WIN32
	HMODULE hModule = LoadLibraryA(TEXT(libraryName).c_str());
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
		ChatCompletionParameters chatParams;
		chatParams.messages = {
			{"user", "Berapa 1 + 1?"} };
		chatParams.maxNewTokens = 32;
		chatParams.kvCacheFilePath = "kv_cache_1.bin";


		ChatCompletionParameters chatParams2;
		chatParams2.messages = {
			{"user", "Apa itu wololo?"} };
		chatParams2.maxNewTokens = 32;
		chatParams2.kvCacheFilePath = "kv_cache_2.bin";

		// Perform chat completion
		auto start = std::chrono::high_resolution_clock::now();
		int jobId = engine->submitChatCompletionsJob(chatParams);
		int jobId2 = engine->submitChatCompletionsJob(chatParams2);

		size_t lastTextSize = 0;
		size_t tokensSize = 0;

		bool isFinished1 = false;
		bool isFinished2 = false;

		while (!isFinished1 || !isFinished2)
		{
			if (!isFinished1)
			{
				CompletionResult result = engine->getJobResult(jobId);

				if (result.text.size() > lastTextSize) {
					std::string newText = result.text.substr(lastTextSize);
					std::cout << newText << std::flush; // Output new text
					lastTextSize = result.text.size();
				}

				tokensSize = result.tokens.size();

				isFinished1 = engine->isJobFinished(jobId);
			}

			if (!isFinished2)
			{
				isFinished2 = engine->isJobFinished(jobId2);
			}
		}

		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> chatDuration = end - start;

		std::cout << std::endl << "Time taken for chat completion: " << chatDuration.count() << " seconds" << std::endl;
		std::cout << "Tokens: " << tokensSize << std::endl;
		std::cout << "TPS: " << tokensSize / chatDuration.count() << std::endl;

		return 0;
	}
	catch (const std::exception& e)
	{
		std::cerr << "CPU Test Error: " << e.what() << std::endl;
	}
	return 0;
}