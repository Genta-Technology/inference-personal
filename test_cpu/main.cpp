#include "inference.h"
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

typedef InferenceEngine* (GetInferenceEngineFunc)(const char*);

int main(int argc, char* argv[])
{
	if (argc < 2)
	{
		std::cerr << "Usage: " << argv[0] << " <model_path> <backend_name; default: openblas>" << std::endl;
		return 1;
	}

	std::string backendName = "vulkan";
	if (argc > 2)
	{
		backendName = argv[2];
	}
	// lower case the backend name
	std::transform(backendName.begin(), backendName.end(), backendName.begin(), ::tolower);

	std::cout << "Backend name: " << backendName << std::endl;

	std::string modelPath = argv[1];
	std::string libraryName;

	if (backendName == "vulkan")
	{
		libraryName = "InferenceEngineLibVulkan.dll";
	}
	else
	{
		std::cerr << "Invalid backend name: " << backendName << std::endl;
		return 1;
	}

#ifdef _WIN32
	HMODULE hModule = LoadLibraryA(TEXT(libraryName).c_str());
	if (hModule == NULL)
	{
		std::cerr << "Failed to load library: " << libraryName << std::endl;
		return 1;
	}

	GetInferenceEngineFunc* getInferenceEngine = (GetInferenceEngineFunc*)GetProcAddress(hModule, "getInferenceEngine");
	if (!getInferenceEngine)
	{
		std::cerr << "Failed to get the address of getInferenceEngine" << std::endl;
		FreeLibrary(hModule);
		return 1;
	}
#else
	void* handle = dlopen(libraryName.c_str(), RTLD_LAZY);
	if (!handle)
	{
		std::cerr << "Failed to load library: " << dlerror() << std::endl;
		return 1;
	}

	GetInferenceEngineFunc getInferenceEngine = (GetInferenceEngineFunc)dlsym(handle, "getInferenceEngine");
	if (!getInferenceEngine)
	{
		std::cerr << "Failed to get the address of getInferenceEngine" << std::endl;
		dlclose(handle);
		return 1;
	}
#endif

	InferenceEngine* engine = getInferenceEngine(modelPath.c_str());

	try
	{
		// Prepare chat completion parameters
		ChatCompletionParameters chatParams;
		chatParams.messages = {
			{"user", "Jelaskan wololo secara detail"} };
		chatParams.maxNewTokens = 512;

		// Perform chat completion
		auto start = std::chrono::high_resolution_clock::now();
		int jobId = engine->submitChatCompletionsJob(chatParams);

		std::cout << "Chat completion job submitted with ID: 1" << std::endl;

		size_t lastTextSize = 0;
		size_t tokensSize = 0;
		while (true)
		{
			CompletionResult result = engine->getJobResult(jobId);

			if (result.text.size() > lastTextSize) {
				std::string newText = result.text.substr(lastTextSize);
				std::cout << newText << std::flush; // Output new text
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

		return 0;
	}
	catch (const std::exception& e)
	{
		std::cerr << "CPU Test Error: " << e.what() << std::endl;
	}
	return 0;
}