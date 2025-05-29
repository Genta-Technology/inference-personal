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
	std::string modelPath = "C:\\Program Files (x86)\\KolosalAI\\models\\Qwen 3 0.6B\\8-bit";

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
	LoadingParameters lParams;
	lParams.n_batch = 128;
	engine->loadModel(modelPath.c_str(), lParams);

	try
	{
		// Prepare chat completion parameters
		ChatCompletionParameters chatParams;
		chatParams.messages = {
			{"system", "/no_think"},
			{"user", "what's the current weather in mumbai?"},
			{"assistant", R"({
  "tool_calls": [
    {
      "name": "get_current_weather",
      "arguments": {
        "location": "San Francisco"
      }
    }
  ]
})"},
			{"tool_call", "{ \"name\": \"get_current_weather\", \"content\": \"sunny\" }" } };
		chatParams.maxNewTokens = 128;
		chatParams.kvCacheFilePath = "kv_cache.bin";
		chatParams.seqId = 0;
		chatParams.tools = R"([
  {
    "type":"function",
    "function":{
      "name":"get_current_weather",
      "description":"Get the current weather in a given location",
      "parameters":{
        "type":"object",
        "properties":{
          "location":{
            "type":"string",
            "description":"The city and state, e.g. San Francisco, CA"
          }
        },
        "required":["location"]
      }
    }
  }
])";

		// Perform chat completion
		auto start = std::chrono::high_resolution_clock::now();
		int jobId = engine->submitChatCompletionsJob(chatParams);

		std::cout << "Chat completion job submitted with ID: 1\n" << std::endl;

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