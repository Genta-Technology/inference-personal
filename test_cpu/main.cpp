#include "inference.h"
#include <iostream>
#include <chrono>

int main(int argc, char* argv[])
{
	try
	{
		// Check if the model path is provided as a command line argument
		if (argc < 2)
		{
			std::cerr << "Usage: " << argv[0] << " <model_path>" << std::endl;
			return 1;
		}
		std::string modelPath = argv[1];
		// Initialize the inference engine with the provided path
		InferenceEngine engine(modelPath);

		// Prepare completion parameters
		CompletionParameters params;
		params.prompt = "Hello, how are you?";

		// Perform text completion
		auto start = std::chrono::high_resolution_clock::now();
		int jobId = engine.submitCompleteJob(params);
		engine.waitForJob(jobId);
		CompletionResult result = engine.getJobResult(jobId);
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> duration = end - start;

		// Output the completion result
		std::cout << "\n\nCPU Completion Result: " << result.text << std::endl;
		std::cout << "Time taken for text completion: " << duration.count() << " seconds" << std::endl;
		std::cout << "Tokens: " << result.tokens.size() << std::endl;
		std::cout << "-----------------------------------------------------------------------------\n";

		// Prepare chat completion parameters
		ChatCompletionParameters chatParams;
		chatParams.messages = {
			{"user", "Tell me a joke."} };

		// Perform chat completion
		auto chatStart = std::chrono::high_resolution_clock::now();
		int chatJobId = engine.submitChatCompleteJob(chatParams);
		engine.waitForJob(chatJobId);
		CompletionResult chatResult = engine.getJobResult(chatJobId);
		auto chatEnd = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> chatDuration = chatEnd - chatStart;

		// Output the chat completion result
		std::cout << "\n\nCPU Chat Completion Result: " << chatResult.text << std::endl;
		std::cout << "Time taken for chat completion: " << chatDuration.count() << " seconds" << std::endl;
		std::cout << "Tokens: " << chatResult.tokens.size() << std::endl;
	}
	catch (const std::exception& e)
	{
		std::cerr << "CPU Test Error: " << e.what() << std::endl;
	}
	return 0;
}
