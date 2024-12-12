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

		// Prepare chat completion parameters
		ChatCompletionParameters chatParams;
		chatParams.messages = {
			{"user", "Jelaskan wololo secara detail"} };
		chatParams.maxNewTokens = 512;

		// Perform chat completion
		auto start = std::chrono::high_resolution_clock::now();
		int jobId = engine.submitChatCompleteJob(chatParams);

		std::cout << "Chat completion job submitted with ID: 1" << std::endl;

		size_t lastTextSize = 0;
		size_t tokensSize = 0;
		while (true)
		{
			CompletionResult result = engine.getJobResult(jobId);

			if (result.text.size() > lastTextSize) {
				std::string newText = result.text.substr(lastTextSize);
				std::cout << newText << std::flush; // Output new text
				lastTextSize = result.text.size();
			}

			tokensSize = result.tokens.size();

			if (engine.isJobFinished(jobId)) {
				break;
			}

			if (engine.hasJobError(jobId)) {
				std::cerr << "Job error: " << engine.getJobError(jobId) << std::endl;
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
