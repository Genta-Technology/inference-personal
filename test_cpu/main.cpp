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
			{"user", "Tell me a joke."} };
		ChatCompletionParameters chatParams2;
		chatParams2.messages = {
			{"user", "What's the weather today?"} };
		ChatCompletionParameters chatParams3;
		chatParams3.messages = {
			{"user", "What is Wololo"} };
		ChatCompletionParameters chatParams4;
		chatParams4.messages = {
			{"user", "What is the meaning of life?"} };
		ChatCompletionParameters chatParams5;
		chatParams5.messages = {
			{"user", "What is python"} };

		// Perform chat completion
		auto start = std::chrono::high_resolution_clock::now();
		int jobId = engine.submitChatCompleteJob(chatParams);
		int jobId2 = engine.submitChatCompleteJob(chatParams2);
		int jobId3 = engine.submitChatCompleteJob(chatParams3);
		int jobId4 = engine.submitChatCompleteJob(chatParams4);
		int jobId5 = engine.submitChatCompleteJob(chatParams5);

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

		std::cout << std::endl;

		std::cout << "Chat completion job submitted with ID: 2" << std::endl;

		lastTextSize = 0;
		tokensSize = 0;
		while (true)
		{
			CompletionResult result = engine.getJobResult(jobId2);
			if (result.text.size() > lastTextSize) {
				std::string newText = result.text.substr(lastTextSize);
				std::cout << newText << std::flush; // Output new text
				lastTextSize = result.text.size();
			}
			tokensSize = result.tokens.size();
			if (engine.isJobFinished(jobId2)) {
				break;
			}
			if (engine.hasJobError(jobId2)) {
				std::cerr << "Job error: " << engine.getJobError(jobId2) << std::endl;
				break;
			}
		}
		end = std::chrono::high_resolution_clock::now();

		chatDuration = end - start;
		std::cout << std::endl << "Time taken for chat completion: " << chatDuration.count() << " seconds" << std::endl;
		std::cout << "Tokens: " << tokensSize << std::endl;

		std::cout << std::endl;

		std::cout << "Chat completion job submitted with ID: 3" << std::endl;
		
		lastTextSize = 0;
		tokensSize = 0;
		while (true)
		{
			CompletionResult result = engine.getJobResult(jobId3);
			if (result.text.size() > lastTextSize) {
				std::string newText = result.text.substr(lastTextSize);
				std::cout << newText << std::flush; // Output new text
				lastTextSize = result.text.size();
			}
			tokensSize = result.tokens.size();
			if (engine.isJobFinished(jobId3)) {
				break;
			}
			if (engine.hasJobError(jobId3)) {
				std::cerr << "Job error: " << engine.getJobError(jobId3) << std::endl;
				break;
			}
		}
		end = std::chrono::high_resolution_clock::now();
		
		chatDuration = end - start;
		std::cout << std::endl << "Time taken for chat completion: " << chatDuration.count() << " seconds" << std::endl;
		std::cout << "Tokens: " << tokensSize << std::endl;

		std::cout << std::endl;

		std::cout << "Chat completion job submitted with ID: 4" << std::endl;

		lastTextSize = 0;
		tokensSize = 0;
		while (true)
		{
			CompletionResult result = engine.getJobResult(jobId4);
			if (result.text.size() > lastTextSize) {
				std::string newText = result.text.substr(lastTextSize);
				std::cout << newText << std::flush; // Output new text
				lastTextSize = result.text.size();
			}
			tokensSize = result.tokens.size();
			if (engine.isJobFinished(jobId4)) {
				break;
			}
			if (engine.hasJobError(jobId4)) {
				std::cerr << "Job error: " << engine.getJobError(jobId4) << std::endl;
				break;
			}
		}
		end = std::chrono::high_resolution_clock::now();

		chatDuration = end - start;
		std::cout << std::endl << "Time taken for chat completion: " << chatDuration.count() << " seconds" << std::endl;
		std::cout << "Tokens: " << tokensSize << std::endl;

		std::cout << std::endl;

		std::cout << "Chat completion job submitted with ID: 5" << std::endl;

		lastTextSize = 0;
		tokensSize = 0;

		while (true)
		{
			CompletionResult result = engine.getJobResult(jobId5);
			if (result.text.size() > lastTextSize) {
				std::string newText = result.text.substr(lastTextSize);
				std::cout << newText << std::flush; // Output new text
				lastTextSize = result.text.size();
			}
			tokensSize = result.tokens.size();
			if (engine.isJobFinished(jobId5)) {
				break;
			}
			if (engine.hasJobError(jobId5)) {
				std::cerr << "Job error: " << engine.getJobError(jobId5) << std::endl;
				break;
			}
		}
		end = std::chrono::high_resolution_clock::now();

			
		chatDuration = end - start;
		std::cout << std::endl << "Time taken for chat completion: " << chatDuration.count() << " seconds" << std::endl;
		std::cout << "Tokens: " << tokensSize << std::endl;
		return 0;
	}
	catch (const std::exception& e)
	{
		std::cerr << "CPU Test Error: " << e.what() << std::endl;
	}
	return 0;
}
