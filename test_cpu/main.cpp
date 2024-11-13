#include "inference.h"
#include <iostream>

int main(int argc, char *argv[])
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
        auto result = engine.complete(params);

        // Output the completion result
        std::cout << "CPU Completion Result: " << result.text << std::endl;

        // Prepare chat completion parameters
        ChatCompletionParameters chatParams;
        chatParams.messages = {
            {"user", "Tell me a joke."}};

        // Perform chat completion
        auto chatResult = engine.chatComplete(chatParams);

        // Output the chat completion result
        std::cout << "CPU Chat Completion Result: " << chatResult.text << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "CPU Test Error: " << e.what() << std::endl;
    }
    return 0;
}
