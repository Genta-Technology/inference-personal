#include "inference.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <mutex>

std::mutex cout_mutex; // Mutex to synchronize console output

void monitorJob(InferenceEngine& engine, int jobId, const std::string& jobName) {
    size_t lastTextSize = 0;
    while (true) {
        // Check for errors
        if (engine.hasJobError(jobId)) {
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cerr << jobName << " Error: " << engine.getJobError(jobId) << std::endl;
            break;
        }

        // Get the generated text so far
        std::string generatedText = engine.getJobResult(jobId).text;

        // Print new text if available
        if (generatedText.size() > lastTextSize) {
            std::string newText = generatedText.substr(lastTextSize);
            {
                std::lock_guard<std::mutex> lock(cout_mutex);
                std::cout << "[" << jobName << "]: " << newText << std::flush;
            }
            lastTextSize = generatedText.size();
        }

        // Check if the job is finished
        if (engine.isJobFinished(jobId)) {
            break;
        }

        // Sleep briefly to avoid busy waiting
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Print completion message
    {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << std::endl << "[" << jobName << "] completed." << std::endl;
    }
}

int main(int argc, char* argv[])
{
    try
    {
        // Check if the model path is provided as a command-line argument
        if (argc < 2)
        {
            std::cerr << "Usage: " << argv[0] << " <model_path>" << std::endl;
            return 1;
        }
        std::string modelPath = argv[1];

        // Initialize the inference engine with the provided path
        InferenceEngine engine(modelPath);

        // Prepare chat completion parameters for two different jobs
        ChatCompletionParameters chatParams1;
        chatParams1.messages = { {"user", "Tell me a joke."} };

        ChatCompletionParameters chatParams2;
        chatParams2.messages = { {"user", "What's the weather like today?"} };

        // Submit both chat completion jobs
        int jobId1 = engine.submitChatCompleteJob(chatParams1);
        int jobId2 = engine.submitChatCompleteJob(chatParams2);

        // Start monitoring both jobs in separate threads
        std::thread thread1(monitorJob, std::ref(engine), jobId1, "Job1");
        std::thread thread2(monitorJob, std::ref(engine), jobId2, "Job2");

        // Wait for both threads to complete
        thread1.join();
        thread2.join();

        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "CPU Test Error: " << e.what() << std::endl;
    }
    return 0;
}
