#ifndef INFERENCE_H
#define INFERENCE_H

#include <string>
#include <vector>
#include <memory>

//-----------------------------------------------------------------------------------------------
// Data Structures
//-----------------------------------------------------------------------------------------------

struct CompletionParameters
{
    std::string prompt;
    int randomSeed = 42;
    int maxNewTokens = 128;
    int minLength = 8;
    float temperature = 1.0f;
    float topP = 0.5f;
    bool streaming = false;

    bool isValid() const;
};

struct Message
{
    std::string role;
    std::string content;
};

struct ChatCompletionParameters
{
    std::vector<Message> messages;
    int randomSeed = 42;
    int maxNewTokens = 128;
    int minLength = 8;
    float temperature = 1.0f;
    float topP = 0.5f;
    bool streaming = false;

    bool isValid() const;
};

struct CompletionResult
{
    std::vector<int32_t> tokens;
    std::string text;
};

/**
 * @class InferenceEngine
 * @brief A class that provides functionalities for inference operations.
 *
 * The InferenceEngine class is responsible for handling inference tasks such as
 * completion and chat completion. It utilizes a PIMPL (Pointer to Implementation)
 * idiom to hide implementation details.
 */
class InferenceEngine
{
public:
    /**
     * @brief Constructs an InferenceEngine object.
     *
     * @param engineDir The directory where the inference engine is located.
     */
    explicit InferenceEngine(const std::string &engineDir);

    /**
     * @brief Performs a completion operation based on the given parameters.
     *
     * @param params The parameters required for the completion operation.
     * @return CompletionResult The result of the completion operation.
     */
    CompletionResult complete(const CompletionParameters &params);
    /**
     * @brief Performs a chat completion operation based on the given parameters.
     *
     * @param params The parameters required for the chat completion operation.
     * @return CompletionResult The result of the chat completion operation.
     */
    CompletionResult chatComplete(const ChatCompletionParameters &params);

    /**
     * @brief Destructor for the InferenceEngine class.
     */
    ~InferenceEngine();

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;

    /**
     * @brief Checks if a GPU is available for inference operations.
     *
     * @return true if a GPU is available, false otherwise.
     */
    bool isGpuAvailable();
};

#endif // INFERENCE_H