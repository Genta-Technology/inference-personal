#ifndef INFERENCE_H
#define INFERENCE_H

#include <string>
#include <vector>
#include <memory>
#include <future>
#include <mutex>
#include <unordered_map>
#include <atomic>
#include <exception>

//-----------------------------------------------------------------------------------------------
// Data Structures
//-----------------------------------------------------------------------------------------------

/**
 * @brief Parameters for a completion job.
 */
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

/**
 * @brief Parameters for a chat completion job.
 */
struct Message
{
	std::string role;
	std::string content;
};

/**
 * @brief Parameters for a chat completion job.
 */
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

/**
 * @brief Result of a completion job.
 */
struct CompletionResult
{
	std::vector<int32_t> tokens;
	std::string text;
};

/**
 * @brief Interface for an inference engine.
 *
 * This class provides an interface for submitting completion jobs to an inference engine.
 * The engine can be implemented using a CPU or GPU.
 *
 * The engine is responsible for managing the completion jobs and returning the results.
 */
class InferenceEngine
{
public:
	/**
	 * @brief Constructs an InferenceEngine with the specified engine directory.
	 * @param engineDir The directory where the engine is located.
	 */
	explicit InferenceEngine(const std::string& engineDir);

	/**
	 * @brief Submits a completion job and returns the job ID.
	 * @param params The parameters for the completion job.
	 * @return The ID of the submitted job.
	 */
	int submitCompleteJob(const CompletionParameters& params);

	/**
	 * @brief Submits a chat completion job and returns the job ID.
	 * @param params The parameters for the chat completion job.
	 * @return The ID of the submitted job.
	 */
	int submitChatCompleteJob(const ChatCompletionParameters& params);

	/**
	 * @brief Checks if a job is finished.
	 * @param job_id The ID of the job to check.
	 * @return True if the job is finished, false otherwise.
	 */
	bool isJobFinished(int job_id);

	/**
	 * @brief Gets the result of a finished job.
	 * @param job_id The ID of the job to get the result for.
	 * @return The result of the job.
	 * @note This function should only be called if isJobFinished returns true.
	 */
	CompletionResult getJobResult(int job_id);

	/**
	 * @brief Waits for a job to finish.
	 * @param job_id The ID of the job to wait for.
	 */
	void waitForJob(int job_id);

	/**
	 * @brief Destructor for the InferenceEngine.
	 */
	~InferenceEngine();

private:
	struct Impl;
	std::unique_ptr<Impl> pimpl;

	/**
	 * @brief Checks if a GPU is available for inference.
	 * @return True if a GPU is available, false otherwise.
	 */
	bool isGpuAvailable();
};

#endif // INFERENCE_H