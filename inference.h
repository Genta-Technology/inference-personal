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

#ifdef INFERENCE_EXPORTS
#define INFERENCE_API __declspec(dllexport)
#else
#define INFERENCE_API __declspec(dllimport)
#endif

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
class INFERENCE_API InferenceEngine
{
public:
	static InferenceEngine& getInstance(const std::string& engineDir)
	{
		static InferenceEngine instance(engineDir);
		return instance;
	}

	// Delete copy constructor and assignment operator
	InferenceEngine(const InferenceEngine&) = delete;
	InferenceEngine& operator=(const InferenceEngine&) = delete;
	InferenceEngine(InferenceEngine&&) = delete;
	InferenceEngine& operator=(InferenceEngine&&) = delete;

	/**
	 * @brief Submits a completion job and returns the job ID.
	 * @param params The parameters for the completion job.
	 * @return The ID of the submitted job.
	 */
	int submitCompletionsJob(const CompletionParameters& params);

	/**
	 * @brief Submits a chat completion job and returns the job ID.
	 * @param params The parameters for the chat completion job.
	 * @return The ID of the submitted job.
	 */
	int submitChatCompletionsJob(const ChatCompletionParameters& params);

	/**
	 * @brief Checks if a job is finished.
	 * @param job_id The ID of the job to check.
	 * @return True if the job is finished, false otherwise.
	 */
	bool isJobFinished(int job_id);

	/**
	 * @brief Gets the current result of a job.
	 * @param job_id The ID of the job to get the result for.
	 * @return The result of the job.
	 * @note This function would return any results that are currently available, even if the job is not finished.
	 */
	CompletionResult getJobResult(int job_id);

	/**
	 * @brief Waits for a job to finish.
	 * @param job_id The ID of the job to wait for.
	 */
	void waitForJob(int job_id);

	/**
	 * @brief Checks if a job has an error.
	 * @param job_id The ID of the job to check.
	 * @return True if the job has an error, false otherwise.
	 */
	bool hasJobError(int job_id);

	/**
	 * @brief Gets the error message for a job.
	 * @param job_id The ID of the job to get the error message for.
	 * @return The error message for the job.
	 */
	std::string getJobError(int job_id);

	/**
	 * @brief Destructor for the InferenceEngine.
	 */
	~InferenceEngine();

	/**
	 * @brief Resets the singleton instance with a new engine directory.
	 * @param engineDir The new directory where the engine is located.
	 */
	static void resetInstance(const std::string& engineDir)
	{
		std::lock_guard<std::mutex> lock(instanceMutex);
		instance.reset(new InferenceEngine(engineDir));
	}

private:
	/**
	 * @brief Constructs an InferenceEngine with the specified engine directory.
	 * @param engineDir The directory where the engine is located.
	 */
	explicit InferenceEngine(const std::string& engineDir);

	struct Impl;
	std::unique_ptr<Impl> pimpl;

	static std::unique_ptr<InferenceEngine> instance;
	static std::mutex instanceMutex;
};

extern "C" INFERENCE_API InferenceEngine* getInferenceEngine(const char* engineDir);

#endif // INFERENCE_H