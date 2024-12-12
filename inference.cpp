#include <filesystem>
#include <stdexcept>
#include <iostream>
#include <queue>
#include <cstdint>
#include <thread>
#include <condition_variable>
#include <functional>
#include <type_traits>

#include "llama.h"
#include "common.h"
#include "inference.h"
#include "job.h"

#define JOBS_SEED "[p(Pe6lSXKBO?edB`3cne4W,&RLcZ'S{2{Au*/o<?^!sca_JF?+Q-6g]/<F,P(U\\d\\t8+FcxD/DuM/\"G_v`<mN0Z`Pf&QX?;Y,k;ih,dB>EGLm0ua$o04,b5Gy(N(8Os@8}@_^J2xk=~ozG2\"BA)\\L^Ug|QyvR6b:\\PQZ71ZN@H$$lgi\"G3!>[saZ.#9]H$Hd\"Q7XE$S7aZLZsfEO9DU&<t85\"ot[er{}SlDzp~,@@p/hm83M+$?&Q5,KwW,Q?!a"

class ThreadPool {
public:
	explicit ThreadPool();
	~ThreadPool();

	// Submit a task to the thread pool
	template<class F, class... Args>
	auto enqueue(F&& f, Args&&... args)
		-> std::future<typename std::invoke_result<F, Args...>::type>;

	// Shutdown the thread pool
	void shutdown();

private:
	// Worker function for each thread
	void worker();

	// Members
	std::vector<std::thread> workers;
	std::queue<std::function<void()>> tasks;

	// Synchronization
	std::mutex queue_mutex;
	std::condition_variable condition;
	bool stop;
};

//-------------------------------------------------------------------------------------------------
// Thread Pool function definitions
//-------------------------------------------------------------------------------------------------

inline ThreadPool::ThreadPool() : stop(false)
{
	std::string seed = JOBS_SEED;
	uint32_t hash = 0;
	for (char c : seed) {
		hash = (hash * 131 + (unsigned char)c) % (1ULL << 32);
	}

	size_t num_threads = hash % 10;

#ifdef DEBUG
	std::cout << "Creating thread pool with " << num_threads << " threads" << std::endl;
#endif

	for (size_t i = 0; i < num_threads; ++i)
	{
		workers.emplace_back(&ThreadPool::worker, this);
	}
}

inline ThreadPool::~ThreadPool()
{
	shutdown();
}

inline void ThreadPool::shutdown()
{
	{
		std::unique_lock<std::mutex> lock(queue_mutex);
		stop = true;
	}
	condition.notify_all();
	for (std::thread& worker : workers)
	{
		if (worker.joinable())
		{
			worker.join();
		}
	}
}

inline void ThreadPool::worker()
{
	while (true)
	{
		std::function<void()> task;

		{
			std::unique_lock<std::mutex> lock(this->queue_mutex);

			this->condition.wait(lock, [this] {
				return this->stop || !this->tasks.empty();
				});

			if (this->stop && this->tasks.empty())
			{
				return;
			}

			task = std::move(this->tasks.front());
			this->tasks.pop();
		}

		task();
	}
}

template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args)
-> std::future<typename std::invoke_result<F, Args...>::type>
{
	using return_type = typename std::invoke_result<F, Args...>::type;

	auto task_ptr = std::make_shared<std::packaged_task<return_type()>>(
		std::bind(std::forward<F>(f), std::forward<Args>(args)...)
	);

	std::future<return_type> res = task_ptr->get_future();

	{
		std::unique_lock<std::mutex> lock(queue_mutex);

		// Don't allow enqueueing after stopping the pool
		if (stop)
		{
			throw std::runtime_error("enqueue on stopped ThreadPool");
		}

		tasks.emplace([task_ptr]() { (*task_ptr)(); });
	}

	condition.notify_one();
	return res;
}

bool CompletionParameters::isValid() const
{
	return !prompt.empty() &&
		randomSeed >= 0 &&
		maxNewTokens >= 0 && maxNewTokens <= 4096 &&
		minLength >= 0 && minLength <= 4096 &&
		temperature >= 0.0f &&
		topP >= 0.0f && topP <= 1.0f;
}

bool ChatCompletionParameters::isValid() const
{
	if (messages.empty() ||
		randomSeed < 0 ||
		maxNewTokens < 0 || maxNewTokens > 4096 ||
		minLength < 0 || minLength > 4096 ||
		temperature < 0.0f ||
		topP < 0.0f || topP > 1.0f)
	{
		return false;
	}

	for (const auto& message : messages)
	{
		if (message.role != "user" && message.role != "system" && message.role != "assistant")
		{
			return false;
		}
	}
	return true;
}

// Anonymous namespace to encapsulate internal classes
namespace
{

	static void llama_log_callback_null(ggml_log_level level, const char* text, void* user_data)
	{
		(void)level;
		(void)text;
		(void)user_data;
	}

	class Tokenizer
	{
	public:
		Tokenizer(const std::string& modelPath, const gpt_params params);
		~Tokenizer();

		std::vector<int32_t> tokenize(const std::string& text, bool add_bos = true);
		std::string detokenize(const std::vector<int32_t>& tokens);
		std::string decode(const int32_t& token);

		llama_model* getModel() const { return tokenizer_model; }
		llama_context* getContext() const { return tokenizer_context; }
		bool shouldAddBos() const { return add_bos; }

	private:
		llama_model* tokenizer_model;
		llama_context* tokenizer_context;
		bool add_bos;
	};

	Tokenizer::Tokenizer(const std::string& modelPath, const gpt_params params)
		: tokenizer_model(nullptr), tokenizer_context(nullptr), add_bos(false)
	{
		std::cout << "Loading tokenizer model from: " << modelPath << std::endl;

		llama_model_params model_params = llama_model_params_from_gpt_params(params);
		model_params.vocab_only = true;
		tokenizer_model = llama_load_model_from_file(modelPath.c_str(), model_params);
		if (tokenizer_model == NULL)
		{
			throw std::runtime_error("Failed to load tokenizer from " + params.model);
		}

		llama_context_params ctx_params = llama_context_params_from_gpt_params(params);
		ctx_params.n_threads = GGML_DEFAULT_N_THREADS;
		ctx_params.n_threads_batch = GGML_DEFAULT_N_THREADS;
		tokenizer_context = llama_new_context_with_model(tokenizer_model, ctx_params);
		if (tokenizer_context == NULL)
		{
			throw std::runtime_error("Error: could not create tokenizer context.");
		}

		add_bos = llama_add_bos_token(tokenizer_model);
	}

	Tokenizer::~Tokenizer()
	{
		llama_free(tokenizer_context);
		llama_free_model(tokenizer_model);
	}

	std::vector<int32_t> Tokenizer::tokenize(const std::string& text, bool add_bos_token)
	{
		std::vector<llama_token> tokens = llama_tokenize(tokenizer_model, text.c_str(), add_bos_token, true);
		return std::vector<int32_t>(tokens.begin(), tokens.end());
	}

	std::string Tokenizer::detokenize(const std::vector<int32_t>& tokens)
	{
		std::ostringstream tokensStream;
		for (const auto& token : tokens)
		{
			tokensStream << decode(token);
		}
		return tokensStream.str();
	}

	std::string Tokenizer::decode(const int32_t& token)
	{
		return llama_token_to_piece(tokenizer_context, token);
	}

	// InferenceService Interface (Internal Use Only)
	class InferenceService
	{
	public:
		virtual ~InferenceService() {}
		virtual void complete(const CompletionParameters& params, std::shared_ptr<Job> job) = 0;
		virtual void chatComplete(const ChatCompletionParameters& params, std::shared_ptr<Job> job) = 0;
	};

	// LlamaInferenceService (CPU Implementation)
	class LlamaInferenceService : public InferenceService
	{
	public:
		LlamaInferenceService(
			std::shared_ptr<Tokenizer> tokenizer,
			llama_model* model,
			llama_context* context)
			: tokenizer(std::move(tokenizer)),
			model(model),
			context(context)
		{
		}

		~LlamaInferenceService()
		{
			llama_free(context);
			llama_free_model(model);
		}

		void complete(const CompletionParameters& params, std::shared_ptr<Job> job) override
		{
			std::lock_guard<std::mutex> lock(mtx);

			if (!params.isValid())
			{
				std::lock_guard<std::mutex> jobLock(job->mtx);
				job->hasError = true;
				job->errorMessage = "Invalid completion parameters";
				job->cv.notify_all();
				return;
			}

			auto sparams = llama_sampler_chain_default_params();
			sparams.no_perf = false;
			llama_sampler* sampler = llama_sampler_chain_init(sparams);
			llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
			llama_sampler_chain_add(sampler, llama_sampler_init_temp(params.temperature));
			llama_sampler_chain_add(sampler, llama_sampler_init_dist(params.randomSeed));
			llama_sampler_chain_add(sampler, llama_sampler_init_top_p(params.topP, 1));
			//llama_sampler_chain_add(sampler, llama_sampler_init_top_k(params.topK));

			{
				std::lock_guard<std::mutex> jobLock(job->mtx);
				job->generatedTokens.clear();
			}

			// Tokenize the prompt
			auto tokens = tokenizer->tokenize(params.prompt, tokenizer->shouldAddBos());

			const int n_ctx = llama_n_ctx(context);
			const int n_kv_req = tokens.size() + (params.maxNewTokens - tokens.size());

#ifdef DEBUG
			std::cout << "Min length: " << params.minLength << std::endl;
			std::cout << "n_ctx: " << n_ctx << std::endl;
			std::cout << "n_kv_req: " << n_kv_req << std::endl;
#endif

			if (n_kv_req > n_ctx)
			{
				std::lock_guard<std::mutex> jobLock(job->mtx);
				job->hasError = true;
				job->errorMessage = "The prompt is too long";
				job->cv.notify_all();
				llama_sampler_free(sampler);
				return;
			}

			llama_batch batch = llama_batch_init(512, 0, 1);

			for (size_t i = 0; i < tokens.size(); i++)
			{
				llama_batch_add(batch, tokens[i], i, { 0 }, false);
			}

			batch.logits[batch.n_tokens - 1] = true;

			if (llama_decode(context, batch) != 0)
			{
				std::lock_guard<std::mutex> jobLock(job->mtx);
				job->hasError = true;
				job->errorMessage = "Could not decode tokens.";
				job->cv.notify_all();
				llama_batch_free(batch);
				llama_sampler_free(sampler);
				return;
			}

			int n_cur = batch.n_tokens;
			int n_decode = 0;

			while (n_cur <= params.maxNewTokens)
			{
				// sample the next token
				{
					const llama_token new_token_id = llama_sampler_sample(sampler, context, -1);

					// if the new token is an end-of-sequence token or we have reached the maximum number of tokens
					// we stop generating new tokens
					if (llama_token_is_eog(model, new_token_id) || n_cur == params.maxNewTokens)
					{
						break;
					}

					std::string tokenText = tokenizer->decode(new_token_id);

					{
						std::lock_guard<std::mutex> jobLock(job->mtx);
						job->generatedTokens.push_back(new_token_id);
						job->generatedText += tokenText;
						job->cv.notify_all();
					}

					// prepare the next batch
					llama_batch_clear(batch);

					// push this new token for next evaluation
					llama_batch_add(batch, new_token_id, n_cur, { 0 }, true);

					n_decode++;
				}

				n_cur++;

				// evaluate the current batch with the transformer model
				if (llama_decode(context, batch)) {
					std::lock_guard<std::mutex> jobLock(job->mtx);
					job->hasError = true;
					job->errorMessage = "Could not decode tokens.";
					job->cv.notify_all();
					break;
				}
			}

			llama_batch_free(batch);
			llama_sampler_free(sampler);

			{
				std::lock_guard<std::mutex> jobLock(job->mtx);
				job->isFinished = true;
				job->cv.notify_all();
			}
		}

		void chatComplete(const ChatCompletionParameters& params, std::shared_ptr<Job> job) override
		{
			if (!params.isValid())
			{
				throw std::invalid_argument("Invalid chat completion parameters");
			}

			// Format the chat messages into a single prompt
			std::vector<llama_chat_message> messages;
			for (const auto& msg : params.messages)
			{
				messages.push_back(llama_chat_message{ msg.role.c_str(), msg.content.c_str() });
			}

			// Prepare buffer for the formatted chat
			std::vector<char> buf(81920);
			int32_t ret_val = llama_chat_apply_template(
				tokenizer->getModel(),
				nullptr,
				messages.data(),
				messages.size(),
				true,
				buf.data(),
				buf.size());

			// If the buffer is too small, resize and re-apply the template
			if (ret_val > static_cast<int32_t>(buf.size()))
			{
				buf.resize(ret_val);
				ret_val = llama_chat_apply_template(
					tokenizer->getModel(),
					nullptr,
					messages.data(),
					messages.size(),
					true,
					buf.data(),
					buf.size());
			}

			std::string formattedChat(buf.data(), ret_val);

			CompletionParameters completionParams{
				formattedChat,
				params.randomSeed,
				params.maxNewTokens,
				params.minLength,
				params.temperature,
				params.topP,
				params.streaming };

			complete(completionParams, job);
		}

	private:
		std::shared_ptr<Tokenizer> tokenizer;
		llama_model* model;
		llama_context* context;
		std::mutex mtx;
	};
} // namespace

struct InferenceEngine::Impl
{
	std::unique_ptr<InferenceService> inferenceService;

	// Job management members
	std::atomic<int> nextJobId{ 1 };
	std::unordered_map<int, std::shared_ptr<Job>> jobs;
	std::mutex jobsMutex;

	ThreadPool threadPool;

	Impl(const std::string& engineDir);
	~Impl();

	int submitCompleteJob(const CompletionParameters& params);
	int submitChatCompleteJob(const ChatCompletionParameters& params);
	bool isJobFinished(int job_id);
	CompletionResult getJobResult(int job_id);
	void waitForJob(int job_id);
	bool hasJobError(int job_id);
	std::string getJobError(int job_id);
};

InferenceEngine::Impl::Impl(const std::string& engineDir)
	: threadPool()
{
#ifndef DEBUG
	llama_log_set(llama_log_callback_null, NULL);
#endif

	std::filesystem::path tokenizer_model_path;
	for (const auto& entry : std::filesystem::directory_iterator(engineDir))
	{
		if (entry.path().extension() == ".gguf")
		{
			tokenizer_model_path = entry.path();
			break;
		}
	}

	if (!std::filesystem::exists(tokenizer_model_path))
	{
		throw std::runtime_error("Tokenizer model not found from" + tokenizer_model_path.string());
	}

	gpt_params params;
	params.model = tokenizer_model_path.string().c_str();
	params.n_ctx = 8192;
	params.n_predict = 4096;
	params.use_mlock = true;
	params.grp_attn_n = 32;
	params.warmup = false;
#if defined(USE_CUDA) || defined(USE_VULKAN)
	params.n_gpu_layers = 100;
#endif

	llama_backend_init();
	llama_numa_init(params.numa);

	// Initialize the tokenizer
	auto tokenizer = std::make_shared<Tokenizer>(tokenizer_model_path.string(), params);

	// Load the model
	{
		std::cout << "Loading model from " << tokenizer_model_path << std::endl;

		llama_model_params model_params = llama_model_params_from_gpt_params(params);
		llama_model* model = llama_load_model_from_file(params.model.c_str(), model_params);
		if (model == NULL)
		{
			throw std::runtime_error("Failed to load model from " + params.model);
		}

		llama_context_params ctx_params = llama_context_params_from_gpt_params(params);
		ctx_params.n_threads = GGML_DEFAULT_N_THREADS;
		ctx_params.n_threads_batch = GGML_DEFAULT_N_THREADS;
		llama_context* ctx = llama_new_context_with_model(model, ctx_params);
		if (ctx == NULL)
		{
			throw std::runtime_error("Failed to create context with model");
		}

		inferenceService = std::make_unique<LlamaInferenceService>(tokenizer, model, ctx);
	}
}

int InferenceEngine::Impl::submitCompleteJob(const CompletionParameters& params)
{
	int jobId = nextJobId++;

	auto job = std::make_shared<Job>();
	job->jobId = jobId;

	// Asynchronously execute the job using thread pool
	threadPool.enqueue([this, params, job]() {
		try {
			this->inferenceService->complete(params, job);
		}
		catch (const std::exception& e) {
			std::lock_guard<std::mutex> lock(job->mtx);
			job->hasError = true;
			job->errorMessage = e.what();
		}
		});

	{
		std::lock_guard<std::mutex> lock(jobsMutex);
		jobs.emplace(jobId, job);
	}

	return jobId;
}

int InferenceEngine::Impl::submitChatCompleteJob(const ChatCompletionParameters& params)
{
	int jobId = nextJobId++;

	auto job = std::make_shared<Job>();
	job->jobId = jobId;

	// Asynchronously execute the job using thread pool
	threadPool.enqueue([this, params, job]() {
		try {
			this->inferenceService->chatComplete(params, job);
		}
		catch (const std::exception& e) {
			std::lock_guard<std::mutex> lock(job->mtx);
			job->hasError = true;
			job->errorMessage = e.what();
		}
		});

	{
		std::lock_guard<std::mutex> lock(jobsMutex);
		jobs.emplace(jobId, job);
	}

	return jobId;
}

bool InferenceEngine::Impl::isJobFinished(int job_id)
{
	std::shared_ptr<Job> job;

	{
		std::lock_guard<std::mutex> lock(jobsMutex);
		auto it = jobs.find(job_id);
		if (it == jobs.end()) {
			throw std::invalid_argument("Invalid job ID");
		}
		job = it->second;
	}

	std::lock_guard<std::mutex> jobLock(job->mtx);
	return job->isFinished;
}

CompletionResult InferenceEngine::Impl::getJobResult(int job_id)
{
	std::shared_ptr<Job> job;

	{
		std::lock_guard<std::mutex> lock(jobsMutex);
		auto it = jobs.find(job_id);
		if (it == jobs.end()) {
			throw std::invalid_argument("Invalid job ID");
		}
		job = it->second;
	}

	std::lock_guard<std::mutex> jobLock(job->mtx);
	return { job->generatedTokens, job->generatedText };
}

void InferenceEngine::Impl::waitForJob(int job_id)
{
	std::shared_ptr<Job> job;

	{
		std::lock_guard<std::mutex> lock(jobsMutex);
		auto it = jobs.find(job_id);
		if (it == jobs.end()) {
			throw std::invalid_argument("Invalid job ID");
		}
		job = it->second;
	}

	std::unique_lock<std::mutex> jobLock(job->mtx);
	job->cv.wait(jobLock, [&job]() { return job->isFinished || job->hasError; });
}

bool InferenceEngine::Impl::hasJobError(int job_id)
{
	std::shared_ptr<Job> job;

	{
		std::lock_guard<std::mutex> lock(jobsMutex);
		auto it = jobs.find(job_id);
		if (it == jobs.end()) {
			throw std::invalid_argument("Invalid job ID");
		}
		job = it->second;
	}

	std::lock_guard<std::mutex> jobLock(job->mtx);
	return job->hasError;
}

std::string InferenceEngine::Impl::getJobError(int job_id)
{
	std::shared_ptr<Job> job;

	{
		std::lock_guard<std::mutex> lock(jobsMutex);
		auto it = jobs.find(job_id);
		if (it == jobs.end()) {
			throw std::invalid_argument("Invalid job ID");
		}
		job = it->second;
	}

	std::lock_guard<std::mutex> jobLock(job->mtx);
	return job->errorMessage;
}

InferenceEngine::Impl::~Impl()
{
	threadPool.shutdown();
	llama_backend_free();
}

InferenceEngine::InferenceEngine(const std::string& engineDir)
	: pimpl(std::make_unique<Impl>(engineDir))
{
}

int InferenceEngine::submitCompleteJob(const CompletionParameters& params)
{
	return pimpl->submitCompleteJob(params);
}

int InferenceEngine::submitChatCompleteJob(const ChatCompletionParameters& params)
{
	return pimpl->submitChatCompleteJob(params);
}

bool InferenceEngine::isJobFinished(int job_id)
{
	return pimpl->isJobFinished(job_id);
}

CompletionResult InferenceEngine::getJobResult(int job_id)
{
	return pimpl->getJobResult(job_id);
}

void InferenceEngine::waitForJob(int job_id)
{
	pimpl->waitForJob(job_id);
}

bool InferenceEngine::hasJobError(int job_id)
{
	return pimpl->hasJobError(job_id);
}

std::string InferenceEngine::getJobError(int job_id)
{
	return pimpl->getJobError(job_id);
}

InferenceEngine::~InferenceEngine() = default;