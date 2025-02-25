#include <filesystem>
#include <stdexcept>
#include <iostream>
#include <queue>
#include <cstdint>
#include <thread>
#include <condition_variable>
#include <functional>
#include <type_traits>
#ifdef USE_VULKAN
#include <vulkan/vulkan.h>
#endif

#include "llama.h"
#include "common.h"
#include "sampling.h"
#include "inference.h"
#include "job.h"

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

	// Return the number of active threads in the pool
	size_t size() const { return workers.size(); }

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
	size_t num_threads = 1;

#ifdef DEBUG
	std::cout << "[INFERENCE] Creating thread pool with " << num_threads << " threads" << std::endl;
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
#ifdef DEBUG
	std::cout << "[INFERENCE] Enqueueing task to thread pool" << std::endl;
#endif

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
			std::cerr << "[INFERENCE] [ERROR] Enqueue on stopped ThreadPool" << std::endl;
			return res;
		}

		tasks.emplace([task_ptr]() { (*task_ptr)(); });
	}

	condition.notify_one();
	return res;
}

bool CompletionParameters::isValid() const
{
	if (sizeof(prompt) <= 0)
	{
		std::cerr << "[INFERENCE] [ERROR] prompt is empty: " << prompt << std::endl;
		return false;
	}

	if (randomSeed < 0)
	{
		std::cerr << "[INFERENCE] [ERROR] randomSeed is negative: " << randomSeed << std::endl;
		return false;
	}

	if (maxNewTokens < 0 || maxNewTokens > 4096)
	{
		std::cerr << "[INFERENCE] [ERROR] maxNewTokens is out of range: " << maxNewTokens << std::endl;
		return false;
	}

	if (minLength < 0 || minLength > 4096)
	{
		std::cerr << "[INFERENCE] [ERROR] minLength is out of range: " << minLength << std::endl;
		return false;
	}

	if (temperature < 0.0f)
	{
		std::cerr << "[INFERENCE] [ERROR] temperature is negative: " << temperature << std::endl;
		return false;
	}

	if (topP < 0.0f || topP > 1.0f)
	{
		std::cerr << "[INFERENCE] [ERROR] topP is out of range: " << topP << std::endl;
		return false;
	}

	return true;
}

bool ChatCompletionParameters::isValid() const
{
	if (messages.empty())
	{
		std::cerr << "[INFERENCE] [ERROR] messages is empty; size: " << messages.size() << std::endl;
		return false;
	}

	if (randomSeed < 0)
	{
		std::cerr << "[INFERENCE] [ERROR] randomSeed is negative: " << randomSeed << std::endl;
		return false;
	}

	if (maxNewTokens < 0 || maxNewTokens > 4096)
	{
		std::cerr << "[INFERENCE] [ERROR] maxNewTokens is out of range: " << maxNewTokens << std::endl;
		return false;
	}

	if (minLength < 0 || minLength > 4096)
	{
		std::cerr << "[INFERENCE] [ERROR] minLength is out of range: " << minLength << std::endl;
		return false;
	}

	if (temperature < 0.0f)
	{
		std::cerr << "[INFERENCE] [ERROR] temperature is negative: " << temperature << std::endl;
		return false;
	}

	if (topP < 0.0f || topP > 1.0f)
	{
		std::cerr << "[INFERENCE] [ERROR] topP is out of range: " << topP << std::endl;
		return false;
	}

	for (const auto& message : messages)
	{
		if (message.role != "user" && message.role != "system" && message.role != "assistant")
		{
			std::cerr << "[INFERENCE] [ERROR] Invalid role in message: " << message.role << std::endl;
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
		Tokenizer(const std::string& modelPath, common_params& params);
		~Tokenizer();

		std::vector<int32_t>	tokenize(const std::string& text, bool add_bos = true);
		std::string				detokenize(const std::vector<int32_t>& tokens);
		std::string				decode(const int32_t& token);
		std::string				applyTemplate(std::vector<common_chat_msg>& messages);

		const	llama_vocab		*getVocab()		const { return vocab; }
				llama_model		*getModel()		const { return tokenizer_model; }
				llama_context	*getContext()	const { return tokenizer_context; }
				bool			shouldAddBos()	const { return add_bos; }

	private:
		const	llama_vocab		*vocab;
				llama_model		*tokenizer_model;
				llama_context	*tokenizer_context;

		bool add_bos;
	};

	Tokenizer::Tokenizer(const std::string& modelPath, common_params& params)
		: tokenizer_model(nullptr), tokenizer_context(nullptr), add_bos(false)
	{
#ifdef DEBUG
		std::cout << "[INFERENCE] Loading tokenizer model from: " << modelPath << std::endl;
#endif
		llama_model_params model_params = common_model_params_to_llama(params);
		model_params.vocab_only			= true;
		tokenizer_model					= llama_load_model_from_file(modelPath.c_str(), model_params);
		if (tokenizer_model == NULL)
		{
			throw std::runtime_error("[INFERENCE] [ERROR] Could not load tokenizer model from " + modelPath);
		}

		llama_context_params ctx_params = common_context_params_to_llama(params);
		ctx_params.n_threads			= GGML_DEFAULT_N_THREADS;
		ctx_params.n_threads_batch		= GGML_DEFAULT_N_THREADS;
		tokenizer_context				= llama_init_from_model(tokenizer_model, ctx_params);
		if (tokenizer_context == NULL)
		{
			throw std::runtime_error("[INFERENCE] [ERROR] Could not create context with tokenizer model");
		}

		vocab			= llama_model_get_vocab(tokenizer_model);
		add_bos			= llama_add_bos_token(vocab);
	}

	Tokenizer::~Tokenizer()
	{
		llama_free(tokenizer_context);
		llama_free_model(tokenizer_model);
	}

	std::vector<int32_t> Tokenizer::tokenize(const std::string& text, bool add_bos_token)
	{
		std::vector<llama_token> tokens = common_tokenize(tokenizer_context, text.c_str(), true, true);
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
		return common_token_to_piece(tokenizer_context, token);
	}

	std::string Tokenizer::applyTemplate(std::vector<common_chat_msg>& messages)
	{
		return common_chat_apply_template(tokenizer_model, "", messages, true);
	}

	// InferenceService Interface (Internal Use Only)
	class InferenceService
	{
	public:
		virtual ~InferenceService() {}
		virtual void start() = 0;
		virtual void stop() = 0;
		virtual void submitJob(const CompletionParameters& params, std::shared_ptr<Job> job) = 0;
		virtual void complete(const CompletionParameters& params, std::shared_ptr<Job> job) = 0;
		virtual CompletionParameters formatChat(const ChatCompletionParameters& params) = 0;
	};

	// LlamaInferenceService (CPU Implementation)
	class LlamaInferenceService : public InferenceService
	{
	public:
		LlamaInferenceService(std::shared_ptr<Tokenizer> tokenizer, llama_model* model, llama_context* context, 
			common_params params, ggml_threadpool* threadpool)
			: tokenizer(std::move(tokenizer)), model(model), context(context), g_params(params), threadpool(threadpool),
			  n_batch(params.n_batch), n_keep(params.n_keep), n_ctx(llama_n_ctx(context))
		{
#ifdef DEBUG
			std::cout << "Initializing batch with size of: " << g_params.n_batch << std::endl;
#endif

			batch = llama_batch_init(params.n_ctx, 0, g_params.n_batch);

			std::thread inferenceThread(&LlamaInferenceService::start, this);
			inferenceThread.detach();
		}

		~LlamaInferenceService()
		{
			stop();

			llama_free(context);
			llama_free_model(model);

			ggml_threadpool_free(threadpool);
		}

		void stop() override
		{
			should_terminate = true;
		}

		void start() override
		{
			while (!should_terminate)
			{
				std::vector<std::shared_ptr<Job>> current_jobs;
				{
					std::unique_lock<std::mutex> lock(mtx);
					if (jobs.empty()) {
						cv.wait(lock, [this] { return !jobs.empty() || should_terminate; });
					}
					if (should_terminate) break;
					current_jobs = jobs; // Copy jobs to process without holding the lock
				}

				bool batch_has_tokens = false;

				for (auto job : current_jobs)
				{
					std::lock_guard<std::mutex> jobLock(job->mtx);

					if (job->isFinished)
						continue;

					if (checkCancellation(job) || (job->n_remain <= 0 && job->params.maxNewTokens != 0)) {
						saveSession(job);
						common_sampler_free(job->smpl);
						llama_kv_cache_seq_rm(context, 0, 0, -1);
						job->isFinished = true;
						job->cv.notify_all();
						continue;
					}

					if (!job->isDecodingPrompt) {
						if (!ensureContextCapacity(job))
							continue;

						if (!sampleNextToken(job)) {
							saveSession(job);
							common_sampler_free(job->smpl);
							llama_kv_cache_seq_rm(context, 0, 0, -1);
							job->isFinished = true;
							job->cv.notify_all();
							continue;
						}

						job->n_past += 1;
						batch_has_tokens = true;
					}
					else {
						while (job->i_prompt < job->embd_inp.size()) {
							common_batch_add(batch, job->embd_inp[job->i_prompt], job->i_prompt, { 0 }, true);
							common_sampler_accept(job->smpl, job->embd_inp[job->i_prompt], false);
							++(job->i_prompt);
							batch_has_tokens = true;
						}

						job->n_past += job->n_prompt;
						job->isDecodingPrompt = false;
					}
				}

				if (batch_has_tokens) {
					if (llama_decode(context, batch)) {
						for (auto job : current_jobs)
						{
							std::lock_guard<std::mutex> jobLock(job->mtx);
							if (!job->isFinished) {
								job->hasError = true;
								job->errorMessage = "Could not decode next token";
								job->isFinished = true;
								job->cv.notify_all();
							}
						}
					}

					common_batch_clear(batch);
				}
			}
		}

		void submitJob(const CompletionParameters& params, std::shared_ptr<Job> job) override
		{
			if (!validateParameters(params, job)) {
				return;
			}

			job->params = params;

			job->smpl = initializeSampler(params, job);
			if (!job->smpl) {
				return;
			}

			{
				std::lock_guard<std::mutex> jobLock(job->mtx);
				job->generatedTokens.clear();
				job->generatedText.clear();
			}

			job->path_session = params.kvCacheFilePath;
			if (!loadSession(job)) {
				common_sampler_free(job->smpl);
				return;
			}

			job->embd_inp = getInputTokens(params, job->session_tokens);
			if (!ensureNonEmptyInput(job)) {
				common_sampler_free(job->smpl);
				return;
			}

			job->n_matching_session_tokens	= matchSessionTokens(job->session_tokens, job->embd_inp);
			job->n_past						= static_cast<int>(job->n_matching_session_tokens);
			job->n_remain					= params.maxNewTokens;
			job->i_prompt					= static_cast<int>(job->n_matching_session_tokens);
			job->isDecodingPrompt			= true;
			job->isFinished					= false;
			job->n_prompt					= job->embd_inp.size();

			{
				std::lock_guard<std::mutex> lock(mtx);
				jobs.push_back(job);
			}

			cv.notify_one();
		}

		void complete(const CompletionParameters& params, std::shared_ptr<Job> job) override
		{
			submitJob(params, job);

			{
				std::unique_lock<std::mutex> lock(job->mtx);
				job->cv.wait(lock, [&job] { return job->isFinished; });
			}
		}

		CompletionParameters formatChat(const ChatCompletionParameters& params) override
		{
			if (!params.isValid())
			{
				throw std::runtime_error("[INFERENCE] [CHATCOMPLETE] [ERROR] Invalid chat completion parameters\n");
			}

			// Format the chat messages into a single prompt
			std::vector<common_chat_msg> messages;
			for (const auto& msg : params.messages)
			{
				messages.push_back(common_chat_msg{ msg.role, msg.content });
			}

			std::string formatted = tokenizer->applyTemplate(messages);
			CompletionParameters completionParams{
				formatted.c_str(),
				params.randomSeed,
				params.maxNewTokens,
				params.minLength,
				params.temperature,
				params.topP,
				params.streaming,
				params.kvCacheFilePath
			};

			return completionParams;
		}

	private:
		std::shared_ptr<Tokenizer>			tokenizer;
		llama_model*						model;
		llama_context*						context;
		std::mutex							mtx;
		std::condition_variable				cv;
		common_params						g_params;
		ggml_threadpool*					threadpool;
		llama_batch							batch;
		std::vector<std::shared_ptr<Job>>	jobs;
		std::atomic<bool>					should_terminate{ false };
		std::atomic<int>					next_job_id{ 0 };

		const int n_batch;
		const int n_keep;
		const int n_ctx;

		int getNextJobId() {
			return next_job_id++;
		}

		bool validateParameters(const CompletionParameters& params, std::shared_ptr<Job> job) {
			if (!params.isValid()) {
				std::lock_guard<std::mutex> jobLock(job->mtx);
				job->hasError = true;
				job->errorMessage = "Invalid completion parameters";
				job->cv.notify_all();
				return false;
			}
			return true;
		}

		common_sampler* initializeSampler(const CompletionParameters& params, std::shared_ptr<Job> job) {
			auto sparams = g_params.sampling;
			sparams.top_p = params.topP;
			sparams.temp = params.temperature;
			sparams.seed = params.randomSeed;
			// sparams.top_k = params.topK;

#ifdef DEBUG
			sparams.no_perf = false;
#else
			sparams.no_perf = true;
#endif

			common_sampler* sampler = common_sampler_init(model, sparams);
			if (!sampler) {
				std::lock_guard<std::mutex> jobLock(job->mtx);
				job->hasError = true;
				job->errorMessage = "Could not initialize sampler";
				job->cv.notify_all();
				return nullptr;
			}
			return sampler;
		}

		bool loadSession(const std::string& path_session, std::vector<llama_token>& session_tokens, std::shared_ptr<Job> job) {
			if (!path_session.empty()) {
				if (!load_kv_cache(path_session, session_tokens)) {
					std::lock_guard<std::mutex> jobLock(job->mtx);
					job->hasError = true;
					job->errorMessage = "Could not load KV cache from " + path_session;
					job->cv.notify_all();
					return false;
				}
			}
			return true;
		}

		bool loadSession(std::shared_ptr<Job> job) {
			if (!job->path_session.empty()) {
				if (!load_kv_cache(job->path_session, job->session_tokens)) {
					std::lock_guard<std::mutex> jobLock(job->mtx);
					job->hasError = true;
					job->errorMessage = "Could not load KV cache from " + job->path_session;
					job->cv.notify_all();
					return false;
				}
			}
			return true;
		}

		std::vector<llama_token> getInputTokens(const CompletionParameters& params, const std::vector<llama_token>& session_tokens) {
			if (session_tokens.empty() || !params.prompt.empty()) {
				return tokenizer->tokenize(params.prompt, tokenizer->shouldAddBos());
			}
			else {
				return session_tokens;
			}
		}

		bool ensureNonEmptyInput(std::vector<llama_token>& embd_inp, std::shared_ptr<Job> job) {
			if (embd_inp.empty()) {
				if (tokenizer->shouldAddBos()) {
					embd_inp.push_back(llama_token_bos(tokenizer->getVocab()));
				}
				else {
					std::lock_guard<std::mutex> jobLock(job->mtx);
					job->hasError = true;
					job->errorMessage = "Empty prompt and no BOS token available.";
					job->cv.notify_all();
					return false;
				}
			}
			return true;
		}

		bool ensureNonEmptyInput(std::shared_ptr<Job> job) {
			if (job->embd_inp.empty()) {
				if (tokenizer->shouldAddBos()) {
					job->embd_inp.push_back(llama_token_bos(tokenizer->getVocab()));
				}
				else {
					std::lock_guard<std::mutex> jobLock(job->mtx);
					job->hasError = true;
					job->errorMessage = "Empty prompt and no BOS token available.";
					job->cv.notify_all();
					return false;
				}
			}
			return true;
		}

		bool checkCancellation(std::shared_ptr<Job> job) {
			if (job->cancelRequested.load()) {
				std::lock_guard<std::mutex> jobLock(job->mtx);
				job->errorMessage = "Generation cancelled by user.";
				job->isFinished = true;
				job->cv.notify_all();
				return true;
			}
			return false;
		}

		bool ensureContextCapacity(std::shared_ptr<Job> job) {
			if (job->n_past + 1 > n_ctx) {
				kv_cache_seq_ltrim(context, n_keep, job->session_tokens, job->n_past, 0);
				if (job->n_past + 1 > n_ctx) {
					std::lock_guard<std::mutex> jobLock(job->mtx);
					job->hasError = true;
					job->errorMessage = "Context overflow even after trimming.";
					job->cv.notify_all();
					return false;
				}
			}
			return true;
		}

		bool ensureContextCapacity(int n_past, int n_tokens_to_add, int n_ctx, int n_keep, std::vector<llama_token>& session_tokens, std::shared_ptr<Job> job) {
			if (n_past + n_tokens_to_add > n_ctx) {
				kv_cache_seq_ltrim(context, n_keep, session_tokens, n_past);
				if (n_past + n_tokens_to_add > n_ctx) {
					std::lock_guard<std::mutex> jobLock(job->mtx);
					job->hasError = true;
					job->errorMessage = "Context overflow even after trimming.";
					job->cv.notify_all();
					return false;
				}
			}
			return true;
		}

		bool processPromptTokens(const std::vector<llama_token>& embd_inp, int& i_prompt, int& n_past, std::vector<llama_token>& session_tokens, common_sampler* sampler, std::shared_ptr<Job> job, const std::string& path_session, const int id = 0) 
		{
			while (i_prompt < embd_inp.size()) {
				common_batch_add(batch, embd_inp[i_prompt], i_prompt, { id }, true);
				common_sampler_accept(sampler, embd_inp[i_prompt], false);
				++i_prompt;
			}

			return true;
		}

		bool processPromptTokens(std::shared_ptr<Job> job)
		{
			while (job->i_prompt < job->embd_inp.size()) {
				common_batch_add(batch, job->embd_inp[job->i_prompt], job->i_prompt, { 0 }, true);
				common_sampler_accept(job->smpl, job->embd_inp[job->i_prompt], false);
				++(job->i_prompt);
			}

			return true;
		}

		bool decodeBatch(common_sampler* sampler, int& n_past, int& n_remain, std::vector<llama_token>& session_tokens, std::shared_ptr<Job> job, int n_ctx, int n_keep, const std::string& path_session) {
			if (!ensureContextCapacity(n_past, 1, n_ctx, n_keep, session_tokens, job)) {
				return false;
			}

			if (n_past > batch.n_tokens) {
				if (!sampleNextToken(sampler, n_past, n_remain, session_tokens, job, path_session)) {
					return false;
				}
			}

			if (llama_decode(context, batch)) {
				std::lock_guard<std::mutex> jobLock(job->mtx);
				job->hasError = true;
				job->errorMessage = "Could not decode next token";
				job->cv.notify_all();
				return false;
			}

			n_past += batch.n_tokens;

			common_batch_clear(batch);
			return true;
		}

		bool sampleNextToken(common_sampler* sampler, int& n_past, int& n_remain, std::vector<llama_token>& session_tokens, std::shared_ptr<Job> job, const std::string& path_session) {
			llama_token id = common_sampler_sample(sampler, context, 0);
			common_sampler_accept(sampler, id, true);
			common_batch_add(batch, id, n_past, { 0 }, true);

			if (llama_token_is_eog(tokenizer->getVocab(), id) || id == llama_token_eos(tokenizer->getVocab())) {
				return false; // Stop generation
			}

			const auto data = llama_perf_context(context);
			const std::string token_str = tokenizer->decode(id);
			{
				std::lock_guard<std::mutex> jobLock(job->mtx);
				job->generatedTokens.push_back(id);
				job->generatedText += token_str;
				job->tps = 1e3 / data.t_eval_ms * data.n_eval;
				job->cv.notify_all();
			}

			if (!path_session.empty()) {
				session_tokens.push_back(id);
			}

			n_remain -= 1;
		}

		bool sampleNextToken(std::shared_ptr<Job> job) {
			llama_token id = common_sampler_sample(job->smpl, context, 0);
			common_sampler_accept(job->smpl, id, false);
			common_batch_add(batch, id, job->n_past, { 0 }, true);

			if (llama_token_is_eog(tokenizer->getVocab(), id) || id == llama_token_eos(tokenizer->getVocab())) {
				return false; // Stop generation
			}

			const auto data = llama_perf_context(context);
			const std::string token_str = tokenizer->decode(id);
			{
				job->generatedTokens.push_back(id);
				job->generatedText += token_str;
				job->tps = 1e3 / data.t_eval_ms * data.n_eval;
				job->cv.notify_all();
			}

			if (!job->path_session.empty()) {
				job->session_tokens.push_back(id);
			}

			job->n_remain -= 1;

			return true;
		}

		void saveSession(const std::string& path_session, const std::vector<llama_token>& session_tokens) {
			if (!path_session.empty()) {
				llama_state_save_file(context, path_session.c_str(), session_tokens.data(), session_tokens.size());
			}
		}

		void saveSession(std::shared_ptr<Job> job) {
			if (!job->path_session.empty()) {
				llama_state_save_file(context, job->path_session.c_str(), job->session_tokens.data(), job->session_tokens.size());
			}
		}

		bool load_kv_cache(const std::string& path, std::vector<llama_token>& session_tokens)
		{
			if (!path.empty()) 
			{
				// Attempt to load
				if (!std::filesystem::exists(path)) 
				{
					// file doesn't exist => no old cache
					printf("[KV] session file does not exist, will create.\n");
				}
				else if (std::filesystem::is_empty(path)) 
				{
					// file is empty => treat as brand-new
					printf("[KV] session file is empty, new session.\n");
				}
				else 
				{
					// The file exists and is not empty
					session_tokens.resize(g_params.n_ctx);  // allocate buffer
					size_t n_token_count_out = 0;

					if (!llama_state_load_file(
						context,
						path.c_str(),
						session_tokens.data(),
						session_tokens.capacity(),
						&n_token_count_out
					)) 
					{
						return false;
					}

					// The llama_state_load_file call gives us how many tokens were in that old session
					session_tokens.resize(n_token_count_out);

#ifdef DEBUG
					printf("[INFERENCE] [KV] loaded session with prompt size: %d tokens\n", (int)session_tokens.size());
					printf("[INFERENCE] [KV] tokens decoded: %s", tokenizer->detokenize(session_tokens).c_str());
#endif
				}
			}
		}

		size_t matchSessionTokens(std::vector<llama_token>& session_tokens, const std::vector<llama_token>& embd_inp, const int id = -1)
		{
			size_t n_matching_session_tokens = 0;

			if (!session_tokens.empty()) {
				const size_t n_preserve = g_params.n_keep;

				if (embd_inp.size() < n_preserve) {
					for (size_t i = 0; i < embd_inp.size() && i < session_tokens.size(); i++) {
						if (embd_inp[i] != session_tokens[i])
							break;
						n_matching_session_tokens++;
					}
				}
				else {
					// First, check the preserved prefix.
					bool prefix_matches = true;
					for (size_t i = 0; i < n_preserve; i++) {
						if (embd_inp[i] != session_tokens[i]) {
							prefix_matches = false;
							break;
						}
					}
					if (!prefix_matches) {
						// Fallback to simple matching from the beginning.
						for (size_t i = 0; i < embd_inp.size() && i < session_tokens.size(); i++) {
							if (embd_inp[i] != session_tokens[i])
								break;
							n_matching_session_tokens++;
						}
					}
					else {
						// The preserved prefix matches.
						// Compute the expected gap (i.e. the number of tokens that were dropped during shifting).
						size_t gap = (embd_inp.size() > session_tokens.size())
							? embd_inp.size() - session_tokens.size()
							: 0;
						// Check the shifted suffix.
						bool shifted_matches = true;
						size_t shifted_tokens = session_tokens.size() > n_preserve ? session_tokens.size() - n_preserve : 0;
						for (size_t i = 0; i < shifted_tokens; i++) {
							size_t prompt_index = n_preserve + gap + i;
							if (prompt_index >= embd_inp.size() || embd_inp[prompt_index] != session_tokens[n_preserve + i]) {
								shifted_matches = false;
								break;
							}
						}
						if (shifted_matches) {
							// We can reuse the whole session_tokens.
							n_matching_session_tokens = session_tokens.size();
#ifdef DEBUG
							std::cout << "[INFERENCE] [KV] Matched preserved prefix and shifted suffix." << std::endl;
#endif
						}
						else {
							// If shifted part doesn't match, fall back to matching as much as possible.
							for (size_t i = 0; i < embd_inp.size() && i < session_tokens.size(); i++) {
								if (embd_inp[i] != session_tokens[i])
									break;
								n_matching_session_tokens++;
							}
						}
					}
				}

#ifdef DEBUG
				if (n_matching_session_tokens == embd_inp.size()) {
					std::cout << "[INFERENCE] Session file has an exact match for the prompt." << std::endl;
				}
				else if (n_matching_session_tokens < (embd_inp.size() / 2)) {
					std::cout << "[INFERENCE] Session file has low similarity to prompt ("
						<< n_matching_session_tokens << " / " << embd_inp.size()
						<< "); will mostly be re-evaluated." << std::endl;
				}
				else {
					std::cout << "[INFERENCE] Session file matches "
						<< n_matching_session_tokens << " / "
						<< embd_inp.size() << " tokens of prompt." << std::endl;
				}
#endif

				if (session_tokens.size() > embd_inp.size() && n_matching_session_tokens > 0)
				{
					--n_matching_session_tokens; // always force to re-evaluate the last token
				}

				// Remove any "future" tokens that don’t match
				// i.e. we only keep the portion that matched
				llama_kv_cache_seq_rm(context, id, n_matching_session_tokens, -1 /*up to end*/);
				session_tokens.resize(n_matching_session_tokens);
#ifdef DEBUG
				printf("[INFERENCE] [KV] removed %d tokens from the cache\n", (int)(session_tokens.size() - n_matching_session_tokens));
				printf("[INFERENCE] [KV] tokens decoded: %s\n", tokenizer->detokenize(session_tokens).c_str());
#endif

				return true;
			}

			return false;
		}

		void kv_cache_seq_ltrim(llama_context* context, int n_keep, std::vector<llama_token>& session_tokens, int& n_past, int id = 0) {
			if (n_past <= n_keep) {
				return;
			}

			int n_left = n_past - n_keep;
			int n_discard = n_left / 2;
			if (n_discard <= 0) {
				return;
			}

#if DEBUG
			std::cout << "\n\nContext full, shifting: n_past = " << n_past
				<< ", n_left = " << n_left
				<< ", n_keep = " << n_keep
				<< ", n_discard = " << n_discard << std::endl << std::endl;
#endif

			llama_kv_cache_seq_rm(context, id, n_keep, n_keep + n_discard);
			llama_kv_cache_seq_add(context, id, n_keep + n_discard, n_past, -n_discard);

			n_past -= n_discard;

			if (!session_tokens.empty() && session_tokens.size() >= (size_t)(n_keep + n_discard)) {
				session_tokens.erase(session_tokens.begin() + n_keep,
					session_tokens.begin() + n_keep + n_discard);
			}
		}
	};
} // namespace

struct InferenceEngine::Impl
{
	std::unique_ptr<InferenceService> inferenceService;

	// Job management members
	std::atomic<int> nextJobId{ 0 };
	std::unordered_map<int, std::shared_ptr<Job>> jobs;
	std::mutex jobsMutex;

	ThreadPool threadPool;

	Impl(const char* engineDir, const int mainGpuId = 0);
	~Impl();

	int submitCompletionsJob(const CompletionParameters& params);
	int submitChatCompletionsJob(const ChatCompletionParameters& params);
	void stopJob(int job_id);
	bool isJobFinished(int job_id);
	CompletionResult getJobResult(int job_id);
	void waitForJob(int job_id);
	bool hasJobError(int job_id);
	std::string getJobError(int job_id);
};

InferenceEngine::Impl::Impl(const char* engineDir, const int mainGpuId)
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
		throw std::runtime_error("[INFERENCE] [ERROR] Tokenizer model not found from " + tokenizer_model_path.string());
	}

	unsigned int inferenceThreads = std::thread::hardware_concurrency() - 1;
	if (inferenceThreads == 0)
		inferenceThreads = 4; // a reasonable default if we cannot detect

	std::cout << "[INFERENCE] Inference threads: " << inferenceThreads << std::endl;

	common_params params;
	params.model						= tokenizer_model_path.string().c_str();
	params.n_ctx						= 4096;
	params.n_keep						= 2048;
	params.use_mlock					= true;
	params.use_mmap						= false;
	params.cont_batching				= false;
	params.warmup						= false;
	params.cpuparams.n_threads			= inferenceThreads;
	//params.flash_attn					= true;
#if defined(USE_CUDA) || defined(USE_VULKAN)
	std::cout << "[INFERENCE] Using CUDA or Vulkan" << std::endl;

	params.n_gpu_layers = 100;
#endif

	std::cout << "[INFERENCE] Using main GPU ID: " << params.main_gpu << std::endl;

	llama_backend_init();
	llama_numa_init(params.numa);

	// Initialize the tokenizer
	// TODO: tokenizer should be handled by the inference service
	auto tokenizer = std::make_shared<Tokenizer>(tokenizer_model_path.string(), params);

	// Load the model
	{
#ifdef DEBUG
		std::cout << "[INFERENCE] Loading model from " << tokenizer_model_path << std::endl;
#endif
		// Load model and apply lora adapters, if any
		common_init_result llama_init = common_init_from_params(params);

		llama_model*		model = llama_init.model.release();
		llama_context*		ctx	  = llama_init.context.release();

		if (model == NULL || ctx == NULL)
		{
			throw std::runtime_error("[INFERENCE] [ERROR] Failed to load model from " + params.model);
		}

		struct ggml_threadpool_params threadpool_params;
		ggml_threadpool_params_init(&threadpool_params, inferenceThreads);
		threadpool_params.prio = GGML_SCHED_PRIO_REALTIME;
		set_process_priority(GGML_SCHED_PRIO_REALTIME);
		struct ggml_threadpool* threadpool			= ggml_threadpool_new(&threadpool_params);
		llama_attach_threadpool(ctx, threadpool, nullptr);

		inferenceService = std::make_unique<LlamaInferenceService>(tokenizer, model, ctx, params, threadpool);
	}
}

int InferenceEngine::Impl::submitCompletionsJob(const CompletionParameters& params)
{
	int jobId = nextJobId++;

	auto job = std::make_shared<Job>();
	job->jobId = jobId;

	// Asynchronously execute the job using thread pool
	try {
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
	}
	catch (const std::exception& e)
	{
		std::cerr << "[INFERENCE] [ERROR] " << e.what() << std::endl;
		return -1;
	}

	{
		std::lock_guard<std::mutex> lock(jobsMutex);
		jobs.emplace(jobId, job);
	}

	return jobId;
}

int InferenceEngine::Impl::submitChatCompletionsJob(const ChatCompletionParameters& params)
{
	int jobId = nextJobId++;

	auto job = std::make_shared<Job>();
	job->jobId = jobId;

#ifdef DEBUG
	std::cout << "[INFERENCE] Submitting chat completions job to queue" << std::endl;
#endif

	// Asynchronously execute the job using thread pool
	try
	{
		threadPool.enqueue([this, params, job]() {
			try {
#ifdef DEBUG
				std::cout << "[INFERENCE] Processing completion task to engine" << std::endl;
#endif

				this->inferenceService->complete(this->inferenceService->formatChat(params), job);
			}
			catch (const std::exception& e) {
				std::lock_guard<std::mutex> lock(job->mtx);
				job->hasError = true;
				job->errorMessage = e.what();

				std::cerr << "[INFERENCE] [ERROR] " << e.what() << "\n" << std::endl;
			}
			});
	}
	catch (const std::exception& e)
	{
		std::cerr << "[INFERENCE] [ERROR] " << e.what() << std::endl;
		return -1;
	}

	{
		std::lock_guard<std::mutex> lock(jobsMutex);
		jobs.emplace(jobId, job);
	}

	return jobId;
}

void InferenceEngine::Impl::stopJob(int job_id) {
	std::lock_guard<std::mutex> lock(jobsMutex);
	auto it = jobs.find(job_id);
	if (it != jobs.end()) {
		it->second->cancelRequested.store(true);
		// Optionally notify any waiting threads:
		std::lock_guard<std::mutex> jobLock(it->second->mtx);
		it->second->cv.notify_all();
	}
}

bool InferenceEngine::Impl::isJobFinished(int job_id)
{
	std::shared_ptr<Job> job;

	{
		std::lock_guard<std::mutex> lock(jobsMutex);
		auto it = jobs.find(job_id);
		if (it == jobs.end())
		{
			std::cerr << "[INFERENCE] [ERROR] Invalid job ID\n" << std::endl;
			return false;
		}
		job = it->second;
	}

	std::lock_guard<std::mutex> jobLock(job->mtx);
	bool isFinished = job->isFinished;
	if (isFinished)
	{
		std::lock_guard<std::mutex> lock(jobsMutex);
		jobs.erase(job_id);
	}
	return isFinished;
}

CompletionResult InferenceEngine::Impl::getJobResult(int job_id)
{
	std::shared_ptr<Job> job;

	{
		std::lock_guard<std::mutex> lock(jobsMutex);
		auto it = jobs.find(job_id);
		if (it == jobs.end()) 
		{
			std::cerr << "[INFERENCE] [ERROR] Invalid job ID\n" << std::endl;
			return { {}, "" };
		}
		job = it->second;
	}

	std::lock_guard<std::mutex> jobLock(job->mtx);
	return { job->generatedTokens, job->generatedText, job->tps };
}

void InferenceEngine::Impl::waitForJob(int job_id)
{
	std::shared_ptr<Job> job;

	{
		std::lock_guard<std::mutex> lock(jobsMutex);
		auto it = jobs.find(job_id);
		if (it == jobs.end()) 
		{
			throw std::runtime_error("[INFERENCE] [ERROR] Invalid job ID\n");
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
		if (it == jobs.end()) 
		{
			std::cerr << "[INFERENCE] [ERROR] Invalid job ID\n" << std::endl;
			return false;
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
		if (it == jobs.end()) 
		{
			std::cerr << "[INFERENCE] [ERROR] Invalid job ID\n" << std::endl;
			return "";
		}
		job = it->second;
	}

	std::lock_guard<std::mutex> jobLock(job->mtx);
	return job->errorMessage;
}

InferenceEngine::Impl::~Impl()
{
	threadPool.shutdown();
	jobs.clear();
	llama_backend_free();

	inferenceService.reset();
}

INFERENCE_API InferenceEngine::InferenceEngine()
	: pimpl(nullptr)
{
}

INFERENCE_API bool InferenceEngine::loadModel(const char* engineDir, const int mainGpuId)
{
#ifdef DEBUG
	std::cout << "[INFERENCE] Loading model from " << engineDir << std::endl;
#endif
	this->pimpl.reset();

	try {
		this->pimpl = std::make_unique<Impl>(engineDir, mainGpuId);
	}
	catch (const std::exception& e) {
		std::cerr << "[INFERENCE] [ERROR] Could not load model from: " << engineDir << "\nError: " << e.what() << "\n" << std::endl;
		return false;
	}
	return true;
}

INFERENCE_API bool InferenceEngine::unloadModel()
{
	if (!this->pimpl)
	{
		std::cerr << "[INFERENCE] [ERROR] Model not loaded\n" << std::endl;
		return false;
	}

	this->pimpl.reset();
	return true;
}

INFERENCE_API int InferenceEngine::submitCompletionsJob(const CompletionParameters& params)
{
#ifdef DEBUG
	std::cout << "[INFERENCE] Submitting completions job" << std::endl;
#endif

	return pimpl->submitCompletionsJob(params);
}

INFERENCE_API int InferenceEngine::submitChatCompletionsJob(const ChatCompletionParameters& params)
{
#ifdef DEBUG
	std::cout << "[INFERENCE] Submitting chat completions job" << std::endl;
#endif

	return pimpl->submitChatCompletionsJob(params);
}

INFERENCE_API void InferenceEngine::stopJob(int job_id)
{
	pimpl->stopJob(job_id);
}

INFERENCE_API bool InferenceEngine::isJobFinished(int job_id)
{
	return pimpl->isJobFinished(job_id);
}

INFERENCE_API CompletionResult InferenceEngine::getJobResult(int job_id)
{
	return pimpl->getJobResult(job_id);
}

INFERENCE_API void InferenceEngine::waitForJob(int job_id)
{
	pimpl->waitForJob(job_id);
}

INFERENCE_API bool InferenceEngine::hasJobError(int job_id)
{
	return pimpl->hasJobError(job_id);
}

INFERENCE_API std::string InferenceEngine::getJobError(int job_id)
{
	return pimpl->getJobError(job_id);
}

INFERENCE_API InferenceEngine::~InferenceEngine() = default;

extern "C" INFERENCE_API IInferenceEngine* createInferenceEngine()
{
	return new InferenceEngine();
}

extern "C" INFERENCE_API void destroyInferenceEngine(IInferenceEngine* engine)
{
	delete static_cast<InferenceEngine*>(engine);
}