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
		virtual void complete(const CompletionParameters& params, std::shared_ptr<Job> job) = 0;
		virtual void chatComplete(const ChatCompletionParameters& params, std::shared_ptr<Job> job) = 0;
	};

	// LlamaInferenceService (CPU Implementation)
	class LlamaInferenceService : public InferenceService
	{
	public:
		LlamaInferenceService(std::shared_ptr<Tokenizer> tokenizer, llama_model* model, llama_context* context, 
			common_params params, ggml_threadpool* threadpool)
			: tokenizer(std::move(tokenizer)), model(model), context(context), g_params(params), threadpool(threadpool)
		{
		}

		~LlamaInferenceService()
		{
			llama_free(context);
			llama_free_model(model);

			ggml_threadpool_free(threadpool);
		}

		void complete(const CompletionParameters& params, std::shared_ptr<Job> job) override
		{
#ifdef DEBUG
			std::cout << "[INFERENCE] [COMPLETE] Starting completion" << std::endl;

			// Print the parameters
			std::cout << "[INFERENCE] [COMPLETE] Chat completion parameters: " << std::endl;
			std::cout
				<< "[INFERENCE] [COMPLETE] Random seed: "		<< params.randomSeed << std::endl
				<< "[INFERENCE] [COMPLETE] Max new tokens: "	<< params.maxNewTokens << std::endl
				<< "[INFERENCE] [COMPLETE] Min length: "		<< params.minLength << std::endl
				<< "[INFERENCE] [COMPLETE] Temperature: "		<< params.temperature << std::endl
				<< "[INFERENCE] [COMPLETE] Top P: "				<< params.topP << std::endl;

			std::cout << "[INFERENCE] [COMPLETE] Prompt: "		<< params.prompt << std::endl;
#endif

			std::lock_guard<std::mutex> lock(mtx);

			if (!params.isValid())
			{
				std::lock_guard<std::mutex> jobLock(job->mtx);
				job->hasError = true;
				job->errorMessage = "Invalid completion parameters";
				job->cv.notify_all();
				return;
			}

			auto sparams		= g_params.sampling;
			sparams.top_p		= params.topP;
			sparams.temp		= params.temperature;
			sparams.seed		= params.randomSeed;
			// sparams.top_k	= params.topK;

#ifdef DEBUG
			sparams.no_perf = false;
#else
			sparams.no_perf = true;
#endif

			common_sampler* sampler = common_sampler_init(model, sparams);
			if (!sampler) 
			{
				std::lock_guard<std::mutex> jobLock(job->mtx);
				job->hasError = true;
				job->errorMessage = "Could not initialize sampler";
				job->cv.notify_all();
				return;
			}

			{
				std::lock_guard<std::mutex> jobLock(job->mtx);
				job->generatedTokens.clear();
				job->generatedText.clear();
			}

			// Attempt to load existing session (KV cache) if "kvCacheFilePath" is set
			std::string path_session = params.kvCacheFilePath;
			std::vector<llama_token> session_tokens;  // The tokens we previously saved

			if (!path_session.empty()) {
				// Attempt to load
				if (!std::filesystem::exists(path_session)) {
					// file doesn't exist => no old cache
					printf("[KV] session file does not exist, will create.\n");
				}
				else if (std::filesystem::is_empty(path_session)) {
					// file is empty => treat as brand-new
					printf("[KV] session file is empty, new session.\n");
				}
				else {
					// The file exists and is not empty
					session_tokens.resize(g_params.n_ctx);  // allocate buffer
					size_t n_token_count_out = 0;

					if (!llama_state_load_file(
						context,
						path_session.c_str(),
						session_tokens.data(),
						session_tokens.capacity(),
						&n_token_count_out
					)) {
						// If load fails, handle error
						std::lock_guard<std::mutex> jobLock(job->mtx);
						job->hasError = true;
						job->errorMessage = "Failed to load session file: " + path_session;
						job->cv.notify_all();
						common_sampler_free(sampler);
						return;
					}

					// The llama_state_load_file call gives us how many tokens were in that old session
					session_tokens.resize(n_token_count_out);

#ifdef DEBUG
					printf("[INFERENCE] [KV] loaded session with prompt size: %d tokens\n", (int)session_tokens.size());
					printf("[INFERENCE] [KV] tokens decoded: %s", tokenizer->detokenize(session_tokens).c_str());
#endif
				}
			}

#ifdef DEBUG
			std::cout << "[INFERENCE] [COMPLETE] Tokenizing prompt" << std::endl;
#endif

			std::vector<llama_token> embd_inp;  // "embedding input"
			if (session_tokens.empty() || !params.prompt.empty()) {
				// If the session cache is empty OR we have a brand new prompt,
				// we tokenize from scratch.
#ifdef DEBUG
				std::cout << "[INFERENCE] Tokenizing new prompt" << std::endl;
#endif
				embd_inp = tokenizer->tokenize(params.prompt, tokenizer->shouldAddBos());
			}
			else {
#ifdef DEBUG
				std::cout << "[INFERENCE] Reusing session tokens from disk" << std::endl;
#endif
				// If we want to re-use exactly what was loaded, we do:
				embd_inp = session_tokens;
			}

			if (embd_inp.empty()) {
				// Attempt to add BOS if configured
				if (tokenizer->shouldAddBos()) {
					embd_inp.push_back(llama_token_bos(tokenizer->getVocab()));
				}
				else {
					std::lock_guard<std::mutex> jobLock(job->mtx);
					job->hasError = true;
					job->errorMessage = "Empty prompt and no BOS token available.";
					job->cv.notify_all();
					return;
				}
			}

			size_t n_matching_session_tokens = 0;
			if (!session_tokens.empty()) {
				for (llama_token id : session_tokens) {
					if (n_matching_session_tokens >= embd_inp.size()
						|| id != embd_inp[n_matching_session_tokens])
					{
						break;
					}
#ifdef DEBUG
					std::cout << "[INFERENCE] [KV] Matched token: "
						<< id << " == " << embd_inp[n_matching_session_tokens] << std::endl;
#endif

					n_matching_session_tokens++;
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

				if (session_tokens.size() > embd_inp.size())
				{
					--n_matching_session_tokens; // always force to re-evaluate the last token
				}

				// Remove any "future" tokens that don’t match
				// i.e. we only keep the portion that matched
				llama_kv_cache_seq_rm(context, -1 /*any key_id*/, n_matching_session_tokens, -1 /*up to end*/);
				session_tokens.resize(n_matching_session_tokens);
#ifdef DEBUG
				printf("[INFERENCE] [KV] removed %d tokens from the cache\n", (int)(session_tokens.size() - n_matching_session_tokens));
				printf("[INFERENCE] [KV] tokens decoded: %s\n", tokenizer->detokenize(session_tokens).c_str());
#endif
			}


			int n_past		= (int)n_matching_session_tokens;   // how many tokens are “in” the model’s cache
			int n_ctx		= llama_n_ctx(context);
			int n_batch		= g_params.n_batch;					// how many tokens to evaluate at once
			int n_remain	= params.maxNewTokens;				// generation budget
			int i_prompt	= (int)n_matching_session_tokens;	// how many from embd_inp have been consumed

			// Context Trimming Configuration
			int n_keep		= g_params.n_keep;

#ifdef DEBUG
			std::cout << "[INFERENCE] [COMPLETE] Starting decode loop\n"
				<< " - n_ctx:    " << n_ctx << "\n"
				<< " - n_past:   " << n_past << "\n"
				<< " - n_remain: " << n_remain << "\n";
#endif

			std::vector<llama_token> embd;  // batch of tokens to evaluate
			
			while (true) 
			{
				if (job->cancelRequested.load()) {
					{
						std::lock_guard<std::mutex> jobLock(job->mtx);
						job->errorMessage = "Generation cancelled by user.";
						job->isFinished = true;
						job->cv.notify_all();
					}
					break;
				}

				// 1. If we still have prompt tokens left to feed, push them into `embd`:
				if (i_prompt < (int)embd_inp.size()) 
				{
					while (i_prompt < (int)embd_inp.size() && (int)embd.size() < n_batch) 
					{
						embd.push_back(embd_inp[i_prompt]);
						++i_prompt;
					}

					// Evaluate prompt tokens in batch:
					if (!embd.empty()) 
					{
						if (n_past + (int)embd.size() > n_ctx) 
						{
							kv_cache_seq_ltrim(context, n_keep, session_tokens, n_past);

							if (n_past + (int)embd.size() > n_ctx) {
								std::lock_guard<std::mutex> jobLock(job->mtx);
								job->hasError = true;
								job->errorMessage = "Context overflow even after trimming.";
								job->cv.notify_all();
								common_sampler_free(sampler);
								return;
							}
						}

						// **Accept** these tokens in the sampler so it accounts them in repetition-penalty, etc.
						// For prompt tokens, use `accept_grammar = false` in typical llama.cpp usage:
						for (auto t : embd) 
						{
							common_sampler_accept(sampler, t, /*accept_grammar=*/false);
						}

						llama_decode(context, llama_batch_get_one(embd.data(), (int)embd.size()));
						n_past += (int)embd.size();

						if (!path_session.empty()) 
						{
							session_tokens.insert(session_tokens.end(), embd.begin(), embd.end());
						}

						embd.clear();
					}
					continue;
				}

				// 2. We have consumed all prompt tokens, so now we generate the model’s output
				if (n_remain <= 0) {
					// done generating
					break;
				}

				if (job->cancelRequested.load()) {
					{
						std::lock_guard<std::mutex> jobLock(job->mtx);
						job->errorMessage = "Generation cancelled by user.";
						job->isFinished = true;
						job->cv.notify_all();
					}
					break;
				}


				// check for context overflow
				if (n_past + 1 > n_ctx)
				{
					kv_cache_seq_ltrim(context, n_keep, session_tokens, n_past);

					if (n_past + 1 > n_ctx) {
						std::lock_guard<std::mutex> jobLock(job->mtx);
						job->hasError = true;
						job->errorMessage = "Context overflow even after trimming.";
						job->cv.notify_all();
						common_sampler_free(sampler);
						return;
					}

					//std::cout << "\n\n[CONTEXT SHIFT]\n\n";
				}

				// sample the next token
				llama_token id = common_sampler_sample(sampler, context, -1);

				// accept it into the sampler for subsequent penalty calculations
				common_sampler_accept(sampler, id, /*accept_grammar=*/true);

				// append this token to `embd`
				embd.push_back(id);

				// evaluate in a batch (here, we do 1 at a time for simplicity):
				if (llama_decode(context, llama_batch_get_one(&id, 1))) {
					std::lock_guard<std::mutex> jobLock(job->mtx);
					job->hasError = true;
					job->errorMessage = "Could not decode next token";
					job->cv.notify_all();
					break;
				}
				n_past += 1;
				n_remain -= 1;

				// check end-of-sequence or EOG
				if (llama_token_is_eog(tokenizer->getVocab(), id) ||
					(id == llama_token_eos(tokenizer->getVocab())))
				{
					break;
				}

				const auto data = llama_perf_context(context);

				// add the generated token to the Job so the caller sees partial results
				{
					const std::string token_str = tokenizer->decode(id);
					std::lock_guard<std::mutex> jobLock(job->mtx);
					job->generatedTokens.push_back(id);
					job->generatedText += token_str;
					job->tps = 1e3 / data.t_eval_ms * data.n_eval;
					job->cv.notify_all();
				}

				// if session saving is enabled, record new token
				if (!path_session.empty()) {
					session_tokens.push_back(id);
				}

				// we used that single token, so empty `embd` for the next iteration
				embd.clear();
			}

			// Save the final KV cache to disk
			if (!path_session.empty()) {
#ifdef DEBUG
				std::cout << "[INFERENCE] [KV] Saving final session to file: " << path_session << std::endl;
				std::cout << "[INFERENCE] [KV] Session final size: " << session_tokens.size() << std::endl;
#endif
				llama_state_save_file(context,
					path_session.c_str(),
					session_tokens.data(),
					session_tokens.size());
			}

#ifdef DEBUG
			std::cout << "[INFERENCE] [COMPLETE] Decoding completed" << std::endl;
			common_perf_print(context, sampler);
#endif

			common_sampler_free(sampler);
			llama_kv_cache_clear(context);

			{
				std::lock_guard<std::mutex> jobLock(job->mtx);
				job->isFinished = true;
				job->cv.notify_all();
			}
		}

		void chatComplete(const ChatCompletionParameters& params, std::shared_ptr<Job> job) override
		{
#ifdef DEBUG
			std::cout << "[INFERENCE] [CHATCOMPLETE] Starting chat completion" << std::endl;

			// Print the parameters
			std::cout << "[INFERENCE] [CHATCOMPLETE] Chat completion parameters: " << std::endl;
			std::cout
				<< "[INFERENCE] [CHATCOMPLETE] Random seed: " << params.randomSeed << std::endl
				<< "[INFERENCE] [CHATCOMPLETE] Max new tokens: " << params.maxNewTokens << std::endl
				<< "[INFERENCE] [CHATCOMPLETE] Min length: " << params.minLength << std::endl
				<< "[INFERENCE] [CHATCOMPLETE] Temperature: " << params.temperature << std::endl
				<< "[INFERENCE] [CHATCOMPLETE] Top P: " << params.topP << std::endl;

			for (const auto& message : params.messages)
			{
				std::cout << "[INFERENCE] [CHATCOMPLETE] Role: " << message.role << " Content: " << message.content << std::endl;
			}
#endif

			if (!params.isValid())
			{
				throw std::runtime_error("[INFERENCE] [CHATCOMPLETE] [ERROR] Invalid chat completion parameters\n");
			}

#ifdef DEBUG
			std::cout << "[INFERENCE] [CHATCOMPLETE] Applying chat completion template" << std::endl;
#endif

			// Format the chat messages into a single prompt
			std::vector<common_chat_msg> messages;
			for (const auto& msg : params.messages)
			{
				messages.push_back(common_chat_msg{ msg.role, msg.content });
			}

			std::cout << "[INFERENCE] [CHATCOMPLETE] Messages" << std::endl;
			for (const auto& msg : messages)
			{
				std::cout << "[INFERENCE] [CHATCOMPLETE] Role: " << msg.role << " Content: " << msg.content << std::endl;
			}

			llama_model* model = tokenizer->getModel();
			if (!model)
			{
				throw std::runtime_error("[INFERENCE] [CHATCOMPLETE] [ERROR] Could not get model from tokenizer\n");
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

			complete(completionParams, job);
		}

	private:
		std::shared_ptr<Tokenizer>	tokenizer;
		llama_model*				model;
		llama_context*				context;
		std::mutex					mtx;
		common_params				g_params;
		ggml_threadpool*			threadpool;

		void kv_cache_seq_ltrim(llama_context* context,
			int n_keep,
			std::vector<llama_token>& session_tokens,
			int& n_past) {
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

			llama_kv_cache_seq_rm(context, 0, n_keep, n_keep + n_discard);
			llama_kv_cache_seq_add(context, 0, n_keep + n_discard, n_past, -n_discard);

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
	std::atomic<int> nextJobId{ 1 };
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

				this->inferenceService->chatComplete(params, job);
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