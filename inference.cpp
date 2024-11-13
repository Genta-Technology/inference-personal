#include <filesystem>
#include <stdexcept>
#include <iostream>
#include <mutex>
#ifdef USE_GPU
#include <cuda_runtime.h>
#endif

#include "llama.h"
#include "common.h"
#include "inference.h"
#ifdef USE_GPU
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"
#endif

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
        return false;
    for (const auto &message : messages)
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

    static void llama_log_callback_null(ggml_log_level level, const char *text, void *user_data)
    {
        (void)level;
        (void)text;
        (void)user_data;
    }

    class Tokenizer
    {
    public:
        Tokenizer(const std::string &modelPath);
        ~Tokenizer();

        std::vector<int32_t> tokenize(const std::string &text, bool add_bos = true);
        std::string detokenize(const std::vector<int32_t> &tokens);

        llama_model *getModel() const { return tokenizer_model; }
        llama_context *getContext() const { return tokenizer_context; }
        bool shouldAddBos() const { return add_bos; }

    private:
        llama_model *tokenizer_model;
        llama_context *tokenizer_context;
        bool add_bos;
    };

    Tokenizer::Tokenizer(const std::string &modelPath)
        : tokenizer_model(nullptr), tokenizer_context(nullptr), add_bos(false)
    {
#ifdef DEBUG
        llama_log_set(llama_log_callback_null, NULL);
#endif
        llama_backend_init();

        std::cout << "Loading tokenizer model from: " << modelPath << std::endl;

        llama_model_params model_params = llama_model_default_params();
        model_params.vocab_only = true;

        tokenizer_model = llama_load_model_from_file(modelPath.c_str(), model_params);
        if (!tokenizer_model)
        {
            throw std::runtime_error("Could not load tokenizer model");
        }

        llama_context_params ctx_params = llama_context_default_params();
        tokenizer_context = llama_new_context_with_model(tokenizer_model, ctx_params);
        if (!tokenizer_context)
        {
            throw std::runtime_error("Error: could not create tokenizer context.");
        }

        add_bos = llama_should_add_bos_token(tokenizer_model);
    }

    Tokenizer::~Tokenizer()
    {
        llama_free(tokenizer_context);
        llama_free_model(tokenizer_model);
    }

    std::vector<int32_t> Tokenizer::tokenize(const std::string &text, bool add_bos_token)
    {
        std::vector<llama_token> tokens = llama_tokenize(tokenizer_model, text.c_str(), add_bos_token, true);
        return std::vector<int32_t>(tokens.begin(), tokens.end());
    }

    std::string Tokenizer::detokenize(const std::vector<int32_t> &tokens)
    {
        std::ostringstream tokensStream;
        for (const auto &token : tokens)
        {
            std::string tokenStr = llama_token_to_piece(tokenizer_context, token);
            tokensStream << tokenStr;
        }
        return tokensStream.str();
    }

    // InferenceService Interface (Internal Use Only)
    class InferenceService
    {
    public:
        virtual ~InferenceService() {}
        virtual CompletionResult complete(const CompletionParameters &params) = 0;
        virtual CompletionResult chatComplete(const ChatCompletionParameters &params) = 0;
    };

    // CpuInferenceService (CPU Implementation)
    class CpuInferenceService : public InferenceService
    {
    public:
        CpuInferenceService(std::shared_ptr<Tokenizer> tokenizer)
            : tokenizer(std::move(tokenizer)) {}

        CompletionResult complete(const CompletionParameters &params) override
        {
            std::cout << "Using CPU (Dummy) inference service." << std::endl;

            if (!params.isValid())
            {
                throw std::invalid_argument("Invalid completion parameters");
            }

            // Tokenize the prompt
            auto tokens = tokenizer->tokenize(params.prompt, tokenizer->shouldAddBos());

            // For demonstration, return the tokenized input as a string
            std::string tokenizedText = tokenizer->detokenize(tokens);

            // Create a dummy response (e.g., echo the tokenized prompt)
            return {tokens, tokenizedText};
        }

        CompletionResult chatComplete(const ChatCompletionParameters &params) override
        {
            std::cout << "Using CPU (Dummy) inference service." << std::endl;

            if (!params.isValid())
            {
                throw std::invalid_argument("Invalid chat completion parameters");
            }

            // Format the chat messages into a single prompt
            std::vector<llama_chat_message> messages;
            for (const auto &msg : params.messages)
            {
                messages.push_back(llama_chat_message{msg.role.c_str(), msg.content.c_str()});
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

            // Tokenize the formatted chat
            CompletionParameters completionParams{
                formattedChat,
                params.randomSeed,
                params.maxNewTokens,
                params.minLength,
                params.temperature,
                params.topP,
                params.streaming};

            return complete(completionParams);
        }

    private:
        std::shared_ptr<Tokenizer> tokenizer;
    };

#ifdef USE_GPU

    // GpuInferenceService (GPU Implementation)
    class GpuInferenceService : public InferenceService
    {
    public:
        GpuInferenceService(
            std::shared_ptr<tensorrt_llm::executor::Executor> executor,
            std::shared_ptr<Tokenizer> tokenizer)
            : executor(std::move(executor)),
              tokenizer(std::move(tokenizer))
        {
        }

        ~GpuInferenceService()
        {
            // Resources are cleaned up by shared pointers and destructors
        }

        CompletionResult complete(const CompletionParameters &params) override
        {
            std::lock_guard<std::mutex> lock(mtx);

            if (!params.isValid())
            {
                throw std::invalid_argument("Invalid completion parameters");
            }

            // Tokenize the prompt
            auto tokenIds = tokenizer->tokenize(params.prompt, tokenizer->shouldAddBos());

            auto request = tensorrt_llm::executor::Request(tokenIds, params.maxNewTokens);
            auto samplingConfig = tensorrt_llm::executor::SamplingConfig(
                1, std::nullopt, params.topP, std::nullopt, std::nullopt,
                std::nullopt, params.randomSeed, params.temperature, params.minLength);
            request.setSamplingConfig(samplingConfig);
            request.setStreaming(false);
            request.setEndId(llama_token_eos(tokenizer->getModel()));
            request.setPadId(llama_token_pad(tokenizer->getModel()));

            auto requestId = executor->enqueueRequest(request);
            auto responses = executor->awaitResponses(requestId);
            auto response = responses.at(0);

            if (!response.hasError())
            {
                auto outputTokens = response.getResult().outputTokenIds.at(0);
                std::string outputText = tokenizer->detokenize(outputTokens);

                return {outputTokens, outputText};
            }
            else
            {
                throw std::runtime_error(response.getErrorMsg());
            }
        }

        CompletionResult chatComplete(const ChatCompletionParameters &params) override
        {
            std::lock_guard<std::mutex> lock(mtx);

            if (!params.isValid())
            {
                throw std::invalid_argument("Invalid chat completion parameters");
            }

            std::vector<llama_chat_message> messages;
            for (const auto &msg : params.messages)
            {
                messages.push_back(llama_chat_message{msg.role.c_str(), msg.content.c_str()});
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
                params.streaming};

            return complete(completionParams);
        }

    private:
        std::shared_ptr<tensorrt_llm::executor::Executor> executor;
        std::shared_ptr<Tokenizer> tokenizer;
        std::mutex mtx;
    };

#endif // USE_GPU

} // namespace

struct InferenceEngine::Impl
{
    std::unique_ptr<InferenceService> inferenceService;
    bool use_gpu;

    Impl(const std::string &engineDir, bool use_gpu);
    ~Impl();

    CompletionResult complete(const CompletionParameters &params)
    {
        return inferenceService->complete(params);
    }

    CompletionResult chatComplete(const ChatCompletionParameters &params)
    {
        return inferenceService->chatComplete(params);
    }
};

InferenceEngine::Impl::Impl(const std::string &engineDir, bool use_gpu)
    : use_gpu(use_gpu)
{
#ifdef USE_GPU
    std::filesystem::path tokenizer_model_path = std::filesystem::path(engineDir) / "tokenizer.gguf";
#else
    // If USE_GPU is not defined, find any .gguf file in the engine directory
    std::filesystem::path tokenizer_model_path;
    for (const auto &entry : std::filesystem::directory_iterator(engineDir))
    {
        if (entry.path().extension() == ".gguf")
        {
            tokenizer_model_path = entry.path();
            break;
        }
    }
#endif

    if (!std::filesystem::exists(tokenizer_model_path))
    {
        throw std::runtime_error("Tokenizer model not found from" + tokenizer_model_path.string());
    }

    // Initialize the tokenizer
    auto tokenizer = std::make_shared<Tokenizer>(tokenizer_model_path.string());

    if (use_gpu)
    {
#ifdef USE_GPU
        std::cout << "GPU detected. Initializing GPU inference." << std::endl;
        // Initialize TensorRT LLM plugins
        initTrtLlmPlugins();

#ifdef DEBUG
        tensorrt_llm::common::TLLM_LOG(tensorrt_llm::common::Logger::Level::INFO);
#else
        tensorrt_llm::common::TLLM_LOG(tensorrt_llm::common::Logger::Level::ERROR);
#endif

        std::filesystem::path trtllmEnginePath = engineDir;
        if (!std::filesystem::exists(trtllmEnginePath))
        {
            throw std::runtime_error("Engine folder not found");
        }

        auto config = tensorrt_llm::executor::ExecutorConfig(
            1,
            tensorrt_llm::executor::SchedulerConfig(
                tensorrt_llm::executor::CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT),
            tensorrt_llm::executor::KvCacheConfig(),
            false,
            true,
            tensorrt_llm::executor::kDefaultIterStatsMaxIterations,
            tensorrt_llm::executor::kDefaultRequestStatsMaxIterations,
            tensorrt_llm::executor::BatchingType::kINFLIGHT);

        auto executor = std::make_shared<tensorrt_llm::executor::Executor>(
            trtllmEnginePath, tensorrt_llm::executor::ModelType::kDECODER_ONLY,
            config);

        auto isLeaderProcess = executor->isLeaderProcess();

        if (!isLeaderProcess)
        {
            throw std::runtime_error("Failed to initialize the executor.");
        }

        inferenceService = std::make_unique<GpuInferenceService>(
            executor, tokenizer);

#else
        // If USE_GPU is not defined but GPU is available, notify the user.
        std::cerr << "GPU is available but the application was not compiled with GPU support." << std::endl;
        use_gpu = false;
#endif
    }

    if (!use_gpu)
    {
        // Initialize CPU inference if GPU is not available
        std::cout << "GPU not detected or not usable. Falling back to CPU inference." << std::endl;
        inferenceService = std::make_unique<CpuInferenceService>(tokenizer);
    }
}

InferenceEngine::Impl::~Impl()
{
    // Resources are cleaned up by destructors
}

InferenceEngine::InferenceEngine(const std::string &engineDir)
    : pimpl(std::make_unique<Impl>(engineDir, isGpuAvailable()))
{
}

CompletionResult InferenceEngine::complete(const CompletionParameters &params)
{
    return pimpl->complete(params);
}

CompletionResult InferenceEngine::chatComplete(const ChatCompletionParameters &params)
{
    return pimpl->chatComplete(params);
}

InferenceEngine::~InferenceEngine() = default;

bool InferenceEngine::isGpuAvailable()
{
#ifdef USE_GPU
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    return (error_id == cudaSuccess && deviceCount > 0);
#else
    return false;
#endif
}