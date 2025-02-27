
#include "llama.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

#include <filesystem>
#include <variant>

#ifdef ENABLE_PAPI
#include <papi.h>
#endif

using json = nlohmann::json;
namespace fs = std::filesystem;

using ValueType = std::variant<int, std::string>;

const std::string DEFAULT_MODE   = "default";
const std::string MIX_MODAL_MODE = "mix_modal";
const std::string BENCHMARK_MODE = "benchmark";


#ifdef ENABLE_PAPI
void handle_error(int retval) {
    std::cerr << "PAPI error: " << retval << ", " << PAPI_strerror(retval) << std::endl;
}

void save_papi_events_to_json_file(const std::string save_file_path,long long *values, int preset_event_count, int *presetEventCodes, long long real_time, long long user_time){
    PAPI_event_info_t preset_event_info;
    json papi_results;
    papi_results["PAPI_cycles"] = real_time;
    papi_results["PAPI_usec"] = user_time;
    papi_results["PAPI_TOT_INS"] = values[0];
    for (int i = 0; i < preset_event_count; i++) {
        char eventName[PAPI_MAX_STR_LEN];
        PAPI_event_code_to_name(presetEventCodes[i], eventName);        
        PAPI_get_event_info(presetEventCodes[i], &preset_event_info);
        papi_results[eventName]["event_description"] = preset_event_info.long_descr;
        papi_results[eventName]["event_value"] = values[i + 1];
    }
    std::ofstream save_file(save_file_path);
    save_file << papi_results.dump(4);
    save_file.close();
}
#endif

std::vector<float> load_input_embeddings(const std::string& embd_file_path) {
    std::ifstream file(embd_file_path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + embd_file_path);
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<float> embeddings(size / sizeof(float));
    file.read(reinterpret_cast<char*>(embeddings.data()), size);

    return embeddings;
}

std::vector<std::string> getEmbeddingBinFiles(const std::string& directoryPath) {
    std::vector<std::string> binFiles;

    try {
        for (const auto& entry : fs::directory_iterator(directoryPath)) {
            if (entry.is_regular_file() && entry.path().extension() == ".bin") {
                binFiles.push_back(entry.path().filename().string());
            }
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return binFiles;
}

static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    %s -m model.gguf [-n n_predict] [-ngl n_gpu_layers] [prompt]\n", argv[0]);
    printf("\n");
}

int run_mix_modal_model_with_embeddings(std::unordered_map<std::string, std::string> config){
    fs :: path repo_path = fs :: current_path();
    const std::string common_data_path = repo_path.string() + "/config.json";
    json common_data;
    std::ifstream json_file(common_data_path);
    if (json_file.is_open()) {
        json_file >> common_data;
    } else {
        fprintf(stderr, "Failed to open common data file\n");
        return 1;
    }
    std::string papi_results_save_file_path = config["papi_res_path"];
    const int n_embd = common_data["n_embd"];
    std :: string embd_file_path=  config["embd_file_path"];

    // path to the model gguf file
    std::string model_path = config["model_gguf_file_path"];
    // number of tokens to predict
    int n_predict = std::stoi(config["n_tokens"]);

    // number of layers to offload to the GPU
    int ngl = std::stoi(config["n_gpu_layers"]);
    // parse command line arguments

    std :: string papi_event_name;

    // load the input embeddings
    std::vector<float> input_embeddings = load_input_embeddings(embd_file_path);
    const int n_tokens = input_embeddings.size() / n_embd;
    const int n_prompt = n_tokens;


    // load dynamic backends

    ggml_backend_load_all();

    // initialize the model

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;

    

    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }

    // initialize the context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ubatch = n_tokens;
    // n_ctx is the context size
    ctx_params.n_ctx = n_prompt + n_predict - 1;
    // n_batch is the maximum number of tokens that can be processed in a single call to llama_decode
    ctx_params.n_batch = n_prompt;
    // enable performance counters
    ctx_params.no_perf = false;

    llama_context * ctx = llama_init_from_model(model, ctx_params);

    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    // initialize the sampler

    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);

    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // prepare a batch for the prompt

    llama_batch batch = llama_batch_init(n_tokens, n_embd,1);
    batch.n_tokens = n_tokens;
    memcpy(batch.embd, input_embeddings.data(), n_embd * n_tokens * sizeof(float));
    
    for (int i = 0; i < n_tokens; i++) {
        batch.pos[i] = i;  
        batch.n_seq_id[i] = 1;  
        batch.seq_id[i][0] = 0; 
        batch.logits[i] = (i == n_tokens - 1);
    }


    // main loop

    const auto t_main_start = ggml_time_us();
    int n_decode = 0;
    llama_token new_token_id;

    
    #ifdef ENABLE_PAPI

        int EventSet = PAPI_NULL,retval=0,preset_event_count=0;
        unsigned int preset_event = 0x0;
        PAPI_event_info_t preset_event_info;
        int presetEventCodes[PAPI_MAX_PRESET_EVENTS]; 
        long long papi_start_cycles, papi_end_cycles, papi_start_usec, papi_end_usec;

        retval = PAPI_library_init(PAPI_VER_CURRENT);
        if (retval != PAPI_VER_CURRENT)
            handle_error(retval);

        retval = PAPI_multiplex_init();
        if (retval != PAPI_OK)
            handle_error(retval);

        retval = PAPI_create_eventset(&EventSet);
        if (retval != PAPI_OK)
            handle_error(retval);

        retval = PAPI_add_event(EventSet, PAPI_TOT_INS);
        if (retval != PAPI_OK)
            handle_error(retval);

        retval = PAPI_set_multiplex(EventSet);
        if (retval != PAPI_OK)
            handle_error(retval);

        printf("PAPI preset events:\n");
        for (int i = 0; i < PAPI_MAX_PRESET_EVENTS; i++) {
            preset_event = PAPI_PRESET_MASK | i;
            if ((PAPI_query_event(preset_event) == PAPI_OK) && (preset_event != PAPI_TOT_INS)) {
                retval = PAPI_add_event(EventSet, preset_event);
                if (retval == PAPI_OK) {
                    presetEventCodes[preset_event_count++] = preset_event;
                    char eventName[PAPI_MAX_STR_LEN];
                    PAPI_event_code_to_name(preset_event, eventName);
                    printf("  - %s\n", eventName);
                    retval = PAPI_get_event_info(preset_event, &preset_event_info);
                    printf("    - %s\n", preset_event_info.long_descr);
                }
            }
        }

        if (preset_event_count == 0) {
            printf("⚠️ Nothing added.\n");
            exit(1);
        }

        retval = PAPI_start(EventSet);
        if (retval != PAPI_OK)
            handle_error(retval);

        papi_start_cycles = PAPI_get_real_cyc();
        papi_start_usec = PAPI_get_real_usec();
    
    #endif


    for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + n_predict; ) {
        // evaluate the current batch with the transformer model
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }

        n_pos += batch.n_tokens;

        // sample the next token
        {
            new_token_id = llama_sampler_sample(smpl, ctx, -1);

            // is it an end of generation?
            // intentionally force to produce as given n_tokens token to profile better, enable later
            if (llama_vocab_is_eog(vocab, new_token_id)) {
                break;
            }

            char buf[128];
            int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n < 0) {
                fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
                return 1;
            }
            std::string s(buf, n);
            printf("%s", s.c_str());
            fflush(stdout);

            // prepare the next batch with the sampled token
            batch = llama_batch_get_one(&new_token_id, 1);

            n_decode += 1;
        }
    }

    printf("\n");
    printf("n_decode: %d\n", n_decode);

    #ifdef ENABLE_PAPI

        papi_end_cycles = PAPI_get_real_cyc();
        papi_end_usec = PAPI_get_real_usec();

        long long values[preset_event_count + 1];
        retval = PAPI_stop(EventSet, values);
        if (retval != PAPI_OK)
            handle_error(retval);
        printf("\n--- PAPI Event Results ---\n");
        printf("PAPI_TOT_INS: %lld\n", values[0]);
        for (int i = 0; i < preset_event_count; i++) {
            char eventName[PAPI_MAX_STR_LEN];
            PAPI_event_code_to_name(presetEventCodes[i], eventName);
            printf("%s: %lld\n", eventName, values[i + 1]);
        }

        printf("\n\033[0;32mPAPI Profiling Completed!\n\033[0m");

        save_papi_events_to_json_file(papi_results_save_file_path,values,preset_event_count,presetEventCodes,papi_end_cycles - papi_start_cycles,papi_end_usec - papi_start_usec);
    #endif

    const auto t_main_end = ggml_time_us();

    fprintf(stderr, "%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
            __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));

    fprintf(stderr, "\n");
    llama_perf_sampler_print(smpl);
    llama_perf_context_print(ctx);
    fprintf(stderr, "\n");

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);  
    
    return 0;
}



int run_default_mode(std::unordered_map<std::string,std::string> config) {
    // path to the model gguf file
    std::string model_path = config["model_gguf_file_path"];
    // prompt to generate text from
    std::string prompt = config["user_prompt"];
    // number of layers to offload to the GPU
    int ngl = std::stoi(config["n_gpu_layers"]);
    // number of tokens to predict
    int n_predict = std::stoi(config["n_tokens"]);
    // load dynamic backends

    ggml_backend_load_all();

    // initialize the model

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }

    // tokenize the prompt

    // find the number of tokens in the prompt
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);

    // allocate space for the tokens and tokenize the prompt
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
        return 1;
    }

    // initialize the context

    llama_context_params ctx_params = llama_context_default_params();
    // n_ctx is the context size
    ctx_params.n_ctx = n_prompt + n_predict - 1;
    // n_batch is the maximum number of tokens that can be processed in a single call to llama_decode
    ctx_params.n_batch = n_prompt;
    // enable performance counters
    ctx_params.no_perf = false;

    llama_context * ctx = llama_init_from_model(model, ctx_params);

    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    // initialize the sampler

    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);

    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // print the prompt token-by-token

    for (auto id : prompt_tokens) {
        char buf[128];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
            return 1;
        }
        std::string s(buf, n);
        printf("%s", s.c_str());
    }

    // prepare a batch for the prompt

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    // main loop

    const auto t_main_start = ggml_time_us();
    int n_decode = 0;
    llama_token new_token_id;

    for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + n_predict; ) {
        // evaluate the current batch with the transformer model
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }

        n_pos += batch.n_tokens;

        // sample the next token
        {
            new_token_id = llama_sampler_sample(smpl, ctx, -1);

            // is it an end of generation?
            if (llama_vocab_is_eog(vocab, new_token_id)) {
                break;
            }

            char buf[128];
            int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n < 0) {
                fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
                return 1;
            }
            std::string s(buf, n);
            printf("%s", s.c_str());
            fflush(stdout);

            // prepare the next batch with the sampled token
            batch = llama_batch_get_one(&new_token_id, 1);

            n_decode += 1;
        }
    }

    printf("\n");

    const auto t_main_end = ggml_time_us();

    fprintf(stderr, "%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
            __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));

    fprintf(stderr, "\n");
    llama_perf_sampler_print(smpl);
    llama_perf_context_print(ctx);
    fprintf(stderr, "\n");

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);

    return 0;
}

int run_benchmark(std::unordered_map<std::string, std::string> config){
    fs :: path repo_path = fs :: current_path();
    const std::string common_data_path = repo_path.string() + "/config.json";
    json common_data;
    std::ifstream json_file(common_data_path);
    if (json_file.is_open()) {
        json_file >> common_data;
    } else {
        fprintf(stderr, "Failed to open common data file\n");
        return 1;
    }
    const int n_embd = common_data["n_embd"];

    // path to the model gguf file
    std::string model_path =config["model_gguf_file_path"];
    std::string model_id = config["model_id"];
    // number of tokens to predict
    int n_predict = std::stoi(config["n_tokens"]);
    
    // number of layers to offload to the GPU
    int ngl =std::stoi(config["n_gpu_layers"]);

    
    std::string embd_dir_key = model_id+"_embeddings_dir_path";
    std::cout << "Model ID: " << embd_dir_key << std::endl;
    std::string embd_dir_path = common_data[embd_dir_key];
    std::vector<std::string> benchmark_embeddings = getEmbeddingBinFiles(embd_dir_path);
    
    // load dynamic backends

    ggml_backend_load_all();

    // initialize the model

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;

    

    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_context_params ctx_params = llama_context_default_params();
    llama_context * ctx = nullptr;

    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }

    for(int i=0; i<benchmark_embeddings.size(); i++){
        std::cout << "Model ID: " << benchmark_embeddings[i] << std::endl;
        std::vector<float> input_embeddings = load_input_embeddings(embd_dir_path+"/"+benchmark_embeddings[i]);
        const int n_tokens = input_embeddings.size() / n_embd;
        const int n_prompt = n_tokens;
        // initialize the context
        ctx_params.n_ubatch = n_tokens;
        // n_ctx is the context size
        ctx_params.n_ctx = n_prompt + n_predict - 1;
        // n_batch is the maximum number of tokens that can be processed in a single call to llama_decode
        ctx_params.n_batch = n_prompt;
        // enable performance counters
        ctx_params.no_perf = false;

        ctx = llama_init_from_model(model, ctx_params);

        if (ctx == NULL) {
            fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
            return 1;
        }

        // initialize the sampler

        auto sparams = llama_sampler_chain_default_params();
        sparams.no_perf = false;
        llama_sampler * smpl = llama_sampler_chain_init(sparams);

        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

        // prepare a batch for the prompt

        llama_batch batch = llama_batch_init(n_tokens, n_embd,1);
        batch.n_tokens = n_tokens;
        memcpy(batch.embd, input_embeddings.data(), n_embd * n_tokens * sizeof(float));
        
        for (int i = 0; i < n_tokens; i++) {
            batch.pos[i] = i;  
            batch.n_seq_id[i] = 1;  
            batch.seq_id[i][0] = 0; 
            batch.logits[i] = (i == n_tokens - 1);
        }


        // main loop

        const auto t_main_start = ggml_time_us();
        int n_decode = 0;
        llama_token new_token_id;

        for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + n_predict; ) {
            // evaluate the current batch with the transformer model
            if (llama_decode(ctx, batch)) {
                fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
                return 1;
            }

            n_pos += batch.n_tokens;

            // sample the next token
            {
                new_token_id = llama_sampler_sample(smpl, ctx, -1);

                // is it an end of generation?
                // intentionally force to produce as given n_tokens token to profile better, enable later
                if (llama_vocab_is_eog(vocab, new_token_id)) {
                    break;
                }

                char buf[128];
                int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
                if (n < 0) {
                    fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
                    return 1;
                }
                std::string s(buf, n);
                printf("%s", s.c_str());
                fflush(stdout);

                // prepare the next batch with the sampled token
                batch = llama_batch_get_one(&new_token_id, 1);

                n_decode += 1;
            }
        }

        printf("\n");
        printf("n_decode: %d\n", n_decode);

        const auto t_main_end = ggml_time_us();

        fprintf(stderr, "%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
                __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));

        fprintf(stderr, "\n");
        llama_perf_sampler_print(smpl);
        llama_perf_context_print(ctx);
        fprintf(stderr, "\n");

        llama_sampler_free(smpl);
        llama_free(ctx);

        if(i == benchmark_embeddings.size()-1){
            llama_model_free(model);  
        }
    }


    
    return 0;
}


void print_help(){
    printf("  -mode <mode>             Mode of operation (mix_modal or default)\n");
    printf("  --help                   Display this help message\n");
    printf("Options:\n");
    printf("Default Mode:\n");
    printf("  -m <model_path>          Path to the model gguf file\n");
    printf("  -n <n_predict>           Number of tokens to predict (default: 32)\n");
    printf("  -ngl <n_gpu_layers>      Number of layers to offload to the GPU (default: 99)\n");
    printf("  -prompt <user_prompt>    Text prompt to model");
    printf("Mix Modal Mode\n");
    printf("  -m <model_path>          Path to the model gguf file\n");
    printf("  -n <n_predict>           Number of tokens to predict (default: 32)\n");
    printf("  -ngl <n_gpu_layers>      Number of layers to offload to the GPU (default: 99)\n");
    printf("  -embd <embd_file_path>   Path to the input embeddings file\n");
    printf("  -papi <0|1>              Enable or disable PAPI profiling (0: disable, 1: enable)\n");
    printf("  -papi_res_pth <path>     Path to save PAPI results in JSON format\n");
    printf("Benchmark Mode\n");
    printf("  -m_id <model_name          Model name for now, deepseek and chameleon available");
    printf("  -m <model_path>          Path to the model gguf file\n");
    printf("  -n <n_predict>           Number of tokens to predict (default: 32)\n");
    printf("  -ngl <n_gpu_layers>      Number of layers to offload to the GPU (default: 99)\n");
    printf("  -bench <benchmark_name>  Name of the benchmark to run\n");
}

int main(int argc, char ** argv) {
    std::unordered_map<std::string,std::string> mix_modal_modal_mode_config, default_mode_config,benchmark_mode_config;
    std::string model_gguf_file_path, user_prompt, embd_file_path, papi_result_dir_path, model_id, benchmark_name, program_mode,n_tokens, n_gpu_layers;

    for(int i=1; i<argc; i++){
        if(strcmp(argv[i],"--help")==0){
            printf("Usage: %s [options]\n", argv[0]);
            print_help();
            return 1;
        }
        else if(strcmp(argv[i],"-mode")==0){
            if (i + 1 < argc) {
                program_mode = argv[++i];
            } else {
                print_help();
                return 1;
            }
        }
        else if(strcmp(argv[i],"-m")==0){
            if (i + 1 < argc) {
                model_gguf_file_path = argv[++i];
            } else {
                print_help();
                return 1;
            }
        }
        else if(strcmp(argv[i],"-n")==0){
            if (i + 1 < argc) {
                n_tokens = argv[++i];
            } else {
                print_help();
                return 1;
            }
        }
        else if(strcmp(argv[i],"-ngl")==0){
            if (i + 1 < argc) {
                n_gpu_layers = argv[++i];
            } else {
                print_help();
                return 1;
            }
        }
        else if(strcmp(argv[i],"-prompt")==0){
            if (i + 1 < argc) {
                user_prompt = argv[++i];
            } else {
                print_help();
                return 1;
            }
        }
        else if(strcmp(argv[i],"-m_id")==0){
            if (i + 1 < argc) {
                model_id = argv[++i];
            } else {
                print_help();
                return 1;
            }
        }
        else if(strcmp(argv[i],"-embd")==0){
            if (i + 1 < argc) {
                embd_file_path = argv[++i];
            } else {
                print_help();
                return 1;
            }
        }
        else if(strcmp(argv[i],"-papi_res_pth")==0){
            if (i + 1 < argc) {
                papi_result_dir_path = argv[++i];
            } else {
                print_help();
                return 1;
            }
        }
        else if(strcmp(argv[i],"-bench")==0){
            if (i + 1 < argc) {
                benchmark_name = argv[++i];
            } else {
                print_help();
                return 1;
            }
        }
        else{
            print_help();
            return 1;
        }
    }

    default_mode_config["model_gguf_file_path"]=model_gguf_file_path;
    default_mode_config["n_tokens"]=n_tokens;
    default_mode_config["n_gpu_layers"]=n_gpu_layers;
    default_mode_config["user_prompt"]=user_prompt;

    mix_modal_modal_mode_config["model_gguf_file_path"]=model_gguf_file_path;
    mix_modal_modal_mode_config["n_tokens"]=n_tokens;
    mix_modal_modal_mode_config["n_gpu_layers"]=n_gpu_layers;
    mix_modal_modal_mode_config["embd_file_path"]=embd_file_path;
    mix_modal_modal_mode_config["papi_result_dir_path"]=papi_result_dir_path;

    benchmark_mode_config["model_gguf_file_path"]=model_gguf_file_path;
    benchmark_mode_config["n_tokens"]=n_tokens;
    benchmark_mode_config["n_gpu_layers"]=n_gpu_layers;
    benchmark_mode_config["model_id"]=model_id;
    benchmark_mode_config["benchmark_name"]=benchmark_name;



    if(program_mode == MIX_MODAL_MODE)
        run_mix_modal_model_with_embeddings(mix_modal_modal_mode_config);
    else if(program_mode == DEFAULT_MODE)
        run_default_mode(default_mode_config);
    else if(program_mode == BENCHMARK_MODE)
        run_benchmark(benchmark_mode_config);
    else
        print_help();

    return 0;
}