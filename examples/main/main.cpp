
#include "llama.h"
#include "llama-context.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include "nlohmann/json.hpp"
#include <chrono>
#include <algorithm>
#include <random>


#include <filesystem>
#include <variant>

#ifdef ENABLE_PAPI
#include <papi.h>
#endif

using json = nlohmann::json;
namespace fs = std::filesystem;

using ValueType = std::variant<int, std::string>;

const std::string MIX_MODAL_MODE = "mix_modal";

std::vector<float> load_input_embeddings(const std::string& embd_file_path);
int run_mix_modal_model_with_embeddings(std::map<std::string, std::string> config);
void print_help();
#ifdef ENABLE_PAPI
void handle_error(int retval);
void save_papi_events_to_json_file(const std::string save_file_path,std::vector<long long> values, int preset_event_count, int *presetEventCodes, long long real_time, long long user_time);
#endif


bool PAPI_INIT_STATUS = false;
void init_papi_library() {
    if (!PAPI_INIT_STATUS){
        int retval = PAPI_library_init(PAPI_VER_CURRENT);
        if (retval != PAPI_VER_CURRENT)
            handle_error(retval);

        retval = PAPI_multiplex_init();
        if (retval != PAPI_OK)
            handle_error(retval);
        
        retval = PAPI_thread_init(pthread_self);
        if (retval != PAPI_OK)
            handle_error(retval);

        PAPI_INIT_STATUS = true;
    }
}

#ifdef ENABLE_PAPI
void handle_error(int retval) {
    std::cerr << "PAPI error: " << retval << ", " << PAPI_strerror(retval) << std::endl;
}

void save_papi_events_to_json_file(const std::string save_file_path,std::vector<long long> values, int preset_event_count, int *presetEventCodes, long long real_time, long long user_time){
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
    file.close();
    return embeddings;
}


int run_mix_modal_model_with_embeddings(std::map<std::string, std::string> config){
    const char* llama_repo_path_env = std::getenv("MULTI_MIX_MODAL_TRANSFORMERS_REPO_PATH");
    fs::path repo_path = (llama_repo_path_env != nullptr) ? fs::path(llama_repo_path_env) : fs::current_path();
    const std::string common_data_path = repo_path.string() + "/config.json";
    json common_data;
    std::ifstream json_file(common_data_path);
    if (json_file.is_open()) {
        json_file >> common_data;
    } else {
        fprintf(stderr, "Failed to open common data file\n");
        return 1;
    }
    
    const int n_embd                        = common_data["n_embd"];
    std :: string embd_file_path            = config["embd_file_path"];
    std :: string model_id                  = config["model_id"];
    std::string papi_results_save_file_path = config["papi_res_dir_path"];
    std::string model_path = config["model_gguf_file_path"];
    int n_predict = std::stoi(config["n_tokens"]);
    int ngl = std::stoi(config["n_gpu_layers"]);
    std :: string papi_event_name;
    std::vector<float> input_embeddings = load_input_embeddings(embd_file_path);
    const int n_tokens = input_embeddings.size() / n_embd;
    const int n_prompt = n_tokens;



    ggml_backend_load_all();


    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;

    

    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ubatch = n_tokens;
    ctx_params.n_ctx = n_prompt + n_predict - 1;
    ctx_params.n_batch = n_prompt;
    ctx_params.no_perf = false;

    llama_context * ctx = llama_init_from_model(model, ctx_params);

    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }


    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);

    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());


    llama_batch batch = llama_batch_init(n_tokens, n_embd,1);
    batch.n_tokens = n_tokens;
    memcpy(batch.embd, input_embeddings.data(), n_embd * n_tokens * sizeof(float));
    
    for (int i = 0; i < n_tokens; i++) {
        batch.pos[i] = i;  
        batch.n_seq_id[i] = 1;  
        batch.seq_id[i][0] = 0; 
        batch.logits[i] = (i == n_tokens - 1);
    }



    const auto t_main_start = ggml_time_us();
    int n_decode = 0;
    llama_token new_token_id;

    
    #ifdef ENABLE_PAPI
        printf("Enable Papi profiling\n");
        int EventSet = PAPI_NULL,retval=0,preset_event_count=0;
        int preset_event = 0x0;
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

        for (int i = 0; i < PAPI_MAX_PRESET_EVENTS; i++) {
            preset_event = PAPI_PRESET_MASK | i;
            if ((PAPI_query_event(preset_event) == PAPI_OK) && (preset_event != PAPI_TOT_INS)) {
                retval = PAPI_add_event(EventSet, preset_event);
                if (retval == PAPI_OK) {
                    presetEventCodes[preset_event_count++] = preset_event;
                    char eventName[PAPI_MAX_STR_LEN];
                    PAPI_event_code_to_name(preset_event, eventName);
                    retval = PAPI_get_event_info(preset_event, &preset_event_info);
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


    std :: string model_output = "";

    for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + n_predict; ) {
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }

        n_pos += batch.n_tokens;

        {
            new_token_id = llama_sampler_sample(smpl, ctx, -1);

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
            model_output += s.c_str();
            //printf("%s", s.c_str());
            fflush(stdout);

            batch = llama_batch_get_one(&new_token_id, 1);

            n_decode += 1;
        }
    }

    #ifdef ENABLE_PAPI

        papi_end_cycles = PAPI_get_real_cyc();
        papi_end_usec = PAPI_get_real_usec();

        std::vector<long long> values(preset_event_count + 1);
        retval = PAPI_stop(EventSet, values.data());
        if (retval != PAPI_OK)
            handle_error(retval);
        for (int i = 0; i < preset_event_count; i++) {
            char eventName[PAPI_MAX_STR_LEN];
            PAPI_event_code_to_name(presetEventCodes[i], eventName);
        }

        printf("\n\033[0;32mPAPI Profiling Completed!\n\033[0m");

        save_papi_events_to_json_file(papi_results_save_file_path,values,preset_event_count,presetEventCodes,papi_end_cycles - papi_start_cycles,papi_end_usec - papi_start_usec);
    #endif

    const auto t_main_end = ggml_time_us();

    printf("{\n");
    printf("  \"embedding_file_path\": \"%s\",\n", embd_file_path.c_str());
    printf("  \"model_gguf_file_path\": \"%s\",\n", model_path.c_str());
    printf("  \"model_id\": \"%s\",\n", model_id.c_str());
    printf("  \"n_tokens\": %d,\n", n_tokens);
    printf("  \"n_decoded_tokens\": %d,\n", n_decode);
    printf("  \"n_predict\": %d,\n", n_predict);
    printf("  \"n_gpu_layers\": %d,\n", ngl);
    printf("  \"inference_time_sec\": %.4f,\n", (t_main_end - t_main_start) / 1000000.0f);
    printf("  \"output_text\": \"%s\"\n", model_output.c_str()); 
    printf("}\n");

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


void copy_first_token(void);

class PreloadedModel {

    int context_size = 2000;

    llama_sampler * smpl;
    llama_model * model;
    int n_embd=0, ngl;
    const llama_vocab * vocab;
    std::string model_id, model_path;
    llama_context * ctx;

public:
    int papi_event_count = 0;
    int n_predict;
    bool prevent_end_tokens = false;
    std::vector<std::string> papi_event_names;

    PreloadedModel() = default;
    PreloadedModel(int context_size) : context_size(context_size) {;}
    int load_mix_modal_model(std::map<std::string, std::string> config, const std::vector<std::string>& embd_files){
        const char* llama_repo_path_env = std::getenv("MULTI_MIX_MODAL_TRANSFORMERS_REPO_PATH");
        fs::path repo_path = (llama_repo_path_env != nullptr) ? fs::path(llama_repo_path_env) : fs::current_path();
        const std::string common_data_path = repo_path.string() + "/config.json";
        json common_data;
        std::ifstream json_file(common_data_path);
        if (json_file.is_open()) {
            json_file >> common_data;
        } else {
            fprintf(stderr, "Failed to open common data file\n");
            return 1;
        }
        
        n_embd                        = common_data["n_embd"];
        model_id                  = config["model_id"];
        model_path = config["model_gguf_file_path"];
        n_predict = std::stoi(config["n_tokens"]);
        ngl = std::stoi(config["n_gpu_layers"]);
        std :: string papi_event_name;
        



        ggml_backend_load_all();


        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = ngl;

        

        model = llama_model_load_from_file(model_path.c_str(), model_params);
        vocab = llama_model_get_vocab(model);

        std::cerr << "Embeddding shape:";
        for (int i = 0; i < GGML_MAX_DIMS; i++)
            std::cerr << ' ' << model->tok_embd->ne[i];
        std::cerr << std::endl;

        if (100 < model->tok_embd->ne[0] && model->tok_embd->ne[0] < 50000){
            n_embd = model->tok_embd->ne[0];
            std::cerr << "Number of embedding dimensions: " << n_embd << std::endl;
        }

        if (model == NULL) {
            fprintf(stderr , "%s: error: unable to load model\n" , __func__);
            return 1;
        }




        n_predict = std::stoi(config["n_tokens"]);

        int ma = 0;
        for (auto f : embd_files) {
            ma = std::max(ma, (int)load_input_embeddings(f).size());
        }
        std::cerr << ma << ' ' << (ma/n_embd) << ' ' << n_predict << ' ' << (ma / n_embd) + n_predict + 1 << std::endl;
        ma = (ma / n_embd) + n_predict + 1;
        std::cerr << "Maximum tokens in query: " << ma << std::endl;
        context_size = ma;




        auto sparams = llama_sampler_chain_default_params();
        sparams.no_perf = false;

        smpl = llama_sampler_chain_init(sparams);
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
  
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ubatch = context_size;
        ctx_params.n_ctx = context_size;
        ctx_params.n_batch = context_size;
        ctx_params.no_perf = false;

        if (config.count("n_threads")) {
            ctx_params.n_threads = stoi(config["n_threads"]);
            ctx_params.n_threads_batch = stoi(config["n_threads"]);
        }

        ctx = llama_init_from_model(model, ctx_params);

        return 0;
    }

    bool is_end_token(llama_token token_id, const llama_vocab* vocab) {
        if (llama_vocab_is_eog(vocab, token_id)) {
            return true;
        } else if (llama_vocab_is_control(vocab, token_id)) {
            if (false)
                std::cerr << "\nControl detected. Token id: " << token_id << ". Finalizing generation." << std::endl;
            return true;
        }
        return false;
    }

    void remove_control_tokens(int32_t idx) {
        float * logits = llama_get_logits_ith(ctx, idx);

        const int n_vocab = llama_vocab_n_tokens(vocab);

        for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
            if (is_end_token(token_id, vocab))
                logits[token_id] = 0.0;
        }
    }

    int run_model_with_embeddings (std::string embd_file_path, std::string papi_results_save_file_path) {

        std::cerr << "N_Threads: " << ctx->cparams.n_threads << ' ' << ctx->cparams.n_threads_batch << std::endl;

        llama_kv_cache_clear(ctx);

        //std::cerr << "BEGIN" << std::endl;
        std::vector<float> input_embeddings = load_input_embeddings(embd_file_path);
        
        const int n_tokens = input_embeddings.size() / n_embd;
        //std::cerr << "N_TOKENS: " << n_tokens << std::endl;
        const int n_prompt = n_tokens;
        
        assert (n_tokens+n_predict-1 <= context_size), "Assertion failed, hardcoded context size of 2000 is not enough.";

        if (ctx == NULL) {
            fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
            exit(1);
        }

        llama_batch batch = llama_batch_init(n_tokens, n_embd,1);
        batch.n_tokens = n_tokens;
        //std::cerr << "N_TOKENS " << batch.n_tokens << std::endl;
        memcpy(batch.embd, input_embeddings.data(), n_embd * n_tokens * sizeof(float));
        
        for (int i = 0; i < n_tokens; i++) {
            batch.pos[i] = i;  
            batch.n_seq_id[i] = 1;  
            batch.seq_id[i][0] = 0; 
            batch.logits[i] = (i == n_tokens - 1);
        }


        const auto t_main_start = ggml_time_us();
        int n_decode = 0;
        llama_token new_token_id;

        //std::cerr << "END" << std::endl;

        #ifdef ENABLE_PAPI
            
            init_papi_library();
        
        #endif


        std::cerr << "Ouput: ";

        std :: string model_output = "";


        int cnt_iter = 0;
        for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + n_predict; ) {
            //std::cerr << "N_POS: " << n_pos << std::endl;
            if (llama_decode(ctx, batch)) {
                fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
                return 1;
            }

            //std::cerr << "N_TOKENS " << batch.n_tokens << std::endl;
            n_pos += batch.n_tokens;

            {
                if (prevent_end_tokens)
                    remove_control_tokens(-1);

                new_token_id = llama_sampler_sample(smpl, ctx, -1);
                if (cnt_iter == 0) {
                    copy_first_token();
                }
                //std::cerr << "TOK_ID " << new_token_id << std::endl;
                if (is_end_token(new_token_id, vocab))
                    break;

                char buf[128];
                int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
                if (n < 0) {
                    fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
                    return 1;
                }
                std::string s(buf, n);
                std::cerr << s;
                model_output += s.c_str();
                fflush(stdout);

                batch = llama_batch_get_one(&new_token_id, 1);

                n_decode += 1;
            }

            cnt_iter++;
        }

        std::cerr << std::endl;

        const auto t_main_end = ggml_time_us();

        printf("{\n");
        printf("  \"embedding_file_path\": \"%s\",\n", embd_file_path.c_str());
        printf("  \"model_gguf_file_path\": \"%s\",\n", model_path.c_str());
        printf("  \"model_id\": \"%s\",\n", model_id.c_str());
        printf("  \"n_tokens\": %d,\n", n_tokens);
        printf("  \"n_decoded_tokens\": %d,\n", n_decode);
        printf("  \"n_predict\": %d,\n", n_predict);
        printf("  \"n_gpu_layers\": %d,\n", ngl);
        printf("  \"inference_time_sec\": %.4f,\n", (t_main_end - t_main_start) / 1000000.0f);
        printf("  \"output_text\": \"%s\"\n", model_output.c_str()); 
        printf("}\n");

        fprintf(stderr, "%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
                __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));

        fprintf(stderr, "\n");
        llama_perf_sampler_print(smpl);
        llama_perf_context_print(ctx);
        fprintf(stderr, "\n");

        return 0;
    }

    void free_memory() {
        llama_free(ctx);
        llama_sampler_free(smpl);
        llama_model_free(model);  
    }
};

void print_help(){
    printf("  -mode <mode>             Mode of operation (mix_modal or default)\n");
    printf("  --help                   Display this help message\n");
    printf("Options:\n");
    printf("Mix Modal Mode\n");
    printf("  -m <model_path>          Path to the model gguf file\n");
    printf("  -n <n_predict>           Number of tokens to predict (default: 32)\n");
    printf("  -ngl <n_gpu_layers>      Number of layers to offload to the GPU (default: 99)\n");
    printf("  -embd <embd_file_path>   Path to the input embeddings file\n");
    printf("  -m_id <model_id>         Model ID\n");
    printf("  -papi_res_dir <path>     Path to save PAPI profiling results\n");
}

bool count_nodes = false;
bool store_papi_results = true;
bool matrix_statistics = false;
bool profile_first_token = true;
std::vector<int> papi_metric_ids;

void set_custom_papi_metrics(const std::vector<std::string>& metrics) {
    init_papi_library();
    int hw_cnt = PAPI_num_cmp_hwctrs(0);
    for (auto metric : metrics) {
        if (metric == "EVENTSET_SPLIT" && count_nodes) {
            while (papi_metric_ids.size() % hw_cnt != 0)
                papi_metric_ids.push_back(0);
            continue;
        }
        int mtr = 0;
        std::cerr << "Trying to add papi event " << metric << std::endl;
        int retval = PAPI_event_name_to_code(metric.data(), &mtr);
        if (retval != PAPI_OK) {
            std::cerr << "Unable to add event " << metric << ". Error when trying to get code from name." << std::endl;
            continue;
        }
        retval = PAPI_query_event(mtr);
        if (retval != PAPI_OK) {
            std::cerr << "Unable to add event " << metric << ". Query event didn't return ok." << std::endl;
            continue;
        }
        papi_metric_ids.push_back(mtr);
    }
}

void output_stream_matrix_stats(std::ostream& out) {
    /** Matrix statistics:
     *      Zero ratio
     *      Zero count
     *      Element count
     */
    if (!matrix_statistics) return;
    out.precision(10);
    int layer_cnt_zr, layer_cnt_zc, layer_cnt_ec;
    layer_cnt_zr = layer_cnt_zc = layer_cnt_ec = 0;
    float* zratio_arr = get_node_zero_ratio_array(&layer_cnt_zr);
    int64_t* zcount_arr = get_node_zero_count_arr(&layer_cnt_zc);
    int64_t* ecount_arr = get_node_element_count_arr(&layer_cnt_ec);
    out << "ZRatio " << layer_cnt_zr;
    for (int i = 0; i < layer_cnt_zr; i++)
        out << ' ' << zratio_arr[i];
    out << std::endl;
    out << "ZCount " << layer_cnt_zc;
    for (int i = 0; i < layer_cnt_zc; i++)
        out << ' ' << zcount_arr[i];
    out << std::endl;
    out << "ECount " << layer_cnt_ec;
    for (int i = 0; i < layer_cnt_ec; i++)
        out << ' ' << ecount_arr[i];
    out << std::endl;
}

std::vector<long long> first_token_counters;
void copy_first_token() {
    if (!profile_first_token) return;
    int cnt_sz = 0, papi_cnt=0, num_th=0;
    long long* node_time_counter = get_node_time_counter(&cnt_sz, &papi_cnt, &num_th);
    //std::cerr << "3 things " << cnt_sz << ' ' << papi_cnt << ' ' << num_th << std::endl;
    first_token_counters = std::vector<long long>(cnt_sz*papi_cnt*num_th);
    for (int i = 0; i < first_token_counters.size(); i++){
        first_token_counters[i] = node_time_counter[i];
        //std::cerr << first_token_counters[i] << ' ';
        assert(first_token_counters[i] >= 0);
    }
    //std::cerr << std::endl;
    /*int name_num, str_sz;
    char* layer_names = get_layer_names(&name_num, &str_sz);
    for (int i = 0; i < name_num; i++)
        std::cerr << &layer_names[i*str_sz] << ' ';
    std::cerr << std::endl;*/
    //memcpy(first_token_counters.data(), node_time_counter, cnt_sz*papi_cnt*num_th*sizeof(long long));
}

void add_first_token_counters(json& dict) {
    if (!profile_first_token) return;
    /*for (int i = 0; i < first_token_counters.size(); i++){
        std::cerr << first_token_counters[i] << ' ';
        assert(first_token_counters[i] >= 0);
    }
    std::cerr << std::endl;*/
    std::vector<std::string> lnames, enames;
    int cnt_sz = 0, papi_cnt=0, num_th=0;
    get_node_time_counter(&cnt_sz, &papi_cnt, &num_th);
    //std::cerr << "3 things " << cnt_sz << ' ' << papi_cnt << ' ' << num_th << std::endl;
    //assert(papi_cnt == model.papi_event_count);
    int name_num, str_sz;
    char* layer_names = get_layer_names(&name_num, &str_sz);
    /*for (int i = 0; i < name_num; i++)
        std::cerr << &layer_names[i*str_sz] << ' ';
    std::cerr << std::endl;*/
    for (int i = 0; i < name_num; i++)
        lnames.push_back(std::string(&layer_names[i*str_sz]));
    char* event_names = get_papi_events(&name_num, &str_sz);
    enames.push_back(std::string("time_ns"));
    enames.push_back(std::string("papi_cyc"));
    for (int i = 0; i < name_num; i++)
        enames.push_back(std::string(&event_names[i*str_sz]));
    std::vector<std::vector<std::vector<long long>>> res(cnt_sz, std::vector<std::vector<long long>>(papi_cnt, std::vector<long long>(num_th)));
    for (int i = 0; i < cnt_sz; i++)
        for (int j = 0; j < papi_cnt; j++)
            for (int k = 0; k < num_th; k++){
                res[i][j][k] = first_token_counters[i*papi_cnt*num_th+j*num_th+k];
                assert(0 <= i*papi_cnt*num_th+j*num_th+k && i*papi_cnt*num_th+j*num_th+k < first_token_counters.size());
                assert(first_token_counters[i*papi_cnt*num_th+j*num_th+k] >= 0);
                assert(res[i][j][k] >= 0);
            }
    if (count_nodes)
        dict["layer_num"] = cnt_sz;
    dict["thread_num"] = num_th;
    if (!dict.contains("events")) {
        dict["events"] = json::object();
    } else {
        //std::cerr << dict["events"].dump(4) << std::endl;
    }
    PAPI_event_info_t preset_event_info;
    for (auto s : enames) {
        int code = 0;
        PAPI_event_name_to_code(s.data(), &code);
        PAPI_get_event_info(code, &preset_event_info);
        dict["events"][s] = preset_event_info.long_descr;
    }
    dict["events"]["time_ns"] = "Time taken by tensor operations from PAPI_get_real_ns().";    
    dict["events"]["papi_cyc"] = "Cycles taken by tensor operations from PAPI_get_real_cyc().";  
    dict["attr_num"] = dict["events"].size();
    //std::cerr << dict["events"].dump(4) << std::endl;
    for (int i = 0; i < enames.size(); i++) {
        std::vector<long long> su(num_th);
        for (int t = 0; t < num_th; t++) {
            for (int l = 0; l < lnames.size(); l++) {
                su[t] += res[l][i][t];
            }
        }
        long long tsu = 0;
        for (long long x : su)
            tsu += x;
        //std::cerr << enames[i] << ' ' << tsu << std::endl;
        if (!(tsu >= 0 && tsu <= 1000000000000000000ll)) std::cout << "Should crash" << std::endl;
        assert(tsu >= 0 && tsu <= 1000000000000000000ll);
        //if (!dict.contains(enames[i]) || !count_nodes){
        dict[enames[i]] = tsu;
        //}
        //if (!dict.contains(enames[i]+"_th") || !count_nodes){
        dict[enames[i]+"_th"] = su; 
        //}
    }
    if (count_nodes) {
        dict["layers"] = lnames;
        if (!dict.contains("prof_layers")) {
            dict["prof_layers"] = json::array();
        }
        for (int l = 0; l < lnames.size(); l++) {
            if (dict["prof_layers"].size() <= l)
                dict["prof_layers"].push_back(json::object());
        }
        for (int l = 0; l < lnames.size(); l++) {
            for (int e = 0; e < enames.size(); e++) {
                std::vector<int64_t> thv(num_th);
                int64_t su = 0;
                for (int t = 0; t < num_th; t++) {
                    thv[t] = res[l][e][t];
                    su += thv[t];
                }
                assert(su >= 0);
                dict["prof_layers"][l][enames[e]] = su;
                assert(dict["prof_layers"][l][enames[e]] >= 0);
                dict["prof_layers"][l][enames[e]+"_th"] = thv;
            }
        }
        /*for (int i = 0; i < dict["layer_num"]; i++) {
            assert(dict["prof_layers"][i]["PAPI_TOT_CYC"] >= 0);
        }*/
    }
}

void add_node_counters(json& dict) {
    std::cerr << "Storing node counters..." << std::endl;
    std::vector<std::string> lnames, enames;
    int cnt_sz = 0, papi_cnt=0, num_th=0;
    long long* node_time_counter = get_node_time_counter(&cnt_sz, &papi_cnt, &num_th);
    //assert(papi_cnt == model.papi_event_count);
    int name_num, str_sz;
    char* layer_names = get_layer_names(&name_num, &str_sz);
    for (int i = 0; i < name_num; i++)
        lnames.push_back(std::string(&layer_names[i*str_sz]));
    char* event_names = get_papi_events(&name_num, &str_sz);
    enames.push_back(std::string("time_ns"));
    enames.push_back(std::string("papi_cyc"));
    for (int i = 0; i < name_num; i++)
        enames.push_back(std::string(&event_names[i*str_sz]));
    std::vector<std::vector<std::vector<long long>>> res(cnt_sz, std::vector<std::vector<long long>>(papi_cnt, std::vector<long long>(num_th)));
    for (int i = 0; i < cnt_sz; i++)
        for (int j = 0; j < papi_cnt; j++)
            for (int k = 0; k < num_th; k++){
                res[i][j][k] = node_time_counter[i*papi_cnt*num_th+j*num_th+k];
                assert(node_time_counter[i*papi_cnt*num_th+j*num_th+k] >= 0);
                assert(res[i][j][k] >= 0);
            }
    if (count_nodes)
        dict["layer_num"] = cnt_sz;
    dict["thread_num"] = num_th;
    if (!dict.contains("events")) {
        dict["events"] = json::object();
    } else {
        //std::cerr << dict["events"].dump(4) << std::endl;
    }
    PAPI_event_info_t preset_event_info;
    for (auto s : enames) {
        int code = 0;
        PAPI_event_name_to_code(s.data(), &code);
        PAPI_get_event_info(code, &preset_event_info);
        dict["events"][s] = preset_event_info.long_descr;
    }
    dict["events"]["time_ns"] = "Time taken by tensor operations from PAPI_get_real_ns().";    
    dict["events"]["papi_cyc"] = "Cycles taken by tensor operations from PAPI_get_real_cyc().";    
    dict["attr_num"] = dict["events"].size();
    //std::cerr << dict["events"].dump(4) << std::endl;
    for (int i = 0; i < enames.size(); i++) {
        std::vector<long long> su(num_th);
        for (int t = 0; t < num_th; t++) {
            for (int l = 0; l < lnames.size(); l++) {
                su[t] += res[l][i][t];
            }
        }
        long long tsu = 0;
        for (long long x : su)
            tsu += x;
        //if (!dict.contains(enames[i]) || !count_nodes){
        dict[enames[i]] = tsu;
        //}
        //if (!dict.contains(enames[i]+"_th") || !count_nodes){
        dict[enames[i]+"_th"] = su; 
        //}
    }
    if (count_nodes) {
        dict["layers"] = lnames;
        if (!dict.contains("prof_layers")) {
            dict["prof_layers"] = json::array();
        }
        for (int l = 0; l < lnames.size(); l++) {
            if (dict["prof_layers"].size() <= l)
                dict["prof_layers"].push_back(json::object());
        }
        for (int l = 0; l < lnames.size(); l++) {
            for (int e = 0; e < enames.size(); e++) {
                std::vector<int64_t> thv(num_th);
                int64_t su = 0;
                for (int t = 0; t < num_th; t++) {
                    thv[t] = res[l][e][t];
                    su += thv[t];
                }
                assert(su >= 0);
                dict["prof_layers"][l][enames[e]] = su;
                assert(dict["prof_layers"][l][enames[e]] >= 0);
                dict["prof_layers"][l][enames[e]+"_th"] = thv;
            }
        }
        assert(dict["layer_num"] >= lnames.size());
        /*for (int i = 0; i < lnames.size(); i++) {
            if (dict["prof_layers"][i]["PAPI_TOT_CYC"] < 0) {
                std::cerr << i << ' ' << dict["prof_layers"][i]["PAPI_TOT_CYC"] << std::endl;
            }
            assert(dict["prof_layers"][i]["PAPI_TOT_CYC"] >= 0);
        }*/
    }
    if (profile_first_token) {
        json first_tk;
        if (dict.contains("first_token"))
            first_tk = dict["first_token"];
        add_first_token_counters(first_tk);
        dict["first_token"] = first_tk;
    }
}

void output_node_counters(std::string file) {
    if (!store_papi_results) return;
    json dict;
    std::cerr << "Opening file " << file << std::endl;
    try {
        if (std::filesystem::exists(file)) {
            std::ifstream fin(file);
            dict = json::parse(fin);
            //std::cerr << "Dict:\n" << dict.dump(4) << std::endl;
            fin.close();
        } else {
            std::cerr << "File doesn't exist. Creating it..." << std::endl;
        }
    } catch (int errorCode) {
        std::cerr << "Error " << errorCode << " ocurred while parsing json. Creating new json file. All previous data in the json will be lost." << std::endl;
    }
    std::ofstream fout(file);
    add_node_counters(dict);
    fout << dict.dump(4);
    fout.close();
    std::cerr << "Done." << std::endl;
}


int main(int argc, char ** argv) {
    std::map<std::string,std::string> mix_modal_modal_mode_config;
    std::string model_gguf_file_path,embd_file_path, model_id,program_mode,n_tokens, n_gpu_layers, papi_res_dir_path;
    std::vector<std::string> embd_files, res_paths, out_files;
    bool multi_input = false;
    bool multi_input_shuffle = true;
    bool set_metrics = false;
    bool adaptive_tok_limit = false;
    std::vector<std::string> papi_metrics;
    std::vector<int> adaptive_toks;
    bool exhaust_token_limit = false;
    for(int i=1; i<argc; i++) {
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
        else if (strcmp(argv[i],"-papi_res_dir")==0){
            if (i + 1 < argc) {
                papi_res_dir_path = argv[++i];
            } else {
                print_help();
                return 1;
            }
        } else if (strcmp(argv[i],"-multi_input")==0) {
            multi_input = true;
            if (argc <= ++i) {
                print_help();
                return 1;
            }
            int n = atoi(argv[i]);
            embd_files = std::vector<std::string>(n);
            res_paths  = std::vector<std::string>(n);
            out_files  = std::vector<std::string>(n);
            for (int j = 0; j < n; j++) {
                if (argc <= i+3) {
                    print_help();
                    return 1;
                }
                embd_files[j] = argv[++i];
                res_paths[j] = argv[++i];
                out_files[j] = argv[++i];
            }
        } else if (strcmp(argv[i],"-multi_input_no_shuffle")==0) {
            if (strcmp(argv[++i], "true") == 0) multi_input_shuffle = false;
        } else if (strcmp(argv[i], "-node_level_statistics")==0) {
            count_nodes = true;
        } else if (strcmp(argv[i], "-specify_papi_metrics")==0) {
            set_metrics = true;
            if (argc <= ++i) {
                print_help();
                return 1;
            }
            int num_metrics = atoi(argv[i]);
            papi_metrics = std::vector<std::string>(num_metrics);
            for (int j = 0; j < num_metrics; j++) {
                if (argc <= ++i) {
                    print_help();
                    return 1;
                }
                papi_metrics[j] = std::string(argv[i]);
            }
        } else if (strcmp(argv[i], "-calc_matrix_stats")==0) {
            store_papi_results = false;
            matrix_statistics = true;
        } else if (strcmp(argv[i], "-n_threads")==0) {
            mix_modal_modal_mode_config["n_threads"] = argv[++i];
        } else if (strcmp(argv[i], "-adaptive_tok_limit")==0) {
            adaptive_tok_limit = true;
            if (argc <= ++i) {
                print_help();
                return 1;
            }
            int num_queries = atoi(argv[i]);
            adaptive_toks = std::vector<int>(num_queries);
            for (int j = 0; j < num_queries; j++) {
                if (argc <= ++i) {
                    print_help();
                    return 1;
                }
                adaptive_toks[j] = atoi(argv[i]);
            }
            int ma = 0;
            for (int x : adaptive_toks)
                ma = std::max(ma, x);
            char buff[20];
            sprintf(buff, "%d", ma);
            n_tokens = std::string(buff);
        } else if (strcmp(argv[i], "-exhaust_token_limit")==0) {
            exhaust_token_limit = true;
        }
        else{
            print_help();
            return 1;
        }
    }

    if (set_metrics) {
        set_custom_papi_metrics(papi_metrics);
    }

    mix_modal_modal_mode_config["model_gguf_file_path"]=model_gguf_file_path;
    mix_modal_modal_mode_config["n_tokens"]=n_tokens;
    mix_modal_modal_mode_config["n_gpu_layers"]=n_gpu_layers;
    mix_modal_modal_mode_config["model_id"]=model_id;
    if(program_mode == MIX_MODAL_MODE) {
        if (!multi_input){
            mix_modal_modal_mode_config["embd_file_path"]=embd_file_path;
            mix_modal_modal_mode_config["papi_res_dir_path"]=papi_res_dir_path;

            run_mix_modal_model_with_embeddings(mix_modal_modal_mode_config);
        } else {
            int n = embd_files.size();
            std::vector<int> perm(n);
            std::iota(perm.begin(), perm.end(), 0);
            if (multi_input_shuffle){
                std::mt19937 g(std::chrono::steady_clock::now().time_since_epoch().count());
                std::shuffle(perm.begin(), perm.end(), g);
            }
            set_node_main_thread_id(pthread_self());
            if (count_nodes){
                start_node_time_counter();
            }
            int num_runs = 1;
            int num_hw_cnts = PAPI_num_cmp_hwctrs(0);
            if (num_hw_cnts == 0) num_hw_cnts = 1000;
            if (set_metrics && count_nodes) {
                num_runs = (papi_metric_ids.size()+num_hw_cnts-1)/num_hw_cnts;
                num_runs = std::max(num_runs, 1);
                n *= num_runs;
            }
            int idx_cnt = 0;
            for (int run = 0; run < num_runs; run++){
                PreloadedModel model;
                model.load_mix_modal_model(mix_modal_modal_mode_config, embd_files);
                count_stats_in_threads();
                if (exhaust_token_limit)
                    model.prevent_end_tokens = true;
                if (set_metrics) {
                    int maxev = 0;
                    int* id_arr = get_papi_event_codes(&maxev);
                    if (count_nodes) {
                        int str = run*num_hw_cnts;
                        int end = std::min((run+1)*num_hw_cnts, (int) papi_metric_ids.size());
                        memcpy(id_arr, papi_metric_ids.data()+str, sizeof(int)*(end-str));
                    } else {
                        if (papi_metric_ids.size() > maxev) {
                            std::cerr << "The number of papi events to count should be <= " << maxev << std::endl;
                            return 1;
                        }
                        memcpy(id_arr, papi_metric_ids.data(), papi_metric_ids.size()*sizeof(int));
                    }
                }
                for (int i : perm) {
                    FILE* out = freopen(out_files[i].c_str(), "w", stdout);
                    if ((void*)out == (void*)NULL) {
                        std::cerr << "Failed opening stream to file " << out_files[i] << ". Aborting execution.";
                        exit(1);
                    }
                    clear_node_time_counter();
                    if (matrix_statistics)
                        set_node_matrix_statistics(true);
                    if (adaptive_tok_limit) {
                        assert(i < adaptive_toks.size());
                        model.n_predict = adaptive_toks[i];
                    }
                    model.run_model_with_embeddings(embd_files[i], res_paths[i]);
                    output_node_counters(res_paths[i]);
                    output_stream_matrix_stats(std::cout);
                    fclose(out);

                    idx_cnt++;
                    std::cerr << "-----------------------------------------------------------------" << std::endl;
                    std::cerr << "Test " << idx_cnt << " of " << n << ". " << (1.0*idx_cnt/n*100) << "% completed." << std::endl;
                }
                model.free_memory();
            }
        }
    } else {
        print_help();
    }
    return 0;
}
