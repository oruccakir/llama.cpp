
#include "llama.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <chrono>
#include <algorithm>
#include <random>
#include "ggml-cpu.h"

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


class PreloadedModel {

    llama_sampler * smpl;
    llama_model * model;
    int n_embd, n_predict, ngl;
    const llama_vocab * vocab;
    std::string model_id, model_path;

public:
    int load_mix_modal_model_with_embeddings(std::map<std::string, std::string> config){
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
        std::cerr << "Nembd: " << n_embd << std::endl;
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

        if (model == NULL) {
            fprintf(stderr , "%s: error: unable to load model\n" , __func__);
            return 1;
        }


        auto sparams = llama_sampler_chain_default_params();
        sparams.no_perf = false;
        smpl = llama_sampler_chain_init(sparams);

        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
  
        return 0;
    }

    int run_model (std::string embd_file_path, std::string papi_results_save_file_path) {
            
        std::cerr << "BEGIN" << std::endl;
        std::vector<float> input_embeddings = load_input_embeddings(embd_file_path);
        
        const int n_tokens = input_embeddings.size() / n_embd;
        //std::cerr << "N_TOKENS: " << n_tokens << std::endl;
        const int n_prompt = n_tokens;


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

        std::cerr << "END" << std::endl;

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
            std::cerr << "N_POS: " << n_pos << std::endl;
            if (llama_decode(ctx, batch)) {
                fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
                return 1;
            }

            std::cerr << "N_TOKENS " << batch.n_tokens << std::endl;
            n_pos += batch.n_tokens;

            {
                new_token_id = llama_sampler_sample(smpl, ctx, -1);
                std::cerr << "TOK_ID " << new_token_id << std::endl;
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
                std::cerr << "s: " << s << std::endl;
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

        llama_free(ctx);

        #ifdef ENABLE_PAPI
            save_papi_events_to_json_file(papi_results_save_file_path,values,preset_event_count,presetEventCodes,papi_end_cycles - papi_start_cycles,papi_end_usec - papi_start_usec);
        #endif

        return 0;
    }

    void free_memory() {
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




int main(int argc, char ** argv) {
    std::map<std::string,std::string> mix_modal_modal_mode_config;
    std::string model_gguf_file_path,embd_file_path, model_id,program_mode,n_tokens, n_gpu_layers, papi_res_dir_path;
    std::vector<std::string> embd_files, res_paths, out_files;
    bool multi_input = false;
    bool multi_input_shuffle = true;
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
            int n = atoi(argv[++i]);
            embd_files = std::vector<std::string>(n);
            res_paths  = std::vector<std::string>(n);
            out_files  = std::vector<std::string>(n);
            for (int j = 0; j < n; j++) {
                embd_files[j] = argv[++i];
                res_paths[j] = argv[++i];
                out_files[j] = argv[++i];
            }
        } else if (strcmp(argv[i],"-multi_input_no_shuffle")==0) {
            if (strcmp(argv[++i], "true") == 0) multi_input_shuffle = false;
        }
        else{
            print_help();
            return 1;
        }
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
            PreloadedModel model;
            model.load_mix_modal_model_with_embeddings(mix_modal_modal_mode_config);
            int n = embd_files.size();
            std::vector<int> perm(n);
            std::iota(perm.begin(), perm.end(), 0);
            if (multi_input_shuffle){
                std::mt19937 g(std::chrono::steady_clock::now().time_since_epoch().count());
                std::shuffle(perm.begin(), perm.end(), g);
            }
            for (int i : perm) {
                std::cerr << "-----------------------------------------------------------------" << std::endl;
                std::cerr << "Test " << i << " of " << n << ". " << (1.0*i/n*100) << "% completed." << std::endl;
                FILE* out = freopen(out_files[i].c_str(), "w", stdout);
                if ((void*)out == (void*)NULL) {
                    std::cerr << "Failed opening stream to file " << out_files[i] << ". Aborting execution.";
                    exit(1);
                }
                model.run_model(embd_files[i], res_paths[i]);
                std::cerr <<  "Node time counters:";
                /*int cnt_sz = 0;
                long long* node_time_counter = get_node_time_counter(&cnt_sz);
                for (int i = 0; i < cnt_sz; i++) {
                    std::cerr << ' ' << node_time_counter[i];
                }
                std::cerr << std::endl;*/
                fclose(out);
                //std::cerr << "FINISHING SINGLE TEST" << std::endl;
                //break;
            }
            model.free_memory();
        }
    } else {
        print_help();
    }
    return 0;
}
