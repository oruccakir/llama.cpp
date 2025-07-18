Where the things happen:<br>
examples/main/main.cpp -> model.run_model() -><br>
src/llama.cpp-> llama_decode() -> llama_decode_impl() -> llama_graph_compute() -><br>
ggml/src/ggml-backend.cpp -> ggml_backend_sched_graph_compute_async() -> ggml_backend_sched_compute_splits() -> ggml_backend_graph_compute_async() -><br>
??? -> backend->iface.graph_compute -><br>
??? -> ... -> ??? -><br>
ggml/src/ggml-cpu/ggml-cpu.c -> ggml_graph_compute_thread() -> [Iterate over graph nodes, for each] -> ggml_compute_forward()