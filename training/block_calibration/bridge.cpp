// Diagnostic wrapper over the exact same Q8 bridge/library as the TV benchmark.
#include "../sampling_tv/q8_bridge.cpp"
// These diagnostic APIs belong to the pinned llama.cpp build, with C++ linkage.
void llama_set_embeddings_layer_inp(llama_context *, uint32_t, bool);
float * llama_get_embeddings_layer_inp(llama_context *, uint32_t);
void llama_set_embeddings_nextn(llama_context *, bool, bool);
float * llama_get_embeddings_nextn_ith(llama_context *, int32_t);
extern "C" {
void block_enable(void *ptr) {
    auto *s = static_cast<State *>(ptr);
    for (uint32_t i=0;i<64;++i) llama_set_embeddings_layer_inp(s->ctx,i,true);
    llama_set_embeddings_nextn(s->ctx,true,false);
}
const float *block_hidden(void *ptr,int32_t layer,int32_t row) {
    auto *s=static_cast<State *>(ptr);
    if (layer==64) return llama_get_embeddings_nextn_ith(s->ctx,row);
    auto *data=llama_get_embeddings_layer_inp(s->ctx,static_cast<uint32_t>(layer));
    return data?data+row*5120:nullptr;
}
}
