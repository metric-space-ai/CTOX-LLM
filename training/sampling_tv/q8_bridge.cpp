// Minimal C ABI over the pinned llama.cpp library; logits never go to disk.
#include "llama.h"
#include <cstdint>
#include <new>
struct State { llama_model *model; llama_context *ctx; int32_t pos; };
extern "C" {
void * tv_open(const char *path, int32_t context) {
    llama_backend_init();
    auto mp = llama_model_default_params(); mp.n_gpu_layers = 99;
    auto *model = llama_model_load_from_file(path, mp);
    if (!model) return nullptr;
    auto cp = llama_context_default_params();
    cp.n_ctx = context; cp.n_batch = 128; cp.n_ubatch = 128;
    cp.n_threads = 2; cp.n_threads_batch = 2;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    auto *ctx = llama_init_from_model(model, cp);
    if (!ctx) { llama_model_free(model); return nullptr; }
    return new State{model, ctx, 0};
}
void tv_close(void *ptr) {
    auto *s = static_cast<State *>(ptr);
    if (s) { llama_free(s->ctx); llama_model_free(s->model); delete s; }
    llama_backend_free();
}
void tv_reset(void *ptr) {
    auto *s = static_cast<State *>(ptr);
    llama_memory_clear(llama_get_memory(s->ctx), true); s->pos = 0;
}
int32_t tv_vocab(void *ptr) {
    return llama_vocab_n_tokens(llama_model_get_vocab(static_cast<State *>(ptr)->model));
}
int32_t tv_tokenize(void *ptr, const char *text, int32_t len, int32_t *out, int32_t cap) {
    return llama_tokenize(llama_model_get_vocab(static_cast<State *>(ptr)->model), text, len, out, cap, false, true);
}
int32_t tv_piece(void *ptr, int32_t token, char *out, int32_t cap) {
    return llama_token_to_piece(llama_model_get_vocab(static_cast<State *>(ptr)->model), token, out, cap, 0, true);
}
bool tv_eog(void *ptr, int32_t token) {
    return llama_vocab_is_eog(llama_model_get_vocab(static_cast<State *>(ptr)->model), token);
}
int32_t tv_eval(void *ptr, const int32_t *tokens, int32_t count) {
    auto *s = static_cast<State *>(ptr);
    for (int32_t offset = 0; offset < count; offset += 128) {
        int32_t n = count - offset; if (n > 128) n = 128;
        auto batch = llama_batch_init(n, 0, 1); batch.n_tokens = n;
        for (int32_t i = 0; i < n; ++i) {
            batch.token[i] = tokens[offset+i]; batch.pos[i] = s->pos+i;
            batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0;
            batch.logits[i] = i == n-1;
        }
        int32_t result = llama_decode(s->ctx, batch); llama_batch_free(batch);
        if (result != 0) return result;
        s->pos += n;
    }
    return 0;
}
const float *tv_logits(void *ptr) {
    return llama_get_logits_ith(static_cast<State *>(ptr)->ctx, -1);
}
}
