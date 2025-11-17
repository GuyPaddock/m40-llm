// cuda/common.h
#pragma once
#include <stdint.h>

#define MAX_TOKENS 4096

// Command types
enum DecodeCommandType : uint32_t {
    DECODE_CMD_NONE = 0,
    DECODE_CMD_PREFILL = 1,    // for initial prompt
    DECODE_CMD_DECODE = 2,     // normal decode step
    DECODE_CMD_STOP = 3        // stop kernel
};

// Single request (sequence)
struct DecodeCommand {
    uint32_t cmd;         // DecodeCommandType
    uint32_t seq_id;      // unique sequence ID
    uint32_t input_len;   // number of input tokens (prefill)
    uint32_t max_new;     // number of tokens to decode
    uint32_t reserved;

    // For PREFILL: tokens
    // For DECODE: last-token-only
    uint32_t tokens[MAX_TOKENS];
};

// GPU writes output tokens here
struct DecodeResult {
    uint32_t seq_id;
    uint32_t token;       // sampled token
    uint32_t done;        // 1 if sequence terminated
};

template<typename T, int Capacity>
struct RingBuffer {
    // Host writes head, GPU writes tail
    volatile uint32_t head; // next slot to write
    volatile uint32_t tail; // next slot to read
    T buffer[Capacity];
};