#!/usr/bin/env bash
# Re-tokenize the exact raw HTTP prompts with the external llama.cpp vocabulary.
set -euo pipefail
if [[ $# != 3 ]]; then
    echo "usage: bash check-context-ids.sh LLAMA_TOKENIZE QWEN_Q8_GGUF NEW_DIRECTORY" >&2
    exit 2
fi
tokenizer=$(realpath "$1")
model=$(realpath "$2")
out=$(realpath -m "$3")
manifest=$(dirname "$(realpath "$0")")/performance-contexts.json
mkdir "$out"
cd "$out"
export CUDA_VISIBLE_DEVICES=-1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
model_hash=4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d
echo "$model_hash  $model" | sha256sum -c - > model-check.txt
ldd "$tokenizer" > libraries.txt
sha256sum "$tokenizer" "$manifest" > tools.sha256
awk '/=> \/.*lib(llama|ggml)/ { print $3 }' libraries.txt | xargs -r sha256sum >> tools.sha256
for name in short medium long; do
    jq -ej --arg name "$name" '.cases[] | select(.name==$name) | .request.prompt' "$manifest" > "$name.prompt"
    prompt_hash=$(jq -er --arg name "$name" '.cases[] | select(.name==$name) | .prompt_sha256' "$manifest")
    echo "$prompt_hash  $name.prompt" | sha256sum -c -
    "$tokenizer" --ids --no-bos --no-escape -m "$model" -f "$name.prompt" > "$name.ids.json" 2> "$name.log"
done
jq -n --slurpfile manifest "$manifest" --slurpfile short short.ids.json \
    --slurpfile medium medium.ids.json --slurpfile long long.ids.json '
    {short:$short,medium:$medium,long:$long} as $observed |
    [$manifest[0].cases[] | . as $case | $observed[.name] as $file |
      {name,expected:.prompt_token_count,observed:($file[0]|length),
       exact:(($file|length)==1 and ($file[0]|type)=="array" and
              $case.prompt_ids==$file[0] and ($case.prompt_ids|length)==$case.prompt_token_count)}]
    ' > comparison.json
cat comparison.json
jq -e 'map(.name)==["short","medium","long"] and all(.[]; .exact)' comparison.json > /dev/null
