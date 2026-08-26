"""serving-stack: the FastAPI service (week 2, CPU, tiny model)."""

from __future__ import annotations

import os
import time
import uuid

import torch
from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer

from schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    HealthResponse,
    ModelList,
    ModelCard,
    Choice,
    ResponseMessage,
    Usage,
)

MODEL_ID = os.environ.get(
    "MODEL_ID",
    "Qwen/Qwen2.5-0.5B-Instruct",
)

app = FastAPI(
    title="serving-stack",
    version="wk2",
)

print(f"loading {MODEL_ID} on cpu ...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
)

model.to("cpu")
model.eval()

print("model ready")


# ---------------------------------------------------------
# GET /health
# ---------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model=MODEL_ID,
    )


# ---------------------------------------------------------
# GET /v1/models
# ---------------------------------------------------------

@app.get("/v1/models", response_model=ModelList)
def list_models() -> ModelList:
    return ModelList(
        data=[
            ModelCard(
                id=MODEL_ID,
                created=int(time.time()),
            )
        ]
    )


# ---------------------------------------------------------
# POST /v1/chat/completions
# ---------------------------------------------------------

@app.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
)
def chat_completions(
    req: ChatCompletionRequest,
) -> ChatCompletionResponse:

    input_ids = tokenizer.apply_chat_template(
        [m.model_dump() for m in req.messages],
        add_generation_prompt=True,
        return_tensors="pt",
    )

    prompt_tokens = input_ids.shape[1]

    do_sample = req.temperature > 0

    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=req.max_tokens,
            do_sample=do_sample,
            temperature=(
                req.temperature
                if do_sample
                else None
            ),
        )

    new_tokens = out[0][prompt_tokens:]

    completion_tokens = len(new_tokens)

    text = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    )

    finish_reason = (
        "length"
        if completion_tokens >= req.max_tokens
        else "stop"
    )

    return ChatCompletionResponse(
        id="chatcmpl-" + uuid.uuid4().hex,
        created=int(time.time()),
        model=req.model,
        choices=[
            Choice(
                index=0,
                message=ResponseMessage(
                    role="assistant",
                    content=text,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=(
                prompt_tokens
                + completion_tokens
            ),
        ),
    )