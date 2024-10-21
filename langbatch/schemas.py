from typing import Dict, List, Union, Iterable, Optional
from typing_extensions import Literal, override
from pydantic import BaseModel

from openai.types.chat_model import ChatModel
from openai.types.chat import completion_create_params
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_tool_choice_option_param import ChatCompletionToolChoiceOptionParam

class OpenAIChatCompletionRequest(BaseModel):
    messages: Iterable[ChatCompletionMessageParam]
    model: Union[str, ChatModel]
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, int]] = None
    logprobs: Optional[bool] = None
    max_tokens: Optional[int] = None
    n: Optional[int] = None
    parallel_tool_calls: Optional[bool] = None
    presence_penalty: Optional[float] = None
    response_format: Optional[completion_create_params.ResponseFormat] = None
    seed: Optional[int] = None
    service_tier: Optional[Literal["auto", "default"]] = None
    stop: Optional[Union[str, List[str]]] = None
    temperature: Optional[float] = None
    tool_choice: Optional[ChatCompletionToolChoiceOptionParam] = None
    tools: Optional[Iterable[ChatCompletionToolParam]] = None
    top_logprobs: Optional[int] = None
    top_p: Optional[float] = None
    user: Optional[str] = None

class OpenAIEmbeddingRequest(BaseModel):
    input: Union[str, List[str], Iterable[int], Iterable[Iterable[int]]]
    model: Union[str, Literal["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"]]
    dimensions: Optional[int] = None
    encoding_format: Optional[Union[str, Literal["float", "base64"]]] = None
    user: Optional[str] = None

class VertexAIChatCompletionRequest(BaseModel):
    messages: Iterable[ChatCompletionMessageParam]
    model: Union[str, ChatModel]
    frequency_penalty: Optional[float] = None
    max_tokens: Optional[int] = None
    n: Optional[int] = None
    presence_penalty: Optional[float] = None
    response_format: Optional[completion_create_params.ResponseFormat] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    temperature: Optional[float] = None
    tools: Optional[Iterable[ChatCompletionToolParam]] = None
    top_p: Optional[float] = None

class AnthropicChatCompletionRequest(BaseModel):
    messages: Iterable[ChatCompletionMessageParam]
    model: Union[str, ChatModel]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None
    tool_choice: Optional[ChatCompletionToolChoiceOptionParam] = None
    tools: Optional[Iterable[ChatCompletionToolParam]] = None
    parallel_tool_calls: Optional[bool] = None
    user: Optional[str] = None