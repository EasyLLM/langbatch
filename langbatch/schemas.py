from typing import Dict, List, Union, Iterable, Optional
from typing_extensions import Literal, override
from pydantic import BaseModel

from openai.types.chat_model import ChatModel
from openai.types.chat import completion_create_params
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_tool_choice_option_param import ChatCompletionToolChoiceOptionParam

class NotGiven:
    def __bool__(self) -> Literal[False]:
        return False

    @override
    def __repr__(self) -> str:
        return "NOT_GIVEN"

NOT_GIVEN = NotGiven()

class OpenAIChatCompletionRequest(BaseModel):
    messages: Iterable[ChatCompletionMessageParam]
    model: Union[str, ChatModel]
    frequency_penalty: Optional[float] | NotGiven = NOT_GIVEN
    function_call: completion_create_params.FunctionCall | NotGiven = NOT_GIVEN
    functions: Iterable[completion_create_params.Function] | NotGiven = NOT_GIVEN
    logit_bias: Optional[Dict[str, int]] | NotGiven = NOT_GIVEN
    logprobs: Optional[bool] | NotGiven = NOT_GIVEN
    max_tokens: Optional[int] | NotGiven = NOT_GIVEN
    n: Optional[int] | NotGiven = NOT_GIVEN
    parallel_tool_calls: bool | NotGiven = NOT_GIVEN
    presence_penalty: Optional[float] | NotGiven = NOT_GIVEN
    response_format: completion_create_params.ResponseFormat | NotGiven = NOT_GIVEN
    seed: Optional[int] | NotGiven = NOT_GIVEN
    service_tier: Optional[Literal["auto", "default"]] | NotGiven = NOT_GIVEN
    stop: Union[Optional[str], List[str]] | NotGiven = NOT_GIVEN
    temperature: Optional[float] | NotGiven = NOT_GIVEN
    tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN
    tools: Iterable[ChatCompletionToolParam] | NotGiven = NOT_GIVEN
    top_logprobs: Optional[int] | NotGiven = NOT_GIVEN
    top_p: Optional[float] | NotGiven = NOT_GIVEN
    user: str | NotGiven = NOT_GIVEN