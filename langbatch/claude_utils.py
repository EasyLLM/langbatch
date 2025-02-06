from typing import Any, Dict, List, Optional
from langbatch.utils import get_web_image
from langbatch.schemas import AnthropicChatCompletionRequest
import time

def convert_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    converted_messages = []
    for message in messages:
        if message["role"] == "assistant" and message["tool_calls"]:
            converted_tool_calls = []
            for tool_call in message["tool_calls"]:
                converted_tool_call = {
                    "type": "tool_use",
                    "id": tool_call["id"],
                    "name": tool_call["function"]["name"],
                    "input": tool_call["function"]["arguments"]
                }
                converted_tool_calls.append(converted_tool_call)
            converted_message = {"role": "assistant", "content": [converted_tool_call]}
        elif message["role"] == "tool":
            converted_message = {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": message["tool_call_id"],
                        "content": message["content"]
                    }
                ]
            }
        else:
            converted_message = {
                "role": message["role"],
                "content": convert_content(message["content"])
            }
        converted_messages.append(converted_message)
    return converted_messages

def convert_content(content: Any) -> List[Dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    elif isinstance(content, list):
        converted_content = []
        for item in content:
            if isinstance(item, str):
                converted_content.append({"type": "text", "text": item})
            elif isinstance(item, dict):
                if item["type"] == "text":
                    converted_content.append(item)
                elif item["type"] == "image_url":
                    image_url = item["image_url"]["url"]
                    if image_url.startswith("data:"):
                        image_media_type = image_url.split(";")[0].split(":")[-1]
                        image_data = image_url.split(",")[1]
                    else:
                        image_media_type, image_data = get_web_image(image_url)

                    converted_content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image_media_type,
                            "data": image_data
                        }
                    })

        return converted_content
    return []

def convert_tools(tools: Optional[List[Dict[str, Any]]]):
    if not tools:
        return None
    
    converted_tools = []
    for tool in tools:
        if tool["type"] == "function":
            converted_tool = {
                "name": tool["function"]["name"],
                "input_schema": tool["function"]["parameters"]
            }
            if tool["function"]["description"]:
                converted_tool["description"] = tool["function"]["description"]
            converted_tools.append(converted_tool)
    return converted_tools

def convert_tool_choice(tools_given: bool, tool_choice: Optional[Dict[str, Any]], parallel_tool_calls: Optional[bool]):
    tool_choice_obj = None
    if tool_choice is None and tools_given:
        tool_choice_obj = {"type": "auto"}
    
    if isinstance(tool_choice, str):
        match tool_choice:
            case "auto":
                tool_choice_obj = {"type": "auto"}
            case "required":
                tool_choice_obj = {"type": "any"}
            case "none":
                tool_choice_obj = {"type": "auto"} if tools_given else None
    elif isinstance(tool_choice, dict):
        if tool_choice["type"] == "function":
            return {"type": "tool", "name": tool_choice["function"]["name"]}
    
    # Handle parallel_tool_calls
    if parallel_tool_calls and tool_choice_obj:
        tool_choice_obj["disable_parallel_tool_use"] = parallel_tool_calls
    
    return tool_choice_obj

def convert_request(req: dict):
    request = AnthropicChatCompletionRequest(**req["body"])

    messages = []
    system = None
    for message in request.messages:
        if message["role"] == "system":
            if isinstance(message["content"], str):
                system = message["content"]
            elif isinstance(message["content"], dict):
                try:
                    system = message["content"]["text"]
                except KeyError:
                    pass
        else:
            messages.append(message)

    messages = convert_messages(messages)

    req = {
        "model": request.model,
        "messages": messages,
    }
    if system:
        req["system"] = system

    if request.max_tokens:
        req["max_tokens"] = request.max_tokens
    else:
        req["max_tokens"] = 1000
    if request.temperature:
        req["temperature"] = request.temperature
    if request.top_p:
        req["top_p"] = request.top_p
    if request.stop:
        req["stop_sequences"] = request.stop
    if request.tools:
        tools = convert_tools(request.tools)
        tool_choice = convert_tool_choice(tools is not None, request.tool_choice, request.parallel_tool_calls)
        req["tools"] = tools
        req["tool_choice"] = tool_choice

    return req

def convert_response_message(message):
    if isinstance(message['content'], str):
        return {
            'role': message['role'],
            'content': message['content']
        }
    elif isinstance(message['content'], list):
        tool_calls = []
        content = None
        for item in message['content']:
            if item['type'] == 'tool_use':
                tool_calls.append({
                    'type': 'function',
                    'id': item['id'],
                    'function': {
                        'name': item['name'],
                        'arguments': item['input']
                    }
                })
            else:
                content = item
        
        message = {
            'role': message['role'],
            'content': content,
        }
        if len(tool_calls) > 0:
            message['tool_calls'] = tool_calls
        return message

def convert_message(message, custom_id) -> dict:
    choice = {
        'index': 0,
        'logprobs': None,
        'finish_reason': message['stop_reason'].lower(),
        'message': convert_response_message(message)
    }
    choices = [choice]
    usage = {
        'prompt_tokens': message['usage']['input_tokens'],
        'completion_tokens': message['usage']['output_tokens'],
        'total_tokens': message['usage']['input_tokens'] + message['usage']['output_tokens']
    }
    body = {
        'id': message['id'],
        'object': 'chat.completion',
        'created': int(time.time()),
        'model': message['model'],
        'system_fingerprint': None,
        'choices': choices,
        'usage': usage
    }
    res = {
        'request_id': custom_id,
        'status_code': 200,
        'body': body,
    }
    return res

def convert_response(response) -> dict:
    if response['result']['type'] == 'succeeded':
        message = response['result']['message']
        res = convert_message(message, response['custom_id'])
        error = None 
    elif response['result']['type'] == 'errored':
        error = {
            'message': response['result']['error']['type'],
            'code': response['result']['error']['type']
        }
        res = None
    elif response['result']['type'] == 'expired':
        error = {
            'message': 'Request expired',
            'code': 'request_expired'
        }
        res = None
        
    output = {
        'id': f'{response["custom_id"]}',
        'custom_id': response['custom_id'],
        'response': res,
        'error': error
    }
    return output