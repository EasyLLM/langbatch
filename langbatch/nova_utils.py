from typing import Any, Dict, List, Optional
import time
from langbatch.schemas import AnthropicChatCompletionRequest

def convert_content_nova(content: Any) -> List[Dict[str, Any]]:
    if isinstance(content, str):
        return [{"text": content}]
    elif isinstance(content, list):
        converted_content = []
        for item in content:
            if isinstance(item, str):
                converted_content.append({"text": item})
            elif isinstance(item, dict):
                if item["type"] == "text":
                    converted_content.append({"text": item["text"]})
                elif item["type"] == "image_url":
                    # TODO: Handle image conversion from URL to bytes/base64
                    pass
        return converted_content
    return []

def convert_tools_nova(tools: Optional[List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    if not tools:
        return None
    
    converted_tools = []
    for tool in tools:
        if tool["type"] == "function":
            converted_tool = {
                "toolSpec": {
                    "name": tool["function"]["name"],
                    "description": tool["function"].get("description", ""),
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": tool["function"]["parameters"].get("properties", {}),
                            "required": tool["function"]["parameters"].get("required", [])
                        }
                    }
                }
            }
            converted_tools.append(converted_tool)
    
    return {"tools": converted_tools, "toolChoice": {"auto": {}}} if converted_tools else None

def convert_request_nova(req: dict) -> dict:
    request = AnthropicChatCompletionRequest(**req["body"])
    
    messages = []
    system_content = None
    
    for message in request.messages:
        if message["role"] == "system":
            system_content = convert_content_nova(message["content"])
        else:
            messages.append({
                "role": message["role"],
                "content": convert_content_nova(message["content"])
            })
    
    req = {
        "messages": messages
    }
    
    if system_content:
        req["system"] = system_content

    # Convert inference config
    inference_config = {}
    if request.max_tokens:
        inference_config["max_new_tokens"] = request.max_tokens
    if request.temperature:
        inference_config["temperature"] = request.temperature
    if request.top_p:
        inference_config["top_p"] = request.top_p
    if request.stop:
        inference_config["stopSequences"] = request.stop
    
    if inference_config:
        req["inferenceConfig"] = inference_config
        
    # Convert tools
    if request.tools:
        tool_config = convert_tools_nova(request.tools)
        if tool_config:
            req["toolConfig"] = tool_config
            
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
            if 'toolUse' in item:
                tool_calls.append({
                    'type': 'function',
                    'id': item['toolUse']['toolUseId'],
                    'function': {
                        'name': item['toolUse']['name'],
                        'arguments': item['toolUse']['input']
                    }
                })
            elif 'text' in item:
                content = item['text']
        
        message = {
            'role': message['role'],
            'content': content,
        }
        if len(tool_calls) > 0:
            message['tool_calls'] = tool_calls
        return message

def convert_message(message, custom_id, model) -> dict:
    choice = {
        'index': 0,
        'logprobs': None,
        'finish_reason': message['stopReason'].lower(),
        'message': convert_response_message(message['output']['message'])
    }
    choices = [choice]
    usage = {
        'prompt_tokens': message['usage']['inputTokens'],
        'completion_tokens': message['usage']['outputTokens'],
        'total_tokens': message['usage']['totalTokens']
    }
    body = {
        'id': custom_id,
        'object': 'chat.completion',
        'created': int(time.time()),
        'model': model,
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

def convert_response_nova(response, model) -> dict:
    message = response['modelOutput']
    custom_id = response['recordId']
    res = convert_message(message, custom_id, model)
    error = None
        
    output = {
        'id': custom_id,
        'custom_id': custom_id,
        'response': res,
        'error': error
    }
    return output