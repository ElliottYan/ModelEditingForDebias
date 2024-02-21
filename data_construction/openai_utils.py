import openai
import time
import os

if "OPENAI_API_KEY" not in os.environ:
    openai.api_key = "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"
else:
    openai.api_key = os.environ['OPENAI_API_KEY']

def chatgpt(messages):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-1106",
        messages = messages,
        temperature=0,
        max_tokens=4096,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None)
    return  response

def gpt_fun(messages):
    time.sleep(0.5)
    try:
        response = chatgpt(messages)
    except openai.error.APIError as e:
        print(f"OpenAI API returned an API Error: {e}")
        response = "ERROR"
        pass
    except openai.error.APIConnectionError as e:
        print(f"Failed to connect to OpenAI API: {e}")
        time.sleep(30)
        response = chatgpt(messages)
        pass
    except openai.error.RateLimitError as e:
        print(f"OpenAI API request exceeded rate limit: {e}")
        time.sleep(30)
        response = chatgpt(messages)
        pass
    except Exception as e: 
        print(e)
        response = "ERROR"
        pass
    if response =="ERROR":
        response = "CONTENT_FILTER."
    else:
        if response["choices"][0]["finish_reason"]=="content_filter":
            response = "CONTENT_FILTER."
        else:
            response = response["choices"][0]["message"]["content"]    
    return response
