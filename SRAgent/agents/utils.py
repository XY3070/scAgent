import os
import re
import sys
import asyncio
import os
from functools import wraps
from importlib import resources
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dynaconf import Dynaconf

import openai

def load_settings() -> Dict[str, Any]:
    """
    Load settings from settings.yml file
    
    Returns:
        Dictionary containing settings for the specified environment
    """
    # get path to settings
    # 获取当前文件所在的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 向上两级目录到达项目根目录 (SRAgent/SRAgent/agents/utils.py -> SRAgent/SRAgent -> SRAgent)
    project_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
    # 构建 settings.yml 的绝对路径
    s_path = os.path.join(project_root, "SRAgent", "settings.yml")
    if not os.path.exists(s_path):
        raise FileNotFoundError(f"Settings file not found: {s_path}")
    # Determine the environment to load
    current_env = os.getenv("DYNACONF_ENV", "default") # Use DYNACONF_ENV as the environment variable for Dynaconf

    settings = Dynaconf(
        settings_files=[s_path], 
        environments=True, 
        env_switcher="DYNACONF_ENV", # Use DYNACONF_ENV as the switcher
        current_env=current_env # Set the current environment
    )
    return settings

def async_retry_on_flex_timeout(func):
    """
    Async decorator to retry with default tier if flex tier times out.
    """
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        # Check if we're using flex tier
        service_tier = getattr(self, '_service_tier', None)
        model_name = getattr(self, 'model_name', None)
        
        if service_tier != "flex":
            # Not using flex tier, just call the function normally
            return await func(self, *args, **kwargs)
        
        try:
            # Try with flex tier first
            return await func(self, *args, **kwargs)
        except (asyncio.TimeoutError, openai.APITimeoutError, openai.APIConnectionError) as e:
            print(f"Flex tier timeout for model {model_name}, retrying with standard tier...", file=sys.stderr)
            
            # Create a new instance with default tier
            if hasattr(self, '_fallback_model'):
                # Use pre-created fallback model if available
                fallback_model = self._fallback_model
            else:
                # Create fallback model on the fly
                fallback_kwargs = {
                    "model_name": self.model_name,
                    "temperature": getattr(self, 'temperature', None),
                    "max_tokens": getattr(self, 'max_tokens', None),
                }
                # Add reasoning_effort if it's an o-model
                if hasattr(self, 'reasoning_effort'):
                    fallback_kwargs["reasoning_effort"] = self.reasoning_effort
                    fallback_kwargs["temperature"] = None
                fallback_model = ChatOpenAI(**fallback_kwargs)
            
            # Retry with default tier
            return await fallback_model.ainvoke(*args, **kwargs)
        except Exception as e:
            # For other exceptions, just raise them
            raise
    
    return wrapper

def sync_retry_on_flex_timeout(func):
    """
    Sync decorator to retry with default tier if flex tier times out.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Check if we're using flex tier
        service_tier = getattr(self, '_service_tier', None)
        model_name = getattr(self, 'model_name', None)
        
        if service_tier != "flex":
            # Not using flex tier, just call the function normally
            return func(self, *args, **kwargs)
        
        try:
            # Try with flex tier first
            return func(self, *args, **kwargs)
        except (openai.APITimeoutError, openai.APIConnectionError) as e:
            print(f"Flex tier timeout for model {model_name}, retrying with standard tier...", file=sys.stderr)
            
            # Create a new instance with default tier
            if hasattr(self, '_fallback_model'):
                # Use pre-created fallback model if available
                fallback_model = self._fallback_model
            else:
                # Create fallback model on the fly
                fallback_kwargs = {
                    "model_name": self.model_name,
                    "temperature": getattr(self, 'temperature', None),
                    "max_tokens": getattr(self, 'max_tokens', None),
                }
                # Add reasoning_effort if it's an o-model
                if hasattr(self, 'reasoning_effort'):
                    fallback_kwargs["reasoning_effort"] = self.reasoning_effort
                    fallback_kwargs["temperature"] = None
                fallback_model = ChatOpenAI(**fallback_kwargs)
            
            # Retry with default tier
            return fallback_model.invoke(*args, **kwargs)
        except Exception as e:
            # For other exceptions, just raise them
            raise
    
    return wrapper

class FlexTierChatOpenAI(ChatOpenAI):
    """
    Extended ChatOpenAI that supports automatic fallback from flex to default tier.
    """
    def __init__(self, *args, service_tier: Optional[str] = None, openai_api_base: Optional[str] = None, openai_api_key: Optional[str] = None, **kwargs):
        super().__init__(*args, openai_api_base=openai_api_base, openai_api_key=openai_api_key, **kwargs)
        self._service_tier = service_tier
        
        # Create fallback model if using flex tier
        if service_tier == "flex":
            fallback_kwargs = kwargs.copy()
            fallback_kwargs.pop('service_tier', None)
            fallback_kwargs.pop('timeout', None)
            self._fallback_model = ChatOpenAI(openai_api_base=openai_api_base, openai_api_key=openai_api_key, **fallback_kwargs)
    
    @async_retry_on_flex_timeout
    async def ainvoke(self, *args, **kwargs):
        return await super().ainvoke(*args, **kwargs)
    
    @sync_retry_on_flex_timeout
    def invoke(self, *args, **kwargs):
        return super().invoke(*args, **kwargs)

def set_model(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    reasoning_effort: Optional[str] = None,
    agent_name: str = "default",
    max_tokens: Optional[int] = None,
    service_tier: Optional[str] = None,
) -> Any:
    """
    Create a model instance with settings from configuration
    Args:
        model_name: Override model name from settings
        temperature: Override temperature from settings
        reasoning_effort: Override reasoning effort from settings
        agent_name: Name of the agent to get settings for
        max_tokens: Maximum number of tokens to use for the model
        service_tier: Service tier to use for the model
    Returns:
        Configured model instance
    """
    # Load settings
    settings = load_settings()
    
    # Use provided params or get from settings
    if model_name is None:
        try:
            model_name = settings["models"][agent_name]
        except KeyError:
            # try default
            try:
                model_name = settings["models"]["default"]
            except KeyError:
                raise ValueError(f"No model name was provided for agent '{agent_name}'")
    if temperature is None:
        try:
            temperature = settings["temperature"][agent_name]
        except KeyError:
            try:
                temperature = settings["temperature"]["default"]
            except KeyError:
                raise ValueError(f"No temperature was provided for agent '{agent_name}'")
    if reasoning_effort is None:
        try:
            reasoning_effort = settings["reasoning_effort"][agent_name]
        except KeyError:
            try:
                reasoning_effort = settings["reasoning_effort"]["default"]
            except KeyError:
                if temperature is None:
                    raise ValueError(f"No reasoning effort or temperature was provided for agent '{agent_name}'")
    if max_tokens is None:
        try:
            max_tokens = settings["max_tokens"][agent_name]
        except KeyError:
            try:
                max_tokens = settings["max_tokens"]["default"]
            except KeyError:
                pass # No max_tokens provided
    if service_tier is None:
        try:
            service_tier = settings["service_tier"][agent_name]
        except (KeyError, TypeError):
            try:
                service_tier = settings["service_tier"]["default"]
            except (KeyError, TypeError):
                try:
                    service_tier = settings["service_tier"]
                except (KeyError, TypeError):
                    service_tier = "default"  # fallback to default service tier

    # Get timeout from settings (optional)
    timeout = None
    try:
        timeout = settings["flex_timeout"][agent_name]
    except (KeyError, TypeError):
        try:
            timeout = settings["flex_timeout"]["default"]
        except (KeyError, TypeError):
            try:
                timeout = settings["flex_timeout"]
            except (KeyError, TypeError):
                timeout = 180.0  # Default value

    # Validate service_tier for OpenAI models
    if service_tier == "flex" and not re.search(r"^o[0-9]", model_name):
        raise ValueError(f"Service tier 'flex' only works with o3 and o4-mini models, not {model_name} (agent: {agent_name})")

    # Check model provider and initialize appropriate model
    if model_name.startswith("claude"): # e.g.,  "claude-3-7-sonnet-20250219"
        if reasoning_effort == "low":
            think_tokens = 1024
        elif reasoning_effort == "medium":
            think_tokens = 1024 * 2
        elif reasoning_effort == "high":
            think_tokens = 1024 * 4
        else:
            think_tokens = 0
        if think_tokens > 0:
            if not max_tokens:
                max_tokens = 1024
            max_tokens += think_tokens
            thinking = {"type": "enabled", "budget_tokens": think_tokens}
            temperature = None
        else:
            thinking = {"type": "disabled"}
            if temperature is None:
                raise ValueError(f"Temperature is required for Claude models if reasoning_effort is not set")
        if not max_tokens:
            max_tokens = 1024
        model = ChatAnthropic(model=model_name, temperature=temperature, thinking=thinking, max_tokens=max_tokens)
    elif model_name.startswith("gpt-4"):
        # GPT-4o models use temperature but not reasoning_effort
        # Use FlexTierChatOpenAI for automatic fallback support
        model = FlexTierChatOpenAI(
            model_name=model_name, 
            temperature=temperature, 
            reasoning_effort=None, 
            max_tokens=max_tokens, 
            service_tier=service_tier,
            timeout=timeout if service_tier == "flex" else None,
            openai_api_base=settings.get("qwen_api_base"),
            openai_api_key=settings.get("qwen_api_key")
        )
    elif re.search(r"^o[0-9]", model_name):
        # o[0-9] models use reasoning_effort but not temperature
        # Use FlexTierChatOpenAI for automatic fallback support
        model = FlexTierChatOpenAI(
            model_name=model_name, 
            temperature=None, 
            reasoning_effort=reasoning_effort, 
            max_tokens=max_tokens, 
            service_tier=service_tier,
            timeout=timeout if service_tier == "flex" else None,
            openai_api_base=settings.get("qwen_api_base"),
            openai_api_key=settings.get("qwen_api_key")
        )
    elif model_name.startswith("Qwen"):
        # 使用OpenAI兼容接口调用本地部署的QWEN模型
        # 创建自定义请求头
        default_headers = {}
        if reasoning_effort is not None:
            default_headers["X-Qwen-Enable-Thinking"] = "true"
        else:
            default_headers["X-Qwen-Enable-Thinking"] = "false"
            
        model = ChatOpenAI(
            model_name=model_name,
            openai_api_base=settings["qwen_api_base"],
            openai_api_key=settings["qwen_api_key"],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.8,
            presence_penalty=1.5,
            default_headers=default_headers
        )
    else:
        raise ValueError(f"Model {model_name} not supported")

    return model


# main
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(override=True)

    # load settings
    settings = load_settings()
    print(settings)

    # set model
    model = set_model(model_name="Qwen3-235B-A22B", agent_name="default")
    print(model)