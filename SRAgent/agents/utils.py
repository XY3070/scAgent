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
from SRAgent.config import settings as app_settings

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
    settings: Optional[dict] = None,
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
    print(f"Loaded settings: {settings.as_dict()}")

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
                raise ValueError(f"No reasoning effort was provided for agent '{agent_name}'")

    # Get API base URLs
    openai_api_base = settings.get("openai_api_base", os.getenv("OPENAI_API_BASE", app_settings.MODEL_API_URL))
    qwen_api_base = settings.get("qwen_api_base", os.getenv("QWEN_API_BASE"))

    # Set API key and base URL from settings or environment variables
    openai_api_key = settings.get("openai_api_key", os.getenv("OPENAI_API_KEY"))
    anthropic_api_key = settings.get("anthropic_api_key", os.getenv("ANTHROPIC_API_KEY"))
    
    # For Qwen models, prioritize qwen_api_key from settings or environment
    if "Qwen" in model_name:
        openai_api_key = settings.get("qwen_api_key", os.getenv("QWEN_API_KEY", openai_api_key))


    # Determine the model class and arguments
    if "Qwen" in model_name:
        model_class = FlexTierChatOpenAI
        model_kwargs = {
            "model_name": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "openai_api_base": qwen_api_base,  # Pass the Qwen API base URL
            "openai_api_key": openai_api_key,
            "service_tier": service_tier,
            "request_timeout": settings.get("flex_timeout") if service_tier == "flex" else settings.get("db_timeout"),
        }
    elif "claude" in model_name:
        model_class = ChatAnthropic
        model_kwargs = {
            "model_name": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "anthropic_api_key": anthropic_api_key,
        }
    else:
        model_class = FlexTierChatOpenAI
        model_kwargs = {
            "model_name": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "openai_api_key": openai_api_key,
            "service_tier": service_tier,
            "request_timeout": settings.get("flex_timeout") if service_tier == "flex" else settings.get("db_timeout"),
        }

    # Add reasoning_effort if it's an o-model
    if reasoning_effort:
        model_kwargs["reasoning_effort"] = reasoning_effort
        model_kwargs["temperature"] = None # Reasoning effort overrides temperature

    return model_class(**model_kwargs)

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(override=True)

    # load settings
    settings = load_settings()
    print(settings)

    # set model
    model = set_model(model_name="Qwen3-235B-A22B", agent_name="default")
    print(model)