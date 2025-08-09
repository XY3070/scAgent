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
                raise ValueError(f"No reasoning_effort was provided for agent '{agent_name}'")

    # Get max_tokens from settings if not provided
    if max_tokens is None:
        try:
            max_tokens = settings["max_tokens"][agent_name]
        except KeyError:
            try:
                max_tokens = settings["max_tokens"]["default"]
            except KeyError:
                max_tokens = 4096 # Default to 4096 if not specified

    # Get service_tier from settings if not provided
    if service_tier is None:
        if isinstance(settings.get("service_tier"), dict):
            try:
                service_tier = settings["service_tier"][agent_name]
            except KeyError:
                try:
                    service_tier = settings["service_tier"]["default"]
                except KeyError:
                    service_tier = "default" # Default to default if not specified
        else:
            service_tier = settings.get("service_tier", "default") # If service_tier is a string, use it directly

    # Determine API base URL based on model type or explicit setting
    openai_api_base = settings.get("openai_api_base", os.getenv("OPENAI_API_BASE", app_settings.MODEL_API_URL))

    # Initialize the model based on model_name
    if "gpt" in model_name:
        # OpenAI models
        openai_api_key = settings.get("openai_api_key", os.getenv("OPENAI_API_KEY"))
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set or not found in settings.")

        if service_tier == "flex":
            model = FlexTierChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                openai_api_base=openai_api_base,
                openai_api_key=openai_api_key,
                service_tier=service_tier,
                request_timeout=10.0 # Set a timeout for flex tier
            )
        else:
            model = ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                openai_api_base=openai_api_base,
                openai_api_key=openai_api_key,
            )
    elif "claude" in model_name:
        # Anthropic models
        anthropic_api_key = settings.get("anthropic_api_key", os.getenv("ANTHROPIC_API_KEY"))
        if not anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set or not found in settings.")
        model = ChatAnthropic(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            anthropic_api_key=anthropic_api_key,
        )
    elif "o-model" in model_name:
        # O-models (e.g., for reasoning)
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        model = ChatOpenAI(
            model_name=model_name,
            temperature=None,  # O-models use reasoning_effort instead of temperature
            max_tokens=max_tokens,
            openai_api_base=openai_api_base,
            openai_api_key=openai_api_key,
        )
        model.reasoning_effort = reasoning_effort
    elif "Qwen" in model_name or "qwen" in model_name:
        # For Qwen models, use ChatOpenAI with custom api_base
        qwen_api_base = settings.get("qwen_api_base")
        if not qwen_api_base and "claude" in settings:
            qwen_api_base = settings["claude"].get("qwen_api_base")
        if not qwen_api_base:
            raise ValueError("Qwen API base URL not set or not found in settings.")
        model = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_base=qwen_api_base,
            openai_api_key="default_key"  # Placeholder key for Qwen models
        )
    else:
        # Default to ChatOpenAI for other models, assuming OpenAI-compatible API
        model = FlexTierChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_base=openai_api_base,
            service_tier=service_tier,
            timeout=settings["flex_timeout"] if service_tier == "flex" else app_settings.DB_TIMEOUT,
        )

    return model

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(override=True)

    # load settings
    settings = load_settings()
    print(settings)

    # set model
    model = set_model(model_name="Qwen3-235B-A22B", agent_name="default")
    print(model)