"""Task-aware routing; provider models are configured, never silently substituted."""
import os


def select_tier(role: str, text: str, retry: bool = False) -> str:
    if retry or role == 'reviewer' or len(text) > 24000:
        return 'claude'
    if role in {'supervisor', 'analyst', 'ideator'} or len(text) > 6000:
        return 'deepseek'
    return 'qwen'


class ModelRouter:
    def __init__(self):
        self.models = {}

    def get(self, role, text='', retry=False):
        tier = select_tier(role, text, retry)
        if tier not in self.models:
            prefix = tier.upper()
            key = os.environ[prefix + '_API_KEY']
            model = os.environ[prefix + '_MODEL']
            if tier == 'claude':
                from langchain_anthropic import ChatAnthropic
                self.models[tier] = ChatAnthropic(model=model, api_key=key, max_tokens=4096,
                                                   timeout=90, max_retries=2)
            else:
                from langchain_openai import ChatOpenAI
                self.models[tier] = ChatOpenAI(model=model, api_key=key,
                    base_url=os.environ[prefix + '_BASE_URL'], temperature=0,
                    timeout=90, max_retries=2)
        return self.models[tier]

    async def ask(self, role, prompt, retry=False):
        result = await self.get(role, prompt, retry).ainvoke(prompt)
        if isinstance(result.content, str):
            return result.content
        return '\n'.join(b.get('text', '') for b in result.content if isinstance(b, dict))
