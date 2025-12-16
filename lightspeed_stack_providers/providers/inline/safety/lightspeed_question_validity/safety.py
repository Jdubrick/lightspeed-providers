import logging
from typing import Any
from string import Template

from lightspeed_stack_providers.providers.inline.safety.lightspeed_question_validity.config import (
    QuestionValidityShieldConfig,
)
from llama_stack.apis.safety.safety import ModerationObject, ModerationObjectResults
from llama_stack.apis.datatypes import Api
from llama_stack.providers.datatypes import ShieldsProtocolPrivate
from llama_stack.apis.safety import (
    Safety,
    RunShieldResponse,
    SafetyViolation,
    ViolationLevel,
)
from llama_stack.apis.inference import (
    Inference,
    UserMessage,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAIUserMessageParam,
)

log = logging.getLogger(__name__)

SUBJECT_REJECTED = "REJECTED"
SUBJECT_ALLOWED = "ALLOWED"


class QuestionValidityShieldImpl(Safety, ShieldsProtocolPrivate):
    def __init__(self, config: QuestionValidityShieldConfig, deps) -> None:
        self.config = config
        self.model_prompt_template = Template(self.config.model_prompt)
        self.inference_api: Inference = deps[Api.inference]

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def run_moderation(self, input: str | list[str], model: str) -> ModerationObject:
        return ModerationObject(
            id="noop",
            model=model,
            results=[
                ModerationObjectResults(
                    flagged=False,
                    categories={},
                    category_scores={},
                    category_applied_input_types={},
                    user_message=None,
                    metadata={"note": "run_moderation stubbed; use run_shield"},
                )
            ],
        )

    async def run_shield(
        self,
        shield_id: str,
        messages,
        params: dict[str, Any] | None = None,
    ) -> RunShieldResponse:
        # Safely extract the last user message
        user_messages = [m for m in messages if m.role == "user"]
        if not user_messages:
            return RunShieldResponse(violation=None)

        message = UserMessage(content=user_messages[-1].content)
        log.debug(f"Shield UserMessage: {message.content}")

        impl = QuestionValidityRunner(
            model_id=self.config.model_id,
            model_prompt_template=self.model_prompt_template,
            invalid_question_response=self.config.invalid_question_response,
            inference_api=self.inference_api,
        )
        return await impl.run(message)


class QuestionValidityRunner:
    def __init__(
        self,
        model_id: str,
        model_prompt_template: Template,
        invalid_question_response: str,
        inference_api: Inference,
    ):
        self.model_id = model_id
        self.model_prompt_template = model_prompt_template
        self.invalid_question_response = invalid_question_response
        self.inference_api = inference_api

    def build_prompt(self, message: UserMessage) -> str:
        prompt = self.model_prompt_template.substitute(
            allowed=SUBJECT_ALLOWED,
            rejected=SUBJECT_REJECTED,
            message=message.content,
        )
        log.debug(f"Shield prompt: {prompt}")
        return prompt

    def parse_model_response(self, response: str) -> RunShieldResponse:
        response = response.strip()
        log.debug(f"Shield response: {response}")

        if response == SUBJECT_ALLOWED:
            return RunShieldResponse(violation=None)

        return RunShieldResponse(
            violation=SafetyViolation(
                violation_level=ViolationLevel.ERROR,
                user_message=self.invalid_question_response,
            )
        )

    async def run(self, message: UserMessage) -> RunShieldResponse:
        prompt = self.build_prompt(message)

        response = await self.inference_api.openai_chat_completion(
            OpenAIChatCompletionRequestWithExtraBody(
                model=self.model_id,
                messages=[
                    OpenAIUserMessageParam(role="user", content=prompt)
                ],
            )
        )

        content = response.choices[0].message.content.strip()
        return self.parse_model_response(content)
