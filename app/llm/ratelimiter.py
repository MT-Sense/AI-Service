import random
import time

from google.genai.errors import ClientError

from app.llm.client import client


def call_llm_with_backoff(
    prompt: str,
    model: str = "gemini-3.5-flash-lite",
    max_retries: int = 5,
    base_delay: float = 5.0,
):
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={
                    "temperature": 0,
                    "response_mime_type": "application/json",
                },
            )
            return response

        except ClientError as e:
            is_last_attempt = attempt == max_retries - 1
            error_str = str(e)

            if "RESOURCE_EXHAUSTED" in error_str or "UNAVAILABLE" in error_str:
                if is_last_attempt:
                    raise

                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"โดน rate limit (ครั้งที่ {attempt + 1}/{max_retries}) "
                      f"รอ {delay:.1f} วินาทีก่อนลองใหม่...")
                time.sleep(delay)
            else:
                raise

    raise RuntimeError("Retry ครบจำนวนแล้วแต่ไม่สำเร็จ")