from .config import Config
from .prompts import Prompts

def generate_chat_response(client, user_input, context, history=[], img_count=0, is_visual=False):
    """
    Generates a response using templates from prompts.py and logic for mode selection.
    """
    # 1. Intent Triangulation
    query_lower = user_input.lower()
    
    has_visual_intent = any(t in query_lower for t in Config.VISUAL_TRIGGERS)
    has_example_intent = any(t in query_lower for t in Config.EXAMPLE_TRIGGERS)
    is_short = len(user_input.split()) <= 6
    has_info_intent = any(t in query_lower for t in Config.INFO_TRIGGERS) or (len(user_input.split()) > 8)

    # 2. Visual Context Flag
    if is_visual and img_count > 0:
        visual_info = f"\n[SYSTEM: {img_count} relevant visual(s) detected and currently displayed. Reference them directly.]"
    else:
        visual_info = ""

    # 3. Mode Selection (Conditions)
    if has_visual_intent:
        if has_info_intent or (has_example_intent and not is_short):
            length_instruction = Prompts.CONDITION_HYBRID
        elif is_short:
            length_instruction = Prompts.CONDITION_VISUAL_ONLY
        else:
            length_instruction = Prompts.CONDITION_HYBRID
    elif has_example_intent:
        length_instruction = Prompts.CONDITION_TEXT_ONLY
    else:
        length_instruction = Prompts.CONDITION_TEXT_ONLY
        visual_info = ""

    # 4. Format History
    history_str = ""
    for exchange in history[-4:]:
        u = exchange.get('user', '')
        b = exchange.get('bot', '')
        history_str += f"User: {u}\nAI: {b}\n"
    if not history_str: history_str = "No previous history."

    # 5. Construct Final Prompts
    sys_content = Prompts.SYSTEM_PROMPT.format(
        length_instruction=length_instruction,
        visual_info=visual_info
    )
    user_content = Prompts.USER_PROMPT.format(
        history=history_str,
        context=context,
        question=user_input
    )

    messages = [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": user_content}
    ]

    # 6. Execute API Call
    try:
        completion = client.chat.completions.create(
            messages=messages,
            model=Config.GROQ_MODEL,
            max_tokens=Config.MAX_TOKENS
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"
