import json
import re
import logging
from langchain_ollama import OllamaLLM

MODEL = "llama3.2:1b"
llm = OllamaLLM(model=MODEL, temperature=0.0)

logger = logging.getLogger(__name__)

# ---------- base helper ----------
def _call_llm(prompt: str, template_piece):
    raw = llm.invoke(prompt)
    raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception as e:
        # Try to extract the largest JSON object/array from the output
        m = re.search(r"\{[\s\S]*\}|\[[\s\S]*\]", raw)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception as e2:
                logger.error(f"Failed to parse extracted JSON: {e2}\nRaw: {raw}")
        else:
            logger.error(f"No JSON found in LLM output.\nRaw: {raw}")
        # Always return the template if parsing fails
        return template_piece

# ---------- section prompts ----------
def prompt_personal_details(text, tpl):
    p = (
        "You are an expert résumé parser."
        "Extract the PersonalDetails section into the exact JSON below. "
        "Return ONLY valid JSON, no markdown."
        "No delimter errors, no markdown.\n\n"
        f"JSON template:\n{json.dumps(tpl, indent=2)}\n\n"
        "Resume text:\n"
        f"{text[:7000]}"
    )
    return _call_llm(p, tpl)

def prompt_contact_details(text, tpl):
    p = (
        "You are an expert résumé parser."
        "Extract ContactDetails (email, phone, websites, address) "
        "into the exact JSON below. Return only valid JSON."
        "No delimter errors, no markdown.\n\n"
        f"JSON template:\n{json.dumps(tpl, indent=2)}\n\n"
        "Resume text:\n"
        f"{text[:7000]}"
    )
    return _call_llm(p, tpl)

def prompt_passport_id(text, tpl):
    p = (
        "You are an expert résumé parser. "
        "Extract PassportDetails, Identification, CategoryDetails "
        "into the exact JSON below. Return only valid JSON."
        "No delimter errors, no markdown.\n\n"
        f"JSON template:\n{json.dumps(tpl, indent=2)}\n\n"
        "Resume text:\n"
        f"{text[:7000]}"
    )
    return _call_llm(p, tpl)

def prompt_education(text, tpl):
    p = (
        "You are an expert résumé parser."
        "Extract Education entries into the exact JSON array below. "
        "Return only valid JSON."
        "No delimter errors, no markdown.\n\n"
        f"JSON template:\n{json.dumps(tpl, indent=2)}\n\n"
        "Resume text:\n"
        f"{text[:7000]}"
    )
    return _call_llm(p, tpl)

def prompt_work_experience(text, tpl):
    p = (
        "You are an expert résumé parser."
        "Extract WorkExperience entries into the exact JSON array below. "
        "Return only valid JSON."
        "No delimter errors, no markdown.\n\n"
        f"JSON template:\n{json.dumps(tpl, indent=2)}\n\n"
        "Resume text:\n"
        f"{text[:7000]}"
    )
    return _call_llm(p, tpl)

def prompt_projects(text, tpl):
    p = (
        "You are an expert résumé parser."
        "Extract Projects entries into the exact JSON array below. "
        "Return only valid JSON."
        "No delimter errors, no markdown.\n\n"
        f"JSON template:\n{json.dumps(tpl, indent=2)}\n\n"
        "Resume text:\n"
        f"{text[:7000]}"
    )
    return _call_llm(p, tpl)

def prompt_skills(text, tpl):
    p = (
        "You are an expert résumé parser."
        "Extract TechnicalSkills and SoftSkills with numeric proficiency (1-10). "
        "Return JSON like:\n"
        '{"TechnicalSkills":[{"skillName":"Python","competencyLevel":{"Beginner":1,"Intermediate":6,"Advanced":10}}\n\n'
        "Resume text:\n"
        "No delimter errors, no markdown.\n\n"
        f"{text[:7000]}"
    )
    return _call_llm(p, tpl)

def prompt_misc(text, tpl):
    p = (
        "You are an expert résumé parser. "
        "Extract remaining résumé sections exactly as provided in the JSON below. "
        "Return only valid JSON."
        "No delimter errors, no markdown.\n\n"
        f"JSON template:\n{json.dumps(tpl, indent=2)}\n\n"
        "Resume text:\n"
        f"{text[:7000]}"
    )
    return _call_llm(p, tpl)

# ---------- standalone competency level ----------
def competency_level_from_resume(text: str) -> str:
    prompt = (
        "You are an expert résumé evaluator. "
        "Based ONLY on the résumé text below, determine the overall competency level "
        "as Beginner, Intermediate, or Advanced by considering:\n"
        "• Total years of professional experience\n"
        "• Highest education level attained\n"
        "• Relevant certifications or courses\n"
        "• Complexity and scope of projects described\n"
        "• Seniority of roles held\n\n"
        "Return **only one** of these exact words:\n"
        "Beginner\n"
        "Intermediate\n"
        "Advanced\n\n"
        "No markdown, no extra text.\n\n"
        "Resume text:\n"
        f"{text[:7000]}"
    )
    raw = llm.invoke(prompt).strip()
    if raw.lower() in {"beginner", "intermediate", "advanced"}:
        return raw.capitalize()
    return "Intermediate"