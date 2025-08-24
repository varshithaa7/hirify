import json
import uuid
import os
import sys
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List

from file_handler import extract_text
from llm_handler import (
    prompt_personal_details,
    prompt_contact_details,
    prompt_passport_id,
    prompt_education,
    prompt_work_experience,
    prompt_projects,
    prompt_skills,
    prompt_misc,
    competency_level_from_resume,
)

_TPL_PATH = Path(__file__).with_name("resume_json.json")
with open(_TPL_PATH, "r", encoding="utf-8") as f:
    TEMPLATE = json.load(f)

OUTPUT_DIR = Path("./json_outputs_llama3.1_2")
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------- schema enforcer (strict) ----------
def _enforce(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """fill base with patch keys only (no new keys)."""
    if isinstance(base, dict):
        return {k: _enforce(v, patch.get(k, {})) for k, v in base.items()}
    if isinstance(base, list) and base and isinstance(base[0], dict):
        return [_enforce(base[0], item) for item in (patch or [])]
    return patch if patch is not None else base


# ---------- processing ----------
def _process_single(file_path: Path) -> Dict[str, Any]:
    text = extract_text(str(file_path))
    if not text:
        return {}

    rpd = TEMPLATE["ResumeParserData"].copy()
    rpd["ResumeFileName"] = file_path.name
    rpd["ParsingDate"] = datetime.now().isoformat()

    tasks = {
        "personal": lambda: prompt_personal_details(text, rpd["PersonalDetails"]),
        "contact": lambda: prompt_contact_details(text, rpd["ContactDetails"]),
        "passport": lambda: prompt_passport_id(
            text,
            {**rpd["PassportDetails"], **rpd["Identification"], **rpd["CategoryDetails"]},
        ),
        "education": lambda: prompt_education(text, rpd["Education"]),
        "work": lambda: prompt_work_experience(text, rpd["WorkExperience"]),
        "projects": lambda: prompt_projects(text, rpd["Projects"]),
        "skills": lambda: prompt_skills(text, rpd["SegregatedSkill"]),
        "misc": lambda: prompt_misc(
            text,
            {
                **rpd["WorkedPeriod"],
                "Achievements": rpd["Achievements"],
                "Publications": rpd["Publications"],
                "Hobbies": rpd["Hobbies"],
                "References": rpd["References"],
                "LanguagesKnown": rpd["LanguagesKnown"],
            },
        ),
        "global_level": lambda: competency_level_from_resume(text),
    }

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {k: pool.submit(fn) for k, fn in tasks.items()}
        results = {k: f.result() for k, f in futures.items()}

    # merge results as before
    rpd["PersonalDetails"].update(results["personal"])
    rpd["ContactDetails"].update(results["contact"])
    for k in ("PassportDetails", "Identification", "CategoryDetails"):
        rpd[k].update(results["passport"].get(k, {}))
    rpd["Education"] = results["education"]
    rpd["WorkExperience"] = results["work"]
    rpd["Projects"] = results["projects"]
    rpd["SegregatedSkill"].update(results["skills"])
    misc = results["misc"]
    for k in misc:
        if k in rpd:
            rpd[k] = misc[k]
    level = results["global_level"]
    for skill_list in (rpd["SegregatedSkill"]["TechnicalSkills"], rpd["SegregatedSkill"]["SoftSkills"]):
        for skill in skill_list:
            skill["competencyLevel"] = level

    final = _enforce(TEMPLATE, {"ResumeParserData": rpd})
    return {"document_id": str(uuid.uuid4()), "resume_data": final["ResumeParserData"]}


def _process_folder(folder: Path) -> None:
    files = [f for f in folder.rglob("*") if f.suffix.lower() in {".pdf", ".docx", ".doc", ".txt"}]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {f: pool.submit(_process_single, f) for f in files}
        for f, fut in futures.items():
            res = fut.result()
            out = OUTPUT_DIR / f"{f.stem}_json_output.json"
            out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
            print(f"✅ {out.name} saved")


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else input("File or folder: ").strip()
    target = Path(target).expanduser().resolve()

    if target.is_file():
        res = _process_single(target)
        out = OUTPUT_DIR / f"{target.stem}_json_output.json"
        out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
        print(f"✅ {out.name} saved")
    elif target.is_dir():
        _process_folder(target)
    else:
        print("❌ Path not found")


if __name__ == "__main__":
    main()