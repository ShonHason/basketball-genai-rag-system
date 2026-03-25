import glob
import json
import os
import re
import sys
import time
import random
from collections import defaultdict
from typing import Any, Dict, List

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from synthetic_data_generator import generate_data


POSITIONS = {"PG", "SG", "SF", "PF", "C"}
TEMPERAMENTS = {"calm", "balanced", "intense", "hot-headed"}
LOCKER_ROOM_ROLES = {
    "leader",
    "mentor",
    "glue-guy",
    "quiet-worker",
    "joker",
    "challenger",
    "neutral",
}
TEAM_STATUSES = {
    "franchise_star",
    "star",
    "starter",
    "sixth_man",
    "role_player",
    "defensive_specialist",
    "bench_prospect",
    "veteran_anchor",
    "rookie_project",
}

# ---------------------------------------------------------
# GLOBAL REGISTRIES
# ---------------------------------------------------------
NAME_COUNT = defaultdict(int)
MAX_SAME_NAME = 3

LAST_NAME_REGISTRY = defaultdict(list)
FORMER_PLAYER_NAMES = []

POSITION_HEIGHT_RANGES = {
    "PG": (178, 188),
    "SG": (185, 195),
    "SF": (193, 203),
    "PF": (200, 210),
    "C": (208, 218),
}


def compact_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (ValueError, TypeError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def normalize_position(position: Any) -> str:
    position = str(position).strip().upper() if position is not None else ""
    return position if position in POSITIONS else "SG"


def normalize_score(value: Any, default: int = 50) -> int:
    score = safe_int(value, default)
    return max(0, min(100, score))


def normalize_percentage(value: Any, default: float = 0.0) -> float:
    pct = safe_float(value, default)
    if pct < 0:
        return 0.0
    if pct > 100:
        return 100.0
    return round(pct, 1)


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


# ---------------------------------------------------------
# NAME CONTROL
# ---------------------------------------------------------
def limit_name_duplicates(name: str) -> str:
    key = name.lower().strip()
    NAME_COUNT[key] += 1

    if NAME_COUNT[key] <= MAX_SAME_NAME:
        return name

    return f"{name} Jr.{NAME_COUNT[key]}"


# ---------------------------------------------------------
# HEIGHT / POSITION REALISM
# ---------------------------------------------------------
def generate_realistic_height(position: str, fallback_height: int = 195) -> int:
    position = normalize_position(position)
    low, high = POSITION_HEIGHT_RANGES.get(position, (fallback_height - 5, fallback_height + 5))
    return random.randint(low, high)


def enforce_position_height_consistency(player: Dict[str, Any]) -> Dict[str, Any]:
    height = safe_int(player.get("height_cm"), 195)
    position = normalize_position(player.get("position"))

    # If height is wildly unrealistic for stated position, fix height first.
    low, high = POSITION_HEIGHT_RANGES[position]
    if height < low or height > high:
        height = generate_realistic_height(position, height)
        player["height_cm"] = height

    # Wingspan should generally exceed height.
    wingspan = safe_int(player.get("wingspan_cm"), height + 8)
    if wingspan < height + 4:
        wingspan = height + random.randint(6, 14)
    player["wingspan_cm"] = wingspan

    # Weight rough realism by position / height.
    weight = safe_int(player.get("weight_kg"), 92)
    if position == "PG":
        weight = max(72, min(weight, 92))
    elif position == "SG":
        weight = max(78, min(weight, 100))
    elif position == "SF":
        weight = max(85, min(weight, 110))
    elif position == "PF":
        weight = max(95, min(weight, 122))
    elif position == "C":
        weight = max(102, min(weight, 135))
    player["weight_kg"] = weight

    return player


# ---------------------------------------------------------
# FAMILY RELATIONS
# ---------------------------------------------------------
def maybe_add_family_relation(player: Dict[str, Any]) -> Dict[str, Any]:
    name = str(player.get("name", "")).strip()
    if not name:
        return player

    last_name = name.split()[-1]
    relation = None

    # Sometimes same surname means a family link
    if LAST_NAME_REGISTRY[last_name] and random.random() < 0.12:
        related_player = random.choice(LAST_NAME_REGISTRY[last_name])
        if related_player != name:
            relation = {
                "type": random.choice(["brother", "cousin"]),
                "related_player": related_player,
            }

    # Small chance to be son of former player
    if relation is None and random.random() < 0.06:
        former_name = f"Retired {last_name}"
        FORMER_PLAYER_NAMES.append(former_name)
        relation = {
            "type": "son_of_former_player",
            "related_player": former_name,
        }

    LAST_NAME_REGISTRY[last_name].append(name)

    if relation:
        player["family_relation"] = relation

    return player


def rewrite_internal_name_references(
    roster: List[Dict[str, Any]], rename_map: Dict[str, str]
) -> List[Dict[str, Any]]:
    def map_name(name: str) -> str:
        if not isinstance(name, str):
            return name
        return rename_map.get(name.strip(), name)

    for player in roster:
        chemistry = player.get("chemistry", {})

        for item in safe_list(chemistry.get("best_teammates")):
            if isinstance(item, dict):
                item["player"] = map_name(item.get("player", ""))

        chemistry["neutral_teammates"] = [
            map_name(x) for x in safe_list(chemistry.get("neutral_teammates"))
        ]

        for item in safe_list(chemistry.get("low_chemistry_teammates")):
            if isinstance(item, dict):
                item["player"] = map_name(item.get("player", ""))

        if isinstance(player.get("family_relation"), dict):
            rel = player["family_relation"]
            related_player = rel.get("related_player")
            rel["related_player"] = map_name(related_player)

    return roster


# ---------------------------------------------------------
# BUILD ROSTER SEED
# ---------------------------------------------------------
def build_roster_seed(team: Dict[str, Any]) -> List[Dict[str, Any]]:
    roster = safe_list(team.get("roster"))
    cleaned = []

    for idx, player in enumerate(roster, start=1):
        if not isinstance(player, dict):
            continue

        position = normalize_position(player.get("position"))
        fallback_height = safe_int(player.get("height_cm"), 195)

        cleaned.append(
            {
                "name": player.get("name", f"Unknown Player {idx}"),
                "number": safe_int(player.get("number"), idx),
                "position": position,
                "height_cm": generate_realistic_height(position, fallback_height),
                "age": safe_int(player.get("age"), 24),
                "nationality": player.get("nationality", "Unknown"),
            }
        )

    return cleaned


# ---------------------------------------------------------
# MARKDOWN FOR TEAM SUMMARY
# ---------------------------------------------------------
def create_team_players_markdown(team: Dict[str, Any]) -> str:
    city = team.get("city", "Unknown")
    name = team.get("name", "Team")
    team_full_name = f"{city} {name}"
    roster = safe_list(team.get("roster"))

    # --- MATH SECTION (KEEP AS IS) ---
    total_ppg = sum(p.get("season_projection", {}).get("ppg", 0) for p in roster)
    total_rpg = sum(p.get("season_projection", {}).get("rpg", 0) for p in roster)
    total_apg = sum(p.get("season_projection", {}).get("apg", 0) for p in roster)
    avg_age = round(sum(p.get("age", 0) for p in roster) / len(roster), 1) if roster else 0
    avg_height = round(sum(p.get("height_cm", 0) for p in roster) / len(roster)) if roster else 0

    scoring_leader = max(roster, key=lambda x: x.get("season_projection", {}).get("ppg", 0), default={})
    rebound_leader = max(roster, key=lambda x: x.get("season_projection", {}).get("rpg", 0), default={})
    assist_leader = max(roster, key=lambda x: x.get("season_projection", {}).get("apg", 0), default={})

    # --- UPDATED MARKDOWN STRUCTURE ---
    lines = [
        f"# {team_full_name} Full Roster and Analytics",
        f"**Team ID:** {team.get('id', 'unknown')}",
        f"**Conference/Division:** {team.get('conference', '?')} | {team.get('division', '?')}",
        f"**Canonical Slug:** {slugify(team_full_name)}",
        f"**Overview:** {compact_text(team.get('description', 'No description'))}",
        "",
        "## Team Performance Statistics",
        f"* **Total Project PPG:** {round(total_ppg,1)}",
        f"* **Total Project RPG:** {round(total_rpg,1)}",
        f"* **Total Project APG:** {round(total_apg,1)}",
        f"* **Average Roster Age:** {avg_age} years",
        f"* **Average Roster Height:** {avg_height} cm",
        "",
        "## Team Statistical Leaders",
        f"* **Points Leader:** {scoring_leader.get('name','?')} ({scoring_leader.get('season_projection',{}).get('ppg','?')} PPG)",
        f"* **Rebounds Leader:** {rebound_leader.get('name','?')} ({rebound_leader.get('season_projection',{}).get('rpg','?')} RPG)",
        f"* **Assists Leader:** {assist_leader.get('name','?')} ({assist_leader.get('season_projection',{}).get('apg','?')} APG)",
        "",
        "## Player Roster and Personnel Details",
    ]

    for p in roster:
        # ... (Your existing logic for personality, role, chemistry, etc.) ...
        personality = p.get("personality", {})
        role = p.get("role_in_team", {})
        stats = p.get("season_projection", {})
        family = p.get("family_relation", {})
        
        # Best Teammates logic (shortened for the line)
        best_names = [item.get("player") if isinstance(item, dict) else item for item in p.get("chemistry", {}).get("best_teammates", [])]
        
        family_text = "None"
        if isinstance(family, dict) and family.get("type"):
            family_text = f"{family.get('type')} of {family.get('related_player')}"

        # SELF-DESCRIBING LINE: Every line starts with the Team Name.
        # This is CRITICAL if the roster is split into multiple chunks.
        lines.append(
            f"* **{team_full_name} Player:** #{p.get('number', '?')} {p.get('name', 'Unknown')} | "
            f"Pos: {p.get('position', '?')} | {p.get('height_cm', '?')}cm | Age: {p.get('age', '?')} | "
            f"Status: {role.get('status', 'unknown')} | "
            f"Temperament: {personality.get('temperament', 'unknown')} | "
            f"Stats: {stats.get('ppg', '?')} PPG, {stats.get('rpg', '?')} RPG, {stats.get('apg', '?')} APG | "
            f"Family: {family_text} | "
            f"Chemistry: Closest to {', '.join(best_names[:2]) if best_names else 'None'}"
        )

    return "\n".join(lines)

# ---------------------------------------------------------
# MARKDOWN FOR PLAYER FILE
# Slightly improved for RAG:
# - stable labeled fields
# - canonical slug
# - search tags
# ---------------------------------------------------------
def create_player_markdown(team: Dict[str, Any], player: Dict[str, Any]) -> str:
    # --- Extraction & Prep ---
    personality = player.get("personality", {})
    role = player.get("role_in_team", {})
    chemistry = player.get("chemistry", {})
    basketball = player.get("basketball_profile", {})
    personal_info = player.get("personal_info", {})
    background = player.get("background_story", {})
    projection = player.get("season_projection", {})
    contract = player.get("contract", {})
    health = player.get("health_profile", {})
    development = player.get("development_profile", {})
    media = player.get("media_and_reputation", {})
    family = player.get("family_relation", {})

    player_name = player.get('name', 'Unknown')
    team_full_name = f"{team.get('city', 'Unknown')} {team.get('name', 'Team')}"

    # Helper for Teammates
    best_text = ", ".join([f"{x.get('player')} ({x.get('score')})" for x in chemistry.get("best_teammates", []) if isinstance(x, dict)]) or "None"
    low_text = ", ".join([f"{x.get('player')} ({x.get('score')})" for x in chemistry.get("low_chemistry_teammates", []) if isinstance(x, dict)]) or "None"

    family_text = "None"
    if isinstance(family, dict) and family.get("type") and family.get("related_player"):
        family_text = f"{family.get('type')}: {family.get('related_player')}"

    search_tags = [player_name, team_full_name, player.get("position"), player.get("nationality")]

    # --- Markdown Construction ---
    # We add the player name to headers so the MarkdownHeaderTextSplitter captures it in metadata.
    # We use bullet points for lists so the Recursive splitter breaks cleanly between lines.

    return "\n".join(
        [
            f"# Player Profile: {player_name}",
            f"**Full Name:** {player_name}",
            f"**Current Team:** {team_full_name}",
            f"**Canonical Player Slug:** {slugify(player_name)}",
            f"**Player ID:** {player.get('player_id', 'unknown')}",
            f"**Search Tags:** {', '.join([str(x) for x in search_tags if x])}",
            "",
            f"## {player_name} - Personal & Physical Info",
            f"* **Identity:** #{player.get('number', '?')} | {player.get('position', '?')} | {player.get('nationality', 'Unknown')}",
            f"* **Dimensions:** Height: {player.get('height_cm', '?')}cm | Weight: {player.get('weight_kg', '?')}kg | Wingspan: {player.get('wingspan_cm', '?')}cm",
            f"* **Bio:** Age: {player.get('age', '?')} | Born: {personal_info.get('birth_city', 'Unknown')}, {personal_info.get('country_of_origin', 'Unknown')}",
            f"* **Languages:** {', '.join(safe_list(personal_info.get('languages')))}",
            f"* **Background:** {compact_text(personal_info.get('family_background', ''))}",
            f"* **Family Relation:** {family_text}",
            "",
            f"## {player_name} - Team Role",
            f"* **Status:** {role.get('status', 'unknown')} (Star: {role.get('is_star', False)})",
            f"* **Rotation:** {role.get('rotation_role', 'unknown')} | Expected MPG: {role.get('minutes_expectation', '?')}",
            "",
            f"## {player_name} - Basketball Profile",
            f"* **Play Style:** {', '.join(safe_list(basketball.get('play_style')))}",
            f"* **Strengths:** {', '.join(safe_list(basketball.get('strengths')))}",
            f"* **Weaknesses:** {', '.join(safe_list(basketball.get('weaknesses')))}",
            f"* **Attributes:** Basketball IQ: {basketball.get('basketball_iq', '?')} | Motor: {basketball.get('motor', '?')}",
            "",
            f"## {player_name} - Personality & Leadership",
            f"* **Metrics:** Leadership: {personality.get('leadership', '?')} | Ego: {personality.get('ego_level', '?')} | Work Ethic: {personality.get('work_ethic', '?')}",
            f"* **Temperament:** {personality.get('temperament', 'unknown')} | Confidence: {personality.get('confidence', '?')}",
            f"* **Locker Room Role:** {personality.get('locker_room_role', 'unknown')}",
            "",
            f"## {player_name} - Season Projections",
            f"* **Volume:** {projection.get('games', '?')} G | {projection.get('games_started', '?')} GS | {projection.get('minutes', '?')} MPG",
            f"* **Primary Stats:** {projection.get('ppg', '?')} PPG | {projection.get('rpg', '?')} RPG | {projection.get('apg', '?')} APG",
            f"* **Secondary Stats:** {projection.get('spg', '?')} SPG | {projection.get('bpg', '?')} BPG | {projection.get('topg', '?')} TOPG",
            f"* **Efficiency:** FG%: {projection.get('fg_pct', '?')} | 3P%: {projection.get('three_pct', '?')} | FT%: {projection.get('ft_pct', '?')}",
            "",
            f"## {player_name} - Contract & Market Value",
            f"* **Financials:** {contract.get('years_remaining', '?')} years remaining | ${contract.get('annual_salary_m', '?')}M Annual",
            f"* **Market Value:** Tier: {contract.get('contract_tier', 'unknown')} | Trade Value: {contract.get('trade_value', '?')}",
            "",
            f"## {player_name} - Health & Development",
            f"* **Health Profile:** Durability: {health.get('durability', '?')} | Injury Risk: {health.get('injury_risk', '?')} | Conditioning: {health.get('conditioning', '?')}",
            f"* **Potential:** Ceiling: {development.get('ceiling', '?')} | Current Impact: {development.get('current_impact', '?')}",
            f"* **Focus Areas:** {', '.join(safe_list(development.get('improvement_areas')))}",
            "",
            f"## {player_name} - Chemistry & Reputation",
            f"* **Fan Popularity:** {media.get('fan_popularity', '?')} | Media Attention: {media.get('media_attention', '?')}",
            f"* **Reputation:** {compact_text(background.get('public_reputation', ''))}",
            f"* **Teammate Links:** Best: {best_text} | Low: {low_text} | Neutral: {', '.join(safe_list(chemistry.get('neutral_teammates')))}",
            "",
            f"## {player_name} - History & Summary",
            f"**Path to Team:** {compact_text(background.get('path_to_team', ''))}",
            f"**Off Court Interests:** {', '.join(safe_list(background.get('off_court_interests')))}",
            f"**Profile Summary:** {compact_text(player.get('profile_summary', ''))}",
        ]
    )

# ---------------------------------------------------------
# VALIDATION + ENRICHMENT CLEANUP
# ---------------------------------------------------------
def validate_player_enrichment(
    enriched_roster: List[Dict[str, Any]], original_roster: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    original_by_name = {
        str(p.get("name", "")).strip().lower(): p
        for p in original_roster
        if isinstance(p, dict)
    }

    validated = []
    rename_map: Dict[str, str] = {}

    for idx, player in enumerate(enriched_roster, start=1):
        if not isinstance(player, dict):
            continue

        original_name = str(player.get("name", "")).strip()
        if not original_name:
            continue

        final_name = limit_name_duplicates(original_name)
        rename_map[original_name] = final_name

        base = original_by_name.get(original_name.lower(), {})

        personality = player.get("personality", {})
        role_in_team = player.get("role_in_team", {})
        chemistry = player.get("chemistry", {})
        basketball_profile = player.get("basketball_profile", {})
        personal_info = player.get("personal_info", {})
        background_story = player.get("background_story", {})
        season_projection = player.get("season_projection", {})
        contract = player.get("contract", {})
        health_profile = player.get("health_profile", {})
        development_profile = player.get("development_profile", {})
        media_and_reputation = player.get("media_and_reputation", {})
        family_relation = player.get("family_relation", {})

        position = normalize_position(player.get("position", base.get("position")))
        input_height = safe_int(player.get("height_cm", base.get("height_cm")), 195)
        height = generate_realistic_height(position, input_height)

        validated_player = {
            "player_id": player.get("player_id", base.get("player_id", f"unknown_player_{idx:02d}")),
            "name": final_name,
            "number": safe_int(player.get("number", base.get("number")), idx),
            "position": position,
            "height_cm": height,
            "age": safe_int(player.get("age", base.get("age")), 24),
            "nationality": player.get("nationality", base.get("nationality", "Unknown")),
            "weight_kg": safe_int(player.get("weight_kg"), 92),
            "wingspan_cm": safe_int(
                player.get("wingspan_cm"),
                height + 8,
            ),
            "personal_info": {
                "birth_city": personal_info.get("birth_city", "Unknown"),
                "country_of_origin": personal_info.get(
                    "country_of_origin",
                    player.get("nationality", base.get("nationality", "Unknown")),
                ),
                "languages": safe_list(personal_info.get("languages")) or ["English"],
                "family_background": compact_text(
                    personal_info.get("family_background", "No notable background.")
                ),
            },
            "basketball_profile": {
                "play_style": safe_list(basketball_profile.get("play_style"))[:8],
                "strengths": safe_list(basketball_profile.get("strengths"))[:8],
                "weaknesses": safe_list(basketball_profile.get("weaknesses"))[:6],
                "basketball_iq": normalize_score(basketball_profile.get("basketball_iq"), 60),
                "motor": normalize_score(basketball_profile.get("motor"), 65),
            },
            "personality": {
                "leadership": normalize_score(personality.get("leadership"), 50),
                "ego_level": normalize_score(personality.get("ego_level"), 40),
                "competitiveness": normalize_score(personality.get("competitiveness"), 65),
                "work_ethic": normalize_score(personality.get("work_ethic"), 65),
                "confidence": normalize_score(personality.get("confidence"), 60),
                "temperament": personality.get("temperament", "balanced")
                if personality.get("temperament", "balanced") in TEMPERAMENTS
                else "balanced",
                "locker_room_role": personality.get("locker_room_role", "neutral")
                if personality.get("locker_room_role", "neutral") in LOCKER_ROOM_ROLES
                else "neutral",
            },
            "role_in_team": {
                "status": role_in_team.get("status", "role_player")
                if role_in_team.get("status", "role_player") in TEAM_STATUSES
                else "role_player",
                "is_star": bool(role_in_team.get("is_star", False)),
                "rotation_role": compact_text(role_in_team.get("rotation_role", "rotation")),
                "minutes_expectation": safe_int(role_in_team.get("minutes_expectation"), 18),
            },
            "season_projection": {
                "games": safe_int(season_projection.get("games"), 72),
                "games_started": safe_int(season_projection.get("games_started"), 20),
                "minutes": round(safe_float(season_projection.get("minutes"), 18.0), 1),
                "ppg": round(safe_float(season_projection.get("ppg"), 8.0), 1),
                "rpg": round(safe_float(season_projection.get("rpg"), 4.0), 1),
                "apg": round(safe_float(season_projection.get("apg"), 2.0), 1),
                "spg": round(safe_float(season_projection.get("spg"), 0.8), 1),
                "bpg": round(safe_float(season_projection.get("bpg"), 0.5), 1),
                "topg": round(safe_float(season_projection.get("topg"), 1.4), 1),
                "fg_pct": normalize_percentage(season_projection.get("fg_pct"), 45.0),
                "three_pct": normalize_percentage(season_projection.get("three_pct"), 34.0),
                "ft_pct": normalize_percentage(season_projection.get("ft_pct"), 76.0),
            },
            "contract": {
                "years_remaining": safe_int(contract.get("years_remaining"), 2),
                "annual_salary_m": round(safe_float(contract.get("annual_salary_m"), 4.5), 1),
                "contract_tier": compact_text(contract.get("contract_tier", "rotation-level")),
                "trade_value": normalize_score(contract.get("trade_value"), 50),
            },
            "health_profile": {
                "durability": normalize_score(health_profile.get("durability"), 70),
                "injury_risk": normalize_score(health_profile.get("injury_risk"), 35),
                "conditioning": normalize_score(health_profile.get("conditioning"), 72),
            },
            "development_profile": {
                "ceiling": normalize_score(development_profile.get("ceiling"), 65),
                "current_impact": normalize_score(development_profile.get("current_impact"), 58),
                "improvement_areas": safe_list(development_profile.get("improvement_areas"))[:6],
            },
            "media_and_reputation": {
                "fan_popularity": normalize_score(media_and_reputation.get("fan_popularity"), 50),
                "media_attention": normalize_score(media_and_reputation.get("media_attention"), 45),
            },
            "chemistry": {
                "best_teammates": [
                    {
                        "player": item.get("player", "Unknown"),
                        "score": normalize_score(item.get("score"), 70),
                        "reason": compact_text(item.get("reason", "Strong mutual respect.")),
                    }
                    for item in safe_list(chemistry.get("best_teammates"))
                    if isinstance(item, dict) and item.get("player")
                ][:3],
                "neutral_teammates": [
                    str(x)
                    for x in safe_list(chemistry.get("neutral_teammates"))
                    if str(x).strip()
                ][:5],
                "low_chemistry_teammates": [
                    {
                        "player": item.get("player", "Unknown"),
                        "score": normalize_score(item.get("score"), 35),
                        "reason": compact_text(
                            item.get("reason", "Different personalities or competing roles.")
                        ),
                    }
                    for item in safe_list(chemistry.get("low_chemistry_teammates"))
                    if isinstance(item, dict) and item.get("player")
                ][:3],
            },
            "background_story": {
                "path_to_team": compact_text(
                    background_story.get(
                        "path_to_team",
                        "Joined through standard team-building process."
                    )
                ),
                "off_court_interests": safe_list(background_story.get("off_court_interests"))[:6],
                "public_reputation": compact_text(
                    background_story.get("public_reputation", "Respected professional.")
                ),
            },
            "profile_summary": compact_text(player.get("profile_summary", "")),
        }

        if isinstance(family_relation, dict) and family_relation.get("type") and family_relation.get("related_player"):
            validated_player["family_relation"] = {
                "type": compact_text(family_relation.get("type")),
                "related_player": compact_text(family_relation.get("related_player")),
            }

        validated_player = enforce_position_height_consistency(validated_player)
        validated_player = maybe_add_family_relation(validated_player)

        validated.append(validated_player)

    if len(validated) != len(original_roster):
        validated_names = {
            str(p.get("name", "")).strip().lower()
            for p in validated
            if isinstance(p, dict)
        }

        for idx, base_player in enumerate(original_roster, start=1):
            base_name = str(base_player.get("name", "")).strip()
            if base_name and base_name.lower() not in validated_names:
                position = normalize_position(base_player.get("position"))
                height = generate_realistic_height(position, safe_int(base_player.get("height_cm"), 195))

                fallback_player = {
                    "player_id": base_player.get("player_id", f"unknown_player_{idx:02d}"),
                    "name": limit_name_duplicates(base_name),
                    "number": safe_int(base_player.get("number"), idx),
                    "position": position,
                    "height_cm": height,
                    "age": safe_int(base_player.get("age"), 24),
                    "nationality": base_player.get("nationality", "Unknown"),
                    "weight_kg": 92,
                    "wingspan_cm": height + 8,
                    "personal_info": {
                        "birth_city": "Unknown",
                        "country_of_origin": base_player.get("nationality", "Unknown"),
                        "languages": ["English"],
                        "family_background": "No notable background.",
                    },
                    "basketball_profile": {
                        "play_style": [],
                        "strengths": [],
                        "weaknesses": [],
                        "basketball_iq": 60,
                        "motor": 65,
                    },
                    "personality": {
                        "leadership": 50,
                        "ego_level": 40,
                        "competitiveness": 65,
                        "work_ethic": 65,
                        "confidence": 60,
                        "temperament": "balanced",
                        "locker_room_role": "neutral",
                    },
                    "role_in_team": {
                        "status": "role_player",
                        "is_star": False,
                        "rotation_role": "rotation",
                        "minutes_expectation": 18,
                    },
                    "season_projection": {
                        "games": 72,
                        "games_started": 20,
                        "minutes": 18.0,
                        "ppg": 8.0,
                        "rpg": 4.0,
                        "apg": 2.0,
                        "spg": 0.8,
                        "bpg": 0.5,
                        "topg": 1.4,
                        "fg_pct": 45.0,
                        "three_pct": 34.0,
                        "ft_pct": 76.0,
                    },
                    "contract": {
                        "years_remaining": 2,
                        "annual_salary_m": 4.5,
                        "contract_tier": "rotation-level",
                        "trade_value": 50,
                    },
                    "health_profile": {
                        "durability": 70,
                        "injury_risk": 35,
                        "conditioning": 72,
                    },
                    "development_profile": {
                        "ceiling": 65,
                        "current_impact": 58,
                        "improvement_areas": [],
                    },
                    "media_and_reputation": {
                        "fan_popularity": 50,
                        "media_attention": 45,
                    },
                    "chemistry": {
                        "best_teammates": [],
                        "neutral_teammates": [],
                        "low_chemistry_teammates": [],
                    },
                    "background_story": {
                        "path_to_team": "Joined through standard team-building process.",
                        "off_court_interests": [],
                        "public_reputation": "Respected professional.",
                    },
                    "profile_summary": "",
                }

                fallback_player = enforce_position_height_consistency(fallback_player)
                fallback_player = maybe_add_family_relation(fallback_player)

                validated.append(fallback_player)

    validated = rewrite_internal_name_references(validated, rename_map)
    validated.sort(key=lambda x: (x.get("number", 999), x.get("name", "")))
    return validated


# ---------------------------------------------------------
# ROSTER CONSTRAINTS
# ---------------------------------------------------------
def enforce_roster_constraints(roster: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

    if not roster:
        return roster

    # ---- AGE LIMIT ----
    for p in roster:
        age = safe_int(p.get("age"), 24)

        if age < 18:
            age = 18
        if age > 38:
            age = 38

        p["age"] = age

        if age == 18:
            p.setdefault("role_in_team", {})
            p["role_in_team"]["status"] = "rookie_project"

        p["position"] = normalize_position(p.get("position"))
        p["height_cm"] = safe_int(p.get("height_cm"), 195)
        p["weight_kg"] = safe_int(p.get("weight_kg"), 92)
        p["wingspan_cm"] = safe_int(p.get("wingspan_cm"), p["height_cm"] + 8)

        p = enforce_position_height_consistency(p)

    # ---- ENSURE STAR ----
    stars = [p for p in roster if p.get("role_in_team", {}).get("is_star")]

    if not stars:

        def star_score(p):
            stats = p.get("season_projection", {})
            role = p.get("role_in_team", {})
            personality = p.get("personality", {})

            return (
                stats.get("ppg", 0) * 2
                + stats.get("apg", 0)
                + role.get("minutes_expectation", 0) * 0.6
                + personality.get("confidence", 0) * 0.3
                + personality.get("leadership", 0) * 0.3
            )

        star = max(roster, key=star_score)

        star.setdefault("role_in_team", {})
        star["role_in_team"]["is_star"] = True
        star["role_in_team"]["status"] = "star"

    # ---- TEAM SCORING NORMALIZATION ----
    total_ppg = sum(
        p.get("season_projection", {}).get("ppg", 0)
        for p in roster
    )

    if total_ppg == 0:
        return roster

    target = 105
    factor = target / total_ppg

    for p in roster:
        p["season_projection"]["ppg"] = round(
            p["season_projection"]["ppg"] * factor, 1
        )

    total_ppg = sum(p["season_projection"]["ppg"] for p in roster)

    star = max(
        roster,
        key=lambda x: x.get("season_projection", {}).get("ppg", 0)
    )

    if total_ppg < 95:
        star["season_projection"]["ppg"] += 95 - total_ppg

    if total_ppg > 120:
        star["season_projection"]["ppg"] -= total_ppg - 120

    return roster


# ---------------------------------------------------------
# ENRICH TEAM PLAYERS
# ---------------------------------------------------------
def enrich_team_players(team: Dict[str, Any]) -> Dict[str, Any]:
    roster_seed = build_roster_seed(team)

    if not roster_seed:
        print(f"Skipping {team.get('city', '')} {team.get('name', '')}: no roster found")
        return team

    for i, p in enumerate(roster_seed, start=1):
        p["player_id"] = f"{team.get('id', 'team_unknown')}_player_{i:02d}"

    player_names = [p["name"] for p in roster_seed]
    team_name = f"{team.get('city', 'Unknown')} {team.get('name', 'Team')}"
    roster_json = json.dumps(roster_seed, ensure_ascii=False, indent=2)

    prompt = f"""
You are enriching player profiles for a fictional professional basketball team.

TEAM:
{team_name}

CONTEXT:
- Conference: {team.get('conference', 'Unknown')}
- Division: {team.get('division', 'Unknown')}
- Team description: {team.get('description', 'No description')}
- Head coach: {team.get('coaching_staff', {}).get('head_coach', 'Unknown')}
- Market size: {team.get('market', {}).get('market_size', 'unknown')}
- Branding colors: {team.get('branding', {}).get('colors', [])}
- Arena: {team.get('arena', {}).get('name', 'Unknown')}
- Team identity: defensive, offensive, balanced, gritty, fast-paced, physical, technical, disciplined, or youth-focused as appropriate

ROSTER (DO NOT CHANGE):
{roster_json}

Chemistry relationships must reference ONLY players from this list:
{player_names}

If referencing teammates in chemistry, the name must exactly match one of the players in the provided roster list.

These players already exist in the team.

You MUST enrich the SAME players.

You are NOT allowed to:
- add players
- remove players
- rename players
- change jersey numbers
- change positions

IMPORTANT RULES:

NAME VARIETY
- Reusing names across the league is acceptable, but avoid excessive repetition.
- If multiple players share the same name, they should differ in background, nationality, age, or role.

HEIGHT / POSITION REALISM
- PG should usually be around 178-188 cm
- SG should usually be around 185-195 cm
- SF should usually be around 193-203 cm
- PF should usually be around 200-210 cm
- C should usually be around 208-218 cm
- Do not produce unrealistic combinations like a 183 cm center

AGE RULES
- Age must be between 18 and 38.
- Age 18 players should usually be rookies.
- Players over 34 should usually have declining athleticism.

STAR PLAYER RULES
- Every team must have at least one star-level player.
- A star does NOT have to lead the team in points.
- A star is defined by presence, usage, leadership, minutes, media attention, and overall impact.
- Stars usually play 30+ minutes and have strong confidence and reputation.
- Only 1–2 stars per team.

TEAM SCORING BALANCE
- Total team scoring should roughly fall between 95 and 120 points per game.
- Bench players should score less.
- Role players should not outscore primary options.

ROLE REALISM
- Minutes must match role.
- Bench prospects should not play starter minutes.
- Defensive specialists may have low scoring but high defensive impact.

FAMILY / LEAGUE LORE
- It is allowed, but not required, for a player to be a brother, cousin, or son of a former player.
- If you include family background, keep it concise and believable.

CHEMISTRY RULES
- Players do not all get along.
- Some players compete for minutes.
- Some mentor younger players.
- Some personalities clash.

RETURN JSON ONLY.

Schema:
{{
  "players": [
    {{
      "player_id": "...",
      "name": "...",
      "number": 0,
      "position": "PG|SG|SF|PF|C",
      "height_cm": 0,
      "age": 0,
      "nationality": "...",
      "weight_kg": 0,
      "wingspan_cm": 0,
      "personal_info": {{
        "birth_city": "...",
        "country_of_origin": "...",
        "languages": ["..."],
        "family_background": "..."
      }},
      "basketball_profile": {{
        "play_style": ["..."],
        "strengths": ["..."],
        "weaknesses": ["..."],
        "basketball_iq": 0,
        "motor": 0
      }},
      "personality": {{
        "leadership": 0,
        "ego_level": 0,
        "competitiveness": 0,
        "work_ethic": 0,
        "confidence": 0,
        "temperament": "calm|balanced|intense|hot-headed",
        "locker_room_role": "leader|mentor|glue-guy|quiet-worker|joker|challenger|neutral"
      }},
      "role_in_team": {{
        "status": "franchise_star|star|starter|sixth_man|role_player|defensive_specialist|bench_prospect|veteran_anchor|rookie_project",
        "is_star": false,
        "rotation_role": "...",
        "minutes_expectation": 0
      }},
      "season_projection": {{
        "games": 0,
        "games_started": 0,
        "minutes": 0,
        "ppg": 0,
        "rpg": 0,
        "apg": 0,
        "spg": 0,
        "bpg": 0,
        "topg": 0,
        "fg_pct": 0,
        "three_pct": 0,
        "ft_pct": 0
      }},
      "contract": {{
        "years_remaining": 0,
        "annual_salary_m": 0,
        "contract_tier": "...",
        "trade_value": 0
      }},
      "health_profile": {{
        "durability": 0,
        "injury_risk": 0,
        "conditioning": 0
      }},
      "development_profile": {{
        "ceiling": 0,
        "current_impact": 0,
        "improvement_areas": ["..."]
      }},
      "media_and_reputation": {{
        "fan_popularity": 0,
        "media_attention": 0
      }},
      "chemistry": {{
        "best_teammates": [
          {{
            "player": "...",
            "score": 0,
            "reason": "..."
          }}
        ],
        "neutral_teammates": ["..."],
        "low_chemistry_teammates": [
          {{
            "player": "...",
            "score": 0,
            "reason": "..."
          }}
        ]
      }},
      "background_story": {{
        "path_to_team": "...",
        "off_court_interests": ["..."],
        "public_reputation": "..."
      }},
      "family_relation": {{
        "type": "...",
        "related_player": "..."
      }},
      "profile_summary": "2-4 sentence summary"
    }}
  ]
}}
"""

    data = generate_data(prompt)
    players = data.get("players", [])
    players = safe_list(players)
    enriched = validate_player_enrichment(players, roster_seed)
    enriched = enforce_roster_constraints(enriched)

    team["roster"] = enriched
    team["team_dynamics"] = build_team_dynamics(team_name, enriched)

    return team


# ---------------------------------------------------------
# TEAM DYNAMICS
# ---------------------------------------------------------
def build_team_dynamics(team_name: str, roster: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not roster:
        return {
            "team_name": team_name,
            "locker_room_leaders": [],
            "star_players": [],
            "mentor_pairs": [],
            "notable_friendships": [],
            "internal_tensions": [],
        }

    leaders = sorted(
        roster,
        key=lambda p: (
            p.get("personality", {}).get("leadership", 0),
            p.get("personality", {}).get("work_ethic", 0),
        ),
        reverse=True,
    )[:3]

    stars = [p["name"] for p in roster if p.get("role_in_team", {}).get("is_star", False)]
    if not stars:
        stars = [
            p["name"]
            for p in sorted(
                roster,
                key=lambda x: (
                    x.get("role_in_team", {}).get("minutes_expectation", 0),
                    x.get("personality", {}).get("confidence", 0),
                ),
                reverse=True,
            )[:2]
        ]

    mentor_pairs = []
    friendships = []
    tensions = []
    by_name = {p["name"]: p for p in roster if p.get("name")}

    for player in roster:
        role = player.get("personality", {}).get("locker_room_role", "")

        if role == "mentor":
            for mate in player.get("chemistry", {}).get("best_teammates", []):
                mate_name = mate.get("player")
                if mate_name and mate_name in by_name:
                    mentor_pairs.append(
                        {
                            "mentor": player["name"],
                            "teammate": mate_name,
                            "reason": mate.get("reason", "Strong veteran guidance."),
                        }
                    )

        for mate in player.get("chemistry", {}).get("best_teammates", []):
            mate_name = mate.get("player")
            if mate_name and mate_name in by_name:
                pair = tuple(sorted([player["name"], mate_name]))
                friendships.append(
                    {
                        "players": list(pair),
                        "reason": mate.get("reason", "Strong on-court and off-court connection."),
                    }
                )

        for mate in player.get("chemistry", {}).get("low_chemistry_teammates", []):
            mate_name = mate.get("player")
            if mate_name and mate_name in by_name:
                pair = tuple(sorted([player["name"], mate_name]))
                tensions.append(
                    {
                        "players": list(pair),
                        "reason": mate.get("reason", "Role overlap or personality tension."),
                    }
                )

    def unique_pairs(items: List[Dict[str, Any]], key_name: str) -> List[Dict[str, Any]]:
        seen = set()
        unique = []
        for item in items:
            pair = tuple(item.get(key_name, []))
            if pair and pair not in seen:
                seen.add(pair)
                unique.append(item)
        return unique

    friendships = unique_pairs(friendships, "players")[:8]
    tensions = unique_pairs(tensions, "players")[:8]

    seen_mentor = set()
    uniq_mentor_pairs = []
    for item in mentor_pairs:
        key = (item["mentor"], item["teammate"])
        if key not in seen_mentor:
            seen_mentor.add(key)
            uniq_mentor_pairs.append(item)

    return {
        "team_name": team_name,
        "locker_room_leaders": [p["name"] for p in leaders],
        "star_players": stars[:3],
        "mentor_pairs": uniq_mentor_pairs[:6],
        "notable_friendships": friendships,
        "internal_tensions": tensions,
    }


# ---------------------------------------------------------
# GENERATE PLAYERS
# ---------------------------------------------------------
def generate_players():
    total_start = time.time()

    team_files = sorted(glob.glob("data/teams/*.json"))
    if not team_files:
        raise FileNotFoundError("No team JSON files found in data/teams")

    os.makedirs("data/players", exist_ok=True)

    updated_count = 0

    for path in team_files:
        team_start = time.time()

        with open(path, "r", encoding="utf-8") as f:
            team = json.load(f)

        team_name = f"{team.get('city', 'Unknown')} {team.get('name', 'Team')}"
        team_slug = slugify(team_name)

        print(f"\nEnriching players for {team_name}...")

        try:
            enriched_team = enrich_team_players(team)
        except Exception as e:
            print(f"Failed on {team_name}: {e}")
            continue

        with open(path, "w", encoding="utf-8") as f:
            json.dump(enriched_team, f, indent=2, ensure_ascii=False)
        team_players_dir = os.path.join("data/players", team_slug)
        os.makedirs(team_players_dir, exist_ok=True)

        os.makedirs("data/team_summaries", exist_ok=True)

        team_summary_md_path = os.path.join("data/team_summaries", f"{team_slug}.md")
        with open(team_summary_md_path, "w", encoding="utf-8") as f:
            f.write(create_team_players_markdown(enriched_team))
        for player in safe_list(enriched_team.get("roster")):
            player_slug = slugify(str(player.get("name", "unknown_player")))
            
            player_json_path = os.path.join(team_players_dir, f"{player_slug}.json")
            with open(player_json_path, "w", encoding="utf-8") as pf:
                json.dump(player, pf, indent=2, ensure_ascii=False)

            player_md_path = os.path.join(team_players_dir, f"{player_slug}.md")
            with open(player_md_path, "w", encoding="utf-8") as pf:
                pf.write(create_player_markdown(enriched_team, player))

        updated_count += 1

        team_elapsed = time.time() - team_start
        print(
            f"Finished {team_name} | "
            f"Players: {len(safe_list(enriched_team.get('roster')))} | "
            f"Time: {team_elapsed:.2f} seconds"
        )

    total_elapsed = time.time() - total_start
    print(f"\n{updated_count} team player profiles generated successfully.")
    print(f"Total generation time: {total_elapsed:.2f} seconds")


if __name__ == "__main__":
    generate_players()