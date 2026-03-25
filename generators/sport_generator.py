import json
import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from synthetic_data_generator import generate_data


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def validate_sport(sport):
    required_fields = [
        "name",
        "description",
        "team_structure",
        "game_structure",
        "timing_rules",
        "scoring_rules",
        "foul_rules",
        "court",
        "recommended_league_structure"
    ]

    for field in required_fields:
        if field not in sport:
            raise ValueError(f"Missing required field: {field}")

    league = sport["recommended_league_structure"]

    if league["min_teams"] >= league["max_teams"]:
        raise ValueError("Invalid league structure: min_teams >= max_teams")

def create_markdown(sport):
    canonical_slug = slugify(sport["name"])
    search_tags = [
        sport["name"],
        canonical_slug,
        sport["game_structure"]["period_type"],
        str(sport["team_structure"]["players_on_court"]),
        str(sport["timing_rules"]["shot_clock_seconds"]),
    ]

    # Added explicit double newlines (\n\n) between sections.
    # Used bold keys and bullet points to help the Recursive splitter 
    # find logical line breaks if a section exceeds 600 chars.
    return f"""# {sport["name"]} Rulebook

**Metadata**
* Canonical Sport Slug: {canonical_slug}
* Sport ID: {sport.get("id", "unknown")}
* Search Tags: {", ".join(search_tags)}

---

## Overview
* **Name:** {sport["name"]}
* **Description:** {sport["description"]}

## Team Structure
* **Players On Court:** {sport["team_structure"]["players_on_court"]}
* **Max Roster Size:** {sport["team_structure"]["max_roster_size"]}

## Game Structure
* **Period Type:** {sport["game_structure"]["period_type"]}
* **Periods:** {sport["game_structure"]["periods"]}
* **Minutes Per Period:** {sport["game_structure"]["minutes_per_period"]}
* **Overtime Minutes:** {sport["game_structure"]["overtime_minutes"]}

## Timing Rules
* **Shot Clock:** {sport["timing_rules"]["shot_clock_seconds"]} seconds
* **Backcourt:** {sport["timing_rules"]["backcourt_seconds"]} seconds

## Scoring Rules
* **Free Throw:** {sport["scoring_rules"]["free_throw"]} points
* **Field Goal:** {sport["scoring_rules"]["field_goal"]} points
* **Long Range:** {sport["scoring_rules"]["long_range_shot"]} points

## Foul Rules
* **Max Personal Fouls:** {sport["foul_rules"]["max_personal_fouls"]}
* **Bonus After:** {sport["foul_rules"]["team_foul_bonus_after"]} fouls

## Court Dimensions
* **Length:** {sport["court"]["length_meters"]}m
* **Width:** {sport["court"]["width_meters"]}m
* **Rim Height:** {sport["court"]["rim_height_meters"]}m

## League Structure
* **Min Teams:** {sport["recommended_league_structure"]["min_teams"]}
* **Max Teams:** {sport["recommended_league_structure"]["max_teams"]}
* **Recommended Roster:** {sport["recommended_league_structure"]["recommended_roster_size"]}
"""


def generate_sport():
    prompt = """
You are designing a professional team sport inspired by basketball.

The sport must be realistic and internally consistent.

Return JSON only.

Important constraints:

- players_on_court must be between 3 and 7
- max_roster_size must be greater than players_on_court
- periods must be between 2 and 4
- minutes_per_period must be between 8 and 15
- overtime_minutes must be between 3 and 10
- shot_clock_seconds must be between 10 and 30
- min_teams must be less than max_teams
- min_teams between 8 and 20
- max_teams between 10 and 32

Schema:

{
 "sport":{

   "name":"...",
   "description":"short explanation of the sport",

   "team_structure":{
      "players_on_court":0,
      "max_roster_size":0
   },

   "game_structure":{
      "period_type":"quarters or halves",
      "periods":0,
      "minutes_per_period":0,
      "overtime_minutes":0
   },

   "timing_rules":{
      "shot_clock_seconds":0,
      "backcourt_seconds":0
   },

   "scoring_rules":{
      "free_throw":0,
      "field_goal":0,
      "long_range_shot":0
   },

   "foul_rules":{
      "max_personal_fouls":0,
      "team_foul_bonus_after":0
   },

   "court":{
      "length_meters":0,
      "width_meters":0,
      "rim_height_meters":0
   },

   "recommended_league_structure":{
      "min_teams":0,
      "max_teams":0,
      "recommended_roster_size":0
   }

 }
}
"""

    data = generate_data(prompt)

    sport = data.get("sport")

    if not sport:
        raise ValueError("Model did not return sport data")
        
    sport["id"] = "sport_0001"

    validate_sport(sport)

    os.makedirs("data/sport", exist_ok=True)

    with open("data/sport/sport.json", "w", encoding="utf-8") as f:
        json.dump(sport, f, indent=2, ensure_ascii=False)

    md_content = create_markdown(sport)

    with open("data/sport/sport.md", "w", encoding="utf-8") as f:
        f.write(md_content)

    print("Sport generated successfully.")


if __name__ == "__main__":
    generate_sport()