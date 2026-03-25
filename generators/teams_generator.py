import json
import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from synthetic_data_generator import generate_data
from name_generator import generate_name_by_country

def slugify(text: str):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"

import random


def enforce_unique_player_names(roster):

    for player in roster:

        nationality = player.get("nationality", "USA")

        name = player.get("name", "").strip()

        # אם אין שם → צור שם לפי מדינה
        if not name:
            name = generate_name_by_country(nationality)

        # אם השם כבר קיים בליגה → צור שם חדש
        while name.lower() in USED_PLAYER_NAMES:
            name = generate_name_by_country(nationality)

        USED_PLAYER_NAMES.add(name.lower())

        player["name"] = name

    return roster

def create_markdown(team):
    coaching = team.get("coaching_staff", {})
    assistants_list = coaching.get("assistant_coaches", [])
    assistants = ", ".join(assistants_list) if assistants_list else "None"

    # Formatting Rivalries: Added the team name to each line for redundancy
    team_name = team.get('name','Team')
    rivalries_list = team.get("rivalries", [])
    rivalries = "\n".join([
        f"* **{team_name} Rival:** {r.get('team','Unknown')} | Type: {r.get('type','unknown')}" 
        for r in rivalries_list
    ]) if rivalries_list else "* No major rivalries recorded."

    # Formatting Roster: This is the most important part for splitting.
    # Every line now starts with the Team Name + Player Name.
    roster_list = team.get("roster", [])
    roster = "\n".join([
        f"* **{team_name} Player:** {p.get('name','Unknown')} (#{p.get('number','?')}) | "
        f"Pos: {p.get('position','?')} | H: {p.get('height_cm','?')}cm | "
        f"Age: {p.get('age','?')} | Nat: {p.get('nationality','Unknown')}"
        for p in roster_list
    ]) if roster_list else "* No roster data available."

    sponsors = team.get("sponsors", {})
    ownership = team.get("ownership", {})
    arena = team.get("arena", {})
    market = team.get("market", {})
    branding = team.get("branding", {})
    history = team.get("history", {})

    team_full_name = f"{team.get('city','Unknown')} {team.get('name','Team')}"
    canonical_slug = slugify(team_full_name)
    search_tags = [team.get('name'), team.get('city'), team.get('conference'), "sports"]
    search_tags = [t for t in search_tags if t] # Remove None values
    canonical_slug = slugify(team_full_name)

    return f"""# {team_full_name} Team Profile

**Team Metadata**
* **Canonical Slug:** {canonical_slug}
* **Team ID:** {team.get("id","unknown")}
* **Search Tags:** {", ".join(search_tags)}

---

## Team Overview
* **Full Name:** {team_full_name}
* **Description:** {team.get("description","No description available.")}

## Location and Market
* **Home City:** {team.get("city","Unknown")} ({team.get("country","Unknown")})
* **Conference:** {team.get("conference","?")} | **Division:** {team.get("division","?")}
* **Market Size:** {market.get("market_size","unknown")}
* **Population:** {market.get("city_population","?")}

## Venue and Sponsorship
* **Arena:** {arena.get("name","Unknown Arena")} (Capacity: {arena.get("capacity","?")})
* **Primary Sponsor:** {sponsors.get("primary","Unknown")}
* **Secondary Sponsor:** {sponsors.get("secondary","Unknown")}

## Leadership and Coaching
* **Owner:** {ownership.get("owner","Unknown")}
* **GM:** {ownership.get("general_manager","Unknown")}
* **Head Coach:** {coaching.get("head_coach","Unknown")}
* **Assistants:** {assistants}
* **Strength Coach:** {coaching.get("strength_coach","Unknown")}

## Branding and History
* **Colors:** {", ".join(branding.get("colors",["Unknown"]))}
* **Mascot:** {branding.get("mascot","Unknown")}
* **Championships:** {history.get("championships",0)}
* **Conference Titles:** {history.get("conference_titles",0)}
* **Playoff Appearances:** {history.get("playoff_appearances",0)}

## Roster
{roster}

## Rivalries
{rivalries}
"""


def team_is_valid(team):

    required_fields = [
        "arena",
        "ownership",
        "branding",
        "market",
        "coaching_staff",
        "sponsors",
        "description",
        "roster"
    ]

    for field in required_fields:
        if field not in team:
            return False

    roster = team.get("roster")

    if not isinstance(roster, list) or len(roster) < 8:
        return False

    arena = team.get("arena", {})
    if not arena.get("name"):
        return False

    ownership = team.get("ownership", {})
    if not ownership.get("owner"):
        return False

    coaching = team.get("coaching_staff", {})
    if not coaching.get("head_coach"):
        return False

    sponsors = team.get("sponsors", {})
    if not sponsors.get("primary"):
        return False

    return True

def generate_valid_team_batch(prompt, expected_teams, max_attempts=5):

    for attempt in range(max_attempts):

        print(f"\nTeam generation attempt {attempt+1}")

        data = generate_data(prompt)

        teams = data.get("teams")

        if not isinstance(teams, list):
            print("Invalid response structure")
            continue

        if len(teams) != expected_teams:
            print(f"Wrong number of teams: {len(teams)} expected {expected_teams}")
            continue

        valid = True

        for team in teams:

            if not team_is_valid(team):
                print("Invalid team detected")
                valid = False
                break

        if valid:
            print("Valid teams generated")
            return teams

    raise Exception("Failed to generate valid teams after retries")


USED_PLAYER_NAMES = set()

def generate_teams():

    with open("data/league/league.json") as f:
        league = json.load(f)

    teams_structure = league["teams"]
    champions = league["history"]["championship_history"]

    championship_count = {}
    for c in champions:
        team = c["champion"]
        championship_count[team] = championship_count.get(team, 0) + 1

    generated_teams = []

    for base in teams_structure:

        print(f"\nGenerating team: {base['city']} {base['name']}")

        prompt = f"""
You are generating a single fictional basketball team.

LEAGUE CONTEXT:
{json.dumps(league, indent=2)}

ALREADY GENERATED TEAMS:
{json.dumps(generated_teams, indent=2)}

CURRENT TEAM:
{json.dumps(base, indent=2)}

The CURRENT TEAM already contains:
city
country
name
conference
division

DO NOT change these values.

You must ADD additional information.

------------------------------------------------

ROSTER RULES

- Each team must have 8–12 players
- Jersey numbers must be unique within each team
- Jersey numbers must be realistic basketball numbers between 0 and 99
- Prefer common basketball numbers such as 0, 00, 1–55
- Avoid sequential numbering like 1,2,3,4,5,6,7
- Numbers should appear randomly distributed like real teams- Positions must be PG, SG, SF, PF, or C
- Heights must be realistic

------------------------------------------------

NATIONALITY RULES
- At least 4 players from USA
- Maximum 6 players from the same nationality
- Minimum 3 nationalities per team



PLAYER NAME RULES

All player names must be fictional.

Do NOT generate real players such as:
LeBron James
Stephen Curry
Nikola Jokic
Luka Doncic

Names must match nationality.

If you see the exact name in other team roster change to another name.

------------------------------------------------

IMPORTANT

Do not generate placeholders such as:
Replacement Player
Unknown

------------------------------------------------

Return JSON ONLY in this format:

{{
 "team": {{
   "arena":{{"name":"string","capacity":number}},
   "ownership":{{"owner":"string","president":"string","general_manager":"string"}},
   "branding":{{"colors":[],"mascot":"string"}},
   "market":{{"market_size":"string","city_population":number}},
   "history":{{"conference_titles":number,"division_titles":number,"playoff_appearances":number}},
   "coaching_staff":{{"head_coach":"string","assistant_coaches":[],"strength_coach":"string"}},
   "roster":[{{"name":"string","number":number,"position":"PG|SG|SF|PF|C","height_cm":number,"age":number,"nationality":"string"}}],
   "rivalries":[{{"team":"string","type":"string"}}],
   "sponsors":{{"primary":"string","secondary":"string"}},
   "description":"string"
 }}
}}
"""

        data = generate_data(prompt)

        team_extra = data.get("team", {})

        team = {**base, **team_extra}

        roster = team.get("roster", [])
        roster = enforce_unique_player_names(roster)

        if len(roster) < 8:
            print("Roster too small — fixing")
            while len(roster) < 8:
                roster.append({
                    "name": f"Generated Player {len(roster)+1}",
                    "number": 90 + len(roster),
                    "position": "SG",
                    "height_cm": 195,
                    "age": 24,
                    "nationality": "USA"
                })

        if len(roster) > 12:
            roster = roster[:12]

        team["roster"] = roster

        team_full_name = f"{team['city']} {team['name']}"
        championships = championship_count.get(team_full_name, 0)

        team.setdefault("history", {})
        team["history"]["championships"] = championships

        generated_teams.append(team)

    os.makedirs("data/teams", exist_ok=True)

    for i, team in enumerate(generated_teams, start=1):

        team["id"] = f"team_{i:04d}"

        filename = slugify(f"{team['city']} {team['name']}")

        json_path = f"data/teams/{filename}.json"
        md_path = f"data/teams/{filename}.md"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(team, f, indent=2, ensure_ascii=False)

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(create_markdown(team))

    print(f"{len(generated_teams)} teams generated successfully.")


if __name__ == "__main__":
    generate_teams()