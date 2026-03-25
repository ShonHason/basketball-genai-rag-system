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
    

def create_markdown(league):
    structure = league.get("structure", {})
    rules = league.get("rules", {})
    leadership = league.get("leadership", {})
    business = league.get("business", {})
    history = league.get("history", {})

    # Using Bullet points for lists ensures the Recursive splitter breaks 
    # BETWEEN items rather than in the middle of a line.
    conferences = "\n".join([f"* **Conference:** {c}" for c in structure.get("conferences", [])])
    divisions = "\n".join([f"* **Division:** {d}" for d in structure.get("divisions", [])])
    tv = "\n".join([f"* **TV Partner:** {t}" for t in business.get("tv_partners", [])])

    # Enhanced Teams list: Every line contains the core keywords.
    # This ensures that even if the list is split, every chunk is self-descriptive.
    teams = "\n".join([
        (
            f"* **Team:** {t.get('city','Unknown')} {t.get('name','Team')} | "
            f"Location: {t.get('country','Unknown')} | "
            f"Group: {t.get('conference','?')} - {t.get('division','?')}"
        )
        for t in league.get("teams", [])
    ])

    champions = "\n".join([
        f"* **{c.get('year')} Champion:** {c.get('champion')}"
        for c in history.get("championship_history", [])
    ])

    championship_table = "\n".join([
        f"* **{row.get('team')}:** {row.get('champions_count')} Titles"
        for row in history.get("championship_table", [])
    ])

    canonical_slug = slugify(league.get("name", "league"))
    search_tags = [
        league.get("name", ""),
        canonical_slug,
        league.get("sport", ""),
        str(structure.get("number_of_teams", "")),
        str(league.get("founded_year", "")),
    ]

    # Structure: Double newlines ensure the Recursive splitter treats sections as blocks.
    return f"""# {league.get("name")} League Profile

**League Metadata**
* **Canonical Slug:** {canonical_slug}
* **League ID:** {league.get("id", "unknown")}
* **Sport:** {league.get("sport")}
* **Founded:** {league.get("founded_year")}
* **Status Year:** {league.get("current_year")}
* **Tags:** {", ".join(search_tags)}

---

## League Overview
* **Name:** {league.get("name")}
* **Description:** {league.get("description","No description available.")}

## League Structure
* **Total Teams:** {structure.get("number_of_teams")}
{conferences}
{divisions}

## Teams and Affiliations
{teams}

## Rules and Regulations
* **Roster Size:** {rules.get("roster_size")} players
* **Salary Cap:** {rules.get("salary_cap")}
* **Playoff Format:** {rules.get("playoff_format")}

## League Leadership
* **Commissioner:** {leadership.get("commissioner")}
* **Deputy Commissioner:** {leadership.get("deputy_commissioner")}
* **President:** {leadership.get("league_president")}

## Business Operations
* **Headquarters:** {business.get("headquarters_city")}
* **Annual Revenue:** {business.get("annual_revenue")}
**Television Partners:**
{tv}

## Championship History
{champions}

## Historical Records
* **Most Titles:** {", ".join(history.get("most_championships_teams", []))}
* **Defending Champion:** {history.get("most_recent_champion")}

## Championship Leaderboard
{championship_table}
"""

def generate_league():

    with open("data/sport/sport.json") as f:
        sport = json.load(f)

    structure = sport.get("recommended_league_structure", {})
    min_teams = structure.get("min_teams", 12)
    max_teams = structure.get("max_teams", 32)

    prompt = f"""
You are generating a professional sports league.

SPORT:
{json.dumps(sport, indent=2)}

IMPORTANT RULES

Cities MUST be real world cities.



Team names MUST be fictional.

Team name must be ONE fictional word.

Cities should not repeat.

Constraints:
- A country may not appear more than 6 times in the teams list
- At least 5 teams must be from cities located in USA
- At least 4 teams must be from cities located in Europe
- At least 1 teams must be from cities located in Israel
- number_of_teams between {min_teams} and {max_teams}
- conferences usually 2
- divisions between 4 and 6

Generate championship history.

Return JSON only.

Schema:

{{
 "league": {{
  "name": "...",
  "sport": "{sport["name"]}",
  "founded_year": 0, (before 2020)
  "current_year": 2026,

  "description": "...",

  "structure": {{
    "number_of_teams": 0,
    "conferences": ["..."],
    "divisions": ["..."]
  }},

  "teams": [
    {{
      "city": "...",
      "country": "...",
      "name": "...",
      "conference": "...",
      "division": "..."
    }}
  ],

  "rules": {{
    "roster_size": 0,
    "salary_cap": 0,
    "playoff_format": "..."
  }},

  "leadership": {{
    "commissioner": "...",
    "deputy_commissioner": "...",
    "league_president": "..."
  }},

  "business": {{
    "headquarters_city": "...",
    "annual_revenue": 0,
    "tv_partners": ["..."]
  }},

  "history": {{
    "championship_history": [
      {{
        "year": 0,
        "champion": "City Team"
      }}
    ],
    "most_recent_champion": "...",
  
 }}
}}
}}
"""

    data = generate_data(prompt)

    league = data.get("league")

    if not league:
        raise ValueError("Model did not return league data")

    history = league.get("history", {})
    champions = history.get("championship_history", [])

    counts = {}

    for c in champions:
        team = c.get("champion")
        counts[team] = counts.get(team, 0) + 1

    table = [
        {"team": team, "champions_count": count}
        for team, count in counts.items()
    ]

    history["championship_table"] = table

    max_titles = max(counts.values())
    history["most_championships_teams"] = [
        team for team, count in counts.items() if count == max_titles
    ]

    history["most_recent_champion"] = champions[-1]["champion"]
    if not league:
        raise ValueError("Model did not return league data")

    # --------------------------
    # Validate team count
    # --------------------------

    teams = league.get("teams", [])
    expected = league.get("structure", {}).get("number_of_teams")

    if expected and len(teams) != expected:
        print(f"Model returned {len(teams)} teams but expected {expected}")

    league["id"] = "league_0001"

    os.makedirs("data/league", exist_ok=True)

    with open("data/league/league.json", "w", encoding="utf-8") as f:
        json.dump(league, f, indent=2, ensure_ascii=False)

    md = create_markdown(league)

    with open("data/league/league.md", "w", encoding="utf-8") as f:
        f.write(md)

    print("League generated successfully")


if __name__ == "__main__":
    generate_league()