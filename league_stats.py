import os
import re
import pandas as pd
from pathlib import Path

# Paths configuration
PLAYERS_PATH = "data/players" 
KB_OUTPUT_PATH = "data/league_stats"

def safe_extract(pattern, content, default="0.0"):
    """Helper to extract data using regex."""
    match = re.search(pattern, content)
    return match.group(1).strip() if match else default

def generate_kb():
    """Generates RAG-optimized KB files with 3 separate tables per file. 
       Includes Team, Age, Nat, Pos, and PPG for maximum context in every row."""
    os.makedirs(KB_OUTPUT_PATH, exist_ok=True)
    all_players = []

    if not os.path.exists(PLAYERS_PATH):
        print(f"❌ Error: Folder {PLAYERS_PATH} not found!")
        return

    # Data extraction loop
    for root, dirs, files in os.walk(PLAYERS_PATH):
        for file in files:
            if file.endswith(".md"):
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    content = f.read()
                    name = Path(file).stem.replace('_', ' ').title()
                    try:
                        # Extract basic info
                        team = safe_extract(r"\*\*Current Team:\*\* (.*)", content, "Free Agent")
                        
                        # Parsing Identity line: #21 | SF | India
                        identity_line = safe_extract(r"\*\*Identity:\*\* (.*)", content, "")
                        parts = [p.strip() for p in identity_line.split("|")]
                        pos = parts[1] if len(parts) > 1 else "N/A"
                        nat = parts[2] if len(parts) > 2 else "N/A"
                        
                        # Stats and Personal Info
                        age = int(safe_extract(r"Age: (\d+)", content, "0"))
                        ppg = float(safe_extract(r"([\d\.]+) PPG", content))
                        rpg = float(safe_extract(r"([\d\.]+) RPG", content))
                        apg = float(safe_extract(r"([\d\.]+) APG", content))
                        spg = float(safe_extract(r"([\d\.]+) SPG", content))
                        
                        # Financials
                        salary_match = re.search(r"\$([\d\.]+)M Annual", content)
                        salary = float(salary_match.group(1)) if salary_match else 0.0

                        all_players.append({
                            "Name": name, "Team": team, "Pos": pos, 
                            "Age": age, "Nat": nat, "PPG": ppg, "RPG": rpg, 
                            "APG": apg, "SPG": spg, "Salary(M)": salary
                        })
                    except Exception as e:
                        print(f"⚠️ Error parsing {file}: {e}")

    df = pd.DataFrame(all_players)

    def save_3_tables_in_one_file(filename, sort_by, title, ascending=False):
        """Creates one markdown file containing 3 distinct tables.
           Columns: Rank, Name, Team, Age, Nat, Pos, PPG, [Target Stat]"""
        sorted_df = df.sort_values(by=sort_by, ascending=ascending).copy()
        sorted_df.insert(0, 'Rank', range(1, len(sorted_df) + 1))
        
        # Consistent columns for all tables to ensure RAG accuracy.
        # We always include Team, Age, Nat, Pos, and PPG for maximum context.
        if sort_by == "PPG":
            display_cols = ['Rank', 'Name', 'Team', 'Age', 'Nat', 'Pos', 'PPG']
        else:
            display_cols = ['Rank', 'Name', 'Team', 'Age', 'Nat', 'Pos', 'PPG', sort_by]
            
        clean_df = sorted_df[display_cols]

        n = len(clean_df)
        parts = [
            (clean_df.iloc[:n//3], "TOP TIER"),
            (clean_df.iloc[n//3 : 2*n//3], "MIDDLE TIER"),
            (clean_df.iloc[2*n//3:], "BOTTOM TIER")
        ]

        with open(f"{KB_OUTPUT_PATH}/{filename}.md", "w", encoding="utf-8") as f:
            f.write(f"# {title} - Full Rankings\n")
            f.write(f"**Keywords:** {sort_by} ranking, leaderboard, basketball positions, player age, nationality\n\n")
            
            for part_df, label in parts:
                f.write(f"## {label} ({sort_by})\n")
                f.write(part_df.to_markdown(index=False, tablefmt="pipe"))
                f.write("\n\n---\n\n")

    # --- Generate All Required Files ---
    save_3_tables_in_one_file("kb_scoring", "PPG", "League Scoring Leaders")
    save_3_tables_in_one_file("kb_rebounds", "RPG", "League Rebounding Leaders")
    save_3_tables_in_one_file("kb_assists", "APG", "League Assists Leaders")
    save_3_tables_in_one_file("kb_steals", "SPG", "League Steals Leaders")
    save_3_tables_in_one_file("kb_salaries", "Salary(M)", "League Salary Rankings")
    save_3_tables_in_one_file("kb_age", "Age", "League Age Rankings", ascending=False)

    # Nationality Summary Table with Keywords
    nat_counts = df['Nat'].value_counts().reset_index()
    nat_counts.columns = ['Nationality', 'Count']
    with open(f"{KB_OUTPUT_PATH}/kb_nationalities.md", "w", encoding="utf-8") as f:
        f.write("# League Nationality Distribution Summary\n")
        f.write("**Keywords:** country origin, nationality breakdown, global players, player roots\n\n")
        f.write(nat_counts.to_markdown(index=False, tablefmt="pipe"))

    print(f"✅ KB updated! All tables now include Nat, Pos, Age, Team, and PPG with Keywords.")

if __name__ == "__main__":
    generate_kb()