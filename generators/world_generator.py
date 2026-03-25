import os
import sys
import shutil

# מאפשר import מה-root של הפרויקט
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# -----------------------------
# RESET DATA
# -----------------------------

def reset_data():

    if os.path.exists("data"):
        print("Deleting old data folder...")
        shutil.rmtree("data")

    if os.path.exists("vector_db"):
        print("Deleting old vector database...")
        shutil.rmtree("vector_db")

    print("Old data cleared\n")


# -----------------------------
# CREATE PROJECT STRUCTURE
# -----------------------------

from setup_universe import create_project_structure


# -----------------------------
# WORLD GENERATION
# -----------------------------

from generators.sport_generator import generate_sport
from generators.league_generator import generate_league
from generators.teams_generator import generate_teams
from generators.players_generator import generate_players
from league_stats import generate_ultimate_league_stats

def generate_world():

    print("Generating sport...")
    generate_sport()

    print("Generating league...")
    generate_league()

    print("Generating teams...")
    generate_teams()

    print("Generating players...")
    generate_players()

    print("Generating stats...")
    generate_ultimate_league_stats()
     
    print("World generation complete\n")


# -----------------------------
# VECTOR DATABASE
# -----------------------------

from rag_setup import build_knowledge_base


def build_vectors():

    print("Building vector database...")
    build_knowledge_base()
    print("Vector database ready\n")


# -----------------------------
# CHAT
# -----------------------------

from chatbot import answer_question

def start_chat():

    print("\nAeroBall RAG Assistant")
    print("Type 'exit' to quit\n")

    history = []

    while True:

        question = input("You: ")

        if question.lower() in ["exit", "quit"]:
            break

        answer = answer_question(question, history)

        print("\nAssistant:", answer)
        print()


# -----------------------------
# FULL PIPELINE
# -----------------------------

def run_pipeline():

    reset_data()

    create_project_structure()

    try:
        generate_world()
    except Exception as e:
        print("World generation failed:", e)
        return

    build_vectors()

    start_chat()


# -----------------------------
# MAIN
# -----------------------------

if __name__ == "__main__":

    run_pipeline()