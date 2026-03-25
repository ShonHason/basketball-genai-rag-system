import os


def create_project_structure():

    folders = [
        "data",
        "data/sport",
        "data/league",
        "data/teams",
        "data/players"
    ]

    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    print("Project structure created successfully.")


if __name__ == "__main__":
    create_project_structure()