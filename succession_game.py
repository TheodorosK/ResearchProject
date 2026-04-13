import time


def print_slow(text):
    for char in text:
        print(char, end='', flush=True)
        time.sleep(0.01)
    print("\n")


class SuccessionGame:
    def __init__(self):
        # Core Resources
        self.stability = 50      # Mathematics & AI Accuracy
        self.pluralism = 50      # Decoloniality & Diversity
        self.corporate = 50      # C-Suite Approval & Traditional Power
        self.budget = 3          # R&D Budget for Tech Tree
        self.turn = 1
        self.max_turns = 3

    def display_dashboard(self):
        print("\n" + "=" * 50)
        print(f"YEAR: {self.turn} | R&D BUDGET: {self.budget} Units")
        print(f"Algorithmic Stability : {self.stability}/100")
        print(f"Epistemic Pluralism   : {self.pluralism}/100")
        print(f"Corporate Power       : {self.corporate}/100")
        print("=" * 50 + "\n")

    def check_game_over(self):
        if self.stability <= 0:
            print_slow(
                "GAME OVER: Your algorithms destabilized. "
                "The succession pipeline collapsed into chaos."
            )
            return True
        elif self.pluralism <= 0:
            print_slow(
                "GAME OVER (Epistemicide): You created a perfect algorithmic monoculture. "
                "Diverse talent abandoned the company, and market disruption destroyed the firm."
            )
            return True
        elif self.corporate <= 0:
            print_slow("GAME OVER: The C-Suite lost faith in your project and fired you.")
            return True
        return False

    def event_globe_study(self):
        print_slow(
            "[CRISIS] The GLOBE study reveals your algorithm's definition of 'assertiveness' "
            "is penalizing non-Western leaders."
        )
        print("1. Rewrite objective functions (Gain Pluralism, Lose Corporate Power)")
        print("2. Suppress the report (Gain Corporate Power, Lose Pluralism)")
        choice = input("Enter 1 or 2: ")

        if choice == '1':
            self.pluralism += 20
            self.corporate -= 15
        else:
            self.corporate += 20
            self.pluralism -= 20

    def event_ghost_workers(self):
        print_slow(
            "[CRISIS] Your training data is being labeled by underpaid 'ghost workers' "
            "in low-regulation geographies—a modern colonial extraction."
        )
        print("1. Enforce fair labor standards & delay launch (Gain Pluralism, Lose Corporate Power)")
        print("2. Commit 'ethics shirking' to meet targets (Gain Corporate Power, Lose Pluralism, Lose Stability)")
        choice = input("Enter 1 or 2: ")

        if choice == '1':
            self.pluralism += 15
            self.corporate -= 15
        else:
            self.corporate += 15
            self.pluralism -= 20
            self.stability -= 10

    def tech_tree(self):
        if self.budget <= 0:
            print_slow("No R&D budget left for this year.")
            return

        print_slow("[TECH TREE] Allocate R&D Budget to improve the system:")
        print("1. Spiking Neural Networks (SNN): Captures non-linear career paths. "
              "(Cost: 1) -> [+Stability, +Pluralism]")
        print("2. Fuzzy Parameter ZNN: Replaces rigid binaries with degrees of truth. "
              "(Cost: 1) -> [+Pluralism]")
        print("3. Retain 1970s Competency Models: Traditional matrices. "
              "(Cost: 0) -> [+Corporate, -Pluralism]")
        choice = input("Enter 1, 2, or 3: ")

        if choice == '1' and self.budget >= 1:
            self.budget -= 1
            self.stability += 15
            self.pluralism += 10
            print_slow("Deployed SNNs. The algorithm now maps complex career trajectories.")
        elif choice == '2' and self.budget >= 1:
            self.budget -= 1
            self.pluralism += 20
            print_slow(
                "Deployed Fuzzy Logic. "
                "The system now tolerates cultural ambiguity in performance reviews."
            )
        else:
            self.corporate += 15
            self.pluralism -= 10
            print_slow("Maintained legacy models. The C-Suite is pleased, but monoculture deepens.")

    def play(self):
        print_slow(
            "Welcome, Analytics Lead. "
            "Your goal: Save the company from algorithmic monoculture before Year 4."
        )

        while self.turn <= self.max_turns:
            self.display_dashboard()

            if self.turn == 1:
                self.event_globe_study()
            elif self.turn == 2:
                self.event_ghost_workers()
            elif self.turn == 3:
                print_slow(
                    "[CRISIS] The board demands a beta-test of a highly invasive HR tool "
                    "in an overseas branch to avoid local data laws (Ethics Dumping)."
                )
                print("1. Approve it (+Corporate, -Pluralism)")
                print("2. Block it and use participatory design (+Pluralism, -Corporate)")
                if input("Enter 1 or 2: ") == '1':
                    self.corporate += 20
                    self.pluralism -= 25
                else:
                    self.pluralism += 20
                    self.corporate -= 25

            if self.check_game_over():
                return

            self.tech_tree()

            if self.check_game_over():
                return

            self.turn += 1

        self.display_dashboard()
        print_slow("CONGRATULATIONS: You survived the 3-year restructuring period!")
        if self.pluralism > 60 and self.stability > 50:
            print_slow(
                "GOOD ENDING (The Pluralism Protocol): You synthesized technical rigor with "
                "social justice. Your AI learns from the diverse experiences of humanity."
            )
        else:
            print_slow(
                "NEUTRAL ENDING: You survived, but the system remains flawed. "
                "The fight for epistemological pluralism continues."
            )


if __name__ == "__main__":
    game = SuccessionGame()
    game.play()
