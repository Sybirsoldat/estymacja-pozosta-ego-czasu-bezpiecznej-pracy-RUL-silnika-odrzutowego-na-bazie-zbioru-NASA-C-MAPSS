"""
run.py — główny plik uruchomieniowy
Odpal: python run.py
Menu pozwala wybrać co uruchomić.
"""
import subprocess
import sys

STEPS = {
    "1": ("10a_train.py",      "Trening XGBoost + LSTM (Optuna, ~50 min)"),
    "2": ("10b_plots.py",      "Wykresy porównawcze 53–60 (~30 sek)"),
    "3": ("10c_analysis.py",   "Analiza i walidacja 61–73 (~5 min)"),
    "4": ("10d_robustness.py", "Testy odporności 74–78 (~10 min)"),
}

def run_script(filename):
    print(f"\n{'='*70}")
    print(f"  Uruchamiam: {filename}")
    print(f"{'='*70}\n")
    result = subprocess.run([sys.executable, filename])
    if result.returncode != 0:
        print(f"\n  [!] {filename} zakończył się z błędem (kod {result.returncode})")
        return False
    return True

def main():
    print("=" * 70)
    print("  NASA C-MAPSS — RUL Prediction Pipeline")
    print("  Kurs: Zastosowania modeli AI w automatyce")
    print("=" * 70)
    print()
    print("  Co chcesz uruchomić?")
    print()
    print("  [0] Wszystko (1→2→3→4)")
    for key, (fname, desc) in STEPS.items():
        print(f"  [{key}] {desc}")
    print()
    print("  [q] Wyjście")
    print()

    choice = input("  Twój wybór: ").strip().lower()

    if choice == "q":
        print("  Do zobaczenia!")
        return

    if choice == "0":
        # Uruchom wszystko po kolei
        for key in ["1", "2", "3", "4"]:
            fname, desc = STEPS[key]
            ok = run_script(fname)
            if not ok:
                print(f"\n  [!] Zatrzymuję — {fname} zwrócił błąd.")
                break
        print(f"\n{'='*70}")
        print("  GOTOWE — wszystkie fazy zakończone!")
        print(f"{'='*70}")

    elif choice in STEPS:
        fname, desc = STEPS[choice]
        # Sprawdź zależności
        if choice in ["2", "3", "4"]:
            import os
            results_path = "./results_optuna_v3/optuna_v3_results.pkl"
            if not os.path.exists(results_path):
                print(f"\n  [!] Brak wyników treningu ({results_path})")
                print(f"  [!] Najpierw uruchom krok 1 (Trening)")
                return
        run_script(fname)
    else:
        print(f"  [!] Nieznany wybór: {choice}")

if __name__ == "__main__":
    main()