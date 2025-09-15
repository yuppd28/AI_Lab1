# src/train_all.py
import subprocess

def run_command(cmd):
    print(f"\n🚀 Запуск: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ Помилка у: {cmd}")
    else:
        print(f"✅ Завершено: {cmd}")

def main():
    # Тренування Baseline
    run_command("python -m src.train_baseline")

    # Тренування RandomForest
    run_command("python -m src.train_rf")

    # Тренування XGBoost
    run_command("python -m src.train_xgb")

    print("\n🎉 Усі моделі відпрацювали! Перевір reports/tables/ і відкрий Dash:\n")
    print("   python -m dash_app.app")

if __name__ == "__main__":
    main()
