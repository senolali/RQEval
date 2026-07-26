from radar_plot import RadarPlot

def main():

    # 1️⃣ Model sonuçlarını doğru formatta hazırla
    results = [
        {
            "model": "GPT-4",
            "raw_metrics": {
                "correctness": 0.92,
                "consistency": 0.88,
                "robustness": 0.85,
                "logical_coherence": 0.90,
                "efficiency": 0.78,
                "stability": 0.86,
            }
        },
        {
            "model": "LLaMA-3",
            "raw_metrics": {
                "correctness": 0.87,
                "consistency": 0.82,
                "robustness": 0.80,
                "logical_coherence": 0.84,
                "efficiency": 0.81,
                "stability": 0.79,
            }
        }
    ]

    # 2️⃣ RadarPlot objesi oluştur
    plotter = RadarPlot(output_dir="outputs", figsize=(10, 8))

    # 3️⃣ Radar plot üret
    radar_path = plotter.plot(
        results=results,
        filename="model_radar_plot.png",
        title="Reasoning Quality Comparison"
    )

    print(f"Radar plot saved at: {radar_path}")

    # 4️⃣ (Opsiyonel) Bar comparison da üret
    bar_path = plotter.plot_bar_comparison(
        results=results,
        filename="model_bar_plot.png",
        title="Reasoning Metrics by Model"
    )

    print(f"Bar plot saved at: {bar_path}")


if __name__ == "__main__":
    main()