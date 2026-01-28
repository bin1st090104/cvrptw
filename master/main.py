import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from generator import Instance, generate_cvrptw_dataset
from utils.solution import Solution
from utils.visualizer import Visualizer


@hydra.main(config_path="cfg", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # (1) Generate dataset
    dataset: list[Instance] = generate_cvrptw_dataset(
        seed=cfg.dataset.seed,
        n_customers=cfg.dataset.n_customers,
        n_instances=cfg.dataset.n_instances,
        mode=cfg.dataset.mode,
    )

    # (2) Create algorithm
    alg = instantiate(cfg.alg)

    # (3) Create visualizer
    visualizer: Visualizer = instantiate(cfg.visualizer)

    # (3) Solve instances
    solutions: list[Solution] = []
    for i, instance in enumerate(dataset):
        print(f"[Instance {i+1}/{len(dataset)}] Solving...")
        solution: Solution = alg.solve(instance)
        status = solution.status
        print(f"  Status: {status}, Objective: {solution.objective:.2f}")
        solutions.append(solution)
        if solution.__bool__() == True:
            visualizer.plot_solution(instance, solution)
            print(f"  Solution visualized and saved.")

if __name__ == "__main__":
    main()