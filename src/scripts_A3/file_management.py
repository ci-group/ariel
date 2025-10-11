import pickle
import os
 

def save_generation(generation, pop_body_genotype, best_body, best_brain, run_id = 0, sim_config = None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gen_dir = sim_config.data / f"run_{run_id}" / f"generation_{generation:03d}"
    gen_dir.mkdir(parents=True, exist_ok=True)
    
    with open(gen_dir / "body_population.pkl", "wb") as f:
        pickle.dump(pop_body_genotype, f)
    
    best_data = {
        "generation": generation,
        "timestamp": timestamp,
        "best_body_genotype": list(best_body),  # Convert to list for serialization
        "best_brain_genotype": list(best_brain) if best_brain is not None else None,
        "body_fitness": best_body.fitness.values[0] if best_body.fitness.valid else None,
        "brain_fitness": best_brain.fitness.values[0] if hasattr(best_brain, 'fitness') and best_brain.fitness.valid else None,
        "nde": sim_config.nde
    }
    
    with open(gen_dir / "best_performers.pkl", "wb") as f:
        pickle.dump(best_data, f)
    
    print(f"Saved generation {generation} data to {gen_dir}")
    
def load_generation_data(generation, run_id, sim_config):
    """Load saved generation data"""
    gen_dir = sim_config.data / f"run_{run_id}" / f"generation_{generation:03d}"
    
    if not gen_dir.exists():
        raise FileNotFoundError(f"Generation {generation} data not found at {gen_dir}")
    
    with open(gen_dir / "body_population.pkl", "rb") as f:
        population = pickle.load(f)
    
    with open(gen_dir / "best_performers.pkl", "rb") as f:
        best_data = pickle.load(f)
    
    return population, best_data
def load_population_from_generation(sim_config, generation, run_id = 0):
    """Load population and resume from a specific generation"""
    gen_dir = sim_config.data / f"run_{run_id}" / f"generation_{generation:03d}"
    
    if not gen_dir.exists():
        raise FileNotFoundError(f"Generation {generation} data not found at {gen_dir}")
    
    # Load population
    with open(gen_dir / "body_population.pkl", "rb") as f:
        population = pickle.load(f)
    
    # Load best data
    with open(gen_dir / "best_performers.pkl", "rb") as f:
        best_data = pickle.load(f)
    
    # Restore NDE to config
    if "nde" in best_data and best_data["nde"] is not None:
        sim_config.nde = best_data["nde"]
    
    print(f"Resumed from generation {generation} with {len(population)} individuals")
    print(f"Best fitness from that generation: {best_data.get('body_fitness', 'Unknown')}")
    
    return population, best_data

def get_next_run_id(sim_config: EAConfig) -> int:
    """Find the next available run ID to avoid overwriting"""
    run_id = 0
    while (sim_config.data / f"run_{run_id}").exists():
        run_id += 1
    return run_id

def find_latest_generation(sim_config, run_id):
    run_dir = sim_config.data / f"run_{run_id}"
    if not run_dir.exists():
        return -1
    latest_gen = -1
    
    # Look for generation directories
    for gen_dir in run_dir.iterdir():
        if gen_dir.is_dir() and gen_dir.name.startswith("generation_"):
            try:
                gen_num = int(gen_dir.name.split("_")[1])
                if (gen_dir / "body_population.pkl").exists() and (gen_dir / "best_performers.pkl").exists():
                    latest_gen = max(latest_gen, gen_num)
            except (ValueError, IndexError):
                continue
    return latest_gen
