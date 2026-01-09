import asyncio
import hypothesis_engine 

if __name__ == "__main__":
    for i in range(4):
        print("running with standard params")
        asyncio.run(hypothesis_engine.main(n_iterations=9, time_limit=60, use_image_feedback=True, use_large_every=0,
                         param_penalty_weight=0.01,
                         exploration_topology=[1, 2, 3, 4, 5, 6, 7, 0], exploit_point=0.7))