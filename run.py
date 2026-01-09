import asyncio
import hypothesis_engine 

async def _run_many():
    for i in range(4):
        print("running with standard params")
        await hypothesis_engine.main(
            n_iterations=9,
            time_limit=60,
            use_image_feedback=True,
            use_large_every=0,
            param_penalty_weight=0.01,
            exploration_topology=[1, 2, 3, 4, 5, 6, 7, 0],
            exploit_point=0.7,
        )

if __name__ == "__main__":
    asyncio.run(_run_many())