import asyncio
import aiohttp
import time

async def test_concurrency(url, num_requests=100):
    start_time = time.time()
    async with aiohttp.ClientSession() as session:
        tasks = []
        for _ in range(num_requests):
            task = asyncio.ensure_future(session.get(url))
            tasks.append(task)
        responses = await asyncio.gather(*tasks)
    
    total_time = time.time() - start_time
    print(f"Completed {num_requests} requests in {total_time:.2f} seconds")
    print(f"Average response time: {(total_time/num_requests)*1000:.2f}ms")
    print(f"Requests per second: {num_requests/total_time:.2f}")

# Run the test
url = "http://localhost:8000/test/concurrency"
asyncio.run(test_concurrency(url, num_requests=100))