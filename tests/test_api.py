import asyncio
import websockets
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestAPI")

async def test_websocket():
    uri = "ws://localhost:8000/ws"
    logger.info(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            logger.info("Connected!")
            
            # Wait for a few messages
            for i in range(3):
                message = await websocket.recv()
                data = json.loads(message)
                logger.info(f"Received: {data}")
                
    except Exception as e:
        logger.error(f"Connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_websocket())
