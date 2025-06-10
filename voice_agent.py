import logging
import os
import aiohttp
import json
import base64
from dotenv import load_dotenv

from livekit import rtc
from livekit.rtc import Room, RoomOptions

load_dotenv()

logger = logging.getLogger("transcriber")

def validate_room_name(token):
    try:
        # Split the JWT token into parts
        parts = token.split('.')
        if len(parts) != 3:
            return False, "Invalid token format"
        
        # Decode the payload (second part)
        payload = parts[1]
        # Add padding if needed
        payload += '=' * (4 - len(payload) % 4)
        decoded = base64.b64decode(payload)
        data = json.loads(decoded)
        
        # Check if video room info exists
        if 'video' not in data:
            return False, "No room information in token"
        
        room_info = data['video']
        if 'room' not in room_info:
            return False, "No room name in token"
        
        room_name = room_info['room']
        return True, room_name
    except Exception as e:
        return False, f"Error validating token: {str(e)}"

class GroqSTT:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")

    async def transcribe(self, audio_data):
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            # Convert audio data to base64 or appropriate format
            data = {
                "audio": audio_data,
                "model": "whisper-1"  # Using Whisper model through Groq
            }
            async with session.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers=headers,
                json=data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("text", "")
                else:
                    error_text = await response.text()
                    logger.error(f"Transcription failed: {error_text}")
                    return ""

class VoiceAgent:
    def __init__(self):
        self.room = None
        self.audio_track = None
        self.is_connected = False
        self.stt = GroqSTT()

    async def connect(self):
        """Connect to Livekit room"""
        url = os.getenv('LIVEKIT_URL')
        if not url:
            raise ValueError("LIVEKIT_URL must be set in .env")
            
        # Ensure URL uses wss:// protocol
        if not url.startswith('wss://'):
            url = f"wss://{url.replace('https://', '').replace('http://', '')}"
            
        token = os.getenv('LIVEKIT_TOKEN')
        if not token:
            raise ValueError("LIVEKIT_TOKEN must be set in .env")
            
        print(f"Connecting to Livekit at {url}")
        await self.room.connect(url, token)
        self.is_connected = True
        print("Successfully connected to Livekit room")

        # Create and publish audio track
        self.audio_track = rtc.LocalAudioTrack.create()
        await self.room.local_participant.publish_track(self.audio_track)
        print("Audio track published")

        # Set up event handlers
        self.room.on("track_subscribed", self.on_track_subscribed)
        self.room.on("disconnected", self.on_disconnected)
        self.room.on("reconnecting", self.on_reconnecting)
        self.room.on("reconnected", self.on_reconnected)

    async def on_track_subscribed(self, track, publication, participant):
        if track.kind == rtc.TrackKind.AUDIO:
            print(f"Received audio track from {participant.identity}")
            async for frame in track.frames():
                try:
                    # Transcribe audio
                    transcript = await self.stt.transcribe(frame.data)
                    if transcript:
                        print("User:", transcript)
                except Exception as e:
                    print(f"Error processing audio: {str(e)}")

    async def on_disconnected(self):
        print("Disconnected from Livekit room")
        self.is_connected = False

    async def on_reconnecting(self):
        print("Reconnecting to Livekit room...")

    async def on_reconnected(self):
        print("Reconnected to Livekit room")
        self.is_connected = True

    async def disconnect(self):
        if self.room and self.is_connected:
            await self.room.disconnect()
            self.is_connected = False
            print("Disconnected from Livekit room")

async def main():
    agent = VoiceAgent()
    try:
        await agent.connect()
        # Keep the connection alive
        while agent.is_connected:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        await agent.disconnect()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())