import requests
import os
import pandas as pd
from dotenv import load_dotenv
from llama_index.core.tools import FunctionTool
from datetime import date
from huggingface_hub import list_models
import random

load_dotenv()


##### Web search tool using LangSearch API #####

def langsearch_web_search(query: str, fresh="noLimit", summary=True, count=5):
    api_key = os.getenv("LANGSEARCH_API_KEY")
    url = "https://api.langsearch.com/v1/web-search"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "query": query,
        "freshness": fresh,
        "summary": summary,
        "count": count,
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()  # Raises an error if the request failed

    result = []
    for item in response.json()["data"]["webPages"]["value"]:
        result.append({"url": item["url"], "title": item["name"], "snippet": item["snippet"], "summary": item["summary"]})

    return result


def langsearch_tool_fn(query: str, verbose: bool=False) -> str:
    results = langsearch_web_search(query, count=5)


    if verbose:
        text_results = "\n".join([
            f"{r['title']} ({r['url']}): \nSummary: {r['summary']}" for r in results]
            )
        return text_results
    
    text_results = "\n".join([
        f"{r['title']} ({r['url']}): \nSnippet: {r['snippet']}" for r in results]
        )

    return text_results

websearch_tool = FunctionTool.from_defaults(
    fn=langsearch_tool_fn,
    name="langsearch_websearch",
    description="Uses the LangSearch API to search the web and return titles and urls, and snippets or summaries. " \
    "Set verbose to True to return summaries instead of snippets." \
    "Snippets are shorter and more concise, while summaries provide more detailed information about the search results." \
    "Best for genereal information search about a topic, person, event, etc."
)


##### Weather forecast API tool using Open-Meteo API (both current weather and weather forecast for the next 7 days) #####

### weather forecast tool for next 7 days

def get_weather_forecast(latitude: float, longitude: float) -> dict:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ["temperature_2m", "precipitation"],
        "current_weather": False,
    }
    response = requests.get(url, params=params)
    response.raise_for_status()

    return response.json()


def get_weather_forecast_tool_fn(latitude: float, longitude: float) -> str:
    forecast = get_weather_forecast(latitude, longitude)
    hourly_data = forecast.get("hourly", {})
    times = hourly_data.get("time", [])
    temperatures = hourly_data.get("temperature_2m", [])
    precipitations = hourly_data.get("precipitation", [])

    if not times or not temperatures or not precipitations:
        return "No weather forecast data available."

    result = f"Hourly Weather Forecast for next 7 days (from today {date.today()}):\n"
    for time, temp, precip in zip(times, temperatures, precipitations):
        result += f"Time: {time}, Temperature: {temp}°C, Precipitation: {precip}mm\n"

    return result


get_weather_forecast_tool = FunctionTool.from_defaults(
    fn=get_weather_forecast_tool_fn,
    name="get_weather_forecast",
    description="Uses the Open-Meteo Weather Forecast API to get the weather at a location for the next 7 days. " \
    "Returns time (hourly), temperature (Celsius) and precipitation (mm)." \
    "\nBest use get_coordinates before to get the latitude and longitude of the location you want the weather forecast for."
)

### current weather tool

def get_current_weather(latitude: float, longitude: float) -> dict:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": ["temperature_2m", "precipitation", "weather_code", "wind_speed_10m", "cloud_cover"],
        "timezone": "auto"
    }
    response = requests.get(url, params=params)
    response.raise_for_status()

    return response.json()

def get_current_weather_tool_fn(latitude: float, longitude: float) -> str:
    weather = get_current_weather(latitude, longitude)
    current_weather = weather["current"]
    temperature = current_weather["temperature_2m"]
    precipitation = current_weather["precipitation"]
    wind_speed = current_weather["wind_speed_10m"]
    cloudcover = current_weather["cloud_cover"]


    result = f"Current Weather:\nTemperature: {temperature}°C\nPrecipitation: {precipitation}mm\nWind Speed: {wind_speed}km/h\nCloud Cover: {cloudcover}%"
    return result

get_current_weather_tool = FunctionTool.from_defaults(
    fn=get_current_weather_tool_fn,
    name="get_current_weather",
    description="Uses the Open-Meteo Weather Forecast API to get the current weather at a location." \
    "Returns time (hourly), temperature (Celsius), precipitation (mm), wind speed (km/h) and cloud coverage (%)." \
    "\nBest use get_coordinates before to get the latitude and longitude of the location you want the current weather for."
)


##### Image analysis tool using Claude vision #####

import anthropic
import base64

def analyze_image_fn(image_url: str, question: str = "Describe everything you see in this image in detail.") -> str:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    img_response = requests.get(image_url)
    img_response.raise_for_status()
    image_data = base64.standard_b64encode(img_response.content).decode("utf-8")
    media_type = img_response.headers.get("content-type", "image/jpeg").split(";")[0]

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                    {"type": "text", "text": question},
                ],
            }
        ],
    )
    return message.content[0].text

analyze_image_tool = FunctionTool.from_defaults(
    fn=analyze_image_fn,
    name="analyze_image",
    description="Downloads an image from a URL and uses Claude vision to analyze it. "
    "Pass the image URL and an optional specific question about the image. "
    "Use this whenever a question involves an image, chart, diagram, or visual content.",
)


##### Wolfram Alpha tool for mathematical and factual queries #####

def wolfram_alpha_fn(query: str) -> str:
    app_id = os.getenv("WOLFRAM_ALPHA_APP_ID")
    response = requests.get(
        "http://api.wolframalpha.com/v1/result",
        params={"i": query, "appid": app_id},
        timeout=10,
    )
    if response.status_code == 200:
        return response.text
    return f"Wolfram Alpha could not compute an answer for: {query}"

wolfram_alpha_tool = FunctionTool.from_defaults(
    fn=wolfram_alpha_fn,
    name="wolfram_alpha",
    description="Sends a query to Wolfram Alpha and returns a concise computed answer. "
    "Best for mathematical calculations, unit conversions, equations, integrals, "
    "statistics, and factual lookups (e.g. 'integrate x^2 from 0 to 1', '15% of 340', 'sqrt(2) + pi').",
)

##### Get Coordinates tool using Open-Meteo API #####

def get_coordinates_fn(location: str) -> dict:
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": location, "count": 1}
    response = requests.get(url, params=params)
    response.raise_for_status()

    data = response.json()
    if "results" in data and len(data["results"]) > 0:
        result = data["results"][0]
        return {
            "latitude": result.get("latitude"),
            "longitude": result.get("longitude"),
            "name": result.get("name"),
            "country": result.get("country"),
        }
    else:
        return {"error": f"No coordinates found for location: {location}"}
    
get_coordinates_tool = FunctionTool.from_defaults(
    fn=get_coordinates_fn,
    name="get_coordinates",
    description="Uses the Open-Meteo Geocoding API to get the coordinates of a location. " \
    "Returns the latitude and longitude of the location, as well as the name and country if available."
)


##### PDF reader tool #####

import io
from pypdf import PdfReader

def read_pdf_fn(file_url: str) -> str:
    response = requests.get(file_url)
    response.raise_for_status()
    pdf = PdfReader(io.BytesIO(response.content))
    text = ""
    for page in pdf.pages:
        text += page.extract_text() + "\n"
    return text[:15000] if len(text) > 15000 else text

read_pdf_tool = FunctionTool.from_defaults(
    fn=read_pdf_fn,
    name="read_pdf",
    description="Downloads a PDF from a URL and returns the extracted text. "
    "Use this when a question references an attached .pdf file.",
)


##### CSV / Excel reader tool #####

def read_spreadsheet_fn(file_url: str) -> str:
    if ".xls" in file_url.lower():
        df = pd.read_excel(file_url)
    else:
        df = pd.read_csv(file_url)
    return df.to_string(max_rows=100)

read_spreadsheet_tool = FunctionTool.from_defaults(
    fn=read_spreadsheet_fn,
    name="read_spreadsheet",
    description="Downloads and reads a CSV or Excel (.xlsx/.xls) file from a URL and returns its contents as text. "
    "Use this when a question references an attached spreadsheet, CSV, or Excel file.",
)


##### Audio transcription tool using HF Inference API (Whisper) #####

from huggingface_hub import InferenceClient

def transcribe_audio_fn(file_url: str) -> str:
    audio_response = requests.get(file_url)
    audio_response.raise_for_status()
    client = InferenceClient(
        provider="hf-inference",
        api_key=os.getenv("HF_TOKEN"),
    )
    result = client.automatic_speech_recognition(
        audio_response.content,
        model="openai/whisper-large-v3",
    )
    return result.text

transcribe_audio_tool = FunctionTool.from_defaults(
    fn=transcribe_audio_fn,
    name="transcribe_audio",
    description="Downloads an audio file from a URL and transcribes it to text using Whisper. "
    "Use this when a question references an attached audio file (.mp3, .wav, .m4a, etc.).",
)


##### Python code execution tool #####

import sys
from io import StringIO

def execute_python_fn(code: str) -> str:
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        exec(code, {"__builtins__": __builtins__})
        output = sys.stdout.getvalue()
    except Exception as e:
        output = f"Error: {e}"
    finally:
        sys.stdout = old_stdout
    return output.strip() or "Code executed with no printed output."

execute_python_tool = FunctionTool.from_defaults(
    fn=execute_python_fn,
    name="execute_python",
    description="Executes Python code and returns the printed output. "
    "Use this for complex calculations, data manipulation, or logic that other tools cannot handle. "
    "Always use print() to output the result.",
)


##### YouTube transcript tool #####

from youtube_transcript_api import YouTubeTranscriptApi
import re

def get_youtube_transcript_fn(url: str) -> str:
    match = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    if not match:
        return f"Could not extract video ID from URL: {url}"
    video_id = match.group(1)
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([t["text"] for t in transcript])
    except Exception as e:
        return f"Could not retrieve transcript: {e}"

youtube_transcript_tool = FunctionTool.from_defaults(
    fn=get_youtube_transcript_fn,
    name="get_youtube_transcript",
    description="Fetches the transcript/subtitles of a YouTube video given its URL. "
    "Use this whenever a question references a YouTube video link.",
)
