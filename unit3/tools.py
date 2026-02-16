import requests
import os
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

##### Get latest news about a topic using LangSearch API #####

def get_latest_news(topic: str, count=5) -> str:
    query = f"latest news about {topic}"
    results = langsearch_web_search(query, fresh="day", summary=True, count=count)

    text_results = "\n".join([
        f"{r['title']} ({r['url']}): \nSummary: {r['summary']}" for r in results]
        )
    return text_results


get_latest_news_tool = FunctionTool.from_defaults(
    fn=get_latest_news,
    name="get_latest_news",
    description="Uses the LangSearch API to get the latest news about a specific topic. " \
    "Returns the title, url and summary of the latest news articles about the topic." \
    "count specifies how many news articles to return."
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

##### HF stats tool using Hugging Face Hub API #####
def get_hub_stats(author: str) -> str:
    """Fetches the most downloaded model from a specific author on the Hugging Face Hub."""
    try:
        # List models from the specified author, sorted by downloads
        models = list(list_models(author=author, sort="downloads", direction=-1, limit=1))

        if models:
            model = models[0]
            return f"The most downloaded model by {author} is {model.id} with {model.downloads:,} downloads."
        else:
            return f"No models found for author {author}."
    except Exception as e:
        return f"Error fetching models for {author}: {str(e)}"
    
get_most_downloaded_model_by_creator_tool = FunctionTool.from_defaults(
    fn=get_hub_stats,
    name="get_hub_stats",
    description="Returns the most downloaded model from a specified author."
)