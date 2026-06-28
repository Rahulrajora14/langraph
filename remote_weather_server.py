import os
import requests

from dotenv import load_dotenv
from fastmcp import FastMCP

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")

mcp = FastMCP("Weather Server")


# --------------------------
# Helper Function
# --------------------------

def get_weather(city: str):

    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?q={city}"
        f"&appid={API_KEY}"
        f"&units=metric"
    )

    response = requests.get(url)

    if response.status_code != 200:
        return None

    return response.json()


# --------------------------
# Temperature Tool
# --------------------------

@mcp.tool()
def temperature(city: str) -> str:
    """
    Returns current temperature.
    """

    data = get_weather(city)

    if data is None:
        return "City not found."

    temp = data["main"]["temp"]

    return f"{temp} °C"


# --------------------------
# Weather Tool
# --------------------------

@mcp.tool()
def weather(city: str) -> str:
    """
    Returns current weather.
    """

    data = get_weather(city)

    if data is None:
        return "City not found."

    return data["weather"][0]["description"]


# --------------------------
# Humidity Tool
# --------------------------

@mcp.tool()
def humidity(city: str) -> str:
    """
    Returns humidity.
    """

    data = get_weather(city)

    if data is None:
        return "City not found."

    humidity = data["main"]["humidity"]

    return f"{humidity} %"


# --------------------------
# Wind Speed Tool
# --------------------------

@mcp.tool()
def wind_speed(city: str) -> str:
    """
    Returns wind speed.
    """

    data = get_weather(city)

    if data is None:
        return "City not found."

    speed = data["wind"]["speed"]

    return f"{speed} m/s"


# --------------------------
# Run MCP Server
# --------------------------

if __name__ == "__main__":
    mcp.run()