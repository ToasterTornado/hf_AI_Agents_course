from typing import Any, Optional
from smolagents.tools import Tool
import datetime
import requests
import pytz

def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        # Create timezone object
        tz = pytz.timezone(timezone)
        # Get current time in that timezone
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"
    


class GetTimeInTimezone(Tool):

    name = "find_time_of_timezone"
    description = "This tool helps to find the time of a specific timezone"
    inputs = {'timezone': {'type': 'string', 'description': 'The timezone we want to get the time of'}}
    output_type = "string"

    def __init__(self, **kwargs):
        super().__init__()
        
    def forward(self, timezone) -> str:

        tz = pytz.timezone(timezone)
    
        return f"Timezone: {timezone} \nCurrent time: {datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")}"



