from typing import Any, Optional
from smolagents.tools import Tool
from timezonefinder import TimezoneFinder
from geopy.geocoders import Nominatim

class FindTimezone(Tool):
    name = "find_timezone_of_location"
    description = "This tool helps to find the timezone of a specific location"
    inputs = {'query': {'type': 'string', 'description': 'The string desription of the location we want the timezone of.'}}
    output_type = "string"

    def __init__(self, **kwargs):
        super().__init__()

        # defensive import of packages to ensure agent knows if something in the tool goes wrong
        try:
            from timezonefinder import TimezoneFinder
        except ImportError as e:
            raise ImportError("The package timezonefinder is not installed. Please install the package.") from e
        
        try:
            from geopy.geocoders import Nominatim
        except ImportError as e:
            raise ImportError("The package geopy is not installed. Please install the package.") from e
        
        # creating objects for later search
        self.geolocator = Nominatim(user_agent="smolagents_timezone_finder/1.0")
        self.timezone_finder = TimezoneFinder()

    def forward(self, query) -> str:
        try:
            # get the longitude and latitude of the location
            location = self.geolocator.geocode(query, timeout=10)
            if location is None:
                return f"Could not find location: {query}"
            
            tz = self.timezone_finder.timezone_at(lng=location.longitude, lat=location.latitude)
            
            # merge results into a string
            return f"Location: {query}\nLongitude: {location.longitude}\nLatitude: {location.latitude}\nTimezone: {tz}"
        except Exception as e:
            return f"Error finding timezone for {query}: {str(e)}"