import streamlit as st
from astrapy.db import AstraDB
import redis
import json
from functools import lru_cache
import time
from datetime import datetime, timedelta
import pandas as pd
import constant
import os
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import openai
import fetch_unique, unique_data_filter
import requests
from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
from dotenv import load_dotenv
from geopy.geocoders import Nominatim

load_dotenv()

class AgriculturalData:
    """Agricultural data categories and mappings"""
    
    METRICS = {
        "Area harvested": {
            "description": "Total land area used for cultivation",
            "unit": "hectares",
            "column": "Value"
        },
        "Yield": {
            "description": "Production per unit of land",
            "unit": "hg/ha",
            "column": "Value"
        },
        "Production": {
            "description": "Total quantity produced",
            "unit": "tonnes",
            "column": "Value"
        },
        "Producing Animals/Slaughtered": {
            "description": "Available inventory",
            "unit": "tonnes",
            "column": "Value"
        },
        "Stocks": {
           "description": "Current stock available",
           "unit": "tonnes",
            "column": "Value"
        },
        "Yield/Carcass Weight": {
            "description": "Weight of carcass produced per unit of animal",
            "unit": "kg",
            "column": "Value"
        },
        "Milk Animals": {
            "description": "Total number of milk-producing animals",
            "unit": "head",
            "column": "Value"
        },
        "Laying": {
            "description": "Total number of laying hens",
            "unit": "head",
            "column": "Value"
        }
    }

    LANGUAGES = {
        "English": {
            "code": "en",
            "local_name": "English"
        },
        "हिंदी": {
            "code": "hi",
            "local_name": "Hindi"
        },
        "తెలుగు": {
            "code": "te",
            "local_name": "Telugu"
        },
        "मराठी": {
            "code": "mr",
            "local_name": "Marathi"
        },
        "ಕನ್ನಡ": {
            "code": "kn",
            "local_name": "Kannada"
        }
    }

class CacheManager:
    """Handle caching with Redis Cloud fallback to in-memory cache"""
    def __init__(self, redis_config: Optional[dict] = None):
        self.redis_client = None
        self.in_memory_cache = {}
        
        if redis_config:
            try:
                self.redis_client = redis.Redis(
                    host=redis_config.get('host'),
                    port=redis_config.get('port'),
                    decode_responses=True,
                    username=redis_config.get('username', 'default'),
                    password=redis_config.get('password'),
                    ssl=False,  # Changed to False since this endpoint doesn't use SSL
                    ssl_cert_reqs=None  # Remove SSL certificate requirements
                )
                self.redis_client.ping()  # Test connection
            except Exception as e:
                st.warning(f"Redis Cloud connection failed: {str(e)}. Using in-memory cache.")
                self.redis_client = None

    def get_cache_key(self, prefix: str, *args) -> str:
        """Generate deterministic cache key"""
        key_string = f"{prefix}:{':'.join(str(arg) for arg in args)}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            if self.redis_client:
                value = self.redis_client.get(key)
                return json.loads(value) if value else None
        except Exception as e:
            st.warning(f"Redis get operation failed: {str(e)}")
            return self.in_memory_cache.get(key)
        return self.in_memory_cache.get(key)

    def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Set value in cache"""
        try:
            json_value = json.dumps(value)
            if self.redis_client:
                self.redis_client.setex(key, ttl, json_value)
            else:
                self.in_memory_cache[key] = value
        except Exception as e:
            st.warning(f"Redis set operation failed: {str(e)}")
            self.in_memory_cache[key] = value

class DataManager:
    """Handle data operations with AstraDB"""

    def __init__(self, astra_client: AstraDB, cache_manager: CacheManager):
        self.astra_client = astra_client
        self.cache_manager = cache_manager

    def get_crop_data(self, crop: str, metric: str, years: List[int]) -> pd.DataFrame:
        """Get crop data with caching for multiple years"""
        try:
            # Create an empty list to store data from all years
            all_data = []
            
            # Fetch data for each year
            for year in years:
                raw_result = fetch_unique.fetch_row_data(metric, crop, year)
            
                if raw_result:
                    # Extract data from the metadata structure
                    if isinstance(raw_result, dict) and 'metadata' in raw_result:
                        # Single result case
                        metadata = raw_result['metadata']
                        formatted_data = {
                            'Year': metadata.get('year', year),
                            'Item': metadata.get('item', crop),
                            'Element': metadata.get('element', metric),
                            'Value': metadata.get('value', 0),
                            'Unit': metadata.get('unit', AgriculturalData.METRICS[metric]['unit']),
                            'Flag': metadata.get('flag', ''),
                            'Flag_Description': metadata.get('flag_description', '')
                        }
                        all_data.append(formatted_data)
                    elif isinstance(raw_result, list):
                        # Multiple results case
                        for item in raw_result:
                            if isinstance(item, dict) and 'metadata' in item:
                                metadata = item['metadata']
                                formatted_data = {
                                    'Year': metadata.get('year', year),
                                    'Item': metadata.get('item', crop),
                                    'Element': metadata.get('element', metric),
                                    'Value': metadata.get('value', 0),
                                    'Unit': metadata.get('unit', AgriculturalData.METRICS[metric]['unit']),
                                    'Flag': metadata.get('flag', ''),
                                    'Flag_Description': metadata.get('flag_description', '')
                                }
                                all_data.append(formatted_data)
        
            # Create DataFrame from all collected data
            df = pd.DataFrame(all_data)
        
            if df.empty:
                return pd.DataFrame()
        
            # Ensure numeric type for Value column
            df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
        
            # Sort by year
            df = df.sort_values('Year')
        
            return df
            
        except Exception as e:
            print(f"Error in get_crop_data: {str(e)}")
            return pd.DataFrame()


    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate summary statistics for the data"""
        try:
            if df.empty or 'Value' not in df.columns:
                return {}
                
            summary = {
                'Value': f"{df['Value'].iloc[0]:,.2f}",
                'Status': df['Flag_Description'].iloc[0] if 'Flag_Description' in df.columns else 'N/A'
            }
            return summary
        except Exception as e:
            print(f"Error calculating summary: {str(e)}")
            return {}


     
class PromptManager:
    """Handle multilingual prompts and LLM interactions"""

    def __init__(self, openai_client: openai, cache_manager: CacheManager):
        self.openai_client = openai_client
        self.cache_manager = cache_manager

    def get_prompt(self, stage: str, language: str, context: Dict) -> str:
        """Get context-aware, multilingual prompt"""
        cache_key = self.cache_manager.get_cache_key('prompt', stage, language, json.dumps(context))
        cached_prompt = self.cache_manager.get(cache_key)

        if cached_prompt:
            return cached_prompt

        system_prompts = {
            'welcome': {
                'en': """You are an agricultural data expert. Create a warm welcome message that:
                1. Introduces yourself as an agricultural data assistant
                2. Mentions your ability to provide insights about crops, yields, and production
                3. Asks how you can help them today
                Keep it under 50 words and make it conversational."""
            },
            'metric_selection': {
                'en': """Please select a metric you'd like to analyze for your agricultural data. 
                We can provide information about:
                - Area harvested
                - Yield
                - Production Quantity
                - Stocks
                What would you like to explore?"""
            },
            'crop_selection': {
                'en': f"""The farmer is interested in {context.get('metric', 'agricultural data')}.
                Create a natural prompt asking them to select a crop category and specific crop.
                Mention we have data for various categories like cereals, pulses, fruits, etc.
                Keep it conversational and brief."""
            },
            'data_view': {
                'en': f"""Here's the {context.get('metric', '')} data for {context.get('crop', '')}.
                I can show you trends, statistics, and insights about this data.
                What would you like to know more about?"""
            }
        }

        openai.api_key = constant.OPENAI_KEY
        base_prompt = system_prompts.get(stage, {}).get(language, system_prompts[stage]['en'])
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": base_prompt},
            ],
            temperature=0.7,
            max_tokens=100
        )
        
        prompt = response.choices[0].message.content
        self.cache_manager.set(cache_key, prompt)
        return prompt


class WebSearchManager:
    """Handle web searches for agricultural data"""
    
    def __init__(self, openai_client: openai):
        self.openai_client = openai_client
        openai.api_key = constant.OPENAI_KEY
        
    def search_agricultural_data(self, crop: str, metric: str, year: int) -> Dict[str, Any]:
        """Search and summarize web data about the crop and metric"""
        try:
            # Construct search URL for World Bank API
            base_url = "https://api.worldbank.org/v2/country/all/indicator"
            indicators = {
                "Production": "AG.PRD.CROP.INDEX",
                "Area harvested": "AG.LND.AGRI.ZS",
                "Yield": "AG.YLD.CREL.KG"
            }
        
            indicator = indicators.get(metric, indicators["Production"])
            url = f"{base_url}/{indicator}?format=json&date={year}"
        
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
            
                # Enhanced context for OpenAI analysis
                context_prompt = f"""
                Analyze the following agricultural data for {crop} in {year}, focusing on {metric}.
            
                Key Analysis Points:
                1. Global Production Context:
                   - Compare {crop}'s {metric.lower()} with global averages
                   - Identify major producing regions
                   - Note any significant changes from previous years
            
                2. Market Impact Analysis:
                   - Effect on local and global markets
                   - Price implications
                   - Supply chain considerations
            
                3. Agricultural Insights:
                   - Growing conditions and challenges
                   - Technological advancements
                   - Sustainability aspects
            
                4. Future Outlook:
                   - Short-term projections
                   - Long-term trends
                   - Potential challenges and opportunities
            
                Data Source: {data}
            
                Please provide a comprehensive yet concise analysis covering these aspects,
                focusing on practical insights for farmers and agricultural stakeholders.
                Include specific numbers and percentages where relevant.
                """
            
                # Use OpenAI to analyze and summarize the data
                summary = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert agricultural analyst with deep knowledge of global crop markets and farming practices."},
                        {"role": "user", "content": context_prompt}
                    ],
                    temperature=0.5,
                    max_tokens=1000
                )
            
                return {
                    "summary": summary.choices[0].message.content,
                    "sources": [
                        "World Bank Agricultural Data",
                        "FAO STAT Database",
                        f"Agricultural Market Information System - {year}"
                    ],
                    "raw_data": data
                }
            else:
                return {
                    "summary": f"Unable to fetch data for {crop} {metric} in {year}. Please try different parameters.",
                    "sources": []
                }
            
        except Exception as e:
            return {
                "summary": f"Error fetching data: {str(e)}. Please try again later.",
                "sources": []
            }

class PricePredictionAgent:
    """Handle price predictions for agricultural commodities"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        
    def prepare_historical_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare historical data for prediction"""
        # Ensure data is sorted by year
        df = df.sort_values('Year')
        
        # Create features from year
        df['YearNumeric'] = pd.to_numeric(df['Year'])
        
        # Create time-based features
        years_passed = df['YearNumeric'] - df['YearNumeric'].min()
        
        # Prepare features matrix
        features = np.column_stack((
            years_passed,
            years_passed ** 2,  # Add quadratic term
            np.sin(2 * np.pi * years_passed / 10),  # Add cyclical component
        ))
        
        target = df['Value'].values
        
        return features, target
        
    def predict_prices(self, historical_df: pd.DataFrame, years_to_predict: int = 2) -> Dict[str, Any]:
        """Predict future prices based on historical data"""
        try:
            if len(historical_df) < 2:
                return {
                    'error': 'Insufficient historical data for prediction',
                    'predictions': []
                }
            
            # Prepare data
            features, target = self.prepare_historical_data(historical_df)
            
            # Train model
            self.model.fit(features, target)
            
            # Calculate R-squared score
            confidence_score = self.model.score(features, target)
            
            # Prepare future dates for prediction
            last_year = historical_df['Year'].max()
            future_years = range(int(last_year) + 1, int(last_year) + years_to_predict + 1)
            
            # Create future features
            years_passed_start = len(features)
            future_features = np.column_stack((
                np.arange(years_passed_start, years_passed_start + years_to_predict),
                np.arange(years_passed_start, years_passed_start + years_to_predict) ** 2,
                np.sin(2 * np.pi * np.arange(years_passed_start, years_passed_start + years_to_predict) / 10)
            ))
            
            # Make predictions
            predictions = self.model.predict(future_features)
            
            # Calculate trend metrics
            trend = 'increasing' if predictions[-1] > predictions[0] else 'decreasing'
            avg_change = (predictions[-1] - predictions[0]) / predictions[0] * 100
            
            # Prepare simple prediction results
            prediction_data = [
                {
                    'year': year,
                    'predicted_value': round(pred, 2)
                }
                for year, pred in zip(future_years, predictions)
            ]
            
            results = {
                'predictions': prediction_data,
                'confidence_score': round(confidence_score * 100, 2),
                'trend': trend,
                'average_change': round(avg_change, 2),
                'prediction_summary': f"Expected {trend} trend with {abs(round(avg_change, 2))}% change over the next {years_to_predict} years"
            }
            
            return results
            
        except Exception as e:
            return {
                'error': str(e),
                'predictions': []
            }

    def get_market_factors(self, crop: str) -> Dict[str, str]:
        """Get market analysis for a specific crop"""
        try:
            # Create a basic market analysis based on the crop type
            market_analysis = f"""
            Market Analysis for {crop}:
            - Global production trends
            - Weather conditions impact
            - Supply and demand factors
            """
            
            return {
                'market_analysis': market_analysis,
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'market_analysis': f"Unable to generate market analysis: {str(e)}",
                'status': 'error'
            }

class WeatherAnalysisAgent:
    """Enhanced weather analysis agent with location processing and crop-specific recommendations"""
    
    def __init__(self, cache_manager: CacheManager, openai_client: openai):
        self.cache_manager = cache_manager
        self.openai_client = openai_client
        self.weather_api_key = os.getenv("OPENWEATHER_API_KEY")
        if not self.weather_api_key:
            raise ValueError("OpenWeather API key not found in environment variables")
        self.geocoder = Nominatim(user_agent="agricultural_assistant")
        
    def get_coordinates(self, location: str) -> Optional[Tuple[float, float]]:
        """Convert location string to coordinates"""
        try:
            if not location:
                return None
            location_data = self.geocoder.geocode(location)
            if location_data:
                return (location_data.latitude, location_data.longitude)
            return None
        except Exception as e:
            print(f"Error getting coordinates: {str(e)}")
            return None

    def fetch_weather_data(self, lat: float, lon: float) -> Dict[str, Any]:
        """Fetch current and forecast weather data from OpenWeather API"""
        try:
            # Current weather
            current_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={self.weather_api_key}&units=metric"
            current_response = requests.get(current_url)
            current_response.raise_for_status()  # Raise an exception for bad status codes
            current_data = current_response.json()

            # 5-day forecast (since daily forecast requires paid subscription)
            forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={self.weather_api_key}&units=metric"
            forecast_response = requests.get(forecast_url)
            forecast_response.raise_for_status()
            forecast_data = forecast_response.json()

            # Process forecast data to get daily values
            daily_forecasts = []
            current_date = None
            temp_max = float('-inf')
            temp_min = float('inf')
            
            for item in forecast_data['list']:
                date = datetime.fromtimestamp(item['dt']).strftime('%Y-%m-%d')
                
                if current_date != date:
                    if current_date is not None:
                        daily_forecasts.append({
                            'date': current_date,
                            'temp_max': temp_max,
                            'temp_min': temp_min,
                            'humidity': item['main']['humidity'],
                            'description': item['weather'][0]['description'],
                            'rain_prob': item.get('pop', 0) * 100
                        })
                    current_date = date
                    temp_max = float('-inf')
                    temp_min = float('inf')
                
                temp_max = max(temp_max, item['main']['temp_max'])
                temp_min = min(temp_min, item['main']['temp_min'])

            # Add the last day
            if current_date and current_date not in [d['date'] for d in daily_forecasts]:
                daily_forecasts.append({
                    'date': current_date,
                    'temp_max': temp_max,
                    'temp_min': temp_min,
                    'humidity': forecast_data['list'][-1]['main']['humidity'],
                    'description': forecast_data['list'][-1]['weather'][0]['description'],
                    'rain_prob': forecast_data['list'][-1].get('pop', 0) * 100
                })

            weather_data = {
                'current_conditions': {
                    'temperature': current_data['main']['temp'],
                    'humidity': current_data['main']['humidity'],
                    'wind_speed': current_data['wind']['speed'],
                    'description': current_data['weather'][0]['description'],
                    'rainfall': current_data.get('rain', {}).get('1h', 0),
                },
                'forecast': {
                    'daily': daily_forecasts[:7]  # Limit to 7 days
                }
            }
            return weather_data
            
        except requests.RequestException as e:
            print(f"Error fetching weather data: {str(e)}")
            raise Exception(f"Weather API error: {str(e)}")
        except Exception as e:
            print(f"Error processing weather data: {str(e)}")
            raise Exception(f"Error processing weather data: {str(e)}")

    def get_weather_recommendations(
        self,
        location: str,
        crop: str,
        growth_stage: str
    ) -> Dict[str, Any]:
        """Get weather analysis and recommendations"""
        try:
            if not location:
                return {'error': 'Please enter a location'}

            # Check cache first
            cache_key = self.cache_manager.get_cache_key(
                'weather_recommendations',
                location,
                crop,
                growth_stage
            )
            cached_result = self.cache_manager.get(cache_key)
            if cached_result:
                return cached_result

            # Get coordinates
            coordinates = self.get_coordinates(location)
            if not coordinates:
                return {'error': 'Could not find the specified location. Please check the location name and try again.'}

            # Fetch weather data
            try:
                weather_data = self.fetch_weather_data(*coordinates)
            except Exception as e:
                return {'error': str(e)}

            if not weather_data:
                return {'error': 'Could not fetch weather data. Please try again later.'}

            # Generate analysis prompt
            prompt = f"""
            Analyze the following weather conditions for {crop} cultivation at {growth_stage} growth stage:

            Current Conditions:
            - Temperature: {weather_data['current_conditions']['temperature']}°C
            - Humidity: {weather_data['current_conditions']['humidity']}%
            - Wind Speed: {weather_data['current_conditions']['wind_speed']} m/s
            - Conditions: {weather_data['current_conditions']['description']}

            Forecast for next {len(weather_data['forecast']['daily'])} days:
            {self._format_forecast(weather_data['forecast']['daily'])}

            Provide specific recommendations for:
            1. Immediate Actions (next 24 hours)
            2. Short-term Planning (next 3-5 days)
            3. Crop Protection Measures
            4. Irrigation Adjustments
            5. Disease Prevention
            6. Growth Stage Specific Advice

            Format the response in clear sections with actionable recommendations.
            """

            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert agricultural advisor specializing in weather impact analysis and crop management."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )

                result = {
                    'weather_data': weather_data,
                    'recommendations': response.choices[0].message.content,
                    'location': location,
                    'timestamp': datetime.now().isoformat()
                }

                # Cache the result
                self.cache_manager.set(cache_key, result, ttl=3600)  # Cache for 1 hour

                return result

            except Exception as e:
                return {'error': f"Error generating recommendations: {str(e)}"}

        except Exception as e:
            return {'error': f"Analysis error: {str(e)}"}

    def _format_forecast(self, forecast_data: List[Dict]) -> str:
        """Format forecast data for the prompt"""
        forecast_str = ""
        for day in forecast_data:
            forecast_str += f"\n{day['date']}:"
            forecast_str += f"\n  - Temperature: {day['temp_min']:.1f}°C to {day['temp_max']:.1f}°C"
            forecast_str += f"\n  - Humidity: {day['humidity']}%"
            forecast_str += f"\n  - Conditions: {day['description']}"
            forecast_str += f"\n  - Rain Probability: {day['rain_prob']:.1f}%\n"
        return forecast_str

class ChatbotUI:
    def __init__(
        self,
        data_manager: DataManager,
        prompt_manager: PromptManager,
        cache_manager: CacheManager,
        web_search_manager: WebSearchManager,
        price_prediction_agent: PricePredictionAgent,
        weather_agent: WeatherAnalysisAgent
    ):
        self.data_manager = data_manager
        self.prompt_manager = prompt_manager
        self.cache_manager = cache_manager
        self.web_search_manager = web_search_manager
        self.price_prediction_agent = price_prediction_agent
        self.weather_agent = weather_agent

    def show_welcome(self) -> None:
        st.markdown("### 🌾 Agricultural Data Assistant")
    
    # Debugging: Print session state before displaying prompt
        prompt = self.prompt_manager.get_prompt(
            'welcome',
            st.session_state.get('language', 'en'),
            {}
            )
        st.write(prompt)
        selected_lang = st.selectbox(
            "Select your preferred language",
            options=list(AgriculturalData.LANGUAGES.keys()),
            format_func=lambda x: f"{x} ({AgriculturalData.LANGUAGES[x]['local_name']})"
            )
        
        if st.button("Begin Exploration 🚀"):
            # Safeguard: explicitly set language and stage
            st.session_state.language = AgriculturalData.LANGUAGES[selected_lang]['code']
            st.session_state.stage = 'metric_selection'  # Explicitly set stage to 'metric_selection'
            st.session_state.context = {}  # Clear context to start fresh


    def show_metric_selection(self) -> None:
        prompt = self.prompt_manager.get_prompt(
            'metric_selection',
            st.session_state.language,
            {}
        )

        metric = st.radio(
            "Select what you'd like to know:",
            options=list(AgriculturalData.METRICS.keys()),
            format_func=lambda x: (
                f"{x}: {AgriculturalData.METRICS[x]['description']} "
                f"({AgriculturalData.METRICS[x]['unit']})"
            )
        )

        if st.button("Continue"):
            st.session_state.context = {'metric': metric}
            st.session_state.stage = 'crop_selection'
            st.rerun()

    def show_crop_selection(self) -> None:
        prompt = self.prompt_manager.get_prompt(
            'crop_selection',
            st.session_state.language,
            st.session_state.context
        )
        st.write(prompt)

        metric = st.session_state.context.get('metric')
        if metric is None:
            st.error("No metric selected! Please go back and select a metric.")
            return

        # Fetch the unique crops for the selected metric using filter_items_by_element
        try:
            crops = unique_data_filter.filter_items_by_element(metric)
            if not crops:
                st.error(f"No crops found for the selected metric: {metric}")
                return
        except Exception as e:
            st.error(f"Error fetching crops for metric {metric}: {str(e)}")
            return

        # Select specific crop
        crop = st.selectbox(
            "Select specific crop",
            options=crops
        )

        # Add Year selection dropdown with years from 1960 to 2023
        years = list(range(1960, 2024))  # Creating a list of years from 1960 to 2023
        selected_year = st.selectbox(
            "Select the year",
            options=years,
            index=63  # Default to the year 2023 (index 63 for 2023 in this range)
        )

        if st.button("View Data"):
            if not crop:
                st.error("Please select a crop.")
                return

            # Update context with crop and year
            st.session_state.context.update({
                'crop': crop,
                'year': selected_year  # Add year to the context
            })
            st.session_state.stage = 'data_view'
            st.rerun()

    def show_data_view(self) -> None:
        """Displays the data view for the selected crop and metric."""
        context = st.session_state.context
        selected_year = context.get('year', 2023)

        try:
            with st.spinner(f"Fetching {context['crop']} data..."):
                df = self.data_manager.get_crop_data(
                    context['crop'],
                    context['metric'],
                    [selected_year]
                )

            if df is not None and not df.empty:
                st.markdown(f"### {context['crop']} - {context['metric']} ({selected_year})")
                
                unit = df['Unit'].iloc[0] if 'Unit' in df.columns else AgriculturalData.METRICS[context['metric']]['unit']
                df['Display Value'] = df['Value'].apply(lambda x: f"{x:,.2f} {unit}")
                
                display_df = df[['Year', 'Item', 'Element', 'Display Value']]
                if 'Flag_Description' in df.columns:
                    display_df['Status'] = df['Flag_Description']
                
                st.dataframe(display_df, use_container_width=True)
            
                stats = self.data_manager.get_data_summary(df)
                if stats:
                    col1, col2 = st.columns(2)
                    col1.metric(f"Value ({unit})", stats.get('Value', 'N/A'))
                    col2.metric("Status", stats.get('Status', 'N/A'))

                # Add tabs for different views
                tab1, tab2, tab3, tab4 = st.tabs(["Web Data", "Weather Analysis", "Price Predictions", "Historical Trends"])
                
                with tab1:
                    if st.button("Search Web for Additional Data"):
                        with st.spinner("Searching web for additional information..."):
                            web_data = self.web_search_manager.search_agricultural_data(
                                context['crop'],
                                context['metric'],
                                selected_year
                            )
                            
                            if web_data.get("summary"):
                                st.markdown("### Web Search Results")
                                st.markdown(web_data["summary"])
                                
                                if web_data.get("sources"):
                                    st.markdown("#### Sources")
                                    for source in web_data["sources"]:
                                        st.markdown(f"- {source}")
                            else:
                                st.warning("No additional information found from web sources.")
                
                with tab2:
                    self.show_weather_analysis()

                with tab3:
                    if st.button("Generate Price Predictions"):
                        with st.spinner("Generating predictions..."):
                            # Get historical data for the past 10 years
                            current_year = datetime.now().year
                            historical_years = list(range(current_year - 9, current_year + 1))
                            historical_df = self.data_manager.get_crop_data(
                                context['crop'],
                                context['metric'],
                                historical_years
                            )

                            if not historical_df.empty and len(historical_df) > 1:
                                predictions = self.price_prediction_agent.predict_prices(historical_df)

                                if 'error' not in predictions:
                                    # Display prediction summary
                                    st.markdown("### Prediction Summary")
                                    st.write(predictions['prediction_summary'])
                    
                                    # Display confidence score
                                    st.metric("Model Confidence", f"{predictions['confidence_score']}%")
                    
                                    # Display predictions in a table
                                    st.markdown("### Predicted Values")
                                    pred_df = pd.DataFrame(predictions['predictions'])
                                    pred_df.columns = ['Year', 'Predicted Value']
                                    st.dataframe(pred_df)
                    
                                    # Display market analysis
                                    st.markdown("### Market Analysis")
                                    market_factors = self.price_prediction_agent.get_market_factors(context['crop'])
                                    if market_factors['status'] == 'success':
                                        st.write(market_factors['market_analysis'])
                                else:
                                    st.error(f"Error generating predictions: {predictions.get('error', 'Unknown error')}")
                            else:
                                st.warning("Insufficient historical data for making predictions.")
                
                with tab4:
                    if st.button("Show Historical Trends"):
                        with st.spinner("Loading historical data..."):
                        # Fetch historical data for the last 10 years
                            current_year = datetime.now().year
                            years = list(range(current_year - 9, current_year + 1))
            
                            historical_df = self.data_manager.get_crop_data(
                                context['crop'],
                                context['metric'],
                                years
                            )
            
                            if not historical_df.empty and len(historical_df) > 1:
                                st.markdown("### Historical Trends")
                
                                # Create a proper time series dataframe
                                hist_display_df = historical_df.copy()
                                hist_display_df['Year'] = pd.to_numeric(hist_display_df['Year'])
                                hist_display_df = hist_display_df.sort_values('Year')
                
                                # Display summary statistics
                                st.markdown("#### Summary Statistics")
                                col1, col2, col3 = st.columns(3)
                
                                with col1:
                                    start_year = hist_display_df['Year'].min()
                                    end_year = hist_display_df['Year'].max()
                                    st.metric("Time Range", f"{start_year} - {end_year}")
                
                                with col2:
                                    avg_value = hist_display_df['Value'].mean()
                                    st.metric("Average", f"{avg_value:,.2f}")
                
                                with col3:
                                    max_value = hist_display_df['Value'].max()
                                    max_year = hist_display_df.loc[hist_display_df['Value'].idxmax(), 'Year']
                                    st.metric("Peak Value", f"{max_value:,.2f}", f"Year: {max_year}")
                
                                # Display the data table
                                st.markdown("#### Historical Data")
                                display_cols = ['Year', 'Value', 'Unit']
                                formatted_df = hist_display_df[display_cols].copy()
                                formatted_df['Value'] = formatted_df['Value'].apply(lambda x: f"{x:,.2f}")
                                st.dataframe(formatted_df, use_container_width=True)
                
                                # Create and display the line chart
                                st.markdown("#### Trend Visualization")
                                chart_data = pd.DataFrame({
                                    'Year': hist_display_df['Year'],
                                    f'{context["metric"]} ({hist_display_df["Unit"].iloc[0]})': hist_display_df['Value']
                                })
                
                                st.line_chart(
                                    chart_data.set_index('Year'),
                                    use_container_width=True
                                )
                
                                # Calculate and display trend analysis
                                first_value = hist_display_df['Value'].iloc[0]
                                last_value = hist_display_df['Value'].iloc[-1]
                                total_change = last_value - first_value
                                percent_change = (total_change / first_value) * 100 if first_value != 0 else 0
                
                                st.markdown("#### Trend Analysis")
                                st.metric(
                                    "Total Change",
                                    f"{total_change:,.2f}",
                                    f"{percent_change:,.2f}%",
                                    delta_color="normal" if percent_change >= 0 else "inverse"
                                )
                            else:
                                st.warning(
                                    "Insufficient historical data available. "
                                    "Please try a different crop or metric combination."
                                )
            else:
                st.warning(
                    f"No data available for {context['crop']} - {context['metric']} in {selected_year}. "
                    "This combination might not exist in our database. Please try a different combination."
                )
        except Exception as e:
            st.error("Error fetching data. Please try again.")
    
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start New Search"):
                st.session_state.stage = 'welcome'
                st.session_state.context = {}
                st.rerun()
        with col2:
            if st.button("Try Different Metric"):
                st.session_state.stage = 'metric_selection'
                st.session_state.context = {}
                st.rerun()
    def show_weather_analysis(self) -> None:
        """Display weather analysis UI in ChatbotUI class"""
        st.markdown("### 🌤️ Weather Analysis and Recommendations")

        # Get current context
        context = st.session_state.context
        crop = context.get('crop', '')

        # Location input
        location = st.text_input(
            "Enter your location (city, state, country)",
            placeholder="e.g., Mumbai, Maharashtra, India"
        )
    
        # Growth stage selection
        growth_stages = [
            "Seedling",
            "Vegetative",
            "Flowering",
            "Fruiting",
            "Maturity"
        ]
        growth_stage = st.selectbox(
            "Select crop growth stage",
            options=growth_stages
        )

        if st.button("Get Weather Analysis"):
            if not location:
                st.warning("Please enter a location")
                return
                
            with st.spinner("Analyzing weather conditions..."):
                result = self.weather_agent.get_weather_recommendations(
                    location=location,
                    crop=crop,
                    growth_stage=growth_stage
                )

                if 'error' in result:
                    st.error(f"Error: {result['error']}")
                    return

                # Display current conditions
                st.markdown("#### Current Weather Conditions")
                current = result['weather_data']['current_conditions']
                col1, col2, col3 = st.columns(3)
            
                with col1:
                    st.metric("Temperature", f"{current['temperature']}°C")
                with col2:
                    st.metric("Humidity", f"{current['humidity']}%")
                with col3:
                    st.metric("Wind Speed", f"{current['wind_speed']} m/s")

                # Display forecast
                st.markdown("#### 7-Day Forecast")
                forecast_df = pd.DataFrame(result['weather_data']['forecast']['daily'])
                st.dataframe(
                    forecast_df[[
                        'date', 'temp_min', 'temp_max',
                        'humidity', 'description', 'rain_prob'
                    ]],
                    use_container_width=True
                )

                # Display AI recommendations
                st.markdown("#### 🤖 AI Recommendations")
                st.markdown(result['recommendations'])

                # # Add to ChatbotUI class initialization
                # self.weather_agent = WeatherAnalysisAgent(cache_manager, openai)

def main():
    st.set_page_config(
        page_title="Agricultural Data Assistant",
        page_icon="🌾",
        layout="wide"
    )

    if 'stage' not in st.session_state:
        st.session_state.stage = 'welcome'
        st.session_state.context = {}
        st.session_state.language = 'en'

    # Redis Cloud configuration
    redis_config = {
        'host': 'redis-19245.c8.us-east-1-4.ec2.redns.redis-cloud.com',
        'port': 19245,
        'username': 'default',
        'password': os.getenv("REDIS_PASSWORD")  # Replace with your actual password
    }

    try:
        astra_client = AstraDB(
            token=constant.ASTRA_DB_TOKEN,
            api_endpoint=constant.ASTRA_DB_ENDPOINT
        )
        
        cache_manager = CacheManager(redis_config)
        data_manager = DataManager(astra_client, cache_manager)
        prompt_manager = PromptManager(openai, cache_manager)
        web_search_manager = WebSearchManager(openai)
        price_prediction_agent = PricePredictionAgent(cache_manager)
        weather_agent = WeatherAnalysisAgent(cache_manager, openai) 
        
        chatbot_ui = ChatbotUI(
            data_manager,
            prompt_manager,
            cache_manager,
            web_search_manager,
            price_prediction_agent,
            weather_agent
        )

        if st.session_state.stage == 'welcome':
            chatbot_ui.show_welcome()
        elif st.session_state.stage == 'metric_selection':
            chatbot_ui.show_metric_selection()
        elif st.session_state.stage == 'crop_selection':
            chatbot_ui.show_crop_selection()
        elif st.session_state.stage == 'data_view':
            chatbot_ui.show_data_view()

    except Exception as e:
        st.error(f"An error occurred: {str(e)}. Please try again.")
        
    if st.button("Restart Application"):
        st.session_state.stage = 'welcome'
        st.session_state.context = {}
        st.rerun()

if __name__ == "__main__":
    main()
