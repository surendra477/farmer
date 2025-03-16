import streamlit as st
import openai
import constant
import time

# Set OpenAI API key (for OpenAI 0.28)
openai.api_key = constant.OPENAI_KEY

# Page configuration
st.set_page_config(
    page_title="FarmAssist - AI Agricultural Assistant",
    page_icon="🌱",
    layout="wide"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-family: 'Trebuchet MS', sans-serif;
        color: #2E7D32;
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
    }
    .chat-container {
        border-radius: 15px;
        padding: 20px;
        background-color: #f1f8e9;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown("<h1 class='main-header'>🌱 FarmAssist: Your AI Agricultural Assistant</h1>", unsafe_allow_html=True)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "conversation_started" not in st.session_state:
    st.session_state.conversation_started = False

# Sidebar for settings and information
with st.sidebar:
    st.header("Settings")
    
    # Language preference
    language_options = ["Automatic", "English", "Hindi", "Hinglish", "Other"]
    preferred_language = st.selectbox("Preferred Language", language_options)
    
    # Model selection
    model_options = {
        "GPT-4o mini": "gpt-4o-mini",
        "GPT-4": "gpt-4-turbo",
        "GPT-3.5 Turbo": "gpt-3.5-turbo",
    }
    selected_model = st.selectbox("AI Model", list(model_options.keys()))
    
    # Response settings
    temperature = st.slider("Creativity (Temperature)", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    max_tokens = st.slider("Max Response Length", min_value=100, max_value=4000, value=500, step=100)
    
    # Location information (optional)
    st.subheader("Location (Optional)")
    region = st.text_input("Region/State")
    
    # Reset conversation
    if st.button("Reset Conversation"):
        st.session_state.messages = []
        st.session_state.conversation_started = False
        st.success("Conversation reset!")

# System prompt for chatbot
system_prompt = """
You are FarmAssist, an expert AI agricultural assistant designed to support farmers worldwide. Your purpose is to provide accurate, practical, and localized farming advice across all aspects of agriculture.

CORE CAPABILITIES:
- Provide expert guidance on crop selection, planting techniques, cultivation practices, harvesting methods, and post-harvest management
- Offer detailed information about soil health, fertilization strategies, crop rotation, and sustainable farming practices
- Deliver pest and disease identification with appropriate management solutions
- Share weather interpretation and agricultural forecasting insights
- Explain irrigation systems, water management, and conservation techniques
- Assist with livestock management, feeding practices, and animal health
- Guide farmers on agricultural equipment selection, usage, and maintenance
- Explain modern farming technologies and precision agriculture techniques
- Provide information on organic farming, permaculture, and regenerative agriculture
- Offer guidance on farm business management, marketing, and certification processes

LANGUAGE HANDLING:
- ALWAYS respond in the same language the user is using in their query
- For Hindi queries, respond completely in Hindi using Devanagari script
- For Hinglish or mixed Hindi-English queries, respond in the same Hinglish style, matching the user's blend of languages
- Support all major world languages, including regional Indian languages
- Use agricultural terminology that matches the local language and dialect
- Maintain consistent language throughout the entire response
- If uncertain about specific agricultural terms in a language, provide both the local term and a description
- When appropriate, include common local/colloquial farming terms from the user's region

INTERACTION GUIDELINES:
- Communicate in the user's preferred language, supporting all major world languages
- Adapt recommendations to the user's geographic location, climate zone, and local conditions
- Consider seasonal context in all advice (planting seasons, monsoons, dry periods, etc.)
- Provide solutions that respect the farmer's resource constraints (water availability, equipment access, etc.)
- Balance traditional knowledge with modern scientific techniques
- Present information in clear, accessible language avoiding unnecessary jargon
- Offer specific, actionable advice rather than vague generalizations
- When appropriate, include numerical data (quantities, timing, measurements) in recommendations
- Recognize and address both small-scale subsistence farming and commercial agricultural operations
- Acknowledge indigenous and traditional farming practices where relevant

LIMITATIONS AND SAFEGUARDS:
- Clarify when advice needs to be adapted to local conditions that you may not have complete information about
- Recommend consulting local agricultural extension services for region-specific guidance when appropriate
- Avoid recommending harmful or illegal agricultural practices
- Acknowledge uncertainty when information is incomplete
- Prioritize environmentally sustainable and economically viable farming approaches

FORMAT FLEXIBILITY:
- Adapt output format based on user needs (concise tips, detailed explanations, step-by-step guides)
- Provide visual descriptions of plants, diseases, or techniques when it adds clarity
- Use numbered lists for sequential processes and bullet points for options or alternatives
- Include both metric and imperial measurements when providing specific recommendations
"""

# Add location context if provided
if region:
    system_prompt += f"\nCONTEXT: The farmer is located in {region}. Adapt advice to this region's climate, soil conditions, and agricultural practices when possible."

# Add language preference if not automatic
if preferred_language != "Automatic":
    system_prompt += f"\nLANGUAGE PREFERENCE: The farmer prefers communication in {preferred_language}."

# Set system message if messages list is empty
if not st.session_state.messages:
    st.session_state.messages = [{"role": "system", "content": system_prompt}]
else:
    # Update system message if it exists and settings changed
    if st.session_state.messages[0]["role"] == "system":
        st.session_state.messages[0]["content"] = system_prompt
    else:
        # Insert system message at the beginning if it doesn't exist
        st.session_state.messages.insert(0, {"role": "system", "content": system_prompt})

# Welcome message - only show once when conversation starts
if not st.session_state.conversation_started:
    st.session_state.conversation_started = True
    
    # Add assistant welcome message based on language preference
    welcome_message = ""
    if preferred_language == "Hindi":
        welcome_message = "नमस्ते! मैं FarmAssist हूँ, आपका कृषि सहायक। आप मुझसे खेती, फसलों, मिट्टी, या मौसम के बारे में कुछ भी पूछ सकते हैं। मैं आपकी कैसे मदद कर सकता हूँ?"
    elif preferred_language == "Hinglish":
        welcome_message = "Namaste! Main FarmAssist hoon, aapka agriculture assistant. Aap mujhse farming, crops, soil, ya weather ke baare mein kuch bhi pooch sakte hain. Main aapki kaise help kar sakta hoon?"
    else:  # Default to English
        welcome_message = "Hello! I'm FarmAssist, your agricultural assistant. You can ask me anything about farming, crops, soil, or weather. How can I help you today?"
    
    # Add welcome message to messages list
    st.session_state.messages.append({"role": "assistant", "content": welcome_message})

# Main chat container
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

# Display chat history (skip system message)
for msg in st.session_state.messages:
    if msg["role"] != "system":  # Don't show system message
        st.chat_message(msg["role"]).write(msg["content"])

# User input
user_input = st.chat_input("Ask me anything about agriculture...")

# Handle user input
if user_input:
    # Store user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)
    
    # Add a spinner while waiting for response
    with st.spinner("FarmAssist is thinking..."):
        try:
            # Call OpenAI API with OpenAI 0.28 format
            response = openai.ChatCompletion.create(
                model=model_options[selected_model],
                messages=st.session_state.messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Get assistant response (OpenAI 0.28 format)
            assistant_response = response["choices"][0]["message"]["content"]
            
            # Store assistant response
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            
            # Display assistant message
            st.chat_message("assistant").write(assistant_response)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            # Log the error
            print(f"Error: {str(e)}")

st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("FarmAssist is an AI-powered agricultural assistant designed to support farmers with accurate, practical advice.")
st.caption("© 2025 FarmAssist | This is a demo application. Always consult with local agricultural experts for advice specific to your region.")